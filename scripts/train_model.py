#!/usr/bin/env python3
"""
Main training script for Android Malware AI Risk Analysis V2
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.data_preprocessor import DataPreprocessor
from src.models.ca_lstm import ChannelAttentionLSTM, DroidetecBaseline
from src.training.trainer import ModelTrainer
from src.training.equilibrium_optimizer import HyperparameterTuner
from src.training.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config: dict) -> tuple:
    """Prepare and preprocess data for training."""
    logger.info("Preparing data...")

    data_config = config.get('data', {})
    preprocessor = DataPreprocessor(data_config)

    # Load and preprocess dataset
    features_path = os.path.join(
        data_config.get('processed_data_path', 'data/processed'),
        data_config.get('features_file', 'features.csv')
    )

    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        logger.info("Please run feature extraction first using scripts/extract_features.py")
        return None

    # Preprocess dataset
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_dataset(
        features_path=features_path,
        output_dir=data_config.get('processed_data_path', 'data/processed')
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def train_ca_lstm(config: dict, data: tuple) -> torch.nn.Module:
    """Train Channel Attention LSTM model."""
    logger.info("Training CA-LSTM model...")

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data

    # Update config with actual input dimension
    model_config = config.get('model', {}).copy()
    model_config['input_dim'] = X_train.shape[1]
    model_config['class_weights'] = preprocessor.class_weights

    # Create model
    model = ChannelAttentionLSTM(model_config)

    # Setup training configuration
    training_config = config.get('training', {}).copy()
    training_config.update(model_config)
    training_config['model_save_path'] = 'models/checkpoints'
    training_config['log_save_path'] = 'models/logs'

    # Create trainer
    trainer = ModelTrainer(model, training_config)

    # Train model
    training_history = trainer.train(X_train, y_train, X_val, y_val)

    logger.info("CA-LSTM training completed!")
    logger.info(f"Best {trainer.monitor}: {training_history['best_metric']:.4f}")

    return model, training_history

def train_droidetec_baseline(config: dict, data: tuple) -> torch.nn.Module:
    """Train Droidetec baseline model."""
    logger.info("Training Droidetec baseline...")

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data

    # Create baseline config
    baseline_config = {
        'input_dim': X_train.shape[1],
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout_rate': 0.3,
        'num_classes': config.get('model', {}).get('num_classes', 3),
        'class_weights': preprocessor.class_weights
    }

    # Create model
    model = DroidetecBaseline(baseline_config)

    # Setup training configuration
    training_config = config.get('training', {}).copy()
    training_config.update(baseline_config)
    training_config['model_save_path'] = 'models/checkpoints'
    training_config['log_save_path'] = 'models/logs'
    training_config['epochs'] = min(training_config.get('epochs', 50), 30)  # Shorter training for baseline

    # Create trainer
    trainer = ModelTrainer(model, training_config)

    # Train model
    training_history = trainer.train(X_train, y_train, X_val, y_val)

    logger.info("Droidetec training completed!")
    logger.info(f"Best {trainer.monitor}: {training_history['best_metric']:.4f}")

    return model, training_history

def hyperparameter_optimization(config: dict, data: tuple) -> dict:
    """Perform hyperparameter optimization using Equilibrium Optimization."""
    if not config.get('optimization', {}).get('enable', False):
        logger.info("Hyperparameter optimization disabled")
        return config.get('model', {})

    logger.info("Starting hyperparameter optimization...")

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data

    # Create tuner
    base_config = config.get('model', {}).copy()
    base_config['input_dim'] = X_train.shape[1]

    tuner = HyperparameterTuner(ChannelAttentionLSTM, config)

    # Run optimization
    best_hyperparams = tuner.tune(X_train, y_train, X_val, y_val)

    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best hyperparameters: {best_hyperparams}")

    return best_hyperparams

def evaluate_models(config: dict, models: dict, data: tuple):
    """Evaluate and compare all models."""
    logger.info("Starting model evaluation...")

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data

    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Compare models
    comparison_results = evaluator.compare_models(models, X_test, y_test)

    # Save results
    os.makedirs('models/evaluation', exist_ok=True)
    evaluator.save_evaluation_report('models/evaluation/evaluation_report.json')
    evaluator.generate_markdown_report('models/evaluation/evaluation_report.md')

    logger.info("Model evaluation completed!")
    logger.info("Results saved to models/evaluation/")

    return comparison_results

def main():
    parser = argparse.ArgumentParser(description='Train Android Malware Detection Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true',
                       help='Enable hyperparameter optimization')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip training baseline model')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing models')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Enable optimization if requested
    if args.optimize:
        config['optimization']['enable'] = True

    logger.info("Starting Android Malware AI V2 Training Pipeline")
    logger.info(f"Configuration: {args.config}")

    # Prepare data
    data = prepare_data(config)
    if data is None:
        return

    trained_models = {}

    if not args.evaluate_only:
        # Hyperparameter optimization (optional)
        if config.get('optimization', {}).get('enable', False):
            best_hyperparams = hyperparameter_optimization(config, data)
            config['model'].update(best_hyperparams)

        # Train CA-LSTM model
        ca_lstm_model, ca_lstm_history = train_ca_lstm(config, data)
        trained_models['CA-LSTM'] = ca_lstm_model

        # Train baseline model (optional)
        if not args.skip_baseline:
            droidetec_model, droidetec_history = train_droidetec_baseline(config, data)
            trained_models['Droidetec_BiLSTM'] = droidetec_model

    else:
        # Load existing models for evaluation
        logger.info("Loading existing models for evaluation...")

        # Try to load CA-LSTM
        ca_lstm_path = 'models/checkpoints/best_model.pth'
        if os.path.exists(ca_lstm_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(ca_lstm_path, map_location=device)

            model_config = checkpoint.get('config', config.get('model', {}))
            model_config['input_dim'] = data[0].shape[1]  # X_train shape

            ca_lstm_model = ChannelAttentionLSTM(model_config)
            ca_lstm_model.load_state_dict(checkpoint['model_state_dict'])
            trained_models['CA-LSTM'] = ca_lstm_model

            logger.info("CA-LSTM model loaded successfully")
        else:
            logger.warning("CA-LSTM model not found")

    # Evaluate models
    if trained_models:
        evaluation_results = evaluate_models(config, trained_models, data)

        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)

        if 'summary' in evaluation_results:
            best_model = evaluation_results['summary'].get('best_overall', 'Unknown')
            logger.info(f"Best Overall Model: {best_model}")

        logger.info("Check models/evaluation/ for detailed results")
        logger.info("Check models/logs/ for training logs")
        logger.info("Check models/checkpoints/ for saved models")

    else:
        logger.error("No models were trained or loaded for evaluation")

if __name__ == "__main__":
    main()