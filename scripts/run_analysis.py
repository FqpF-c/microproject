#!/usr/bin/env python3
"""
Command-line script for analyzing individual APK files
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.apk_analyzer import APKFeatureExtractor
from src.feature_extraction.data_preprocessor import DataPreprocessor
from src.models.ca_lstm import ChannelAttentionLSTM
from src.explainability.explainer import FeatureExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path: str, config: dict) -> tuple:
    """Load trained model and preprocessor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model checkpoint
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', config.get('model', {}))

    # Create and load model
    model = ChannelAttentionLSTM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Load preprocessor
    preprocessor_path = os.path.dirname(model_path).replace('checkpoints', 'processed')
    if not os.path.exists(preprocessor_path):
        preprocessor_path = 'data/processed'

    preprocessor = DataPreprocessor(config.get('data', {}))
    try:
        preprocessor.load_preprocessor(preprocessor_path)
        logger.info("Preprocessor loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load preprocessor: {e}")
        preprocessor = None

    logger.info("Model loaded successfully")
    return model, preprocessor

def analyze_apk(apk_path: str, model, preprocessor, feature_extractor, explainer) -> dict:
    """Analyze a single APK file."""
    logger.info(f"Analyzing APK: {apk_path}")

    # Extract features
    features = feature_extractor.extract_all_features(apk_path)
    if not features:
        logger.error("Failed to extract features")
        return None

    # Preprocess features
    if preprocessor is None:
        logger.error("Preprocessor not available")
        return None

    try:
        features_array = preprocessor.transform_single_sample(features)
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.error(f"Failed to preprocess features: {e}")
        return None

    # Get prediction
    device = next(model.parameters()).device
    features_tensor = features_tensor.to(device)

    with torch.no_grad():
        logits, attention_weights = model(features_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Convert to numpy
    probs = probabilities.cpu().numpy()[0]
    attention = attention_weights.cpu().numpy()[0]

    # Get risk level
    risk_labels = ['Low', 'Medium', 'High']
    predicted_class = np.argmax(probs)
    risk_level = risk_labels[predicted_class]

    # Create prediction result
    prediction_result = {
        'risk_level': risk_level,
        'confidence': float(probs[predicted_class]),
        'probabilities': {
            'Low': float(probs[0]),
            'Medium': float(probs[1]),
            'High': float(probs[2])
        },
        'feature_importance': attention.tolist(),
    }

    # Generate explanation
    if explainer:
        try:
            explanation = explainer.explain_prediction(
                risk_level=risk_level,
                probabilities=probs,
                feature_importance=attention,
                feature_names=preprocessor.feature_names,
                feature_values=features
            )
            prediction_result['explanation'] = explanation
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")

    # Create analysis result
    result = {
        'file_info': {
            'filename': os.path.basename(apk_path),
            'filepath': apk_path,
            'file_size': os.path.getsize(apk_path),
            'analysis_time': datetime.now().isoformat()
        },
        'features': features,
        'feature_count': len(features),
        'prediction': prediction_result,
        'model_info': {
            'model_type': 'Channel Attention LSTM',
            'device': str(device)
        }
    }

    return result

def print_analysis_summary(result: dict):
    """Print a formatted summary of the analysis."""
    if not result:
        return

    print("\n" + "="*60)
    print("APK MALWARE ANALYSIS REPORT")
    print("="*60)

    # File information
    file_info = result['file_info']
    print(f"File: {file_info['filename']}")
    print(f"Size: {file_info['file_size']:,} bytes")
    print(f"Analysis Time: {file_info['analysis_time']}")

    # Risk assessment
    prediction = result['prediction']
    risk_level = prediction['risk_level']
    confidence = prediction['confidence']

    print(f"\nRISK ASSESSMENT:")
    print(f"Risk Level: {risk_level}")
    print(f"Confidence: {confidence:.1%}")

    # Risk level styling
    risk_colors = {
        'Low': '🟢',
        'Medium': '🟡',
        'High': '🔴'
    }
    print(f"Status: {risk_colors.get(risk_level, '⚫')} {risk_level} Risk")

    # Probabilities
    print(f"\nRisk Probabilities:")
    for risk, prob in prediction['probabilities'].items():
        print(f"  {risk}: {prob:.1%}")

    # Feature summary
    features = result['features']
    print(f"\nFEATURE SUMMARY:")
    print(f"Total Features Extracted: {result['feature_count']}")
    print(f"Dangerous Permissions: {features.get('dangerous_permissions_count', 0)}")
    print(f"Suspicious APIs: {features.get('suspicious_api_count', 0)}")
    print(f"Suspicious Strings: {features.get('suspicious_strings', 0)}")

    # Explanation
    if 'explanation' in prediction:
        explanation = prediction['explanation']
        print(f"\nAI EXPLANATION:")
        print(f"{explanation.get('explanations', {}).get('primary', 'No explanation available')}")

        # Top features
        top_features = explanation.get('top_features', [])[:5]
        if top_features:
            print(f"\nTOP CONTRIBUTING FEATURES:")
            for i, feature in enumerate(top_features, 1):
                print(f"  {i}. {feature['description']} (importance: {feature['importance']:.3f})")

        # Recommendations
        recommendations = explanation.get('recommendations', [])
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze Android APK for malware detection')
    parser.add_argument('apk_path', type=str, help='Path to APK file to analyze')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, help='Output file for detailed results (JSON)')
    parser.add_argument('--quiet', action='store_true', help='Only show final result')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Validate APK file
    if not os.path.exists(args.apk_path):
        logger.error(f"APK file not found: {args.apk_path}")
        return

    if not args.apk_path.lower().endswith('.apk'):
        logger.warning("File does not have .apk extension")

    # Load configuration
    config = load_config(args.config)

    # Initialize components
    logger.info("Initializing analysis components...")

    # Feature extractor
    feature_config = config.get('feature_extraction', {})
    feature_extractor = APKFeatureExtractor(feature_config)

    # Load model and preprocessor
    model, preprocessor = load_model(args.model, config)
    if model is None or preprocessor is None:
        logger.error("Failed to load model or preprocessor")
        return

    # Explainer
    explainer_config = config.get('explainability', {})
    explainer = FeatureExplainer(explainer_config)

    # Analyze APK
    result = analyze_apk(args.apk_path, model, preprocessor, feature_extractor, explainer)

    if result is None:
        logger.error("Analysis failed")
        return

    # Print summary
    if not args.quiet:
        print_analysis_summary(result)
    else:
        # Just print the risk level for quiet mode
        print(result['prediction']['risk_level'])

    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {args.output}")

if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues
    main()