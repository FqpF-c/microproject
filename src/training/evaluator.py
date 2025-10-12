import os
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

from src.models.ca_lstm import ChannelAttentionLSTM, DroidetecBaseline
from src.training.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison system.
    Compares CA-LSTM with Droidetec baseline and other models.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.comparison_results = {}

    def evaluate_model(self, model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for identification

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")

        model.eval()
        model = model.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        start_time = time.time()

        # Get predictions
        with torch.no_grad():
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                # Model returns multiple outputs (like CA-LSTM)
                logits, attention_weights = model(X_test)
            else:
                logits = model(X_test)
                attention_weights = None

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        inference_time = time.time() - start_time

        # Convert to numpy
        y_true = y_test.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_proba = probabilities.cpu().numpy()

        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(y_true, y_pred, y_proba)

        # Add timing information
        metrics['inference_time'] = inference_time
        metrics['samples_per_second'] = len(X_test) / inference_time
        metrics['avg_time_per_sample'] = inference_time / len(X_test)

        # Add model-specific information
        if attention_weights is not None:
            metrics['has_attention'] = True
            metrics['attention_entropy'] = self._calculate_attention_entropy(attention_weights)
        else:
            metrics['has_attention'] = False

        # Store results
        self.results[model_name] = metrics

        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")

        return metrics

    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        class_names = ['Low', 'Medium', 'High']
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
                metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]

        # AUC metrics (multi-class)
        try:
            metrics['auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Additional metrics
        metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
        metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(y_true, y_pred)

        return metrics

    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention weights to measure attention diversity."""
        attention_np = attention_weights.cpu().numpy()
        # Normalize attention weights
        attention_norm = attention_np / (np.sum(attention_np, axis=1, keepdims=True) + 1e-8)
        # Calculate entropy
        entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-8), axis=1)
        return float(np.mean(entropy))

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average specificity across classes."""
        cm = confusion_matrix(y_true, y_pred)
        specificities = []

        for i in range(len(cm)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        return np.mean(specificities)

    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        return np.mean(recall_per_class)

    def compare_models(self, models: Dict[str, nn.Module], X_test: torch.Tensor,
                      y_test: torch.Tensor) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels

        Returns:
            Comparison results
        """
        logger.info("Starting model comparison...")

        # Evaluate each model
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)

        # Create comparison
        comparison = self._create_comparison_report()

        # Create visualization
        self._create_comparison_plots()

        logger.info("Model comparison completed")

        return comparison

    def _create_comparison_report(self) -> Dict[str, Any]:
        """Create a comprehensive comparison report."""
        if len(self.results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}

        # Key metrics for comparison
        key_metrics = [
            'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
            'auc_weighted', 'balanced_accuracy', 'specificity'
        ]

        # Performance metrics for comparison
        performance_metrics = [
            'inference_time', 'samples_per_second', 'avg_time_per_sample'
        ]

        comparison = {
            'summary': {},
            'detailed_metrics': {},
            'performance': {},
            'ranking': {},
            'improvements': {}
        }

        # Extract metrics for comparison
        for metric in key_metrics:
            comparison['detailed_metrics'][metric] = {}
            for model_name, results in self.results.items():
                comparison['detailed_metrics'][metric][model_name] = results.get(metric, 0)

        # Performance comparison
        for metric in performance_metrics:
            comparison['performance'][metric] = {}
            for model_name, results in self.results.items():
                comparison['performance'][metric][model_name] = results.get(metric, 0)

        # Create rankings
        for metric in key_metrics:
            ranking = sorted(
                self.results.items(),
                key=lambda x: x[1].get(metric, 0),
                reverse=True
            )
            comparison['ranking'][metric] = [model_name for model_name, _ in ranking]

        # Calculate improvements (assuming first model is baseline)
        if len(self.results) >= 2:
            model_names = list(self.results.keys())
            baseline = model_names[0]  # Typically Droidetec
            best_model = model_names[1]  # Typically CA-LSTM

            for metric in key_metrics:
                baseline_score = self.results[baseline].get(metric, 0)
                best_score = self.results[best_model].get(metric, 0)
                if baseline_score > 0:
                    improvement = ((best_score - baseline_score) / baseline_score) * 100
                    comparison['improvements'][metric] = {
                        'baseline_score': baseline_score,
                        'best_score': best_score,
                        'improvement_percent': improvement
                    }

        # Summary statistics
        comparison['summary'] = {
            'total_models': len(self.results),
            'best_overall': self._get_best_overall_model(),
            'evaluation_date': datetime.now().isoformat()
        }

        self.comparison_results = comparison
        return comparison

    def _get_best_overall_model(self) -> str:
        """Determine the best overall model based on weighted criteria."""
        if not self.results:
            return "None"

        # Weighted scoring
        weights = {
            'accuracy': 0.25,
            'f1_weighted': 0.25,
            'precision_weighted': 0.15,
            'recall_weighted': 0.15,
            'auc_weighted': 0.20
        }

        scores = {}
        for model_name, results in self.results.items():
            score = 0
            for metric, weight in weights.items():
                score += results.get(metric, 0) * weight
            scores[model_name] = score

        best_model = max(scores.items(), key=lambda x: x[1])
        return best_model[0]

    def _create_comparison_plots(self):
        """Create visualization plots for model comparison."""
        if len(self.results) < 2:
            return

        # Create plots directory
        plots_dir = 'models/evaluation_plots'
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Metrics comparison bar plot
        self._plot_metrics_comparison(plots_dir)

        # 2. Confusion matrices
        self._plot_confusion_matrices(plots_dir)

        # 3. Performance comparison
        self._plot_performance_comparison(plots_dir)

        # 4. Radar chart for comprehensive comparison
        self._plot_radar_comparison(plots_dir)

    def _plot_metrics_comparison(self, plots_dir: str):
        """Plot bar chart comparing key metrics."""
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        model_names = list(self.results.keys())

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(metrics))
        width = 0.35

        for i, model_name in enumerate(model_names):
            values = [self.results[model_name].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, plots_dir: str):
        """Plot confusion matrices for all models."""
        model_names = list(self.results.keys())
        n_models = len(model_names)

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        class_names = ['Low', 'Medium', 'High']

        for i, model_name in enumerate(model_names):
            cm = np.array(self.results[model_name]['confusion_matrix'])

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[i]
            )
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, plots_dir: str):
        """Plot performance metrics comparison."""
        metrics = ['inference_time', 'samples_per_second']
        model_names = list(self.results.keys())

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, metric in enumerate(metrics):
            values = [self.results[model_name].get(metric, 0) for model_name in model_names]
            bars = axes[i].bar(model_names, values, alpha=0.8)

            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Time (seconds)' if 'time' in metric else 'Samples/second')
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_radar_comparison(self, plots_dir: str):
        """Create radar chart for comprehensive model comparison."""
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'auc_weighted']
        model_names = list(self.results.keys())

        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for model_name in model_names:
            values = [self.results[model_name].get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Model Comparison\n(Radar Chart)', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_evaluation_report(self, filepath: str):
        """Save comprehensive evaluation report."""
        report = {
            'evaluation_results': self.results,
            'comparison_results': self.comparison_results,
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'device': str(self.device),
                'config': self.config
            }
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {filepath}")

    def generate_markdown_report(self, filepath: str):
        """Generate a markdown report for easy reading."""
        if not self.comparison_results:
            logger.warning("No comparison results available")
            return

        markdown_content = self._create_markdown_content()

        with open(filepath, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Markdown report saved to {filepath}")

    def _create_markdown_content(self) -> str:
        """Create markdown content for the report."""
        content = []

        # Title and summary
        content.append("# Android Malware Detection - Model Evaluation Report\n")
        content.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content.append(f"**Models Evaluated:** {len(self.results)}\n")
        content.append(f"**Best Overall Model:** {self.comparison_results.get('summary', {}).get('best_overall', 'N/A')}\n\n")

        # Key metrics comparison
        content.append("## Performance Comparison\n")
        content.append("| Metric | " + " | ".join(self.results.keys()) + " |\n")
        content.append("|" + "---|" * (len(self.results) + 1) + "\n")

        key_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        for metric in key_metrics:
            row = [metric.replace('_', ' ').title()]
            for model_name in self.results.keys():
                value = self.results[model_name].get(metric, 0)
                row.append(f"{value:.4f}")
            content.append("| " + " | ".join(row) + " |\n")

        # Improvements section
        if 'improvements' in self.comparison_results:
            content.append("\n## Improvements Over Baseline\n")
            for metric, improvement_data in self.comparison_results['improvements'].items():
                content.append(f"**{metric.replace('_', ' ').title()}:**\n")
                content.append(f"- Baseline: {improvement_data['baseline_score']:.4f}\n")
                content.append(f"- Best: {improvement_data['best_score']:.4f}\n")
                content.append(f"- Improvement: {improvement_data['improvement_percent']:.2f}%\n\n")

        # Detailed results
        content.append("## Detailed Results\n")
        for model_name, results in self.results.items():
            content.append(f"### {model_name}\n")
            content.append(f"- **Accuracy:** {results.get('accuracy', 0):.4f}\n")
            content.append(f"- **F1-Score:** {results.get('f1_weighted', 0):.4f}\n")
            content.append(f"- **Precision:** {results.get('precision_weighted', 0):.4f}\n")
            content.append(f"- **Recall:** {results.get('recall_weighted', 0):.4f}\n")
            content.append(f"- **AUC:** {results.get('auc_weighted', 0):.4f}\n")
            content.append(f"- **Inference Time:** {results.get('inference_time', 0):.4f}s\n")
            content.append(f"- **Samples/Second:** {results.get('samples_per_second', 0):.2f}\n\n")

        return "".join(content)


if __name__ == "__main__":
    # Example usage
    from src.models.ca_lstm import ChannelAttentionLSTM, DroidetecBaseline

    config = {
        'input_dim': 150,
        'hidden_units': [128, 64, 32],
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'num_classes': 3
    }

    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Create models
    ca_lstm = ChannelAttentionLSTM(config)
    droidetec = DroidetecBaseline(config)

    models = {
        'Droidetec_BiLSTM': droidetec,
        'CA-LSTM': ca_lstm
    }

    # Create dummy test data
    X_test = torch.randn(500, 150)
    y_test = torch.randint(0, 3, (500,))

    # Compare models
    # comparison_results = evaluator.compare_models(models, X_test, y_test)

    # Save reports
    # evaluator.save_evaluation_report('models/evaluation_report.json')
    # evaluator.generate_markdown_report('models/evaluation_report.md')