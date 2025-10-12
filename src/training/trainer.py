import os
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track training and validation metrics across epochs."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.learning_rates = []
        self.epoch_times = []

    def update(self, epoch_metrics: Dict[str, float]):
        """Update metrics for current epoch."""
        self.train_losses.append(epoch_metrics.get('train_loss', 0))
        self.val_losses.append(epoch_metrics.get('val_loss', 0))
        self.train_accuracies.append(epoch_metrics.get('train_accuracy', 0))
        self.val_accuracies.append(epoch_metrics.get('val_accuracy', 0))
        self.train_f1_scores.append(epoch_metrics.get('train_f1', 0))
        self.val_f1_scores.append(epoch_metrics.get('val_f1', 0))
        self.train_precisions.append(epoch_metrics.get('train_precision', 0))
        self.val_precisions.append(epoch_metrics.get('val_precision', 0))
        self.train_recalls.append(epoch_metrics.get('train_recall', 0))
        self.val_recalls.append(epoch_metrics.get('val_recall', 0))
        self.learning_rates.append(epoch_metrics.get('learning_rate', 0))
        self.epoch_times.append(epoch_metrics.get('epoch_time', 0))

    def get_best_epoch(self, metric: str = 'val_f1') -> int:
        """Get the epoch with the best performance for a given metric."""
        if metric == 'val_f1':
            return np.argmax(self.val_f1_scores)
        elif metric == 'val_accuracy':
            return np.argmax(self.val_accuracies)
        elif metric == 'val_loss':
            return np.argmin(self.val_losses)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        metrics_dict = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_f1_scores': self.train_f1_scores,
            'val_f1_scores': self.val_f1_scores,
            'train_precisions': self.train_precisions,
            'val_precisions': self.val_precisions,
            'train_recalls': self.train_recalls,
            'val_recalls': self.val_recalls,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)


class ModelTrainer:
    """
    Advanced model trainer with multi-epoch support, monitoring, and optimization.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training configuration
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.patience = config.get('patience', 10)
        self.monitor = config.get('monitor', 'val_f1_score')
        self.save_best_only = config.get('save_best_only', True)

        # Learning rate scheduler config
        self.reduce_lr_patience = config.get('reduce_lr_patience', 5)
        self.reduce_lr_factor = config.get('reduce_lr_factor', 0.5)
        self.min_lr = config.get('min_lr', 1e-7)

        # Paths
        self.model_save_path = config.get('model_save_path', 'models/checkpoints')
        self.log_save_path = config.get('log_save_path', 'models/logs')

        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics_tracker = MetricsTracker()
        self.class_weights = config.get('class_weights', None)

        # Move model to device
        self.model = self.model.to(self.device)

        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_save_path, exist_ok=True)

        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get('l2_regularization', 0.001)
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max' if 'f1' in self.monitor else 'min',
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr
        )

        # Loss function with class weights
        if self.class_weights is not None and len(self.class_weights) == 3:
            weights = torch.tensor(list(self.class_weights.values()), dtype=torch.float)
            weights = weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # AUC for multi-class
        if y_proba is not None and len(np.unique(y_true)) > 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                metrics['auc'] = 0.0
        elif y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                metrics['auc'] = 0.0

        return metrics

    def _train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_probabilities = []
        all_targets = []

        n_samples = len(X_train)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)

            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            self.optimizer.zero_grad()

            # Forward pass
            try:
                # Try CA-LSTM style first (returns outputs, attention)
                outputs, _ = self.model(batch_X)
            except ValueError:
                # Fallback for models that only return outputs (like Droidetec)
                outputs = self.model(batch_X)

            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Store predictions for metrics
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / n_batches

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)

        return avg_loss, metrics

    def _validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Validate model for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_probabilities = []
        all_targets = []

        n_samples = len(X_val)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                batch_X = X_val[start_idx:end_idx]
                batch_y = y_val[start_idx:end_idx]

                # Forward pass
                try:
                    # Try CA-LSTM style first (returns outputs, attention)
                    outputs, _ = self.model(batch_X)
                except ValueError:
                    # Fallback for models that only return outputs (like Droidetec)
                    outputs = self.model(batch_X)

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                # Store predictions
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.detach().cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())
                all_targets.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / n_batches

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)

        return avg_loss, metrics

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, Any]:
        """
        Train the model with multi-epoch support and monitoring.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training history and best model info
        """
        logger.info(f"Starting training for {self.epochs} epochs...")

        # Setup training components
        self._setup_training_components()

        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # Training tracking
        best_metric = float('-inf') if 'f1' in self.monitor or 'accuracy' in self.monitor else float('inf')
        best_epoch = 0
        patience_counter = 0
        training_start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training step
            train_loss, train_metrics = self._train_epoch(X_train, y_train)

            # Validation step
            val_loss, val_metrics = self._validate_epoch(X_val, y_val)

            epoch_time = time.time() - epoch_start_time

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Prepare epoch metrics
            epoch_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_recall': val_metrics['recall'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }

            # Update metrics tracker
            self.metrics_tracker.update(epoch_metrics)

            # Monitor metric for early stopping and checkpointing
            if 'f1' in self.monitor:
                current_metric = val_metrics['f1']
                is_better = current_metric > best_metric
            elif 'accuracy' in self.monitor:
                current_metric = val_metrics['accuracy']
                is_better = current_metric > best_metric
            elif 'loss' in self.monitor:
                current_metric = val_loss
                is_better = current_metric < best_metric
            else:
                current_metric = val_metrics['f1']
                is_better = current_metric > best_metric

            # Save best model
            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0

                if self.save_best_only:
                    self._save_checkpoint(epoch, best_metric, 'best_model.pth')

            else:
                patience_counter += 1

            # Learning rate scheduling
            self.scheduler.step(current_metric)

            # Logging
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        training_time = time.time() - training_start_time

        # Save final model and metrics
        self._save_checkpoint(epoch, best_metric, 'final_model.pth')
        self._save_training_history()

        # Training summary
        training_summary = {
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'final_lr': current_lr,
            'metrics_history': self.metrics_tracker.__dict__
        }

        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Best {self.monitor}: {best_metric:.4f} at epoch {best_epoch}")

        return training_summary

    def _save_checkpoint(self, epoch: int, metric_value: float, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric_value': metric_value,
            'config': self.config
        }

        filepath = os.path.join(self.model_save_path, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved: {filepath}")

    def _save_training_history(self):
        """Save training metrics and plots."""
        # Save metrics
        metrics_path = os.path.join(self.log_save_path, 'training_metrics.json')
        self.metrics_tracker.save_metrics(metrics_path)

        # Create training plots
        self._create_training_plots()

    def _create_training_plots(self):
        """Create training and validation plots."""
        epochs = range(1, len(self.metrics_tracker.train_losses) + 1)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        # Loss plot
        axes[0, 0].plot(epochs, self.metrics_tracker.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics_tracker.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(epochs, self.metrics_tracker.train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics_tracker.val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score plot
        axes[0, 2].plot(epochs, self.metrics_tracker.train_f1_scores, 'b-', label='Train F1', linewidth=2)
        axes[0, 2].plot(epochs, self.metrics_tracker.val_f1_scores, 'r-', label='Val F1', linewidth=2)
        axes[0, 2].set_title('F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Precision plot
        axes[1, 0].plot(epochs, self.metrics_tracker.train_precisions, 'b-', label='Train Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.metrics_tracker.val_precisions, 'r-', label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall plot
        axes[1, 1].plot(epochs, self.metrics_tracker.train_recalls, 'b-', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.metrics_tracker.val_recalls, 'r-', label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 2].plot(epochs, self.metrics_tracker.learning_rates, 'g-', linewidth=2)
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.log_save_path, 'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Training plots saved: {plot_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        return checkpoint


if __name__ == "__main__":
    # Example usage
    from src.models.ca_lstm import ChannelAttentionLSTM

    config = {
        'input_dim': 150,
        'hidden_units': [128, 64, 32],
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'num_classes': 3,
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'patience': 5
    }

    # Create model
    model = ChannelAttentionLSTM(config)

    # Create trainer
    trainer = ModelTrainer(model, config)

    # Create dummy data
    X_train = torch.randn(1000, 150)
    y_train = torch.randint(0, 3, (1000,))
    X_val = torch.randn(200, 150)
    y_val = torch.randint(0, 3, (200,))

    # Train model
    # history = trainer.train(X_train, y_train, X_val, y_val)