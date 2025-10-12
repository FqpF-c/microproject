import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChannelAttention(nn.Module):
    """
    Channel Attention mechanism for feature selection and weighting.
    """

    def __init__(self, input_dim: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()

        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        self.reduced_dim = max(1, input_dim // reduction_ratio)

        # Global average pooling and max pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_dim, input_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Attention weights of shape (batch_size, seq_len, input_dim)
        """
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Reshape for pooling: (batch_size * seq_len, input_dim, 1)
        x_reshaped = x.view(-1, input_dim, 1)

        # Global average pooling
        avg_out = self.avg_pool(x_reshaped).view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        avg_out = self.mlp(avg_out)

        # Global max pooling
        max_out = self.max_pool(x_reshaped).view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        max_out = self.mlp(max_out)

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)  # (batch_size * seq_len, input_dim)

        # Reshape back to original shape
        attention = attention.view(batch_size, seq_len, input_dim)

        return attention


class ChannelAttentionLSTM(nn.Module):
    """
    Channel Attention LSTM (CA-LSTM) for Android malware detection.
    Combines LSTM with channel attention mechanism for better feature selection.
    """

    def __init__(self, config: Dict[str, Any]):
        super(ChannelAttentionLSTM, self).__init__()

        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_units = config.get('hidden_units', [128, 64, 32])
        self.attention_dim = config.get('attention_dim', 64)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.l2_regularization = config.get('l2_regularization', 0.001)
        self.num_classes = config.get('num_classes', 3)

        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.hidden_units[0])

        # Channel attention
        self.channel_attention = ChannelAttention(
            input_dim=self.hidden_units[0],
            reduction_ratio=8
        )

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        lstm_input_dim = self.hidden_units[0]

        for i, hidden_dim in enumerate(self.hidden_units):
            lstm_layer = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,  # We'll use separate dropout layers
                bidirectional=True
            )
            self.lstm_layers.append(lstm_layer)
            lstm_input_dim = hidden_dim * 2  # Bidirectional doubles the output size

        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(len(self.hidden_units))
        ])

        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_units[-1] * 2,  # Bidirectional
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )

        # Feature fusion layer
        fusion_input_dim = self.hidden_units[-1] * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.attention_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.attention_dim, self.attention_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_dim // 2, self.attention_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.attention_dim // 4, self.num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.normal_(param, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CA-LSTM model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (logits, attention_weights)
        """
        batch_size = x.size(0)

        # Add sequence dimension (treating each sample as a single timestep)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Input projection
        x = self.input_projection(x)  # (batch_size, 1, hidden_units[0])

        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights  # Element-wise multiplication

        # Store attention weights for explainability
        attention_weights = channel_weights.squeeze(1)  # (batch_size, hidden_units[0])

        # Pass through LSTM layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, _ = lstm_layer(x)
            x = dropout_layer(x)

        # Self-attention
        x, self_attention_weights = self.self_attention(x, x, x)

        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # (batch_size, hidden_units[-1] * 2)

        # Feature fusion
        x = self.feature_fusion(x)

        # Classification
        logits = self.classifier(x)

        return logits, attention_weights

    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """
        Get feature importance scores based on attention weights.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Feature importance scores
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x)

            # Average attention weights across batch
            importance_scores = attention_weights.mean(dim=0).cpu().numpy()

        return importance_scores

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probabilities = F.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class labels.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(x)
        predictions = np.argmax(probabilities, axis=1)

        return predictions


class DroidetecBaseline(nn.Module):
    """
    Droidetec BiLSTM baseline model for comparison.
    """

    def __init__(self, config: Dict[str, Any]):
        super(DroidetecBaseline, self).__init__()

        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.num_classes = config.get('num_classes', 3)

        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.normal_(param, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Droidetec baseline.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor
        """
        batch_size = x.size(0)

        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Global average pooling
        pooled = lstm_out.mean(dim=1)  # (batch_size, hidden_dim * 2)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(x)
        predictions = np.argmax(probabilities, axis=1)

        return predictions


def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('ca-lstm' or 'droidetec')
        config: Model configuration

    Returns:
        Model instance
    """
    if model_type.lower() == 'ca-lstm':
        return ChannelAttentionLSTM(config)
    elif model_type.lower() == 'droidetec':
        return DroidetecBaseline(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    config = {
        'input_dim': 150,
        'hidden_units': [128, 64, 32],
        'attention_dim': 64,
        'dropout_rate': 0.3,
        'l2_regularization': 0.001,
        'num_classes': 3
    }

    # Create CA-LSTM model
    model = ChannelAttentionLSTM(config)

    # Test forward pass
    batch_size = 32
    input_dim = 150
    x = torch.randn(batch_size, input_dim)

    logits, attention_weights = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Test prediction
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")