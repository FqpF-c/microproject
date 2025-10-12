import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Advanced data preprocessing pipeline for Android malware detection.
    Handles feature normalization, class imbalance, and data splitting.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.smote = None
        self.feature_names = None
        self.class_weights = None

        # Configure SMOTE
        if config.get('enable_smote', True):
            self.smote = SMOTE(
                k_neighbors=config.get('smote_k_neighbors', 5),
                sampling_strategy=config.get('smote_sampling_strategy', 'auto'),
                random_state=config.get('random_state', 42)
            )

    def load_dataset(self, features_path: str, labels_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and labels from CSV files."""
        logger.info(f"Loading dataset from {features_path}")

        # Load features
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
        else:
            raise FileNotFoundError(f"Features file not found: {features_path}")

        # Load or extract labels
        if labels_path and os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            labels = labels_df['label']
        else:
            # If no separate labels file, assume labels are in features file
            if 'label' in features_df.columns:
                labels = features_df['label']
                features_df = features_df.drop('label', axis=1)
            else:
                # Create dummy labels based on file names or directory structure
                logger.warning("No labels found. Creating dummy labels.")
                labels = pd.Series(['benign'] * len(features_df))

        logger.info(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
        return features_df, labels

    def create_risk_labels(self, labels: pd.Series) -> pd.Series:
        """Convert labels to risk levels (Low, Medium, High)."""
        risk_labels = []

        for label in labels:
            if isinstance(label, str):
                label_lower = label.lower()
                if 'benign' in label_lower or 'clean' in label_lower:
                    risk_labels.append('Low')
                elif 'malware' in label_lower or 'malicious' in label_lower:
                    risk_labels.append('High')
                elif 'suspicious' in label_lower:
                    risk_labels.append('Medium')
                else:
                    risk_labels.append('Medium')
            else:
                # Numeric labels: 0 = Low, 1 = Medium, 2 = High
                if label == 0:
                    risk_labels.append('Low')
                elif label == 1:
                    risk_labels.append('Medium')
                elif label == 2:
                    risk_labels.append('High')
                else:
                    risk_labels.append('Medium')

        return pd.Series(risk_labels)

    def clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for training."""
        logger.info("Cleaning features...")

        # Remove non-numeric columns (like file_name)
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        text_columns = features_df.select_dtypes(exclude=[np.number]).columns

        if len(text_columns) > 0:
            logger.info(f"Removing non-numeric columns: {list(text_columns)}")
            features_df = features_df[numeric_columns]

        # Handle missing values
        features_df = features_df.fillna(0)

        # Remove constant features
        constant_features = [col for col in features_df.columns
                           if features_df[col].nunique() <= 1]
        if constant_features:
            logger.info(f"Removing constant features: {constant_features}")
            features_df = features_df.drop(constant_features, axis=1)

        # Remove highly correlated features
        correlation_matrix = features_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        high_corr_features = [column for column in upper_triangle.columns
                            if any(upper_triangle[column] > 0.95)]
        if high_corr_features:
            logger.info(f"Removing highly correlated features: {high_corr_features}")
            features_df = features_df.drop(high_corr_features, axis=1)

        logger.info(f"Features after cleaning: {len(features_df.columns)}")
        return features_df

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels to numeric format."""
        logger.info("Encoding labels...")

        # Get unique labels from data
        unique_labels = sorted(labels.unique())
        logger.info(f"Found labels: {unique_labels}")

        # Create mapping based on actual labels present
        if len(unique_labels) == 2:
            # Binary classification
            if 'Low' in unique_labels and 'High' in unique_labels:
                label_order = ['Low', 'High']  # Low=0, High=1
            else:
                label_order = unique_labels
        else:
            # Multi-class
            label_order = ['Low', 'Medium', 'High']

        self.label_encoder.classes_ = np.array(label_order)
        encoded_labels = self.label_encoder.transform(labels)

        # Log class distribution
        unique, counts = np.unique(encoded_labels, return_counts=True)
        class_distribution = dict(zip(
            [self.label_encoder.classes_[i] for i in unique],
            counts
        ))
        logger.info(f"Class distribution: {class_distribution}")

        return encoded_labels

    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE and/or undersampling."""
        logger.info("Balancing dataset...")

        original_counts = np.bincount(y)
        logger.info(f"Original class distribution: {dict(enumerate(original_counts))}")

        if self.smote is not None:
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X, y)

            new_counts = np.bincount(y_resampled)
            logger.info(f"After SMOTE class distribution: {dict(enumerate(new_counts))}")

            return X_resampled, y_resampled

        return X, y

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split dataset into train, validation, and test sets."""
        logger.info("Splitting dataset...")

        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('validation_size', 0.1)
        random_state = self.config.get('random_state', 42)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        logger.info(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray,
                          X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Normalize features using StandardScaler."""
        logger.info("Normalizing features...")

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced learning."""
        logger.info("Computing class weights...")

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))

        logger.info(f"Class weights: {self.class_weights}")
        return self.class_weights

    def preprocess_dataset(self, features_path: str, labels_path: Optional[str] = None,
                          output_dir: str = None) -> Tuple[np.ndarray, ...]:
        """Complete preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline...")

        # Load data
        features_df, labels = self.load_dataset(features_path, labels_path)

        # Convert to risk labels if needed
        if not all(label in ['Low', 'Medium', 'High'] for label in labels.unique()):
            labels = self.create_risk_labels(labels)

        # Clean features
        features_df = self.clean_features(features_df)
        self.feature_names = list(features_df.columns)

        # Encode labels
        y = self.encode_labels(labels)

        # Convert to numpy arrays
        X = features_df.values

        # Split dataset first (before balancing to avoid data leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X, y)

        # Balance training data only
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train)

        # Normalize features
        X_train_norm, X_val_norm, X_test_norm = self.normalize_features(
            X_train_balanced, X_val, X_test
        )

        # Compute class weights
        self.compute_class_weights(y_train_balanced)

        # Save preprocessing objects
        if output_dir:
            self.save_preprocessor(output_dir)

        logger.info("Data preprocessing completed successfully!")

        return (X_train_norm, X_val_norm, X_test_norm,
                y_train_balanced, y_val, y_test)

    def save_preprocessor(self, output_dir: str):
        """Save preprocessing objects for later use."""
        os.makedirs(output_dir, exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))

        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

        # Save feature names
        joblib.dump(self.feature_names, os.path.join(output_dir, 'feature_names.pkl'))

        # Save class weights
        joblib.dump(self.class_weights, os.path.join(output_dir, 'class_weights.pkl'))

        logger.info(f"Preprocessing objects saved to {output_dir}")

    def load_preprocessor(self, input_dir: str):
        """Load preprocessing objects."""
        self.scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(input_dir, 'label_encoder.pkl'))
        self.feature_names = joblib.load(os.path.join(input_dir, 'feature_names.pkl'))
        self.class_weights = joblib.load(os.path.join(input_dir, 'class_weights.pkl'))

        logger.info(f"Preprocessing objects loaded from {input_dir}")

    def transform_single_sample(self, features: Dict[str, Any]) -> np.ndarray:
        """Transform a single sample for prediction."""
        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Ensure all required features are present
        for feature_name in self.feature_names:
            if feature_name not in features_df.columns:
                features_df[feature_name] = 0

        # Select and order features
        features_df = features_df[self.feature_names]

        # Normalize
        features_normalized = self.scaler.transform(features_df.values)

        return features_normalized[0]


if __name__ == "__main__":
    # Example usage
    config = {
        'test_size': 0.2,
        'validation_size': 0.1,
        'random_state': 42,
        'enable_smote': True,
        'smote_k_neighbors': 5,
        'smote_sampling_strategy': 'auto'
    }

    preprocessor = DataPreprocessor(config)

    # Example preprocessing
    # X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_dataset(
    #     "data/processed/features.csv",
    #     "data/processed/labels.csv",
    #     "data/processed"
    # )