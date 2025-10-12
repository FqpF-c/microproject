#!/usr/bin/env python3
"""
Feature extraction script for Android APK files
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.apk_analyzer import APKFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_features_from_directory(apk_directory: str, config: dict) -> pd.DataFrame:
    """Extract features from all APK files in a directory."""
    logger.info(f"Extracting features from APK files in: {apk_directory}")

    # Initialize feature extractor
    feature_config = config.get('feature_extraction', {})
    extractor = APKFeatureExtractor(feature_config)

    # Get list of APK files
    apk_files = []
    for root, dirs, files in os.walk(apk_directory):
        for file in files:
            if file.lower().endswith('.apk'):
                apk_files.append(os.path.join(root, file))

    if not apk_files:
        logger.error(f"No APK files found in {apk_directory}")
        return pd.DataFrame()

    logger.info(f"Found {len(apk_files)} APK files")

    # Extract features
    all_features = []
    failed_files = []

    for apk_file in tqdm(apk_files, desc="Extracting features"):
        try:
            features = extractor.extract_all_features(apk_file)
            if features:
                # Add metadata
                features['file_path'] = apk_file
                features['file_name'] = os.path.basename(apk_file)

                # Try to infer label from directory structure or filename
                label = infer_label_from_path(apk_file)
                features['label'] = label

                all_features.append(features)
            else:
                failed_files.append(apk_file)
                logger.warning(f"Failed to extract features from: {apk_file}")

        except Exception as e:
            failed_files.append(apk_file)
            logger.error(f"Error processing {apk_file}: {e}")

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")

    # Convert to DataFrame
    if all_features:
        df = pd.DataFrame(all_features)
        logger.info(f"Successfully extracted features from {len(all_features)} APK files")
        logger.info(f"Feature dimensions: {df.shape}")
        return df
    else:
        logger.error("No features were extracted")
        return pd.DataFrame()

def infer_label_from_path(file_path: str) -> str:
    """Infer label from file path or filename."""
    path_lower = file_path.lower()

    # Check for common malware/benign indicators in path
    if any(term in path_lower for term in ['malware', 'malicious', 'virus', 'trojan', 'adware']):
        return 'malware'
    elif any(term in path_lower for term in ['benign', 'clean', 'safe', 'legitimate']):
        return 'benign'
    elif 'drebin' in path_lower:
        # Drebin dataset convention
        return 'malware' if 'malware' in path_lower else 'benign'
    else:
        # Default to benign if unclear
        return 'benign'

def create_sample_dataset(output_dir: str, n_samples: int = 100):
    """Create a sample dataset with dummy features for testing."""
    logger.info(f"Creating sample dataset with {n_samples} samples")

    # Sample feature names based on our APK analyzer
    feature_names = [
        'dangerous_permissions_count', 'total_permissions_count',
        'has_perm_read_sms', 'has_perm_send_sms', 'has_perm_read_contacts',
        'has_perm_access_fine_location', 'has_perm_record_audio', 'has_perm_camera',
        'suspicious_api_count', 'crypto_api_count', 'network_api_count', 'sms_api_count',
        'total_opcodes', 'reflection_opcodes',
        'activities_count', 'services_count', 'receivers_count',
        'suspicious_strings', 'url_count', 'avg_string_length',
        'is_debug_cert', 'file_size'
    ]

    # Generate random features
    import numpy as np
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        features = {}

        # Generate realistic feature values
        features['dangerous_permissions_count'] = np.random.randint(0, 15)
        features['total_permissions_count'] = np.random.randint(5, 50)
        features['has_perm_read_sms'] = np.random.choice([0, 1], p=[0.8, 0.2])
        features['has_perm_send_sms'] = np.random.choice([0, 1], p=[0.9, 0.1])
        features['has_perm_read_contacts'] = np.random.choice([0, 1], p=[0.7, 0.3])
        features['has_perm_access_fine_location'] = np.random.choice([0, 1], p=[0.6, 0.4])
        features['has_perm_record_audio'] = np.random.choice([0, 1], p=[0.8, 0.2])
        features['has_perm_camera'] = np.random.choice([0, 1], p=[0.7, 0.3])

        features['suspicious_api_count'] = np.random.randint(0, 30)
        features['crypto_api_count'] = np.random.randint(0, 20)
        features['network_api_count'] = np.random.randint(0, 50)
        features['sms_api_count'] = np.random.randint(0, 10)

        features['total_opcodes'] = np.random.randint(1000, 100000)
        features['reflection_opcodes'] = np.random.randint(0, 500)

        features['activities_count'] = np.random.randint(1, 20)
        features['services_count'] = np.random.randint(0, 10)
        features['receivers_count'] = np.random.randint(0, 15)

        features['suspicious_strings'] = np.random.randint(0, 25)
        features['url_count'] = np.random.randint(0, 100)
        features['avg_string_length'] = np.random.uniform(10, 200)

        features['is_debug_cert'] = np.random.choice([0, 1], p=[0.9, 0.1])
        features['file_size'] = np.random.randint(1000000, 50000000)  # 1MB to 50MB

        # Add some missing features with zeros
        for name in feature_names:
            if name not in features:
                features[name] = 0

        # Generate label based on risk factors
        risk_score = (
            features['dangerous_permissions_count'] * 0.3 +
            features['suspicious_api_count'] * 0.2 +
            features['suspicious_strings'] * 0.1 +
            features['has_perm_send_sms'] * 10 +
            features['is_debug_cert'] * 5
        )

        if risk_score > 15:
            label = 'malware'
        elif risk_score > 8:
            label = 'suspicious'
        else:
            label = 'benign'

        features['label'] = label
        features['file_name'] = f'sample_{i:04d}.apk'
        features['file_path'] = f'/sample/path/sample_{i:04d}.apk'

        data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, 'features.csv')
    df.to_csv(features_path, index=False)

    logger.info(f"Sample dataset created: {features_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Extract features from Android APK files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing APK files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for extracted features')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample dataset instead of processing real APKs')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of samples to create (when using --create-sample)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    logger.info("Starting feature extraction...")

    if args.create_sample:
        # Create sample dataset
        df = create_sample_dataset(args.output, args.sample_size)
    else:
        # Extract features from real APK files
        if not os.path.exists(args.input):
            logger.error(f"Input directory does not exist: {args.input}")
            return

        # Extract features
        df = extract_features_from_directory(args.input, config)

        if df.empty:
            logger.error("No features were extracted. Exiting.")
            return

        # Save features
        os.makedirs(args.output, exist_ok=True)
        features_path = os.path.join(args.output, 'features.csv')
        df.to_csv(features_path, index=False)

        logger.info(f"Features saved to: {features_path}")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("FEATURE EXTRACTION COMPLETED!")
    logger.info("="*50)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Total features: {len(df.columns) - 3}")  # Exclude metadata columns
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info(f"Output directory: {args.output}")

if __name__ == "__main__":
    main()