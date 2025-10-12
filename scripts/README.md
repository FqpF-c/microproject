# Scripts Directory

This directory contains utility scripts for training, evaluation, and analysis.

## Scripts Overview

### 1. `extract_features.py`
Extract features from APK files for training.

```bash
# Extract features from real APK files
python scripts/extract_features.py --input /path/to/apk/files --output data/processed

# Create sample dataset for testing
python scripts/extract_features.py --create-sample --output data/processed --sample-size 1000
```

### 2. `train_model.py`
Main training script with hyperparameter optimization and model comparison.

```bash
# Basic training
python scripts/train_model.py --config config/config.yaml

# With hyperparameter optimization
python scripts/train_model.py --config config/config.yaml --optimize

# Skip baseline training
python scripts/train_model.py --skip-baseline

# Evaluate existing models only
python scripts/train_model.py --evaluate-only
```

### 3. `run_analysis.py`
Analyze individual APK files using trained models.

```bash
# Analyze single APK
python scripts/run_analysis.py /path/to/app.apk

# Quiet mode (only show result)
python scripts/run_analysis.py /path/to/app.apk --quiet

# Save detailed results
python scripts/run_analysis.py /path/to/app.apk --output results.json
```

## Quick Start

1. **Create sample data** (if you don't have APK files):
```bash
python scripts/extract_features.py --create-sample --output data/processed
```

2. **Train models**:
```bash
python scripts/train_model.py --config config/config.yaml
```

3. **Analyze APK**:
```bash
python scripts/run_analysis.py sample.apk
```

## Requirements

- All scripts should be run from the project root directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For real APK analysis, install Android analysis tools