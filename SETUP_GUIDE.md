# Android Malware AI Risk Analysis V2 - Setup Guide

## Overview

This is an advanced Android malware detection system that uses static analysis and AI to classify APK files into risk levels (Low, Medium, High) with explainability features.

## System Components

- **Feature Extraction**: Androguard-based static analysis (100+ features)
- **AI Model**: Channel Attention LSTM with Equilibrium Optimization
- **Explainability**: Gemini API integration for natural language explanations
- **Dashboard**: React-based web interface with real-time analysis
- **API**: Flask backend with RESTful endpoints

## Quick Start

### 1. Installation

```bash
# Clone/Download the project
cd android_malware_ai_v2

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies for frontend
cd src/frontend
npm install
cd ../..
```

### 2. Configuration

1. **Set up Gemini API** (optional, for AI explanations):
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```
1w
2. **Configure system** in `config/config.yaml`:
```yaml
# Adjust paths and parameters as needed
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"

model:
  input_dim: 150
  hidden_units: [128, 64, 32]
  num_classes: 3

api:
  host: "localhost"
  port: 5000
```

### 3. Prepare Data

**Option A: Create sample dataset** (for testing):
```bash
python scripts/extract_features.py --create-sample --output data/processed --sample-size 1000
```

**Option B: Use real APK files**:
```bash
# Place APK files in data/raw/ directory
python scripts/extract_features.py --input data/raw --output data/processed
```

### 4. Train Models

```bash
# Basic training with default settings
python scripts/train_model.py --config config/config.yaml

# Advanced training with hyperparameter optimization
python scripts/train_model.py --config config/config.yaml --optimize
```

### 5. Run the System

**Start Backend API**:
```bash
python src/backend/app.py
```

**Start Frontend Dashboard** (in new terminal):
```bash
cd src/frontend
npm start
```

**Access Dashboard**: Open http://localhost:3000

## Usage Examples

### Command Line Analysis

```bash
# Analyze single APK
python scripts/run_analysis.py /path/to/app.apk

# Quiet mode (just show risk level)
python scripts/run_analysis.py /path/to/app.apk --quiet

# Save detailed results
python scripts/run_analysis.py /path/to/app.apk --output analysis_results.json
```

### Web Dashboard

1. Open http://localhost:3000
2. Navigate to "APK Analysis" page
3. Drag & drop APK file
4. View real-time risk assessment with explanations

### API Endpoints

```bash
# Health check
curl http://localhost:5000/api/health

# Upload APK
curl -X POST -F "file=@app.apk" http://localhost:5000/api/upload

# Analyze APK
curl -X POST -H "Content-Type: application/json" \
     -d '{"filepath": "/path/to/uploaded/app.apk"}' \
     http://localhost:5000/api/analyze
```

## Model Performance

Expected performance metrics:
- **Detection Accuracy**: >95%
- **Precision**: >93%
- **Recall**: >94%
- **F1-Score**: >93%
- **Analysis Time**: ~2-4 seconds per APK

## Architecture Details

### Feature Extraction (100+ features)
- **Permissions**: Dangerous permissions, permission categories
- **API Calls**: Suspicious APIs, crypto APIs, network APIs
- **Opcodes**: Bytecode instruction patterns
- **Manifest**: Activities, services, receivers, intent filters
- **Strings**: Suspicious patterns, URLs, average lengths
- **Certificates**: Debug certificates, signature analysis

### Channel Attention LSTM Model
- **Input Layer**: 150-dimensional feature vector
- **Channel Attention**: Feature selection mechanism
- **BiLSTM Stack**: 3 bidirectional LSTM layers [128, 64, 32]
- **Self-Attention**: Multi-head attention (8 heads)
- **Classification**: 3-class output (Low/Medium/High risk)

### Equilibrium Optimization
- **Bio-inspired**: Physics-based optimization algorithm
- **Hyperparameters**: Batch size, learning rate, hidden units, dropout
- **Population Size**: 50 particles
- **Iterations**: 30 optimization cycles

## Troubleshooting

### Common Issues

1. **APK Analysis Fails**:
   - Ensure APK file is not corrupted
   - Check if Androguard can parse the APK
   - Verify file permissions

2. **Model Training Issues**:
   - Check if data/processed/features.csv exists
   - Ensure sufficient memory (GPU recommended)
   - Verify dataset has balanced classes

3. **Frontend Connection Issues**:
   - Ensure backend is running on port 5000
   - Check CORS settings in Flask app
   - Verify no firewall blocking connections

4. **Gemini API Errors**:
   - Check API key is set correctly
   - Verify API quota and billing
   - System works without Gemini (rule-based explanations)

### Performance Optimization

1. **For Large Datasets**:
   - Use GPU if available
   - Increase batch size
   - Enable data parallel training

2. **For Production**:
   - Use Redis for caching
   - Load balance multiple backend instances
   - Optimize model quantization

## File Structure

```
android_malware_ai_v2/
├── src/
│   ├── feature_extraction/     # APK analysis modules
│   ├── models/                 # AI models (CA-LSTM, Droidetec)
│   ├── training/               # Training and optimization
│   ├── explainability/         # AI explanation system
│   ├── backend/               # Flask API
│   └── frontend/              # React dashboard
├── data/
│   ├── raw/                   # Raw APK files
│   └── processed/             # Extracted features
├── models/
│   ├── checkpoints/           # Trained model weights
│   ├── logs/                  # Training logs and plots
│   └── evaluation/            # Evaluation reports
├── config/                    # Configuration files
├── scripts/                   # Utility scripts
└── docs/                      # Documentation
```

## Security Considerations

- **API Keys**: Store securely, use environment variables
- **File Uploads**: Validate file types, scan for malware
- **Model Security**: Protect model weights from tampering
- **Privacy**: APK analysis should be done securely

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## License

This project is for educational and research purposes. Ensure compliance with applicable laws and regulations when analyzing APK files.

## Support

For issues and questions:
1. Check this setup guide
2. Review error logs in models/logs/
3. Check API health endpoint
4. Verify configuration settings

## Performance Monitoring

The system includes comprehensive monitoring:
- **Training Metrics**: Accuracy, loss, F1-score tracking
- **API Metrics**: Response times, error rates
- **Model Performance**: Confusion matrices, ROC curves
- **System Health**: Component status, resource usage

Access monitoring via:
- Dashboard: http://localhost:3000
- API Health: http://localhost:5000/api/health
- Training Logs: models/logs/training.log