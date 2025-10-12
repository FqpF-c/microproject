import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
from src.feature_extraction.apk_analyzer import APKFeatureExtractor
from src.feature_extraction.data_preprocessor import DataPreprocessor
from src.models.ca_lstm import ChannelAttentionLSTM, DroidetecBaseline, create_model
from src.explainability.explainer import FeatureExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'apk'}

# Global variables
model = None
preprocessor = None
feature_extractor = None
explainer = None
config = None

def load_config():
    """Load configuration from YAML file."""
    global config
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        config = {}

def initialize_components():
    """Initialize all components on startup."""
    global model, preprocessor, feature_extractor, explainer

    try:
        # Load configuration
        load_config()

        # Initialize feature extractor
        feature_config = config.get('feature_extraction', {})
        feature_extractor = APKFeatureExtractor(feature_config)
        logger.info("Feature extractor initialized")

        # Initialize preprocessor
        data_config = config.get('data', {})
        preprocessor = DataPreprocessor(data_config)

        # Load saved preprocessor state
        processed_data_path = data_config.get('processed_data_path', 'data/processed')
        if os.path.exists(processed_data_path):
            preprocessor.load_preprocessor(processed_data_path)
            logger.info("Preprocessor state loaded from saved files")
        else:
            logger.warning("No saved preprocessor state found")

        logger.info("Data preprocessor initialized")

        # Load trained model if available
        model_path = os.path.join('models/checkpoints', 'best_model.pth')
        if os.path.exists(model_path):
            load_trained_model(model_path)
        else:
            logger.warning("No trained model found. Some endpoints will not be available.")

        # Initialize explainer
        explainer_config = config.get('explainability', {})
        explainer = FeatureExplainer(explainer_config)
        logger.info("Explainer initialized")

        # Create upload directory
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        traceback.print_exc()

def load_trained_model(model_path: str):
    """Load a trained model from checkpoint."""
    global model

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Get model configuration
        model_config = checkpoint.get('config', config.get('model', {}))

        # Create model
        model = create_model('ca-lstm', model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)

        logger.info(f"Model loaded successfully from {model_path}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'model_loaded': model is not None,
            'feature_extractor': feature_extractor is not None,
            'preprocessor': preprocessor is not None,
            'explainer': explainer is not None
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_apk():
    """Upload APK file for analysis."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Only APK files are supported.'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"APK uploaded: {filename}")

        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'file_size': os.path.getsize(filepath)
        })

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_apk():
    """Analyze APK file for malware detection."""
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400

        if not feature_extractor:
            return jsonify({'error': 'Feature extractor not initialized'}), 500

        # Extract features
        logger.info(f"Extracting features from {filepath}")
        features = feature_extractor.extract_all_features(filepath)

        if not features:
            return jsonify({'error': 'Failed to extract features from APK'}), 500

        # Predict if model is available
        prediction_result = None
        if model and preprocessor:
            try:
                # Preprocess features
                features_array = preprocessor.transform_single_sample(features)
                features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

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

                prediction_result = {
                    'risk_level': risk_level,
                    'confidence': float(probs[predicted_class]),
                    'probabilities': {
                        'Low': float(probs[0]),
                        'Medium': float(probs[1]),
                        'High': float(probs[2])
                    },
                    'feature_importance': attention.tolist(),
                    'top_features': []
                }

                # Generate explanation if explainer is available
                if explainer:
                    # Handle dimensional mismatch between attention weights and feature names
                    num_features = len(preprocessor.feature_names)
                    if len(attention) > num_features:
                        # Use only the first num_features attention weights
                        feature_importance = attention[:num_features]
                    else:
                        feature_importance = attention

                    explanation = explainer.explain_prediction(
                        risk_level=risk_level,
                        probabilities=probs,
                        feature_importance=feature_importance,
                        feature_names=preprocessor.feature_names,
                        feature_values=features
                    )
                    prediction_result['explanation'] = explanation

                logger.info(f"Analysis completed: {risk_level} risk")

            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                prediction_result = {'error': f'Prediction failed: {str(e)}'}

        return jsonify({
            'filename': os.path.basename(filepath),
            'features': features,
            'feature_count': len(features),
            'prediction': prediction_result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error analyzing APK: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple APK files."""
    try:
        data = request.get_json()
        filepaths = data.get('filepaths', [])

        if not filepaths:
            return jsonify({'error': 'No filepaths provided'}), 400

        results = []
        for filepath in filepaths:
            if os.path.exists(filepath):
                # Analyze each file
                result = analyze_single_file(filepath)
                results.append(result)
            else:
                results.append({
                    'filepath': filepath,
                    'error': 'File not found'
                })

        return jsonify({
            'total_files': len(filepaths),
            'successful': len([r for r in results if 'error' not in r]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return jsonify({'error': str(e)}), 500

def analyze_single_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single APK file."""
    try:
        # Extract features
        features = feature_extractor.extract_all_features(filepath)
        if not features:
            return {'filepath': filepath, 'error': 'Feature extraction failed'}

        # Predict if model is available
        prediction_result = None
        if model and preprocessor:
            try:
                # Preprocess and predict
                features_array = preprocessor.transform_single_sample(features)
                features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

                device = next(model.parameters()).device
                features_tensor = features_tensor.to(device)

                with torch.no_grad():
                    logits, attention_weights = model(features_tensor)
                    probabilities = torch.softmax(logits, dim=1)

                probs = probabilities.cpu().numpy()[0]
                predicted_class = np.argmax(probs)
                risk_labels = ['Low', 'Medium', 'High']
                risk_level = risk_labels[predicted_class]

                prediction_result = {
                    'risk_level': risk_level,
                    'confidence': float(probs[predicted_class]),
                    'probabilities': {
                        'Low': float(probs[0]),
                        'Medium': float(probs[1]),
                        'High': float(probs[2])
                    }
                }

            except Exception as e:
                prediction_result = {'error': f'Prediction failed: {str(e)}'}

        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'features_extracted': len(features),
            'prediction': prediction_result
        }

    except Exception as e:
        return {'filepath': filepath, 'error': str(e)}

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if not model:
        return jsonify({'error': 'No model loaded'}), 404

    try:
        # Get model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        device = next(model.parameters()).device

        return jsonify({
            'model_type': 'Channel Attention LSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(device),
            'input_dim': config.get('model', {}).get('input_dim', 'Unknown'),
            'num_classes': config.get('model', {}).get('num_classes', 'Unknown'),
            'model_loaded': True
        })

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        uploaded_files = []

        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                if filename.endswith('.apk'):
                    filepath = os.path.join(upload_dir, filename)
                    file_stats = os.stat(filepath)
                    uploaded_files.append({
                        'filename': filename,
                        'size': file_stats.st_size,
                        'upload_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    })

        return jsonify({
            'uploaded_files': len(uploaded_files),
            'total_upload_size': sum(f['size'] for f in uploaded_files),
            'recent_uploads': sorted(uploaded_files, key=lambda x: x['upload_time'], reverse=True)[:10],
            'system_info': {
                'torch_version': torch.__version__,
                'device_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        'feature_extraction': config.get('feature_extraction', {}),
        'model': config.get('model', {}),
        'api': config.get('api', {}),
        'explainability': {k: v for k, v in config.get('explainability', {}).items()
                         if k != 'gemini_api_key_env'}  # Don't expose API key
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    # Initialize components
    initialize_components()

    # Get configuration
    api_config = config.get('api', {})
    host = api_config.get('host', 'localhost')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', True)

    logger.info(f"Starting Flask server on {host}:{port}")

    # Run the app
    app.run(host=host, port=port, debug=debug)