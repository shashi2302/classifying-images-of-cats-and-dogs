#!/usr/bin/env python3
"""
Flask API for cat vs dog image classification.
Serves the trained model for inference.
"""

import os
import io
import logging
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import mlflow.sklearn
from flask import Flask, request, jsonify
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / 'models'
MLFLOW_DIR = BASE_DIR / 'mlflow'

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.absolute()}")

# Initialize Flask app
app = Flask(__name__)

# Initialize Prometheus metrics
REQUESTS = Counter('cat_dog_api_requests_total', 'Total number of requests to the API')
PREDICTIONS = Counter('cat_dog_predictions_total', 'Total number of predictions made', ['class'])
LATENCY = Histogram('cat_dog_prediction_latency_seconds', 'Time taken for prediction')

# Global variables
model = None
run_id = None


def load_model():
    """
    Load the latest trained model.
    
    Returns:
        Loaded model
    """
    global model, run_id
    
    # Find the latest model file
    model_files = list(MODEL_DIR.glob("model_*.pkl"))
    if not model_files:
        logger.error("No model files found")
        return None
    
    # Sort by modification time (newest first)
    latest_model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    # Extract run ID from filename
    run_id = latest_model_file.stem.split('_')[1]
    
    try:
        # Load the model
        with open(latest_model_file, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model from {latest_model_file}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def extract_features(image_data):
    """
    Extract features from an image.
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        Feature vector for the image
    """
    try:
        # Open the image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to a standard size
        img = img.resize((100, 100))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract color histograms for each channel
        hist_r, _ = np.histogram(img_array[:, :, 0].flatten(), bins=32, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:, :, 1].flatten(), bins=32, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:, :, 2].flatten(), bins=32, range=(0, 256))
        
        # Normalize histograms
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()
        
        # Combine features
        features = np.concatenate([hist_r, hist_g, hist_b])
        
        return features.reshape(1, -1)  # Reshape for prediction
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({'status': 'ok', 'message': 'API is healthy'}), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects an image file in the request.
    Returns the prediction (cat or dog) and confidence.
    """
    REQUESTS.inc()
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty file provided'}), 400
    
    try:
        # Read image data
        image_data = file.read()
        
        # Extract features
        features = extract_features(image_data)
        
        if features is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction with latency tracking
        start_time = time.time()
        prediction_proba = model.predict_proba(features)[0]
        latency = time.time() - start_time
        LATENCY.observe(latency)
        
        # Get class with highest probability
        class_idx = np.argmax(prediction_proba)
        class_name = 'dog' if class_idx == 1 else 'cat'
        confidence = float(prediction_proba[class_idx])
        
        # Increment prediction counter
        PREDICTIONS.labels(class_name).inc()
        
        # Return prediction
        return jsonify({
            'prediction': class_name,
            'confidence': confidence,
            'latency_ms': latency * 1000
        }), 200
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the loaded model."""
    if model is None or run_id is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get run information from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Extract metrics and parameters
        metrics = run.data.metrics
        params = run.data.params
        
        return jsonify({
            'run_id': run_id,
            'metrics': metrics,
            'parameters': params,
            'model_type': type(model).__name__
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """Main function to run the Flask API."""
    # Load the model
    if load_model() is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main() 