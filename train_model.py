#!/usr/bin/env python3
"""
Model training script for cat vs dog image classification.
Trains a model using scikit-learn and logs metrics with MLflow.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
MLFLOW_DIR = BASE_DIR / 'mlflow'

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.absolute()}")
mlflow.set_experiment("cat-dog-classification")


def extract_features(image_path):
    """
    Extract features from an image.
    
    For simplicity, we'll use basic features like color histograms.
    In a real-world scenario, you might use pre-trained CNN features.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Feature vector for the image
    """
    try:
        # Open the image
        img = Image.open(image_path).convert('RGB')
        
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
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None


def load_data():
    """
    Load the processed data and extract features.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    logger.info("Loading data and extracting features...")
    
    # Load metadata
    metadata_path = PROCESSED_DATA_DIR / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    
    # Initialize feature dictionaries
    features = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    labels = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        split_data = metadata[metadata['split'] == split]
        
        for _, row in split_data.iterrows():
            # Construct image path
            img_path = PROCESSED_DATA_DIR / split / row['label'] / row['filename']
            
            # Extract features
            img_features = extract_features(img_path)
            
            if img_features is not None:
                features[split].append(img_features)
                labels[split].append(1 if row['label'] == 'dog' else 0)
    
    # Convert to numpy arrays
    X_train = np.array(features['train'])
    y_train = np.array(labels['train'])
    X_val = np.array(features['validation'])
    y_val = np.array(labels['validation'])
    X_test = np.array(features['test'])
    y_test = np.array(labels['test'])
    
    logger.info(f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, X_val, y_val):
    """
    Train a model using scikit-learn.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained model
    """
    logger.info("Training model...")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Initialize base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Initialize grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit grid search
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Log parameters
        mlflow.log_params(best_params)
        
        # Log metrics
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = f"confusion_matrix_{run_id}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        # Create and log feature importance plot
        feature_importance = best_model.feature_importances_
        indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        fi_path = f"feature_importance_{run_id}.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        os.remove(fi_path)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        model_path = MODEL_DIR / f"model_{run_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return best_model, run_id


def evaluate_model(model, X_test, y_test, run_id):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        run_id: MLflow run ID
    """
    logger.info("Evaluating model on test set...")
    
    with mlflow.start_run(run_id=run_id):
        # Make predictions
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Test Confusion Matrix')
        cm_path = f"test_confusion_matrix_{run_id}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test precision: {test_precision:.4f}")
        logger.info(f"Test recall: {test_recall:.4f}")
        logger.info(f"Test F1 score: {test_f1:.4f}")


def main():
    """Main function to run the model training pipeline."""
    try:
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = load_data()
        
        # Train model
        model, run_id = train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, run_id)
        
        logger.info("Model training and evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")


if __name__ == "__main__":
    main() 