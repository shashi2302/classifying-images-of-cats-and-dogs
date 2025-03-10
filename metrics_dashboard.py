#!/usr/bin/env python3
"""
Success metrics dashboard for cat vs dog image classification.
Generates a dashboard with key metrics for model performance and system health.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
import mlflow.sklearn
import json
import requests
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = BASE_DIR / 'mlflow'
DASHBOARD_DIR = BASE_DIR / 'dashboard'

# Ensure directories exist
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.absolute()}")


def get_mlflow_metrics():
    """
    Get metrics from MLflow for all runs.
    
    Returns:
        DataFrame with run metrics
    """
    logger.info("Getting metrics from MLflow...")
    
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name("cat-dog-classification")
    if experiment is None:
        logger.error("Experiment not found")
        return pd.DataFrame()
    
    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        logger.warning("No runs found")
        return pd.DataFrame()
    
    # Select relevant columns
    metrics_df = runs[['run_id', 'start_time', 'tags.mlflow.runName', 
                      'metrics.val_accuracy', 'metrics.val_precision', 'metrics.val_recall', 'metrics.val_f1',
                      'metrics.test_accuracy', 'metrics.test_precision', 'metrics.test_recall', 'metrics.test_f1',
                      'metrics.training_time']]
    
    # Rename columns
    metrics_df = metrics_df.rename(columns={
        'tags.mlflow.runName': 'run_name',
        'metrics.val_accuracy': 'val_accuracy',
        'metrics.val_precision': 'val_precision',
        'metrics.val_recall': 'val_recall',
        'metrics.val_f1': 'val_f1',
        'metrics.test_accuracy': 'test_accuracy',
        'metrics.test_precision': 'test_precision',
        'metrics.test_recall': 'test_recall',
        'metrics.test_f1': 'test_f1',
        'metrics.training_time': 'training_time'
    })
    
    # Convert start_time to datetime
    metrics_df['start_time'] = pd.to_datetime(metrics_df['start_time'])
    
    # Sort by start_time
    metrics_df = metrics_df.sort_values('start_time')
    
    logger.info(f"Found {len(metrics_df)} runs")
    
    return metrics_df


def get_api_metrics(api_url='http://localhost:5000'):
    """
    Get metrics from the API.
    
    Args:
        api_url: URL of the API
        
    Returns:
        Dictionary with API metrics
    """
    logger.info("Getting metrics from API...")
    
    try:
        # Get metrics from Prometheus endpoint
        response = requests.get(f"{api_url}/metrics")
        response.raise_for_status()
        
        metrics_text = response.text
        
        # Parse metrics
        metrics = {}
        
        # Parse request count
        for line in metrics_text.split('\n'):
            if line.startswith('cat_dog_api_requests_total'):
                metrics['total_requests'] = float(line.split(' ')[1])
            elif line.startswith('cat_dog_predictions_total{class="cat"'):
                metrics['cat_predictions'] = float(line.split(' ')[1])
            elif line.startswith('cat_dog_predictions_total{class="dog"'):
                metrics['dog_predictions'] = float(line.split(' ')[1])
            elif line.startswith('cat_dog_prediction_latency_seconds_count'):
                metrics['total_predictions'] = float(line.split(' ')[1])
            elif line.startswith('cat_dog_prediction_latency_seconds_sum'):
                metrics['total_latency'] = float(line.split(' ')[1])
        
        # Calculate average latency
        if 'total_predictions' in metrics and 'total_latency' in metrics and metrics['total_predictions'] > 0:
            metrics['avg_latency_ms'] = (metrics['total_latency'] / metrics['total_predictions']) * 1000
        else:
            metrics['avg_latency_ms'] = 0
        
        # Get health status
        health_response = requests.get(f"{api_url}/health")
        health_response.raise_for_status()
        
        health_data = health_response.json()
        metrics['health_status'] = health_data['status']
        
        # Get model info
        model_info_response = requests.get(f"{api_url}/model-info")
        model_info_response.raise_for_status()
        
        model_info = model_info_response.json()
        metrics['model_run_id'] = model_info['run_id']
        metrics['model_type'] = model_info['model_type']
        
        logger.info("API metrics retrieved successfully")
        
        return metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting API metrics: {e}")
        return {
            'total_requests': 0,
            'cat_predictions': 0,
            'dog_predictions': 0,
            'avg_latency_ms': 0,
            'health_status': 'unknown',
            'model_run_id': 'unknown',
            'model_type': 'unknown'
        }


def generate_model_performance_plots(metrics_df, output_dir):
    """
    Generate plots for model performance metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_dir: Directory to save plots
    """
    logger.info("Generating model performance plots...")
    
    # Create a figure for accuracy metrics
    plt.figure(figsize=(12, 8))
    
    # Plot validation and test accuracy
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['start_time'], metrics_df['val_accuracy'], 'o-', label='Validation')
    plt.plot(metrics_df['start_time'], metrics_df['test_accuracy'], 'o-', label='Test')
    plt.title('Accuracy Over Time')
    plt.xlabel('Run Date')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot validation and test precision
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['start_time'], metrics_df['val_precision'], 'o-', label='Validation')
    plt.plot(metrics_df['start_time'], metrics_df['test_precision'], 'o-', label='Test')
    plt.title('Precision Over Time')
    plt.xlabel('Run Date')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Plot validation and test recall
    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['start_time'], metrics_df['val_recall'], 'o-', label='Validation')
    plt.plot(metrics_df['start_time'], metrics_df['test_recall'], 'o-', label='Test')
    plt.title('Recall Over Time')
    plt.xlabel('Run Date')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # Plot validation and test F1 score
    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['start_time'], metrics_df['val_f1'], 'o-', label='Validation')
    plt.plot(metrics_df['start_time'], metrics_df['test_f1'], 'o-', label='Test')
    plt.title('F1 Score Over Time')
    plt.xlabel('Run Date')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png')
    plt.close()
    
    # Create a figure for training time
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['start_time'], metrics_df['training_time'], 'o-')
    plt.title('Training Time Over Time')
    plt.xlabel('Run Date')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time.png')
    plt.close()
    
    logger.info("Model performance plots generated")


def generate_api_metrics_plots(api_metrics, output_dir):
    """
    Generate plots for API metrics.
    
    Args:
        api_metrics: Dictionary with API metrics
        output_dir: Directory to save plots
    """
    logger.info("Generating API metrics plots...")
    
    # Create a figure for prediction distribution
    plt.figure(figsize=(10, 6))
    
    # Plot prediction distribution
    labels = ['Cat', 'Dog']
    values = [api_metrics['cat_predictions'], api_metrics['dog_predictions']]
    
    plt.bar(labels, values)
    plt.title('Prediction Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    
    # Add count labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(int(v)), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png')
    plt.close()
    
    logger.info("API metrics plots generated")


def generate_dashboard(metrics_df, api_metrics, output_dir):
    """
    Generate a dashboard HTML file with all metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        api_metrics: Dictionary with API metrics
        output_dir: Directory to save the dashboard
    """
    logger.info("Generating dashboard...")
    
    # Generate plots
    generate_model_performance_plots(metrics_df, output_dir)
    generate_api_metrics_plots(api_metrics, output_dir)
    
    # Get the latest model metrics
    if not metrics_df.empty:
        latest_metrics = metrics_df.iloc[-1].to_dict()
    else:
        latest_metrics = {
            'val_accuracy': 0,
            'test_accuracy': 0,
            'val_precision': 0,
            'test_precision': 0,
            'val_recall': 0,
            'test_recall': 0,
            'val_f1': 0,
            'test_f1': 0,
            'training_time': 0
        }
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cat vs Dog Classification - Success Metrics Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            .header-right {{
                text-align: right;
            }}
            .metrics-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                flex: 1;
                min-width: 200px;
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.05);
            }}
            .metric-title {{
                font-size: 14px;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .metric-value.good {{
                color: #28a745;
            }}
            .metric-value.warning {{
                color: #ffc107;
            }}
            .metric-value.bad {{
                color: #dc3545;
            }}
            .chart-container {{
                margin-bottom: 30px;
            }}
            .chart {{
                width: 100%;
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Cat vs Dog Classification</h1>
                    <h2>Success Metrics Dashboard</h2>
                </div>
                <div class="header-right">
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Model Type: {api_metrics['model_type']}</p>
                    <p>Health Status: <span class="metric-value {'good' if api_metrics['health_status'] == 'ok' else 'bad'}">{api_metrics['health_status']}</span></p>
                </div>
            </div>
            
            <h3>Model Performance Metrics</h3>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-title">Test Accuracy</div>
                    <div class="metric-value {'good' if latest_metrics['test_accuracy'] >= 0.9 else 'warning' if latest_metrics['test_accuracy'] >= 0.8 else 'bad'}">{latest_metrics['test_accuracy']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test Precision</div>
                    <div class="metric-value {'good' if latest_metrics['test_precision'] >= 0.9 else 'warning' if latest_metrics['test_precision'] >= 0.8 else 'bad'}">{latest_metrics['test_precision']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test Recall</div>
                    <div class="metric-value {'good' if latest_metrics['test_recall'] >= 0.9 else 'warning' if latest_metrics['test_recall'] >= 0.8 else 'bad'}">{latest_metrics['test_recall']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Test F1 Score</div>
                    <div class="metric-value {'good' if latest_metrics['test_f1'] >= 0.9 else 'warning' if latest_metrics['test_f1'] >= 0.8 else 'bad'}">{latest_metrics['test_f1']:.4f}</div>
                </div>
            </div>
            
            <h3>API Performance Metrics</h3>
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-title">Total Requests</div>
                    <div class="metric-value">{int(api_metrics['total_requests'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Cat Predictions</div>
                    <div class="metric-value">{int(api_metrics['cat_predictions'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Dog Predictions</div>
                    <div class="metric-value">{int(api_metrics['dog_predictions'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg. Latency (ms)</div>
                    <div class="metric-value {'good' if api_metrics['avg_latency_ms'] < 100 else 'warning' if api_metrics['avg_latency_ms'] < 200 else 'bad'}">{api_metrics['avg_latency_ms']:.2f}</div>
                </div>
            </div>
            
            <h3>Model Performance Over Time</h3>
            <div class="chart-container">
                <img class="chart" src="model_performance.png" alt="Model Performance">
            </div>
            
            <h3>Training Time</h3>
            <div class="chart-container">
                <img class="chart" src="training_time.png" alt="Training Time">
            </div>
            
            <h3>Prediction Distribution</h3>
            <div class="chart-container">
                <img class="chart" src="prediction_distribution.png" alt="Prediction Distribution">
            </div>
            
            <div class="footer">
                <p>Cat vs Dog Classification Project | MLOps Dashboard</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_dir / 'dashboard.html', 'w') as f:
        f.write(html_content)
    
    logger.info(f"Dashboard generated at {output_dir / 'dashboard.html'}")


def main():
    """Main function to generate the success metrics dashboard."""
    try:
        # Get metrics from MLflow
        metrics_df = get_mlflow_metrics()
        
        # Get metrics from API
        api_metrics = get_api_metrics()
        
        # Generate dashboard
        generate_dashboard(metrics_df, api_metrics, DASHBOARD_DIR)
        
        logger.info("Success metrics dashboard generated successfully")
    except Exception as e:
        logger.error(f"Error generating success metrics dashboard: {e}")


if __name__ == "__main__":
    main() 