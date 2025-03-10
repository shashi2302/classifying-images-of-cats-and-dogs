#!/usr/bin/env python3
"""
Tests for the cat vs dog image classification API.
"""

import os
import sys
import unittest
import json
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app


class TestAPI(unittest.TestCase):
    """Tests for the API functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.base_dir = Path(__file__).resolve().parent.parent
        
        # Create a test directory if it doesn't exist
        self.test_dir = self.base_dir / 'tests' / 'test_data'
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a dummy image for testing
        from PIL import Image
        self.img = Image.new('RGB', (100, 100), color='red')
        self.img_path = self.test_dir / 'test_image.jpg'
        self.img.save(self.img_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if self.img_path.exists():
            os.remove(self.img_path)
    
    def test_health_check(self):
        """Test the health check endpoint."""
        # Mock the model to be loaded
        with patch('src.api.app.model', MagicMock()):
            response = self.app.get('/health')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(data['status'], 'ok')
    
    def test_health_check_no_model(self):
        """Test the health check endpoint when no model is loaded."""
        # Mock the model to be None
        with patch('src.api.app.model', None):
            response = self.app.get('/health')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 503)
            self.assertEqual(data['status'], 'error')
    
    def test_predict_no_file(self):
        """Test the predict endpoint with no file."""
        # Mock the model to be loaded
        with patch('src.api.app.model', MagicMock()):
            response = self.app.post('/predict')
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 400)
            self.assertEqual(data['error'], 'No file provided')
    
    def test_predict_empty_file(self):
        """Test the predict endpoint with an empty file."""
        # Mock the model to be loaded
        with patch('src.api.app.model', MagicMock()):
            response = self.app.post('/predict', data={
                'file': (io.BytesIO(b''), '')
            })
            data = json.loads(response.data)
            
            self.assertEqual(response.status_code, 400)
            self.assertEqual(data['error'], 'Empty file provided')
    
    def test_predict_success(self):
        """Test the predict endpoint with a valid file."""
        # Mock the model and feature extraction
        with patch('src.api.app.model') as mock_model, \
             patch('src.api.app.extract_features') as mock_extract:
            
            # Configure mocks
            mock_extract.return_value = [[0.1, 0.2, 0.3]]  # Dummy features
            mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% dog, 30% cat
            
            # Open the test image
            with open(self.img_path, 'rb') as img_file:
                response = self.app.post('/predict', data={
                    'file': (img_file, 'test_image.jpg')
                })
                
                data = json.loads(response.data)
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(data['prediction'], 'dog')
                self.assertAlmostEqual(data['confidence'], 0.7)
    
    def test_metrics(self):
        """Test the metrics endpoint."""
        response = self.app.get('/metrics')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('cat_dog_api_requests_total', response.data.decode())


if __name__ == '__main__':
    unittest.main() 