#!/usr/bin/env python3
"""
Tests for the cat vs dog image classification model.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_model import extract_features


class TestModel(unittest.TestCase):
    """Tests for the model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_dir = Path(__file__).resolve().parent.parent
    
    def test_extract_features(self):
        """Test feature extraction from an image."""
        # Create a dummy image for testing
        from PIL import Image
        
        # Create a test directory if it doesn't exist
        test_dir = self.base_dir / 'tests' / 'test_data'
        test_dir.mkdir(exist_ok=True)
        
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = test_dir / 'test_image.jpg'
        img.save(img_path)
        
        # Extract features
        features = extract_features(img_path)
        
        # Check that features are not None
        self.assertIsNotNone(features)
        
        # Check that features have the expected shape (32 bins * 3 channels)
        self.assertEqual(len(features), 96)
        
        # Clean up
        os.remove(img_path)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Create a dummy model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy features (96 features as in our implementation)
        X_train = np.random.rand(10, 96)
        y_train = np.random.randint(0, 2, 10)  # Binary classification
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Create a test sample
        X_test = np.random.rand(1, 96)
        
        # Make a prediction
        prediction = model.predict(X_test)
        
        # Check that prediction is either 0 or 1
        self.assertIn(prediction[0], [0, 1])
        
        # Check that prediction probabilities sum to 1
        proba = model.predict_proba(X_test)[0]
        self.assertAlmostEqual(sum(proba), 1.0)


if __name__ == '__main__':
    unittest.main() 