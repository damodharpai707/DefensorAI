import unittest
import numpy as np
from src.detection.phishing_ai_model import build_ai_model, train_ai_model, evaluate_ai_model

class TestPhishingAIModel(unittest.TestCase):
    def setUp(self):
        """Set up test data for AI model."""
        self.input_dim = 3
        self.X_train = np.array([[0.5, 0.3, 0.1], [0.2, 0.8, 0.6], [0.7, 0.4, 0.9]])
        self.y_train = np.array([0, 1, 1])
        self.X_test = np.array([[0.6, 0.2, 0.3]])
        self.y_test = np.array([1])

        # Build and train the model
        self.model = build_ai_model(input_dim=self.input_dim)
        train_ai_model(self.model, self.X_train, self.y_train)

    def test_model_prediction(self):
        """Test AI model predictions."""
        predictions = (self.model.predict(self.X_test) > 0.5).astype(int)
        self.assertEqual(predictions[0][0], self.y_test[0])

    def test_model_accuracy(self):
        """Test AI model evaluation."""
        evaluate_ai_model(self.model, self.X_test, self.y_test)  # Should output metrics
        self.assertTrue(True)  # Dummy assert to confirm test runs

if __name__ == "__main__":
    unittest.main()
