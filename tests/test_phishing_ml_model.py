import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.detection.phishing_ml_model import train_ml_model, evaluate_ml_model

class TestPhishingMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test data for ML model."""
        # Generate dummy data
        data = {
            "length": [10, 50, 60],
            "num_dots": [2, 4, 5],
            "has_suspicious_keywords": [0, 1, 1],
            "label": [0, 1, 1]
        }
        df = pd.DataFrame(data)

        # Features and labels
        self.X = df[["length", "num_dots", "has_suspicious_keywords"]]
        self.y = df["label"]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )

        # Train model
        self.model = train_ml_model(self.X_train, self.y_train)

    def test_model_training(self):
        """Test that the model is trained properly."""
        self.assertIsInstance(self.model, RandomForestClassifier)

    def test_model_evaluation(self):
        """Test ML model evaluation."""
        evaluate_ml_model(self.model, self.X_test, self.y_test)  # Should output metrics
        self.assertTrue(True)  # Dummy assert to confirm test runs

if __name__ == "__main__":
    unittest.main()
