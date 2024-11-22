import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.detection.phishing_ml_model import train_ml_model, evaluate_ml_model

class TestPhishingMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test data for ML model."""
        # Generate dummy data
        data = {
            "length": [10, 50, 60, 45, 25],
            "num_dots": [2, 4, 5, 3, 1],
            "has_suspicious_keywords": [0, 1, 1, 0, 0],
            "label": [0, 1, 1, 0, 0]
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

    def test_model_instance(self):
        """Test that the trained model is an instance of RandomForestClassifier."""
        self.assertIsInstance(self.model, RandomForestClassifier)

    def test_model_training_performance(self):
        """Test if the model achieves acceptable training performance."""
        # Make predictions on the training set
        y_train_pred = self.model.predict(self.X_train)
        report = classification_report(self.y_train, y_train_pred, output_dict=True)
        train_accuracy = report["accuracy"]

        # Ensure training accuracy is above a threshold
        self.assertGreater(train_accuracy, 0.8, "Training accuracy is too low.")

    def test_model_evaluation_metrics(self):
        """Test evaluation metrics on the test set."""
        # Predict probabilities for AUC-ROC calculation
        y_test_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_test_pred_prob)

        # Ensure AUC-ROC is above a reasonable threshold
        self.assertGreater(auc_score, 0.75, "AUC-ROC score is below threshold.")

    def test_model_evaluation_output(self):
        """Test evaluation output on the test set."""
        # Evaluate model and ensure no exceptions
        evaluate_ml_model(self.model, self.X_test, self.y_test)
        self.assertTrue(True)  # Dummy assert to confirm test runs

    def test_model_predictions(self):
        """Test that model predictions align with expected output shape."""
        y_test_pred = self.model.predict(self.X_test)

        # Ensure predictions have the same length as the test set
        self.assertEqual(len(y_test_pred), len(self.y_test))

if __name__ == "__main__":
    unittest.main()
