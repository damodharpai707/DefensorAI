import unittest
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from src.detection.phishing_ai_model import build_ai_model, train_ai_model, evaluate_ai_model

class TestPhishingAIModel(unittest.TestCase):
    def setUp(self):
        """Set up test data for AI model."""
        self.input_dim = 3
        self.X_train = np.array([[0.5, 0.3, 0.1], [0.2, 0.8, 0.6], [0.7, 0.4, 0.9], [0.1, 0.5, 0.8]])
        self.y_train = np.array([0, 1, 1, 0])
        self.X_test = np.array([[0.6, 0.2, 0.3], [0.4, 0.7, 0.5]])
        self.y_test = np.array([1, 0])

        # Build and train the model
        self.model = build_ai_model(input_dim=self.input_dim)
        self.history = train_ai_model(self.model, self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test)

    def test_model_prediction(self):
        """Test AI model predictions."""
        predictions = (self.model.predict(self.X_test) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0.5, f"Model accuracy is too low: {accuracy}")
    
    def test_auc_score(self):
        """Test AUC-ROC score for the model."""
        y_pred_prob = self.model.predict(self.X_test).flatten()
        auc_score = roc_auc_score(self.y_test, y_pred_prob)
        self.assertGreaterEqual(auc_score, 0.5, f"AUC-ROC score is too low: {auc_score}")

    def test_model_evaluation_output(self):
        """Test AI model evaluation function outputs metrics without errors."""
        try:
            evaluate_ai_model(self.model, self.X_test, self.y_test)
            self.assertTrue(True)  # Dummy assertion to confirm no errors during evaluation
        except Exception as e:
            self.fail(f"evaluate_ai_model raised an exception: {e}")

    def test_training_history(self):
        """Test the training history for proper logs."""
        self.assertIn('accuracy', self.history.history)
        self.assertIn('loss', self.history.history)
        self.assertGreater(max(self.history.history['accuracy']), 0.5, "Training accuracy is too low.")

if __name__ == "__main__":
    unittest.main()
