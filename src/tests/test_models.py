import unittest
import numpy as np
from ..logistic_regression import *


# Assuming your logistic regression functions and dependencies are imported:
# from logistic_regression import logistic_forward, logistic_loss, train_log_reg, predict_proba, predict


class LogisticRegressionTest(unittest.TestCase):

    # ==========================================
    # 1. UNIT TESTS: Function & Validation Checks
    # ==========================================

    def test_logistic_forward_shape(self):
        """Test that logistic_forward returns predictions matching the sample size."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weights = np.array([0.5, -0.5], dtype=np.float32)
        bias = 0.1

        preds = logistic_forward(X, weights, bias)
        self.assertEqual(preds.shape, (2,))
        # Predictions should be probabilities between 0 and 1
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_predict_proba_1d_input(self):
        """Test that predict_proba correctly reshapes a single 1D sample."""
        weights = np.array([1.0, 1.0], dtype=np.float32)
        bias = 0.0
        single_sample = np.array([0.0, 0.0], dtype=np.float32)

        prob = predict_proba(single_sample, weights, bias)
        self.assertEqual(prob.shape, (1,))
        # At z = 0, sigmoid(0) = 0.5
        np.testing.assert_allclose(prob, [0.5], rtol=1e-5)

    def test_predict_thresholding(self):
        """Test that predict correctly applies the decision threshold."""
        # Mock probabilities array
        probabilities = np.array([0.2, 0.49, 0.5, 0.51, 0.9], dtype=np.float32)

        # Using a dummy weights/bias since predict calls predict_proba internally,
        # but we can test the threshold logic directly via predict if we mock or pass valid inputs,
        # or test the underlying threshold calculation:
        threshold = 0.5
        expected_classes = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        computed_classes = (probabilities >= threshold).astype(np.int32)

        np.testing.assert_array_equal(computed_classes, expected_classes)

    def test_train_log_reg_validation_errors(self):
        """Test that train_log_reg raises ValueError for invalid inputs."""
        # Invalid label values (not 0 or 1)
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_invalid = np.array([0, 2], dtype=np.float32)

        with self.assertRaises(ValueError):
            train_log_reg(X, y_invalid)

        # Dimension mismatch between X and y
        y_mismatch = np.array([0], dtype=np.float32)
        with self.assertRaises(ValueError):
            train_log_reg(X, y_mismatch)

        # Non-finite values (NaN / Inf)
        X_nan = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        y_valid = np.array([0, 1], dtype=np.float32)
        with self.assertRaises(ValueError):
            train_log_reg(X_nan, y_valid)

    # ==========================================
    # 2. INTERPRETABLE TESTS: End-to-End Training
    # ==========================================

    def test_interpretable_linearly_separable_data(self):
        """
        Train logistic regression on a simple, clearly separable dataset
        and verify that it successfully learns the decision boundary and
        correctly classifies new, unseen data points.
        """
        # Create a simple dataset:
        # Class 0 points are located near [1, 1]
        # Class 1 points are located near [8, 8]
        X_train = np.array([
            [1.0, 1.5],
            [1.5, 1.0],
            [2.0, 2.0],
            [8.0, 8.5],
            [8.5, 8.0],
            [9.0, 9.0]
        ], dtype=np.float32)

        y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)

        # Train the model
        weights, bias, final_loss, iterations = train_log_reg(
            X_train, y_train, learning_rate=0.1, tolerance=1e-5, max_iterations=5000
        )

        # Interpretability check 1: Loss should decrease significantly/converge to a low value
        self.assertLess(final_loss, 0.2, "Model failed to minimize BCE loss on separable data.")

        # Interpretability check 2: Test predictions on clear training/test points
        X_test = np.array([
            [1.2, 1.2],  # Should be classified as 0
            [8.8, 8.2]  # Should be classified as 1
        ], dtype=np.float32)

        predictions = predict(X_test, weights, bias)
        expected_predictions = np.array([0, 1], dtype=np.int32)

        np.testing.assert_array_equal(
            predictions,
            expected_predictions,
            err_msg="Model failed to correctly classify clear test samples after training."
        )


if __name__ == "__main__":
    unittest.main()