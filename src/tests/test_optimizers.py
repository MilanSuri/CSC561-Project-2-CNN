import unittest
import numpy as np
from src.optimizers import *


# Used Gemini for generating test cases: https://share.gemini.google/V00UIlfqOt33


# Assuming your functions and custom vector ops are imported:
# from your_module import gradient_descent, logistic_gradient


class GradientDescentAndLogisticGradientTest(unittest.TestCase):

    # ==========================================
    # 1. GRADIENT DESCENT EDGE CASES
    # ==========================================

    def test_gradient_descent_already_converged(self):
        """Test gradient descent when the starting point is already at the minimum (zero gradient)."""
        # Objective: f(x) = x^2, minimum at x = 0
        f = lambda x: np.sum(x ** 2)
        gradient = lambda x: 2.0 * x

        initial = np.array([0.0, 0.0], dtype=np.float32)

        x_opt, loss_opt, iters = gradient_descent(f, gradient, initial, tolerance=1e-5)

        # Should converge immediately at iteration 0 or 1
        np.testing.assert_allclose(x_opt, [0.0, 0.0], atol=1e-5)
        self.assertLessEqual(iters, 1)

    def test_gradient_descent_shape_mismatch_error(self):
        """Test that a ValueError is raised if the gradient function returns an incorrect shape."""
        f = lambda x: np.sum(x ** 2)
        # Incorrect gradient shape (returns 1D array of length 3 instead of 2)
        bad_gradient = lambda x: np.array([1.0, 2.0, 3.0], dtype=np.float32)

        initial = np.array([1.0, 2.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            gradient_descent(f, bad_gradient, initial)

    def test_gradient_descent_max_iterations_limit(self):
        """Test that gradient descent safely terminates when max_iterations is reached without converging."""
        # A flat function where gradient is always non-zero and constant (e.g., f(x) = x, grad = 1)
        f = lambda x: x[0]
        gradient = lambda x: np.array([1.0], dtype=np.float32)

        initial = np.array([10.0], dtype=np.float32)
        max_iters = 5

        x_opt, loss_opt, iters = gradient_descent(
            f, gradient, initial, learning_rate=0.01, tolerance=1e-9, max_iterations=max_iters
        )

        # Should stop exactly at max_iterations
        self.assertEqual(iters, max_iters)

    def test_gradient_descent_zero_learning_rate(self):
        """Test gradient descent with a learning rate of 0 (parameters should remain unchanged)."""
        f = lambda x: np.sum(x ** 2)
        gradient = lambda x: 2.0 * x

        initial = np.array([5.0, -3.0], dtype=np.float32)

        x_opt, loss_opt, iters = gradient_descent(
            f, gradient, initial, learning_rate=0.0, max_iterations=10
        )

        # Weights should not have moved
        np.testing.assert_array_equal(x_opt, initial)

    # ==========================================
    # 2. LOGISTIC GRADIENT EDGE CASES
    # ==========================================

    def test_logistic_gradient_shape(self):
        """Test that logistic_gradient outputs a parameter vector of size (n_features + 1)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)  # 3 samples, 2 features
        y = np.array([0, 1, 1], dtype=np.float32)

        # theta should contain [weight_1, weight_2, bias] -> length 3
        theta = np.array([0.1, -0.2, 0.0], dtype=np.float32)

        grad = logistic_gradient(X, y, theta)

        # Expected shape: n_features (2) + 1 (bias) = 3
        self.assertEqual(grad.shape, (3,))

    def test_logistic_gradient_zero_error(self):
        """Test logistic gradient behavior when predictions perfectly align with targets (near-zero gradients)."""
        X = np.array([[0.0, 0.0]], dtype=np.float32)
        y = np.array([0.0], dtype=np.float32)

        # If bias is very negative, sigmoid(bias) -> 0, matching y = 0 closely.
        theta = np.array([0.0, 0.0, -10.0], dtype=np.float32)

        grad = logistic_gradient(X, y, theta)

        # Gradients should be extremely close to zero
        np.testing.assert_allclose(grad, [0.0, 0.0, 0.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()