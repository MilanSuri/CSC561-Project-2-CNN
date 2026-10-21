import unittest
import numpy as np
from src.activations import *

# Used Gemini for generating test cases: https://share.gemini.google/V00UIlfqOt33

# Assuming your activation functions are imported or defined above:
# from activations import relu, relu_d, sigmoid, sigmoid_d


class ActivationFunctionsTest(unittest.TestCase):

    # ==========================================
    # RELU & RELU DERIVATIVE TESTS (Edge Cases)
    # ==========================================

    def test_relu_exact_zero(self):
        """Test ReLU behavior when input is exactly zero."""
        x = np.array([0.0])
        expected = np.array([0.0], dtype=np.float32)
        np.testing.assert_array_equal(relu(x), expected)

    def test_relu_d_at_zero(self):
        """Test ReLU derivative at x = 0 (should return 0 per implementation)."""
        x = np.array([0.0])
        expected = np.array([0.0], dtype=np.float32)
        np.testing.assert_array_equal(relu_d(x), expected)

    def test_relu_large_numbers(self):
        """Test ReLU with very large positive and negative numbers."""
        x = np.array([-1e10, 1e10], dtype=np.float32)
        expected = np.array([0.0, 1e10], dtype=np.float32)
        np.testing.assert_allclose(relu(x), expected, rtol=1e-5)

    def test_relu_all_negative_1d(self):
        """Test ReLU with a 1D array containing only negative values."""
        x = np.array([-5.5, -0.1, -100.0], dtype=np.float32)
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(relu(x), expected)
        np.testing.assert_array_equal(relu_d(x), expected)

    def test_relu_2d_mixed_matrix(self):
        """Test ReLU and its derivative on a 2D matrix with mixed values."""
        x = np.array([[-1.5, 2.0], [0.0, -3.2]], dtype=np.float32)
        expected_relu = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        expected_deriv = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

        np.testing.assert_array_equal(relu(x), expected_relu)
        np.testing.assert_array_equal(relu_d(x), expected_deriv)

    # ==========================================
    # SIGMOID & SIGMOID DERIVATIVE TESTS (Edge Cases)
    # ==========================================

    def test_sigmoid_zero(self):
        """Test Sigmoid at x = 0 (should evaluate to exactly 0.5)."""
        x = np.array([0.0], dtype=np.float32)
        expected = np.array([0.5], dtype=np.float32)
        np.testing.assert_allclose(sigmoid(x), expected, rtol=1e-5)

    def test_sigmoid_large_positive_stability(self):
        """Test Sigmoid numerical stability with very large positive inputs (prevent overflow)."""
        x = np.array([500.0, 1000.0], dtype=np.float32)
        result = sigmoid(x)
        # Should approach 1.0 without throwing overflow warnings or NaNs
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-5)

    def test_sigmoid_large_negative_stability(self):
        """Test Sigmoid numerical stability with very large negative inputs (branch check)."""
        x = np.array([-500.0, -1000.0], dtype=np.float32)
        result = sigmoid(x)
        # Should approach 0.0 safely using the alternate branch
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-5)

    def test_sigmoid_derivative_bounds(self):
        """Test Sigmoid derivative maximum value at x = 0 (should be 0.25)."""
        x = np.array([0.0], dtype=np.float32)
        expected_d = np.array([0.25], dtype=np.float32)
        np.testing.assert_allclose(sigmoid_d(x), expected_d, rtol=1e-5)

    def test_sigmoid_2d_matrix(self):
        """Test Sigmoid and its derivative on a 2D matrix."""
        x = np.array([[0.0, -2.0], [2.0, 0.0]], dtype=np.float32)
        sig_res = sigmoid(x)
        sig_d_res = sigmoid_d(x)

        # Check shapes and bounds (sigmoid output must be strictly between 0 and 1)
        self.assertEqual(sig_res.shape, (2, 2))
        self.assertTrue(np.all(sig_res >= 0.0) and np.all(sig_res <= 1.0))

        # Check derivative relationship: sigmoid_d(x) == sigmoid(x) * (1 - sigmoid(x))
        expected_d = sig_res * (1.0 - sig_res)
        np.testing.assert_allclose(sig_d_res, expected_d, rtol=1e-5)

    # ==========================================
    # DIMENSION & ERROR HANDLING TESTS
    # ==========================================

    def test_invalid_dimensions_scalar(self):
        """Test that passing a 0D scalar array raises ValueError."""
        scalar_arr = np.array(5.0)  # ndim == 0
        with self.assertRaises(ValueError):
            relu(scalar_arr)
        with self.assertRaises(ValueError):
            sigmoid(scalar_arr)

    def test_invalid_dimensions_3d(self):
        """Test that passing a 3D array raises ValueError."""
        tensor_arr = np.ones((2, 2, 2), dtype=np.float32)  # ndim == 3
        with self.assertRaises(ValueError):
            relu(tensor_arr)
        with self.assertRaises(ValueError):
            sigmoid_d(tensor_arr)


if __name__ == "__main__":
    unittest.main()