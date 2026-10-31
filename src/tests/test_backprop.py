# Milan Suri
# CSC 561
# This file is for testing the backwards pass
# Tests generation link: https://share.gemini.google/K92OLiVouyiB

import unittest
import numpy as np
from src.backprop import *

class TestBackwardPass(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and shapes for testing the backward pass."""
        self.m = 4  # Batch size
        self.n_x = 3  # Input features
        self.n_h = 5  # Hidden layer units
        self.n_y = 1  # Output units (e.g., binary classification)

        # Create dummy inputs and intermediate values
        # Depending on whether your cMath library uses NumPy arrays or custom lists,
        # adjust these initializations accordingly.
        self.X = np.random.randn(self.m, self.n_x)
        self.y = np.random.randint(0, 2, size=(self.m, self.n_y)).astype(float)
        self.W2 = np.random.randn(self.n_h, self.n_y)
        self.predictions = np.random.rand(self.m, self.n_y)
        self.Z1 = np.random.randn(self.m, self.n_h)
        self.A1 = np.random.randn(self.m, self.n_h)

    def test_backward_pass_returns_tuple(self):
        """Test that backward_pass returns a tuple of 4 elements."""
        result = backward_pass(self.X, self.y, self.W2, self.predictions, self.Z1, self.A1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_gradient_shapes(self):
        """Test that the returned gradients match the expected dimensions."""
        dW1, db1, dW2, db2 = backward_pass(
            self.X, self.y, self.W2, self.predictions, self.Z1, self.A1
        )

        # dW1 should have the same shape as W1 (n_x, n_h)
        self.assertEqual(np.shape(dW1), (self.n_x, self.n_h))

        # dW2 should have the same shape as W2 (n_h, n_y)
        self.assertEqual(np.shape(dW2), (self.n_h, self.n_y))

        # db1 should correspond to the hidden layer biases (typically shape of bias vector/sum)
        # Note: Adjust expected shape check if your v_col_sum returns a 1D array or 2D column vector
        self.assertIsNotNone(db1)
        self.assertIsNotNone(db2)

    def test_backward_pass_none_inputs(self):
        """Optional: Test behavior or ensure graceful handling with edge cases (like zero batch size)."""
        # You can add custom assertions here if you want to test edge cases
        pass


if __name__ == '__main__':
    unittest.main()