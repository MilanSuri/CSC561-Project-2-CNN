import unittest
import numpy as np
from ..vector_ops import *


class TestVectorOps(unittest.TestCase):

    def test_v_add_1d(self):
        f = np.array([1, 2, 3])
        g = np.array([4, 5, 6])

        result = v_add(f, g)

        np.testing.assert_array_equal(
            result, np.array([5, 7, 9])
        )

    def test_v_add_2d(self):
        f = np.array([[1, 2], [3, 4]])
        g = np.array([[5, 6], [7, 8]])

        result = v_add(f, g)

        np.testing.assert_array_equal(
            result, np.array([[6, 8], [10, 12]])
        )

    def test_v_scal_mul_1d(self):
        f = np.array([1, 2, 3])

        result = v_scal_mul(f, 2)

        np.testing.assert_array_equal(
            result, np.array([2, 4, 6])
        )

    def test_v_scal_mul_2d(self):
        f = np.array([[1, 2], [3, 4]])

        result = v_scal_mul(f, 3)

        np.testing.assert_array_equal(
            result, np.array([[3, 6], [9, 12]])
        )

    def test_v_scal_mul_negative_scalar(self):
        f = np.array([1, -2, 3])

        result = v_scal_mul(f, -2)

        np.testing.assert_array_equal(
            result, np.array([-2, 4, -6])
        )

    def test_v_elem_mul_1d(self):
        f = np.array([1, 2, 3])
        g = np.array([4, 5, 6])

        result = v_elem_mul(f, g)

        np.testing.assert_array_equal(
            result, np.array([4, 10, 18])
        )

    def test_v_elem_mul_2d(self):
        f = np.array([[1, 2], [3, 4]])
        g = np.array([[5, 6], [7, 8]])

        result = v_elem_mul(f, g)

        np.testing.assert_array_equal(
            result, np.array([[5, 12], [21, 32]])
        )

    def test_v_dot_prod_1d(self):
        f = np.array([1, 2, 3])
        g = np.array([4, 5, 6])

        result = v_dot_prod(f, g)

        self.assertEqual(result, 32)

    def test_v_dot_prod_2d(self):
        f = np.array([[1, 2], [3, 4]])
        g = np.array([[5, 6], [7, 8]])

        result = v_dot_prod(f, g)

        np.testing.assert_array_equal(
            result, np.array([[19, 22], [43, 50]])
        )

    def test_transpose_square_matrix(self):
        f = np.array([[1, 2], [3, 4]])

        result = transpose(f)

        np.testing.assert_array_equal(
            result, np.array([[1, 3], [2, 4]])
        )

    def test_transpose_rectangular_matrix(self):
        f = np.array([[1, 2, 3], [4, 5, 6]])

        result = transpose(f)

        np.testing.assert_array_equal(
            result, np.array([[1, 4], [2, 5], [3, 6]])
        )

    def test_transpose_vector(self):
        f = np.array([1, 2, 3])

        result = transpose(f)

        np.testing.assert_array_equal(
            result, np.array([1, 2, 3])
        )


if __name__ == "__main__":
    unittest.main()
