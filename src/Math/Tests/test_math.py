
import unittest
import numpy as np
import sympy as sp

from ..cu_math import *


class TestMathOps(unittest.TestCase):

    def setUp(self):
        self.f = (
            np.array([2]),          # 2 * exp(s)
            np.array([3]),          # 3
            np.array([4]),          # 4 * log(s)
            [(5, 2)],               # 5 * s**2
        )

        self.g = (
            np.array([1]),          # exp(s)
            np.array([2]),          # 2
            np.array([0]),          # 0 * log(s)
            [(3, 1)],               # 3 * s
        )


    def test_to_expression(self):
        expression = to_expression(self.f)

        expected = (
            2 * sp.exp(s)
            + 3
            + 4 * sp.log(s)
            + 5 * s**2
        )

        self.assertTrue(sp.simplify(expression - expected) == 0)

    def test_to_function(self):
        expression = s**2 + 2*s + 1
        function = to_function(expression)

        self.assertAlmostEqual(float(function(2)), 9.0)

    # Derivative Rules

    def test_derivative_exponential(self):
        result = derivative(
            np.array([2]), np.array([]), np.array([]), []
        )
        self.assertTrue(sp.simplify(result[0][0] - 2*sp.exp(s)) == 0)

    def test_derivative_constant(self):
        result = derivative(
            np.array([]), np.array([5]), np.array([]), []
        )
        self.assertEqual(result[1][0], 0)

    def test_derivative_logarithm(self):
        result = derivative(
            np.array([]), np.array([]), np.array([3]), []
        )
        self.assertTrue(sp.simplify(result[2][0] - 3/s) == 0)

    def test_derivative_power(self):
        result = derivative(
            np.array([]), np.array([]), np.array([]), [(4, 3)]
        )
        self.assertTrue(sp.simplify(result[3][0] - 12*s**2) == 0)

    def test_derivative_zero_power(self):
        result = derivative(
            np.array([]), np.array([]), np.array([]), [(4, 0)]
        )
        self.assertEqual(result[3][0], 0)

    # Full Derivative on Expression

    def test_derivative_expression(self):
        expression = derivative_expression(self.f)

        expected = (
            2 * sp.exp(s)
            + 4/s
            + 10*s
        )

        self.assertTrue(sp.simplify(expression - expected) == 0)

    def test_derivative_function(self):
        function = derivative_function(self.f)

        expected = 2*np.exp(1) + 4 + 10

        self.assertAlmostEqual(float(function(1)), expected)

    # Sum Rule
    def test_differentiate_sum(self):
        result = differentiate_sum(self.f, self.g)

        expected = ( 3*sp.exp(s) + 4/s + 10*s+ 3)

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_differentiate_sum_function(self):
        function = differentiate_sum_function(self.f, self.g)

        expected = 3*np.exp(1) + 4 + 10 + 3

        self.assertAlmostEqual(float(function(1)), expected)

    # Product rule

    def test_product_rule(self):
        result = product(self.f, self.g)

        f_expr = to_expression(self.f)
        g_expr = to_expression(self.g)
        expected = sp.diff(f_expr * g_expr, s)

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_product_function(self):
        function = product_function(self.f, self.g)

        f_expr = to_expression(self.f)
        g_expr = to_expression(self.g)
        expected = sp.lambdify(s, sp.diff(f_expr*g_expr, s), "numpy")

        self.assertAlmostEqual(float(function(2)), float(expected(2)))

    # Chain rule

    def test_chain_rule(self):
        # f(s) = s**2, g(s) = 3*s
        f = ([], [], [], [(1, 2)])
        g = ([], [], [], [(3, 1)])

        result = chain(f, g)
        expected = sp.diff(to_expression(f).subs(s, to_expression(g)), s)

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_chain_function(self):
        f = ([], [], [], [(1, 2)])
        g = ([], [], [], [(3, 1)])

        function = chain_function(f, g)

        # d/ds (3s)**2 = 18s
        self.assertAlmostEqual(float(function(2)), 36.0)

    # Multivariable gradient

    def test_gradient_expression(self):
        x, y = sp.symbols("x y")
        expression = x**2 + 3*x*y + y**2

        result = gradient_expression(expression, [x, y])

        self.assertEqual(
            [sp.simplify(value) for value in result],
            [2*x + 3*y, 3*x + 2*y],
        )

    def test_gradient_function(self):
        x, y = sp.symbols("x y")
        expression = x**2 + 3*x*y + y**2

        function = gradient_function(expression, [x, y])
        result = function(np.array([1, 2]))

        expected = np.array([8, 7], dtype=np.float32)

        np.testing.assert_allclose(result, expected)

    def test_gradient_function_returns_numpy_array(self):
        x, y = sp.symbols("x y")
        expression = x**2 + y**2

        function = gradient_function(expression, [x, y])
        result = function(np.array([3, 4]))

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))


if __name__ == "__main__":
    unittest.main()
