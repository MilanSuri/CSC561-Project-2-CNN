import unittest
from ..cu_math import *


class TestMathOps(unittest.TestCase):

    def setUp(self):
        self.s = sp.Symbol('s')

        self.f = (
            [(2, self.s)],             # 2 * exp(s)
            np.array([3]),             # 3
            [(4, self.s)],             # 4 * log(s)
            [(5, 2, self.s)],          # 5 * s**2
        )

        self.g = (
            [(1, self.s)],             # exp(s)
            np.array([2]),             # 2
            [(0, self.s)],             # 0 * log(s)
            [(3, 1, self.s)],          # 3 * s
        )

    def test_to_expression(self):
        expression = to_expression(self.f)

        expected = (2 * sp.exp(self.s) + 3 + 4 * sp.log(self.s) + 5 * self.s**2)

        self.assertTrue(sp.simplify(expression - expected) == 0)

    def test_to_function(self):
        expression = self.s**2 + 2*self.s + 1
        function = to_function(expression)

        self.assertAlmostEqual(float(function(2)), 9.0)

    # Derivative Rules

    def test_derivative_exponential(self):
        result = derivative(
            [(2, self.s)], np.array([]), [], [], self.s
        )

        self.assertTrue(
            sp.simplify(result[0][0] - 2*sp.exp(self.s)) == 0
        )

    def test_derivative_constant(self):
        result = derivative(
            [], np.array([5]), [], [], self.s
        )

        self.assertEqual(result[1][0], 0)

    def test_derivative_logarithm(self):
        result = derivative(
            [], np.array([]), [(3, self.s)], [], self.s
        )

        self.assertTrue(
            sp.simplify(result[2][0] - 3/self.s) == 0
        )

    def test_derivative_power(self):
        result = derivative(
            [], np.array([]), [], [(4, 3, self.s)], self.s
        )

        self.assertTrue(
            sp.simplify(result[3][0] - 12*self.s**2) == 0
        )

    def test_derivative_zero_power(self):
        result = derivative(
            [], np.array([]), [], [(4, 0, self.s)], self.s
        )

        self.assertEqual(result[3][0], 0)

    def test_derivative_other_variable(self):
        x, y = sp.symbols("x y")

        result = derivative(
            [(2, x)], np.array([]), [(3, y)], [(4, 2, x)], y
        )

        self.assertEqual(result[0][0], 0)
        self.assertTrue(sp.simplify(result[2][0] - 3/y) == 0)
        self.assertEqual(result[3][0], 0)

    # Full Derivative on Expression

    def test_derivative_expression(self):
        expression = derivative_expression(self.f, self.s)

        expected = (2 * sp.exp(self.s) + 4/self.s + 10*self.s)

        self.assertTrue(sp.simplify(expression - expected) == 0)

    def test_derivative_function(self):
        function = derivative_function(self.f, self.s)

        expected = 2*np.exp(1) + 4 + 10

        self.assertAlmostEqual(float(function(1)), expected)

    # Sum Rule

    def test_differentiate_sum(self):
        result = differentiate_sum(self.f, self.g, self.s)

        expected = (3*sp.exp(self.s) + 4/self.s + 10*self.s  + 3)

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_differentiate_sum_function(self):
        function = differentiate_sum_function(self.f, self.g, self.s)

        expected = 3*np.exp(1) + 4 + 10 + 3

        self.assertAlmostEqual(float(function(1)), expected)

    # Product Rule

    def test_product_rule(self):
        result = product(self.f, self.g, self.s)

        f_expr = to_expression(self.f)
        g_expr = to_expression(self.g)
        expected = sp.diff(f_expr * g_expr, self.s)

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_product_function(self):
        function = product_function(self.f, self.g, self.s)

        f_expr = to_expression(self.f)
        g_expr = to_expression(self.g)
        expected = sp.lambdify(
            self.s, sp.diff(f_expr*g_expr, self.s), "numpy"
        )

        self.assertAlmostEqual(
            float(function(2)), float(expected(2))
        )

    # Chain Rule

    def test_chain_rule(self):
        # f(s) = s**2, g(s) = 3*s
        f = ([], [], [], [(1, 2, self.s)])
        g = ([], [], [], [(3, 1, self.s)])

        result = chain(f, g, self.s)

        expected = sp.diff(
            to_expression(f).subs(self.s, to_expression(g)),
            self.s
        )

        self.assertTrue(sp.simplify(result - expected) == 0)

    def test_chain_function(self):
        f = ([], [], [], [(1, 2, self.s)])
        g = ([], [], [], [(3, 1, self.s)])

        function = chain_function(f, g, self.s)

        # d/ds (3s)**2 = 18s
        self.assertAlmostEqual(float(function(2)), 36.0)

    # Multivariable Gradient

    def test_gradient_expression(self):
        x, y = sp.symbols("x y")

        f = (
            [(2, x)],              # 2 * exp(x)
            np.array([]),          # no constants
            [(5, y)],              # 5 * log(y)
            [(3, 2, x), (4, 1, y)]
        )

        result = gradient_expression(f, [x, y])

        expected = [2*sp.exp(x) + 6*x, 5/y + 4]

        self.assertEqual(
            [sp.simplify(value) for value in result],
            expected
        )

    def test_gradient_function(self):
        x, y = sp.symbols("x y")

        f = ([(2, x)], np.array([]), [(5, y)], [(3, 2, x), (4, 1, y)])

        function = gradient_function(f, [x, y])
        result = function(np.array([0, 1]))

        expected = np.array([2, 9], dtype=np.float32)

        np.testing.assert_allclose(result, expected)

    def test_gradient_function_returns_numpy_array(self):
        x, y = sp.symbols("x y")

        f = ([], np.array([]), [], [(2, 2, x), (2, 2, y)])

        function = gradient_function(f, [x, y])
        result = function(np.array([3, 4]))

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))

if __name__ == "__main__":
    unittest.main()