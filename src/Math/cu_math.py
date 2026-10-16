import numpy as np
import sympy as sp

s = sp.Symbol('s')

exp_var = lambda c: c * sp.exp(s)
constant_var = lambda c: 0
ln_var = lambda c: c / s
power_var = lambda c, n: c * (n) * s ** (n - 1) if n != 0 else sp.sympify(0)


def to_function(expression, variables=(s,)):
    return sp.lambdify(variables, expression, modules="numpy")

def derivative(exp_terms: np.array, constant_terms: np.array, ln_terms: np.array, power_terms: np.array):
    differentiated_exp = np.array([exp_var(x) for x in exp_terms])
    differentiated_constant = np.array([constant_var(x) for x in constant_terms])
    differentiated_ln = np.array([ln_var(x) for x in ln_terms])
    differentiated_power = np.array([power_var(x, y) for x, y in power_terms])

    return differentiated_exp, differentiated_constant, differentiated_ln, differentiated_power

def derivative_function(f):
    expression = derivative_expression(f)
    return to_function(expression)

def to_expression(f):
    exp_terms, constant_terms, ln_terms, power_terms = f
    expression = 0

    expression += sum(c * sp.exp(s) for c in exp_terms)
    expression += sum(constant_terms)
    expression += sum(c * sp.log(s) for c in ln_terms)
    expression += sum(c * s ** n for c, n in power_terms)

    return expression


def derivative_expression(f):
    differentiated = derivative(*f)
    expression = 0

    expression += sum(differentiated[0])
    expression += sum(differentiated[1])
    expression += sum(differentiated[2])
    expression += sum(differentiated[3])

    return expression

def differentiate_sum(f: np.array, g: np.array):
    f_prime = derivative_expression(f)
    g_prime = derivative_expression(g)

    return sp.expand(f_prime + g_prime)

def differentiate_sum_function(f, g):
    expression = differentiate_sum(f, g)
    return to_function(expression)

def product(f: np.array, g: np.array):
    f_expr = to_expression(f)
    g_expr = to_expression(g)

    f_prime = derivative_expression(f)
    g_prime = derivative_expression(g)

    return sp.expand(f_prime * g_expr + g_prime * f_expr)

def product_function(f, g):
    expression = product(f, g)
    return to_function(expression)

def chain (f: np.array, g: np.array):
    f_prime = derivative_expression(f)
    g_prime = derivative_expression(g)
    g_expr = to_expression(g)

    return sp.expand (f_prime.subs(s, g_expr) * g_prime)

def chain_function(f, g):
    expression = chain(f, g)
    return to_function(expression)

def gradient_expression(f, variables):
    return [
        sp.diff(f, variable)
        for variable in variables
    ]

def gradient_function(f, variables):
    gradient = gradient_expression(f, variables)
    gradient_eval = to_function(gradient, variables)

    evaluate_gradient = lambda x: np.array(gradient_eval(*x), dtype=np.float32)
    return evaluate_gradient