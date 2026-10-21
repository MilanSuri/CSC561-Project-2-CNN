# Milan Suri
# CSC 561 NN Project
# This file is used for symbolic and numeric differentiation and gradient computation
# https://docs.sympy.org/latest/index.html, Sophia Zhan (used for Math advice + explanations)

import numpy as np
import sympy as sp

s = sp.Symbol('s')

# derivative rules for each supported type of term
exp_var = lambda c, variable: c * sp.exp(variable)
constant_var = lambda c: 0
ln_var = lambda c, variable: c / variable
power_var = lambda c, n, variable: c * (n) * variable ** (n - 1) if n != 0 else sp.sympify(0)


def to_function(expression, variables=(s,)):
    # Convert to numerical function using numpy
    return sp.lambdify(variables, expression, modules="numpy")

def derivative(exp_terms: np.array, constant_terms: np.array, ln_terms: np.array, power_terms: np.array, variable):
    # Diffrentiate with respect to a specific variable for each term
    differentiated_exp = np.array([exp_var(c, v) if v == variable else 0 for c, v in exp_terms])
    differentiated_constant = np.array([constant_var(c) for c in constant_terms])
    differentiated_ln = np.array([ln_var(c, v) if v == variable else 0 for c, v in ln_terms])
    differentiated_power = np.array([power_var(c, n, v) if v == variable else 0 for c, n, v in power_terms])

    return differentiated_exp, differentiated_constant, differentiated_ln, differentiated_power

def derivative_function(f, variable):
    # Convert the derivative symbolic to numpy
    expression = derivative_expression(f, variable)
    return to_function(expression)

def to_expression(f):
    exp_terms, constant_terms, ln_terms, power_terms = f
    expression = 0

    # Combine to one sympy expression
    expression += sum(c * sp.exp(v) for c, v in exp_terms)
    expression += sum(constant_terms)
    expression += sum(c * sp.log(v) for c, v in ln_terms)
    expression += sum(c * v ** n for c, n, v in power_terms)

    return expression

def derivative_expression(f, variable):
    # Sum the derivatives to symbolic derivative
    differentiated = derivative(*f, variable)
    expression = 0

    expression += sum(differentiated[0])
    expression += sum(differentiated[1])
    expression += sum(differentiated[2])
    expression += sum(differentiated[3])

    return expression

def differentiate_sum(f: np.array, g: np.array, variable):
    # Sum rule
    f_prime = derivative_expression(f, variable)
    g_prime = derivative_expression(g, variable)

    return sp.expand(f_prime + g_prime)

def differentiate_sum_function(f, g, variable):
    # Convert to Numpy
    expression = differentiate_sum(f, g, variable)
    return to_function(expression)

def product(f: np.array, g: np.array, v):
    # Apply product rule
    f_expr = to_expression(f)
    g_expr = to_expression(g)

    f_prime = derivative_expression(f, v)
    g_prime = derivative_expression(g, v)

    return sp.expand(f_prime * g_expr + g_prime * f_expr)

def product_function(f, g, v):
    # Convert product to numpy
    expression = product(f, g, v)
    return to_function(expression)

def chain(f: np.array, g: np.array, variable):
    # Chain rule
    f_prime = derivative_expression(f, variable)
    g_prime = derivative_expression(g, variable)
    g_expr = to_expression(g)

    return sp.expand(f_prime.subs(variable, g_expr) * g_prime)

def chain_function(f, g, variable):
    # Convert to numpy
    expression = chain(f, g, variable)
    return to_function(expression)

def gradient_expression(f, variables: list):
    # Compute the gradient by going and taking the partial derivative
    gradient = []

    for variable in variables:
        component = derivative_expression(f, variable)
        gradient.append(component)

    return np.array(gradient, dtype=object)

def gradient_function(f, variables):
    # Convert the gradient to numpy.
    gradient = gradient_expression(f, variables)
    gradient_eval = to_function(list(gradient), tuple(variables))

    def evaluate_gradient(x):
        return np.array(gradient_eval(*x), dtype=np.float32)

    return evaluate_gradient