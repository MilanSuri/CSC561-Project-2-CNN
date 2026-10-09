import numpy as np
import sympy as sp

s = sp.Symbol('s')

# Compute derivatives of a term
exp_var = lambda c: c * sp.exp(s)
constant_var = lambda c: 0
ln_var = lambda c: c / s
power_var = lambda c, n: c * (n) * s ** (n-1) if n != 0 else sp.sympify(0)

def derivative(exp_terms : np.array, constant_terms : np.array, ln_terms : np.array, power_terms : np.array):
    differentiated_exp = np.array([exp_var(x) for x in exp_terms])
    differentiated_constant = np.array([constant_var(x) for x in constant_terms])
    differentiated_ln = np.array([ln_var(x) for x in ln_terms])
    differentiated_power = np.array([power_var(x, y) for x, y in power_terms])

    return differentiated_exp, differentiated_constant, differentiated_ln, differentiated_power