import numpy as np
import sympy as sp

s = sp.Symbol('s')

# Compute Derivatives of the variables
exp_var = lambda c: c * sp.exp(s)
constant_var = lambda c: 0
ln_var = lambda c: c / s
power_var = lambda c, n: c * (n) * s ** (n-1) if n != 0 else 0