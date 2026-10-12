from cu_math import *
import numpy as np

# Can use a negative f/g if I want to do subtraction later
def v_add (f: np.array, g: np.array):
    if f.shape != g.shape:
        raise ValueError ("Vector/Matrix f and g have different shapes, for addition must be the same")

    if f.ndim == 1:
        result = np.empty(len(f), dtype=np.float32)
        for i in range(len(f)):
            result[i] = f[i] + g[i]
    elif f.ndim == 2:
        result = np.empty((len(f), f.shape[1]), dtype=np.float32)
        for i in range(len(f)):
            for j in range(f.shape[1]):
                result[i][j] = f[i][j] + g[i][j]

    return result

def v_scal_mul (f: np.array, scalar: float):
    if f.ndim == 1:
        result = np.empty(len(f), dtype = np.float32)
        for i in range(len(f)):
            result[i] = f[i] * scalar
    elif f.ndim == 2:
        result = np.empty((len(f), f.shape[1]), dtype=np.float32)
        for i in range(len(f)):
            for j in range(f.shape[1]):
                result[i][j] = f[i][j] * scalar

    return result

def v_elem_mul (f : np.array, g: np.array):
    if f.shape != g.shape:
        raise ValueError ("Vector/Matrix f and g have different shapes, for multiplication must be the same")

    if f.ndim == 1:
        result = np.empty(len(f), dtype=np.float32)
        for i in range(len(f)):
            result[i] = f[i] * g[i]
    elif f.ndim == 2:
        result = np.empty((len(f), f.shape[1]), dtype=np.float32)
        for i in range(len(f)):
            for j in range(f.shape[1]):
                result[i][j] = f[i][j] * g[i][j]

    return result