# Milan Suri
# CSC 561
# This is for vector operations
# Source: https://www.geeksforgeeks.org/data-science/vectors-for-ml/

import numpy as np

# Can use a negative f/g if I want to do subtraction later
def v_add (f: np.array, g: np.array):
    # Vector addition

    # Guarding dimentsions
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
    # Scalar multiplication

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
    # Vector multiplication
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

def v_dot_prod (f : np.array, g: np.array):
    # Dot product
    if f.ndim == 1:
        if f.shape != g.shape:
            raise ValueError("Vector/Matrix f and g have different shapes, for multiplication must be the same")

        result = 0.0
        for i in range(len(f)):
            result += f[i] * g[i]
        return result
    elif f.ndim == 2:
        if f.shape[1] != g.shape[0]:
            raise ValueError ("Columns of Matrix f don't match row number matrix g")

        result = np.zeros((len(f), g.shape[1]), dtype=np.float32)
        for i in range(len(f)):
            for j in range(g.shape[1]):
                total_sum = 0
                for k in range(f.shape[1]):
                    total_sum += f[i][k] * g[k][j]
                result[i][j] = total_sum
        return result

def transpose(f: np.array):
    # Transpose a vector - flip rows/cols
    if f.ndim == 1:
        return f.copy()

    rows, cols = f.shape
    result = np.zeros((cols, rows), dtype=f.dtype)

    for i in range(rows):
        for j in range(cols):
            result[j, i] = f[i, j]

    return result

