# Milan Suri
# CSC 561
# This file is for activation functions. These add nonlinearity to models and is used to create decision boundaries.

import math
import numpy as np

# ReLU https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
# Used to get the formula and better understand ReLU

def s_relu (x):
    return max(0, x)

def relu (x : np.ndarray):
    if x.ndim not in (1, 2):
        raise ValueError("ReLU supports only 1D vectors and 2D matrices")

    result = np.empty(x.shape, dtype=np.float32)

    if x.ndim == 1:
        for i in range(x.shape[0]):
            result[i] = s_relu(x[i])

    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = s_relu(x[i, j])

    return result

def s_relu_d(x):
    if x > 0:
        return 1
    else:
        return 0

def relu_d(x : np.ndarray):
    if x.ndim not in (1, 2):
        raise ValueError("ReLU supports only 1D vectors and 2D matrices")

    result = np.empty(x.shape, dtype=np.float32)

    if x.ndim == 1:
        for i in range(x.shape[0]):
            result[i] = s_relu_d(x[i])

    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = s_relu_d(x[i, j])

    return result

# Sigmoid — Used for logistic regression, used class materials.

def s_sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def sigmoid (x: np.ndarray):
    if x.ndim not in (1, 2):
        raise ValueError("Sigmoid supports only 1D vectors and 2D matrices")

    result = np.empty(x.shape, dtype=np.float32)

    if x.ndim == 1:
        for i in range(x.shape[0]):
            result[i] = s_sigmoid(x[i])

    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = s_sigmoid(x[i, j])

    return result

def a_sigmoid_d(x):
    sigmoid = s_sigmoid(x)
    return sigmoid * (1-sigmoid)

def sigmoid_d(x: np.ndarray):
    if x.ndim not in (1, 2):
        raise ValueError("Sigmoid derivative supports only 1D vectors and 2D matrices")

    s = sigmoid(x)
    result = np.empty(x.shape, dtype=np.float32)

    if x.ndim == 1:
        for i in range(x.shape[0]):
            result[i] = s[i] * (1 - s[i])

    elif x.ndim == 2:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i, j] = s[i, j] * (1 - s[i, j])

    return result