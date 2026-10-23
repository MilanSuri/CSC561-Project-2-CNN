# Milan Suri
# CSC 561
'''
This file is my Binary Cross Entropy File. BCE computes the difference between an ML models predicted probas and the
the actual binary labels. [-y log (p) + (1-y) log (1-p)] = BCE.

https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/

https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCELoss.html
'''

import math
import numpy as np

def s_bce(p, y):
    # Clip probabilities to avoid taking the logarithm of DNE
    eps = 1e-7
    p = max(eps, min(1 - eps, p))

    # Calculate binary cross-entropy loss for one prediction
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def bce(p: np.ndarray, y: np.ndarray):
    # Ensure predictions and targets are vectors or matrices
    if p.ndim not in (1, 2) or y.ndim not in (1, 2):
        raise ValueError("BCE supports only 1D vectors and 2D matrices")

    # Predictions and targets must have matching dimensions and shapes
    if p.shape != y.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    total_loss = 0.0
    count = 0

    # Calculate the loss for each element in a 1D vector
    if p.ndim == 1:
        for i in range(p.shape[0]):
            total_loss += s_bce(p[i], y[i])
            count += 1

    # Calculate the loss for each element in a 2D matrix
    elif p.ndim == 2:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                total_loss += s_bce(p[i, j], y[i, j])
                count += 1

    # Prevent division by zero
    if count == 0:
        raise ValueError("BCE cannot calculate loss for empty arrays")

    # Return the average binary cross-entropy loss
    return total_loss / count