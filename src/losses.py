import math
from xml.dom import ValidationErr
import numpy as np

def s_bce(p, y):
    eps = 1e-7
    p = max(eps, min(1 - eps, p))

    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

def bce(p: np.ndarray, y: np.ndarray):
    if p.ndim not in (1, 2) or y.ndim not in (1, 2):
        raise ValueError("BCE supports only 1D vectors and 2D matrices")

    if p.shape != y.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    total_loss = 0.0
    count = 0

    if p.ndim == 1:
        for i in range(p.shape[0]):
            total_loss += s_bce(p[i], y[i])
            count += 1

    elif p.ndim == 2:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                total_loss += s_bce(p[i, j], y[i, j])
                count += 1

    if count == 0:
        raise ValueError("BCE cannot calculate loss for empty arrays")

    return total_loss / count