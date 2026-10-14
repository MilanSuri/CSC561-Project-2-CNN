import math

def a_bce(p, y):
    eps = 1e-7
    p = max(eps, min(1 - eps, p))

    return -(y * math.log(p) + (1 - y) * math.log(1 - p))