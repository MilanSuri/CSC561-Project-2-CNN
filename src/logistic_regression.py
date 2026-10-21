import numpy as np
from Math.vector_ops import *
from activations import sigmoid
from losses import bce
from optimizers import *

def logistic_forward(X, weights, bias):
    weights_col = weights.reshape(-1, 1)
    z = v_dot_prod(X, weights_col)
    z = z.reshape(-1) + bias
    predictions = sigmoid(z)

    return predictions

def logistic_loss(X, y, weights, bias):
    predictions = logistic_forward(X, weights, bias)
    return bce(predictions, y)

def train_log_reg(X, y, learning_rate = 0.01, tolerance=1e-6, max_iterations=10000):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must contain one label per sample")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain finite values")

    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be either 0 or 1")

    n_features = X.shape[1]

    initial = np.zeros(n_features + 1, dtype=np.float32)

    def objective(theta):
        weights = theta[:-1]
        bias = theta[-1]

        return logistic_loss(X, y, weights, bias)

    def gradient():
        grad_weights, grad_bias = logistic_gradient(X, y, theta)

        return np.concatenate([grad_weights, np.array([grad_bias], dtype=np.float32)])

    theta, loss, iterations = gradient_descent(
        objective,
        gradient,
        initial,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_iterations=max_iterations
    )

    weights = theta[:-1]
    bias = theta[-1]

    return weights, bias, loss, iterations

def predict_proba(X, weights, bias):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2 or X.shape[1] != len(weights):
        raise ValueError("Input features do not match model weights")

    return logistic_forward(X, weights, bias)


def predict(X, weights, bias, threshold=0.5):
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(np.int32)