# Milan Suri
# This file is used to develop the functions for logistic regression. This is used to implement the function that
# Are going to be used for the forward propogation, computing the loss, training loops, predictions of log reg.
# Used only in class materials

from src.activations import sigmoid
from src.losses import bce
from src.optimizers import *

def logistic_forward(X, weights, bias):
    # Calculate the weighted sum of inputs and apply the sigmoid function
    weights_col = weights.reshape(-1, 1)
    z = v_dot_prod(X, weights_col)
    z = z.reshape(-1) + bias
    predictions = sigmoid(z)

    return predictions

def logistic_loss(X, y, weights, bias):
    # Measure the difference between predicted probabilities and actual labels
    predictions = logistic_forward(X, weights, bias)
    return bce(predictions, y)

def train_log_reg(X, y, learning_rate = 0.01, tolerance=1e-6, max_iterations=10000):
    # Convert inputs and labels to NumPy arrays with consistent data types
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Validate the input dimensions and label format
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

    # Initialize all weights and the bias to zero
    initial = np.zeros(n_features + 1, dtype=np.float32)

    # Calculate the loss for the current weights and bias
    def objective(theta):
        weights = theta[:-1]
        bias = theta[-1]

        return logistic_loss(X, y, weights, bias)

    # Compute the gradients of the weights and bias for optimization
    def gradient():
        grad_weights, grad_bias = logistic_gradient(X, y, theta)

        return np.concatenate([grad_weights, np.array([grad_bias], dtype=np.float32)])

    # Update parameters using gradient descent to minimize the loss
    theta, loss, iterations = gradient_descent(
        objective,
        gradient,
        initial,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_iterations=max_iterations
    )

    # Separate the trained weights and bias from the optimized parameters
    weights = theta[:-1]
    bias = theta[-1]

    return weights, bias, loss, iterations

def predict_proba(X, weights, bias):
    # Convert input data to a NumPy array and handle a single sample
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Ensure the input features match the trained model
    if X.ndim != 2 or X.shape[1] != len(weights):
        raise ValueError("Input features do not match model weights")

    # Return the predicted probability for each sample
    return logistic_forward(X, weights, bias)


def predict(X, weights, bias, threshold=0.5):
    # Classify samples based on whether their probabilities meet the threshold
    probabilities = predict_proba(X, weights, bias)
    return (probabilities >= threshold).astype(np.int32)