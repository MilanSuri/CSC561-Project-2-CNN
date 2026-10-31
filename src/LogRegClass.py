# Milan Suri
# CSC 561
# Essentially a wrapper class that takes all my methods from the logistic regression and convert it into a class
# that follows essentially the same standards at scikit-learn's Logistic Regression class.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from src.activations import sigmoid
from src.losses import bce
from src.optimizers import *


class LogisticRegression:

    def __init__(self, learning_rate=0.01, threshold=0.5, max_iterations=10000, tolerance=1e-6):
        # User-configurable hyperparameters
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Model parameters (learned during fitting)
        self.weights = None
        self.bias = None

        # Training information
        self.loss_ = None
        self.n_iterations_ = None
        self.n_features_in_ = None

    def _forward(self, X, weights, bias):
        # Calculate weighted sum and apply sigmoid
        weights_col = weights.reshape(-1, 1)

        z = v_dot_prod(X, weights_col)
        z = z.reshape(-1) + bias
        return sigmoid(z)

    def _loss(self, X, y, weights, bias):
        # Calculate binary cross entropy
        predictions = self._forward(X, weights, bias)
        return bce(predictions, y)

    def fit(self, X, y):
        # Convert inputs to NumPy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Validate inputs
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

        self.n_features_in_ = X.shape[1]

        # Initialize weights and bias
        initial = np.zeros(self.n_features_in_ + 1, dtype=np.float32)

        # Objective function
        def objective(theta):
            weights = theta[:-1]
            bias = theta[-1]

            return self._loss(X, y, weights, bias)

        # Gradient function
        def gradient(theta):
            return logistic_gradient(X, y, theta)

        # Perform gradient descent
        theta, loss, iterations = gradient_descent(objective, gradient, initial, learning_rate=self.learning_rate, tolerance=self.tolerance, max_iterations=self.max_iterations)

        # Store learned parameters
        self.weights = theta[:-1]
        self.bias = theta[-1]

        self.loss_ = loss
        self.n_iterations_ = iterations

        return self

    def predict_proba(self, X):

        # Ensure model has been trained
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Input features do not match model weights")

        # Return probabilities
        return self._forward(X, self.weights, self.bias)

    def predict(self, X):

        # Get predicted probabilities
        probabilities = self.predict_proba(X)

        # Apply classification threshold
        return (probabilities >= self.threshold).astype(np.int32)

    def get_params(self, deep=True):
        # Return configurable hyperparameters
        return {
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance
        }

    def set_params(self, **params):
        # Allow users to change hyperparameters
        for key, value in params.items():

            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")

            setattr(self, key, value)

        return self
