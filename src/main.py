
import numpy as np
from sklearn.datasets import load_breast_cancer

from logistic_regression import (
    logistic_loss,
    predict,
    predict_proba
)
from optimizers import gradient_descent, logistic_gradient


# 1. Load dataset using sklearn ONLY
dataset = load_breast_cancer()
X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)

print("Dataset:", dataset.DESCR.splitlines()[0])
print("Samples:", X.shape[0])
print("Features:", X.shape[1])


# 2. Shuffle and split using NumPy
np.random.seed(42)
indices = np.random.permutation(len(X))

split = int(0.8 * len(X))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx].copy()
y_train = y[train_idx]
X_test = X[test_idx].copy()
y_test = y[test_idx]


# 3. Standardize features using NumPy
# Fit scaling parameters on training data only
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Prevent division by zero for constant features
std[std == 0] = 1.0

X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)


# 4. Initialize weights and bias
n_features = X_train.shape[1]
initial = np.zeros(n_features + 1, dtype=np.float32)


# 5. Define objective function
def objective(theta):
    weights = theta[:-1]
    bias = theta[-1]

    return logistic_loss(X_train, y_train, weights, bias)


# 6. Define gradient
def gradient(theta):
    return logistic_gradient(X_train, y_train, theta)


# 7. Train using your gradient descent
theta, loss, iterations = gradient_descent(
    objective,
    gradient,
    initial,
    learning_rate=0.001,
    tolerance=1e-6,
    max_iterations=10000
)

weights = theta[:-1]
bias = theta[-1]


# 8. Check for numerical errors
if not np.all(np.isfinite(theta)) or not np.isfinite(loss):
    raise FloatingPointError(
        "Training produced NaN or infinite values. "
        "Check logistic_gradient and the learning rate."
    )


# 9. Evaluate on test data
predictions = predict(X_test, weights, bias).reshape(-1)
probabilities = predict_proba(X_test, weights, bias).reshape(-1)

accuracy = np.mean(predictions == y_test)

print("\n--- Training Results ---")
print("Final loss:", loss)
print("Iterations:", iterations)
print("Bias:", bias)

print("\n--- Test Results ---")
print("Accuracy:", accuracy)
print("Correct predictions:", np.sum(predictions == y_test))
print("Total test samples:", len(y_test))

print("\n--- First 10 Predictions ---")
for i in range(min(10, len(y_test))):
    print(
        f"Actual: {int(y_test[i])}, "
        f"Predicted: {predictions[i]}, "
        f"Probability: {probabilities[i]:.4f}"
    )
