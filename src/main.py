# Milan Suri
# CSC 561
'''
This is the main pipeline of the project. It combines the logistic regression model with gradient descent to
train my logistic regression model on the breast cancer dataset from sklearn. Achieves these results:

--- Training Results ---
Final loss: 0.10726159
Iterations: 10000
Bias: 0.317636

--- Test Results ---
Accuracy: 0.9912 (99.12%)
Correct predictions: 113
Total test samples: 114

'''

import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from logistic_regression import *
from optimizers import gradient_descent, logistic_gradient


# Load dataset (sklearn is only used to load the data)
dataset = load_breast_cancer()
X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)

print("Dataset:", dataset.DESCR.splitlines()[0])
print("Samples:", X.shape[0])
print("Features:", X.shape[1])


# Shuffle and split the data using NumPy
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))

split = int(0.8 * len(X))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx].copy()
y_train = y[train_idx]
X_test = X[test_idx].copy()
y_test = y[test_idx]


# Standardize features using training data statistics
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1.0

X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)


# Initialize weights and bias
n_features = X_train.shape[1]
initial = np.zeros(n_features + 1, dtype=np.float32)


# Define the objective function
def objective(theta):
    weights = theta[:-1]
    bias = theta[-1]
    return logistic_loss(X_train, y_train, weights, bias)


# Define the gradient function
def gradient(theta):
    return logistic_gradient(X_train, y_train, theta)


# Train logistic regression using your gradient descent
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


# Check for numerical errors
if not np.all(np.isfinite(theta)) or not np.isfinite(loss):
    raise FloatingPointError(
        "Training produced NaN or infinite values. "
        "Check logistic_gradient and the learning rate."
    )


# Evaluate the model on the test data
predictions = np.asarray(predict(X_test, weights, bias)).reshape(-1)

probabilities = np.asarray(predict_proba(X_test, weights, bias)).reshape(-1)

accuracy = np.mean(predictions == y_test)

print("--- Training Results ---")
print("Final loss:", loss)
print("Iterations:", iterations)
print("Bias:", bias)

print("\--- Test Results ---")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("Correct predictions:", int(np.sum(predictions == y_test)))
print("Total test samples:", len(y_test))

# SHAP explanation for the custom logistic regression model
def shap_predict(data):
    return np.asarray(
        predict_proba(data, weights, bias)
    ).reshape(-1)


# Training samples serve as the SHAP background
background = X_train[:30]

# Explain a subset of test samples
X_explain = X_test[:50]

explainer = shap.KernelExplainer(shap_predict, background, feature_names=dataset.feature_names)

shap_values = explainer.shap_values(X_explain, nsamples=200)


# Plot and save the SHAP summary
shap.summary_plot(
shap_values,
    X_explain,
    feature_names=dataset.feature_names,
    max_display=15,
    show=False
)

plt.title("SHAP Summary Plot - Logistic Regression")
plt.tight_layout()

output_file = "shap_summary.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSHAP summary plot saved to: {output_file}")