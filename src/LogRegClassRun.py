# Milan Suri
# CSC 561
# This file is the main pipeline for running the logistic regression class implementation that matches or is close
# to the scikit-learn implementation. Uses random data, and then trains a log regression model, allows you to
# apply default or specify hyperparameters and then fits and runs.

import numpy as np
from src.LogRegClass import LogisticRegression

X_train = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [2.0, 1.0],
    [3.0, 3.5],
    [3.5, 4.0],
    [4.0, 3.0]
])

y_train = np.array([0, 0, 0, 1, 1, 1])


# Initialize model with default parameters
model = LogisticRegression()

# Specify custom hyperparameters
# model = LogisticRegression(
#     learning_rate=0.05,
#     threshold=0.6,
#     max_iterations=5000,
#     tolerance=1e-8
# )

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
probabilities = model.predict_proba(X_train)


# Display results
print("Weights:", model.weights)
print("Bias:", model.bias)

print("Final Loss:", model.loss_)
print("Iterations:", model.n_iterations_)

print("\nPredicted Probabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual Labels:")
print(y_train)
