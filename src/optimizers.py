from Math.vector_ops import *

def gradient_descent(f, gradient, initial: np.array, learning_rate: float = 0.01, tolerance: float = 1e-6, max_iterations: int = 10000):
    # Get starting point of GD
    x = np.array(initial, dtype=np.float32)
    iteration = 0

    # Iterate through
    while iteration < max_iterations:
        grad = np.array(gradient(x), dtype=np.float32)

        if grad.shape != x.shape:
            raise ValueError("Gradient and initial vector have different shapes")

        # Check if the gradient is small enough to stop
        grad_norm = np.sqrt(v_dot_prod(grad, grad))

        if grad_norm < tolerance:
            break

        # Move in the opposite direction of the gradient
        scaled_grad = v_scal_mul(grad, learning_rate)
        x = v_add(x, v_scal_mul(scaled_grad, -1.0))

        iteration += 1
    return x, f(x), iteration

def logistic_gradient(X, y, theta):
    m = X.shape[0]
    weights = theta[:-1]
    bias = theta[-1]

    z = v_dot_prod(X, weights.reshape(-1, 1)).reshape(-1) + bias
    predictions = 1.0 / (1.0 + np.exp(-z))

    errors = predictions - y

    grad_weights = v_dot_prod(transpose(X), errors.reshape(-1, 1)).reshape(-1) / m
    grad_bias = np.sum(errors) / m

    return np.concatenate([grad_weights, np.array([grad_bias], dtype=np.float32)]).astype(np.float32)