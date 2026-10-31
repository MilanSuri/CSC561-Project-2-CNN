# Milan Suri
# CSC561
# Backward pass for neural network. This implements the backwards pass or backprop for my NN,
# where it starts at the output and then moves through the network propagating the error in the gradients back
# to the beginning so it can identify the errors and correct.
# https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/

from src.activations import *
from cMath.vector_ops import *


def backward_pass(X, y, W2, predictions, Z1, A1):
    '''Implements the backpropogation algorithm which calculates the gradients for the loss function
     with respect to the weights and biases and then starts at output layer and moves through the network using
      chain rule. Finally, it returns the gradients. '''
    m = X.shape[0]

    # Output layer error (Sigmoid + BCE)
    dZ2 = v_scal_mul( v_add(predictions, -y), 1 / m )

    # Output layer gradients
    dW2 = v_dot_prod(transpose(A1), dZ2)
    db2 = v_col_sum(dZ2)

    # Propagate error backwards
    dA1 = v_dot_prod(dZ2, transpose(W2))

    # Apply ReLU derivative
    dZ1 = v_elem_mul(dA1, relu_d(Z1))

    # Hidden layer gradients
    dW1 = v_dot_prod(transpose(X), dZ1)
    db1 = v_col_sum(dZ1)

    return dW1, db1, dW2, db2