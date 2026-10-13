# Add ReLU, ReLU derivative, Sigmoid, Sigmoid derivative
def a_relu (x):
    return max(0, x)

def a_relu_d(x):
    if x > 0:
        return 1
    else:
        return 0

def a_sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def a_sigmoid_d(x):
    sigmoid = a_sigmoid(x)
    return sigmoid * (1-sigmoid)