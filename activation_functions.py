import numpy as np

def relu(x):
    x[x < 0] = 0
    return x

def relu_backprop(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x

def sigmoid(x):
    y = np.exp(-x)
    return 1 / (1 + y)

def sigmoid_backprop(x):
    k = sigmoid(x)
    return k*(1-k)
    
def tanh(x):
    y = np.exp(x)
    z = np.exp(-x)
    return (y-z)/(y+z)

def softmax(x):
    z = np.exp(x)
    return z / np.sum(z)

def softmax_backprop(x):
    return x