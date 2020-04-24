import numpy as np
import matplotlib.pyplot as plt
from activation_functions import relu, sigmoid, softmax, relu_backprop, sigmoid_backprop, softmax_backprop

class nn():

    activation_dict = {
        'relu': relu,
        'sigmoid': sigmoid
    }

    activation_backprop_dict = {
        'relu': relu_backprop,
        'sigmoid': sigmoid_backprop
    }

    def __init__(self, hparams):
        self.hparams = hparams
        self.weights = []
        self.biases = []
        self.activation_functions = []

        self.error = []

        self.num_layers = len(hparams)

        for hparam in hparams:
            
            input_dim, output_dim, activation = hparam
            w = np.random.random((output_dim, input_dim))
            b = np.random.random(output_dim)

            self.weights.append(w)
            self.biases.append(b)
            self.activation_functions.append(activation)

    
    def forward(self, x):
        self.activations = []
        self.values = []
        for i in range(self.num_layers):
            self.activations.append(x)
            z = np.matmul(self.weights[i], x) + self.biases[i]
            self.values.append(z)
            x = self.activation_dict[self.activation_functions[i]](z)
        return x

    def backprop(self, y_pred, y_true, learning_rate=0.00001):
        
        loss = (y_pred - y_true)** 2
        self.error.append(loss)

        dLdy = 2 * (y_pred - y_true)
        dydz = self.activation_backprop_dict[self.activation_functions[-1]](self.values[-1])
        dLdz = dLdy * dydz
        
        for i in range(self.num_layers):
            j = self.num_layers - i - 1

            dLdb = dLdz
            dLdw = np.array([dLdz[k]*self.activations[j] for k in range(dLdz.shape[0])])

            dadz = self.activation_backprop_dict[self.activation_functions[j - 1]](self.values[j - 1])
            dzda = self.weights[j]

            # change here
            if j > 0: dLdz = np.sum(np.array([dLdz[k] * dadz * dzda[k] for k in range(dLdz.shape[0])]), 0)
            
            # update weights
            self.biases[j] -= learning_rate*dLdb
            self.weights[j] -= learning_rate*dLdw

    def train(self, x, y):
        y_pred = self.forward(x)
        self.backprop(y_pred, y)


hyper_paramaters = [
    (100, 24, 'relu'),
    (24, 1, 'relu')
]

model = nn(hyper_paramaters)

abc = np.array([3, 6, 5])

N = 100

x = np.random.random((N, 100))

for _ in range(5):
    for i in range(N):
        model.train(x[i], np.sum(x[i]))

plt.plot(model.error)
plt.show()

