import numpy as np
import scipy
class Neural_Network:

    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.10):
        self.w_ih = np.random.randn(n_hidden, n_inputs) * 0.01
        self.w_ho = np.random.randn(n_outputs, n_hidden) * 0.01
        self.b_ih = np.random.randn(n_hidden, 1) * 0.1
        self.b_ho = np.random.randn(n_outputs, 1) * 0.1
        self.lr = lr #learning rate
        self.activation_function = lambda x : scipy.special.expit(x)

    def feed_forward(self, inputs):
        input_arr = np.array([[elem] for elem in inputs], ndmin=2)

        #activation of the hidden layer
        z_1 = self.w_ih.dot(input_arr) + self.b_ih
        a_1 = self.activation_function(z_1)

        #activation of the output layer
        z_2 = self.w_ho.dot(a_1) + self.b_ho
        a_2 = self.activation_function(z_2)

        return [z_1, z_2, a_1, a_2]

    def query(self, inputs):
        return self.feed_forward(inputs)[3]
    
    def backpropogate(self, inputs, targets):
        z_1, z_2, a_1, a_2 = self.feed_forward(inputs)
        inputs = np.array([[elem] for elem in inputs])
        m = len(targets)
        targets = np.array([[elem] for elem in targets], ndmin=2)
        
        #error in the output layer
        #dA2 = a_2 - targets
        dA2 = a_2 - targets
        dZ2 =  dA2 * (a_2 * (1 - a_2))

        #calculate the changes required(gradient descent) for the output layer
        dW2 = (1/m) * dZ2.dot(a_1.copy().T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        #error in the hidden layer
        dA1 = self.w_ho.copy().T.dot(dZ2)
        dZ1 = dA1 * (a_1 * (1 - a_1))

        #changes required for the hidden layer        
        dW1 = (1/m) * dZ1.dot(inputs.copy().T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return [dW1, dW2, db1, db2]
    
    def train(self, inputs, targets):
        dW1, dW2, db1, db2 = self.backpropogate(inputs, targets)

        self.w_ho = self.w_ho - self.lr*dW2
        self.w_ih = self.w_ih - self.lr*dW1
        self.b_ih = self.b_ih - self.lr*db1
        self.b_ho = self.b_ho - self.lr*db2