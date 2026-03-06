"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class ReLU:
    def __init__(self):
        self.last_input = None
    def forward(self, X):
        self.last_input = X
        return np.maximum(0, X)
    def backward(self, dZ):
        dX = dZ.copy()
        dX[self.last_input <= 0] = 0
        return dX

class Sigmoid:
    def __init__(self):
        self.last_output = None
    def forward(self, X):
        self.last_output = 1 / (1 + np.exp(-X))
        return self.last_output
    def backward(self, dZ):
        return dZ * self.last_output * (1 - self.last_output)

class Tanh:
    def __init__(self):
        self.last_output = None
    def forward(self, X):
        self.last_output = np.tanh(X)
        return self.last_output
    def backward(self, dZ):
        return dZ * (1 - self.last_output ** 2)
