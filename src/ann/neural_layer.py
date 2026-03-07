"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class LinearLayer:
    def __init__(self, in_features: int, out_features: int, weight_init: str = "xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        
        if weight_init == "zeros":
            self.W = np.zeros((in_features, out_features))

        elif weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01

        elif weight_init == "xavier":
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))

        else:
            raise ValueError(f"Unsupported weight initialization method: {weight_init}")
        
        
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
        self.last_input = None
        
    
    def forward(self, X):
        
        self.last_input = X
        return X @ self.W + self.b
    
    def backward(self, dZ):
        # Ensure dZ is 2D (batch_size, out_features)
        if dZ.ndim == 1:
            dZ = dZ.reshape(1, -1)
        if self.last_input.ndim == 1:
            self.last_input = self.last_input.reshape(1, -1)
        
        self.grad_W = self.last_input.T @ dZ
        self.grad_b = np.sum(dZ, axis=0)
        return dZ @ self.W.T
    