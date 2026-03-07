"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.y_pred = probs
        self.y_true = y_true
        
        batch_size = logits.shape[0]
        correct_class_probs = probs[np.arange(batch_size), y_true]
        loss = -np.mean(np.log(correct_class_probs + 1e-9))
        
        return loss
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        
        d_logits = self.y_pred.copy()
        d_logits[np.arange(batch_size), self.y_true] -=1
        d_logits /= batch_size
        
        return d_logits

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        if y_true.ndim == 1:
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot
        self.y_pred = y_pred
        self.y_true = y_true
        
        return np.mean((y_pred - y_true) **2)
    
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]
    