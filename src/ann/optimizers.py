"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay
    
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b
                
                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b
    
class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, layers):
        for idx, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue
            if idx not in self.velocities:
                self.velocities[idx] = {
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                }

            g_W = layer.grad_W + self.weight_decay * layer.W
            g_b = layer.grad_b
            
            v_W = self.beta * self.velocities[idx]["v_W"] - self.lr * g_W
            v_b = self.beta * self.velocities[idx]["v_b"] - self.lr * g_b
            
            self.velocities[idx]["v_W"] = v_W
            self.velocities[idx]["v_b"] = v_b

            layer.W += v_W
            layer.b += v_b

class Adam:
    def __init__(self, learning_rate=0.001, beta1 =0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = {}
        self.v = {}
        self.t =0
    
    def step(self, layers):
        self.t += 1
        
        for idx, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue
            if idx not in self.m:
                self.m[idx] = {
                    "m_W": np.zeros_like(layer.W),
                    "m_b": np.zeros_like(layer.b)
                }
                self.v[idx] = {
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                }
            g_W = layer.grad_W + self.weight_decay * layer.W
            g_b = layer.grad_b
                
            self.m[idx]["m_W"] = (
                self.beta1 * self.m[idx]["m_W"]
                + (1 - self.beta1) * g_W
            )
                
            self.m[idx]["m_b"] = (
                self.beta1 * self.m[idx]["m_b"]
                + (1 - self.beta1) * g_b
            )
                
            self.v[idx]["v_W"] = (
                self.beta2 * self.v[idx]["v_W"]
                + (1 - self.beta2) * (g_W **2)
            )
                
            self.v[idx]["v_b"] = (
                self.beta2 * self.v[idx]["v_b"]
                + (1 - self.beta2) * (g_b **2)
            )
            
            m_W_hat = self.m[idx]["m_W"] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m[idx]["m_b"] / (1 - self.beta1 ** self.t)
            
            v_W_hat = self.v[idx]["v_W"] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v[idx]["v_b"] / (1 - self.beta2 ** self.t)
            
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, layers):
        for idx, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if idx not in self.velocities:
                self.velocities[idx] = {
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                }

            v_prev_W = self.velocities[idx]["v_W"]
            v_prev_b = self.velocities[idx]["v_b"]

            # Update velocity
            v_W = self.beta * v_prev_W - self.lr * (layer.grad_W + self.weight_decay * layer.W)
            v_b = self.beta * v_prev_b - self.lr * layer.grad_b

            # Update weights using Nesterov lookahead
            layer.W += -self.beta * v_prev_W + (1 + self.beta) * v_W
            layer.b += -self.beta * v_prev_b + (1 + self.beta) * v_b

            self.velocities[idx]["v_W"] = v_W
            self.velocities[idx]["v_b"] = v_b

class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.v = {}

    def step(self, layers):
        for idx, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if idx not in self.v:
                self.v[idx] = {
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                }

            g_W = layer.grad_W + self.weight_decay * layer.W
            g_b = layer.grad_b
            # Update running average of squared gradients
            self.v[idx]["v_W"] = (
                self.beta * self.v[idx]["v_W"]
                + (1 - self.beta) * (g_W ** 2)
            )

            self.v[idx]["v_b"] = (
                self.beta * self.v[idx]["v_b"]
                + (1 - self.beta) * (g_b ** 2)
            )

            # Update parameters
            layer.W -= self.lr * g_W / (
                np.sqrt(self.v[idx]["v_W"]) + self.epsilon
            )

            layer.b -= self.lr * g_b / (
                np.sqrt(self.v[idx]["v_b"]) + self.epsilon
            )

class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1

        for idx, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue

            if idx not in self.m:
                self.m[idx] = {
                    "m_W": np.zeros_like(layer.W),
                    "m_b": np.zeros_like(layer.b)
                }
                self.v[idx] = {
                    "v_W": np.zeros_like(layer.W),
                    "v_b": np.zeros_like(layer.b)
                }

            g_W = layer.grad_W + self.weight_decay * layer.W
            g_b = layer.grad_b

            # Update biased first moment
            self.m[idx]["m_W"] = (
                self.beta1 * self.m[idx]["m_W"]
                + (1 - self.beta1) * g_W
            )

            self.m[idx]["m_b"] = (
                self.beta1 * self.m[idx]["m_b"]
                + (1 - self.beta1) * g_b
            )

            # Update biased second moment
            self.v[idx]["v_W"] = (
                self.beta2 * self.v[idx]["v_W"]
                + (1 - self.beta2) * (g_W ** 2)
            )

            self.v[idx]["v_b"] = (
                self.beta2 * self.v[idx]["v_b"]
                + (1 - self.beta2) * (g_b ** 2)
            )

            # Bias correction
            m_W_hat = self.m[idx]["m_W"] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m[idx]["m_b"] / (1 - self.beta1 ** self.t)

            v_W_hat = self.v[idx]["v_W"] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v[idx]["v_b"] / (1 - self.beta2 ** self.t)

            # Nadam correction
            m_W_nadam = (
                self.beta1 * m_W_hat
                + ((1 - self.beta1) * g_W) / (1 - self.beta1 ** self.t)
            )

            m_b_nadam = (
                self.beta1 * m_b_hat
                + ((1 - self.beta1) * g_b) / (1 - self.beta1 ** self.t)
            )

            # Update parameters
            layer.W -= self.lr * m_W_nadam / (
                np.sqrt(v_W_hat) + self.epsilon
            )

            layer.b -= self.lr * m_b_nadam / (
                np.sqrt(v_b_hat) + self.epsilon
            )
