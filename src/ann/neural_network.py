"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from ann.neural_layer import LinearLayer
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropyLoss, MSELoss
from ann.optimizers import SGD, Momentum, Adam, NAG, RMSProp, Nadam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Args:
            cli_args: Command-line arguments for configuring the network
                      (expects attributes: hidden_size, activation, loss, optimizer,
                       learning_rate, weight_init, weight_decay)
        """
        self.layers = []

        input_size = 784
        output_size = 10

        # Defensive: allow hidden_size to be string like "128,64" or a list of ints
        hidden_sizes = cli_args.hidden_size
        if isinstance(hidden_sizes, str):
            hidden_sizes = [
                int(x.strip()) for x in hidden_sizes.replace("[", "").replace("]", "").split(",") if x.strip()
            ]
        if hidden_sizes is None or len(hidden_sizes) == 0:
            raise ValueError("hidden_size must contain at least one hidden layer size (e.g., [128,64])")

        activation_name = cli_args.activation
        weight_init = cli_args.weight_init

        if activation_name == "relu":
            activation_class = ReLU
        elif activation_name == "sigmoid":
            activation_class = Sigmoid
        elif activation_name == "tanh":
            activation_class = Tanh
        else:
            raise ValueError("Unsupported activation function")

        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        elif cli_args.loss == "mse":
            self.loss_fn = MSELoss()
        else:
            raise ValueError("Unsupported loss function")

        # Optimizer selection
        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(cli_args.learning_rate, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(cli_args.learning_rate, beta=0.9, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        elif cli_args.optimizer == "adam":
            self.optimizer = Adam(cli_args.learning_rate, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(cli_args.learning_rate, beta=0.9, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(cli_args.learning_rate, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        elif cli_args.optimizer == "nadam":
            self.optimizer = Nadam(cli_args.learning_rate, weight_decay=getattr(cli_args, "weight_decay", 0.0))
        else:
            raise ValueError("Unsupported optimizer (start with sgd)")

        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(LinearLayer(prev_size, hidden_size, weight_init))
            self.layers.append(activation_class())
            prev_size = hidden_size

        # final linear
        self.layers.append(LinearLayer(prev_size, output_size, weight_init))

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients and return scalar loss.
        Assumes loss_fn.forward(logits, y_true) was called here and loss_fn.backward()
        returns gradient wrt logits.
        """
        # compute and store loss (loss_fn should keep what's needed for backward)
        loss = self.loss_fn.forward(y_pred, y_true)

        # get gradient of loss w.r.t. logits (implementation-dependent)
        grad = self.loss_fn.backward()

        # backprop through layers in reverse order
        for layer in reversed(self.layers):
            # layer.backward should return gradient for previous layer's outputs
            grad = layer.backward(grad)
        grad_list = [layer.grad_W for layer in self.layers if hasattr(layer, "grad_W")]

        return loss, grad_list

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Train the network for specified epochs.
        Early stopping based on validation loss with patience.
        """
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0
        num_samples = X_train.shape[0]
        iteration = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i: i + batch_size]
                y_batch = y_train_shuffled[i: i + batch_size]

                logits = self.forward(X_batch)
                loss,_ = self.backward(y_batch, logits)
                
                if iteration < 50:
                    first_layer = self.layers[0]
                    grad_matrix = first_layer.grad_W
                    num_neurons = min(5, grad_matrix.shape[1])

                    for j in range(num_neurons):
                        grad_norm = np.linalg.norm(grad_matrix[:, j])
                        wandb.log({f"grad_neuron_{j}": grad_norm, "iteration": iteration})

                iteration += 1

                self.update_weights()
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            val_metrics = self.evaluate(X_val, y_val)

            # Save best weights (deep copy)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self.best_weights = []
                for layer in self.layers:
                    if hasattr(layer, "W"):
                        self.best_weights.append({"W": layer.W.copy(), "b": layer.b.copy()})
            else:
                patience_counter += 1

            # logging
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
            wandb.log({
                "train_loss": avg_loss,
                "epoch": epoch + 1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"]
            })

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # restore best weights if present
        if hasattr(self, "best_weights"):
            weight_idx = 0
            for layer in self.layers:
                if hasattr(layer, "W"):
                    layer.W = self.best_weights[weight_idx]["W"]
                    layer.b = self.best_weights[weight_idx]["b"]
                    weight_idx += 1

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        Returns dictionary with loss, accuracy, precision, recall, f1 and predictions.
        """
        from sklearn.metrics import confusion_matrix

        logits = self.forward(X)
        loss = self.loss_fn.forward(logits, y)
        predictions = np.argmax(logits, axis=1)
        cm = confusion_matrix(y, predictions)  # kept if needed externally

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average="macro", zero_division=0)
        recall = recall_score(y, predictions, average="macro", zero_division=0)
        f1 = f1_score(y, predictions, average="macro", zero_division=0)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions,
            "confusion_matrix": cm
        }

    def predict(self, X):
        """
        Convenience method: returns class predictions (ints).
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    # -----------------------------------------------------------------
    # weight access helpers (required by updated instructions)
    # -----------------------------------------------------------------
    def get_weights(self):
        """
        Return a dictionary containing all layer weights and biases.
        Keys are of the form "W0", "b0", "W1", "b1", etc., where
        layer 0 is the first layer in `self.layers` that has weights.
        """
        d = {}
        layer_idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                d[f"W{layer_idx}"] = layer.W.copy()
                d[f"b{layer_idx}"] = layer.b.copy()
                layer_idx += 1
        return d

    def set_weights(self, weight_dict):
        """
        Load weights from a dictionary produced by `get_weights`.
        Only updates weights for layers whose keys are present.
        """
        layer_idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                w_key = f"W{layer_idx}"
                b_key = f"b{layer_idx}"
                if w_key in weight_dict:
                    layer.W = weight_dict[w_key].copy()
                if b_key in weight_dict:
                    layer.b = weight_dict[b_key].copy()
                layer_idx += 1