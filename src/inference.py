"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix


def parse_arguments():
    """
    Parse command-line arguments.  Matches the training CLI; defaults
    correspond to the best configuration found so far (128,128 relu,
    adam, etc.).
    """
    parser = argparse.ArgumentParser(description='Run inference or evaluate a model')

    parser.add_argument(
        "--model_path",
        type=str,
        default="src/best_model.npy",
        help="Path to saved model weights (relative)"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=['mnist', 'fashion_mnist'],
        default='mnist',
        help="Dataset to use (default: 'mnist')"
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=64,
        help="Batch size for inference (not used by current implementation)"
    )

    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate (only needed if rebuilding model)"
    )

    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
        default='adam',
        help="Optimizer used during training (for config only)"
    )

    parser.add_argument(
        "--loss",
        "-l",
        type=str,
        choices=['cross_entropy', 'mse'],
        default='cross_entropy',
        help="Loss function (default: 'cross_entropy')"
    )

    parser.add_argument(
        "--hidden_size",
        "-sz",
        type=str,
        default="128,128",
        help="Number of neurons in each hidden layer (comma-separated)"
    )

    parser.add_argument(
        "--num_layers",
        "-nhl",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)"
    )
    
    parser.add_argument(
        "--activation",
        "-a",
        type=str,
        choices=['relu', 'sigmoid', 'tanh'],
        default='relu',
        help="Activation function for hidden layers (default: 'relu')"
    )

    parser.add_argument(
        "--weight_init",
        "-wi",
        type=str,
        choices=['random', 'xavier'],
        default='xavier',
        help="Weight initialization method (default: 'xavier')"
    )

    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=0.0,
        help="Weight decay (L2) factor"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="assignment_1",
        help="W&B project name (unused during inference)"
    )

    return parser.parse_args()


def load_model(model_path, config):
    """
    Load trained model from disk using the dictionary format produced
    by NeuralNetwork.get_weights().
    """
    model = NeuralNetwork(config)

    data = np.load(model_path, allow_pickle=True).item()
    model.set_weights(data)
    return model


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    metrics = model.evaluate(X_test, y_test)
    
    predictions = np.argmax(logits, axis=1)
    
    cm = confusion_matrix(y_test, predictions)

    return {
        "logits": logits,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": cm
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    hidden_sizes = [
        int(x.strip())
        for x in args.hidden_size.replace("[", "").replace("]", "").split(",")
    ]
    
    if len(hidden_sizes) != args.num_layers:
        raise ValueError(f"--num_layers ({args.num_layers}) must match the number of sizes in --hidden_size ({len(hidden_sizes)})")
    
    class Config:
        pass

    config = Config()
    config.hidden_size = hidden_sizes
    config.activation = args.activation
    config.loss = args.loss
    config.weight_init = args.weight_init
    # learning rate / optimizer only matter for building structure
    config.learning_rate = args.learning_rate
    config.optimizer = args.optimizer
    config.weight_decay = args.weight_decay
    
    _, _, X_test, y_test = load_data(args.dataset)

    model = load_model(args.model_path, config)
    
    results = evaluate_model(model, X_test, y_test)

    print("\n===== TEST METRICS =====")
    for key, value in results.items():
        if key == "confusion_matrix":
            print("\nConfusion Matrix:")
            print(value)
        else:
            print(f"{key}: {value}")
    
    print("Evaluation complete!")
    
    return results


if __name__ == '__main__':
    main()
