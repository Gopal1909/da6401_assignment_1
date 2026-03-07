"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
import os
import argparse
import wandb
import json

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Optional positional argument for epochs (for compatibility with test scripts)
    parser.add_argument(
        "epochs_pos",
        nargs="?",
        type=int,
        default=None,
        help="(Optional positional) Number of training epochs"
    )
    
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
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
        help="Mini-batch size (default: 64)"
    )
    
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (default: 0.001)"
    )
    
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
        default='adam',
        help="Optimizer to use (default: 'adam')"
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
        type=int,
        nargs="+",
        default=[128,128],
        help="Number of neurons in each hidden layer (comma-separated, e.g. '128,64')"
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
        help="Activation function for hidden (default: 'relu')"
    )
    
    parser.add_argument(
        "--weight_init",
        "-wi",
        type=str,
        choices=['zeros', 'random', 'xavier'],
        default='xavier',
        help="Weight initialization method (default: 'xavier')"
    )
    
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization) factor (default: 0.0)"
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="da6401_assignment_1-src",
        help="W&B project name for logging"
    )
    
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="src/best_model.npy",
        help="Relative path to save trained model (default: 'src/best_model.npy')"
    )
    
    args = parser.parse_args()
    
    # If positional epochs argument is provided, use it to override --epochs
    if args.epochs_pos is not None:
        args.epochs = args.epochs_pos
    
    return args



def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    
    
    use_wandb = True
    try:
        wandb.init(project=args.wandb_project)
        config = wandb.config
        config.update(vars(args), allow_val_change=True)
        merged = argparse.Namespace(**{**vars(args), **dict(config)})
    except Exception:
        # fallback: no wandb available / offline -- still run
        use_wandb = False
        merged = args

    raw = args.hidden_size
    if isinstance(raw, (list, tuple)):
        hidden_sizes = [int(x) for x in raw]
    else:
        s = str(raw).strip()
        s = s.lstrip("[").rstrip("]")
        hidden_sizes = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(hidden_sizes) != args.num_layers:
        raise ValueError(
            f"--num_layers ({args.num_layers}) must match the number of sizes in --hidden_size ({len(hidden_sizes)})"
        )
    if merged.epochs <=0:
        raise ValueError("epochs must be a positive integer")

    if merged.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if merged.learning_rate <= 0:
        raise ValueError("learning_rate must be a positive float")

    if merged.num_layers <= 0:
        raise ValueError("At least one hidden layer size must be provided via --hidden_size")
    
    X_train, y_train, X_test, y_test = load_data(merged.dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, shuffle=True
    )
    
    model = NeuralNetwork(merged)
    model.train(X_train, y_train, X_val, y_val, merged.epochs, merged.batch_size)
    
    metrics = model.evaluate(X_test, y_test)
    
    try:
        wandb.log({
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds = metrics["predictions"],
                y_true = y_test,
                class_names=[str(i) for i in range(10)]
            )
        })
    except:
        pass
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # ensure save-directory exists (handle paths with no directory component)
    save_dir = os.path.dirname(merged.model_save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    # compute weights dictionary via network helper and record metrics
    best_weights = model.get_weights()

    # base file locations inside src
    model_path = os.path.join("src", "best_model.npy")
    config_path = os.path.join("src", "best_config.json")

    # optionally compare with previous best by F1
    save_model = True
    prev_best = {
        "test_f1": -1.0
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                prev_best = json.load(f)
        except Exception:
            prev_best = {"test_f1": -1.0}
    if metrics["f1"] < prev_best.get("test_f1", -1.0):
        save_model = False

    if save_model:
        np.save(model_path, best_weights)
        # save configuration together with current test metrics
        cfg = vars(args).copy()
        cfg["test_loss"] = metrics["loss"]
        cfg["test_accuracy"] = metrics["accuracy"]
        cfg["test_precision"] = metrics["precision"]
        cfg["test_recall"] = metrics["recall"]
        cfg["test_f1"] = metrics["f1"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Best model and config saved to {model_path} and {config_path}")
    else:
        print("Test F1 did not improve; existing best model retained.")

    wandb.finish()


if __name__ == '__main__':
    main()
