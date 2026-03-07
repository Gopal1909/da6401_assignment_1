"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data(dataset_name):
    if dataset_name == "mnist":
        dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    elif dataset_name == "fashion_mnist":
        dataset = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    else:
        raise ValueError("Unsupported dataset")
    
    X = np.array(dataset['data'], dtype=np.float32) /255.0
    y = np.array(dataset['target'], dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    return X_train, y_train, X_test, y_test