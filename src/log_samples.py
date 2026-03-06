import wandb
import numpy as np
from utils.data_loader import load_data

wandb.init(project="da6401_assignment_1-src", name="mnist_sample_images")

X_train, y_train, _, _ = load_data("mnist")

table = wandb.Table(columns=["image", "label"])

for digit in range(10):
    idx = np.where(y_train == digit)[0][:5]

    for i in idx:
        img = X_train[i].reshape(28, 28)

        table.add_data(
            wandb.Image(img),
            int(y_train[i])
        )

wandb.log({"MNIST Sample Images": table})

wandb.finish()