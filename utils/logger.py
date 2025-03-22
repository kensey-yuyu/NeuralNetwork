import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pip._internal.operations import freeze


class Logger:
    """Logger

    Logger for learning.

    Attributes:
        log: Log for epoch, accuracy and loss.
        cfg: Configurations of model.
        path: Path to save results.

    """

    def __init__(self, cfg, summary) -> None:
        """__init__

        Initialize Logger.

        Args:
            cfg: Configurations of model.
            summary: Summary of model.

        """

        self.log = {
            "epoch": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "train_loss": [],
            "test_loss": [],
        }
        self.cfg = cfg

        # Result path.
        self.path = "./results/" + format(datetime.now(), "%Y-%m-%d_%H:%M:%S")

        # Check results directory.
        os.makedirs(f"{self.path}/codes", exist_ok=True)

        # Save summary.
        with open(f"{self.path}/summary.txt", mode="w", encoding="UTF-8") as file:
            file.write(summary)

        # Save hyperparameters.
        hyperparameters = {
            "Seed": self.cfg.seed,
            "Epoch": self.cfg.epoch,
            "Batch size": self.cfg.batch_size,
            "Learning rate": self.cfg.learning_rate,
            "Scheduler": vars(self.cfg.scheduler) if self.cfg.scheduler is not None else None,
            "Transform": self.cfg.transform,
            "Optimizer": self.cfg.optimizer,
            "Loss function": self.cfg.criterion
        }
        with open(f"{self.path}/hyperparameters.txt", mode="w", encoding="UTF-8") as file:
            for key, value in hyperparameters.items():
                file.write(f"{key}: {value}\n")

        # Save Python files.
        files = [file for file in os.listdir(
            "./") if file.endswith(".py") == True]
        for file in files:
            shutil.copyfile(file, f"{self.path}/codes/{file}")

        # Save Python version.
        with open(f"{self.path}/python-version.lock", mode="w", encoding="UTF-8") as file:
            file.write(f"{sys.version}")

        # Save environments.
        packages = freeze.freeze()
        with open(f"{self.path}/requirements.lock", mode="w", encoding="UTF-8") as file:
            file.writelines("\n".join(packages))
        return

    def record(self, model, epoch, train_accuracy, train_loss, test_accuracy, test_loss) -> None:
        """record

        Record learning history.

        Args:
            model: Model of Neural Network.
            epoch: Current epoch.
            train_accuracy: Train accuracy on current epoch.
            train_loss: Train loss on current epoch.
            test_accuracy: Test accuracy on current epoch.
            test_loss: Test loss on current epoch.

        """

        self.log["epoch"].append(epoch + 1)
        self.log["train_accuracy"].append(train_accuracy)
        self.log["train_loss"].append(train_loss)
        self.log["test_accuracy"].append(test_accuracy)
        self.log["test_loss"].append(test_loss)

        # Save learning history.
        df = pd.DataFrame([self.log["epoch"], self.log["train_accuracy"], self.log["train_loss"], self.log["test_accuracy"],
                           self.log["test_loss"]], index=["Epoch", "Train accuracy", "Train loss", "Test accuracy", "Test loss"]).T
        df.to_csv(f"{self.path}/history.csv", index=False)

        # Save graph of accuracy.
        plt.figure()
        plt.title("Epoch and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(np.arange(1, epoch + 2),
                 self.log["train_accuracy"], label="Train")
        plt.plot(np.arange(1, epoch + 2),
                 self.log["test_accuracy"], label="Test")
        plt.legend()
        plt.savefig(f"{self.path}/Accuracy.png")
        plt.close()

        # Save graph of loss.
        plt.figure()
        plt.title("Epoch and Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.arange(1, epoch + 2),
                 self.log["train_loss"], label="Train")
        plt.plot(np.arange(1, epoch + 2),
                 self.log["test_loss"], label="Test")
        plt.legend()
        plt.savefig(f"{self.path}/Loss.png")
        plt.close()

        # Save model.
        torch.save(model.state_dict(), f"{self.path}/model.pth.tar")
        return
