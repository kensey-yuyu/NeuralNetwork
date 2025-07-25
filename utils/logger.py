import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchinfo
from matplotlib import pyplot as plt
from pip._internal.operations import freeze

from config import Config
from nn import NeuralNetwork


class Logger:
    """
    Logger.
    """

    def __init__(self, cfg: Config, model: NeuralNetwork) -> None:
        """
        Initialize logger.

        Args:
            cfg (Config): Configuration (Hyperparameters) for training model.
            model (NeuralNetwork): Neural network module.
        """

        self.log = {
            "epoch": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "train_loss": [],
            "test_loss": [],
            "best": {
                "epoch": 0,
                "test_accuracy": 0.0
            }
        }
        self.cfg = cfg
        self.model = model

        # Path to save results.
        self.path = "./results/" + format(datetime.now(), "%Y-%m-%d_%H:%M:%S")

        # Make results directory.
        os.makedirs(f"{self.path}/codes", exist_ok=True)

        # Save summary.
        summary = str(torchinfo.summary(
            model=self.model,
            input_size=self.cfg.input_size,
            col_names=["input_size", "output_size",
                       "num_params", "kernel_size", "mult_adds"],
            verbose=0,
            row_settings=["var_names", "depth"]))
        print(summary, "\n")
        with open(f"{self.path}/summary.txt", mode="w", encoding="UTF-8") as file:
            file.write(summary)

        # Save hyperparameters.
        hyperparameters = {
            "Seed": self.cfg.seed,
            "Epoch": self.cfg.epoch,
            "Batch size": self.cfg.batch_size,
            "Transform": self.cfg.transform,
            "Loss function": self.cfg.criterion,
            "Learning rate": self.cfg.learning_rate,
            "Weight decay": self.cfg.weight_decay,
            "Optimizer": self.cfg.optimizer,
            "Scheduler": vars(self.cfg.scheduler) if self.cfg.scheduler is not None else None,
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

    def record(self, epoch: int, train_accuracy: float, train_loss: float, test_accuracy: float, test_loss: float) -> None:
        """
        Record the log of training and evaluation.

        Args:
            epoch (int): Current epoch number.
            train_accuracy (float): Train accuracy on current epoch.
            train_loss (float): Train loss on current epoch.
            test_accuracy (float): Test accuracy on current epoch.
            test_loss (float): Test loss on current epoch.
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
        torch.save({"epoch": epoch + 1,
                    "test_accuracy": test_accuracy,
                    "model_state_dict": self.model.state_dict()}, f"{self.path}/model.pth.tar")

        # Save best model.
        if self.log["best"]["test_accuracy"] < test_accuracy:
            self.log["best"]["test_accuracy"] = test_accuracy
            self.log["best"]["epoch"] = epoch + 1
            torch.save({"epoch": epoch + 1,
                        "test_accuracy": test_accuracy,
                        "model_state_dict": self.model.state_dict()}, f"{self.path}/best_model.pth.tar")
        return
