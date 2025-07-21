import torch

from config import Config
from nn import NeuralNetwork


def train(model: NeuralNetwork, loader: torch.utils.data.DataLoader, cfg: Config) -> tuple[float, float]:
    """
    Train model using mini-batches.

    Args:
        model (NeuralNetwork): Neural network module.
        loader (torch.utils.data.DataLoader): Mini-batches of Train data.
        cfg (Config): Configurations (Hyperparameters) for training.

    Returns:
        tuple[float, float]: Train accuracy and loss.
    """

    # Init.
    model.train()
    corrects = 0
    train_loss = 0

    # Train.
    for batch in loader:
        # Initialize gradients.
        cfg.optimizer.zero_grad()

        # Forward.
        x, t = batch
        x, t = x.to(cfg.device), t.to(cfg.device)
        y = model(x)

        # Calculate loss.
        loss = cfg.criterion(y, t)
        train_loss += loss.item()

        # Backward.
        loss.backward()

        # Update weights with learning rate and gradients.
        cfg.optimizer.step()

        # Calculate corrects.
        corrects += torch.sum(torch.argmax(y, dim=1) == t).item()

    # Calculate train loss.
    train_loss /= len(loader)

    # Calculate train accuracy.
    train_accuracy = 100 * corrects / len(loader.dataset)

    # Update learning rate by scheduler.
    cfg.scheduler.step()
    return train_accuracy, train_loss
