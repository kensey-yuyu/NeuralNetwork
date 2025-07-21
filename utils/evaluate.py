import torch

from config import Config
from nn import NeuralNetwork


def evaluate(model: NeuralNetwork, loader: torch.utils.data.DataLoader, cfg: Config) -> tuple[float, float]:
    """
    Evaluate model.

    Args:
        model (NeuralNetwork): Neural network module.
        loader (torch.utils.data.DataLoader): Data for evaluation.
        cfg (Config): Configuration (Hyperparameters) for evaluation.

    Returns:
        tuple[float, float]: Evaluate accuracy and loss.
    """

    # Init.
    model.eval()
    corrects = 0
    evaluate_loss = 0

    # Evaluate.
    with torch.no_grad():
        for batch in loader:
            # Forward.
            x, t = batch
            x, t = x.to(cfg.device), t.to(cfg.device)
            y = model(x)

            # Calculate loss.
            evaluate_loss += cfg.criterion(y, t).item()

            # Calculate corrects.
            corrects += torch.sum(torch.argmax(y, dim=1) == t).item()

    # Calculate evaluate loss.
    evaluate_loss /= len(loader)

    # Calculate evaluate accuracy.
    evaluate_accuracy = 100 * corrects / len(loader.dataset)
    return evaluate_accuracy, evaluate_loss
