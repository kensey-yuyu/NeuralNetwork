import os
from typing import Tuple

import torch
import torchinfo
import torchvision
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

import utils
from nn import NeuralNetwork


def main() -> None:
    """main

    Main function for Neural Network.
    You can change hyperparameters, transforms, dataset, loss function, optimizer and etc.

    """

    # Init config.
    cfg = utils.Config()
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters.
    cfg.seed = 0
    cfg.epoch = 50
    cfg.batch_size = 128
    cfg.learning_rate = 1e-2
    cfg.weight_decay = 0
    cfg.input_size = (cfg.batch_size, 3, 32, 32)

    # Set seed.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Transforms.
    cfg.transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset.
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=cfg.transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=cfg.transform)

    # Generate Mini-batch.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count())

    # Model.
    model = NeuralNetwork().to(cfg.device)

    # Loss function.
    cfg.criterion = nn.CrossEntropyLoss()

    # Optimizer.
    cfg.optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Learning rate scheduler.
    cfg.scheduler = lr_scheduler.ExponentialLR(
        optimizer=cfg.optimizer, gamma=0.95)

    # Summary.
    summary = str(torchinfo.summary(
        model=model, input_size=cfg.input_size, verbose=0))
    print(summary)

    # Init logger.
    logger = utils.Logger(cfg, summary)

    # Learning.
    for epoch in tqdm(range(cfg.epoch), desc="Learning"):
        train_accuracy, train_loss = train(model, train_loader, cfg)
        test_accuracy, test_loss = test(model, test_loader, cfg)
        logger.record(model, epoch, train_accuracy, train_loss,
                      test_accuracy, test_loss)
        tqdm.write(
            f"Epoch: {epoch}, Train accuracy: {train_accuracy:.3f}, Train loss: {train_loss:.3f}, Test accuracy: {test_accuracy:.3f}, Test loss: {test_loss:.3f}")
    return


def train(model, loader, cfg) -> Tuple[float, float]:
    """train

    Train on learning.

    Args:
        model (nn.Module): Model of Neural Network.
        logger (Logger): Logger for learning.
        cfg (Config): Config for learning.

    Returns:
        Tuple[float, float]: Train accuracy and loss.

    """

    # Init.
    model.train()
    corrects = 0
    train_loss = 0

    # Train.
    for batch in loader:
        # Forward.
        x, t = batch
        x, t = x.to(cfg.device), t.to(cfg.device)
        y = model(x)

        # Calculate loss.
        loss = cfg.criterion(y, t)
        train_loss += loss.item() * x.shape[0]

        # Initialize gradients.
        cfg.optimizer.zero_grad()

        # Backward.
        loss.backward()

        # Update weights with learning rate and gradients.
        cfg.optimizer.step()

        # Calculate corrects.
        corrects += torch.sum(torch.argmax(y, dim=1) == t)

    # Calculate train loss.
    train_loss /= len(loader.dataset)

    # Calculate train accuracy.
    train_accuracy = (100 * corrects / len(loader.dataset)).item()

    # Update learning rate by scheduler.
    cfg.scheduler.step()
    return train_accuracy, train_loss


def test(model, loader, cfg) -> Tuple[float, float]:
    """test

    Test on learning.

    Args:
        model (nn.Module): Model of Neural Network.
        logger (Logger): Logger for learning.
        cfg (Config): Config for learning.

    Returns:
        Tuple[float, float]: Test accuracy and loss.

    """

    # Init.
    model.eval()
    corrects = 0
    test_loss = 0

    # Test.
    with torch.no_grad():
        for batch in loader:
            # Forward.
            x, t = batch
            x, t = x.to(cfg.device), t.to(cfg.device)
            y = model(x)

            # Calculate loss.
            test_loss += cfg.criterion(y, t).item() * x.shape[0]

            # Calculate corrects.
            corrects += torch.sum(torch.argmax(y, dim=1) == t)

    # Calculate test loss.
    test_loss /= len(loader.dataset)

    # Calculate test accuracy.
    test_accuracy = (100 * corrects / len(loader.dataset)).item()
    return test_accuracy, test_loss


if __name__ == "__main__":
    main()
