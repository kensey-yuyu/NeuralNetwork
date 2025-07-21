import os

import torch
import torchvision
from torchvision import transforms


def load_dataset(root: str, train: bool, transform: transforms.Compose, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Load dataset.

    Args:
        root (str): Path of saved dataset.
        train (bool): If it is true, load train dataset. If it is false, load test dataset.
        transform (transforms.Compose): Pre-processes for dataset.
        batch_size (int): The value of batch size.
        shuffle (bool, optional): Whether dataset is shuffled. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Mini-batches of dataset.
    """

    # Load dataset.
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count() // 2, pin_memory=True)
    return loader
