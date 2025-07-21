import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

from nn import NeuralNetwork


class Config:
    """
    Configurations(included Hyperparameters) for neural network training.
    """

    def __init__(self, model: NeuralNetwork) -> None:
        """
        Initialize configurations(hyperparameters).


        Args:
            model (NeuralNetwork): Neural network module for training.
        """

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.root = "./data"
        self.epoch = 50
        self.batch_size = 128
        self.input_size = (self.batch_size, 3, 32, 32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.247, 0.243, 0.261))
        ])
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-2
        self.weight_decay = 0
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.95)

        self.set_seed()
        return

    def set_seed(self) -> None:
        """
        Set a seed value for reproducibility.
        """

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return
