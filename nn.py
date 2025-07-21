from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """
    Neural Network for image classification.
    """

    def __init__(self) -> None:
        """
        Initialize the neural network architecture.
        """

        super(NeuralNetwork, self).__init__()

        # Features.
        self.features = nn.Sequential(
            # Convolutional layer 1.
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # Convolutional layer 2.
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        # Classifier.
        self.classifier = nn.Sequential(
            # Fully connected layer.
            nn.Linear(64*8*8, 10),
            nn.BatchNorm1d(num_features=10),
            nn.Softmax(dim=1)
        )
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the network.
        """

        # Features (ex. conv).
        x = self.features(x)
        # Flatten.
        x = x.view(x.shape[0], -1)
        # Classifier.
        x = self.classifier(x)
        return x
