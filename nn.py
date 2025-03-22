from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """NeuralNetwork 
    """

    def __init__(self) -> None:
        """__init__

        Define the structure of Neural Network.

        """

        super(NeuralNetwork, self).__init__()

        # Conv1.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=128)
        self.actv1 = nn.ReLU()

        # Conv2.
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)
        self.actv2 = nn.ReLU()

        # Conv3.
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(num_features=128)
        self.actv3 = nn.ReLU()

        # Conv4.
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm4 = nn.BatchNorm2d(num_features=128)
        self.actv4 = nn.ReLU()

        # Linear1.
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128*8*8, 10)
        return

    def forward(self, x) -> Tensor:
        """forward 

        Forward processing on learning.

        Args:
            x (Tensor): Input of model.

        Returns:
            Tensor: Output of model.
        """

        # Conv1.
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.actv1(x)

        # Conv2.
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.batch_norm2(x)
        x = self.actv2(x)

        # Conv3.
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.actv3(x)

        # Conv4.
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.batch_norm4(x)
        x = self.actv4(x)

        # Linear1.
        x = self.flatten(x)
        x = self.linear1(x)
        return x
