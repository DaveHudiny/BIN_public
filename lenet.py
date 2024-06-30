# Author = David Hudak
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = lenet.py file of project to subject Biology Inspired Computers. Implements LeNet.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """Implementation of fake LeNet model.

    Args:
        nn (nn.Module): Some Torch stuff.
    """

    def __init__(self):
        """Initialization of neural network model.
        """
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

    def forward(self, x):
        """Forward propagation of LeNet model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Prediction.
        """
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
