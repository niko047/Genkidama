import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from .general_network import GeneralNeuralNet

"""
IDEA FOR TRAINING: Log tree of machines, efficient communication
"""


class ToyNet(nn.Module, GeneralNeuralNet):

    def __init__(self):
        nn.Module.__init__(self)
        GeneralNeuralNet.__init__(self)
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        # Initializes the buffer used to encode and decode the params

        # ToyNet.initialize_layers([self.fc1, self.fc2, self.fc3])

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    @staticmethod
    def loss(output, target):
        return F.mse_loss(output, target)
