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
        self.fc1 = nn.Linear(in_features=2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        # Initializes the buffer used to encode and decode the params

        ToyNet.initialize_layers([self.fc1, self.fc2, self.fc3])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x