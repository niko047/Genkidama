import torch.nn as nn
import torch.nn.functional as F


def initialize(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.constant_(layer.bias, .0)


class NNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

        initialize([self.fc1, self.fc2, self.fc3])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
