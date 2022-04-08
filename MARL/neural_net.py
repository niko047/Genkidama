import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import save as torchsave
from torch import load as torchload
import io

"""
IDEA FOR TRAINING: Log tree of machines, efficient communication
"""

class ToyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        # Initializes the buffer used to encode and decode the params

        ToyNet.initialize_layers([self.fc1, self.fc2, self.fc3])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def encode_parameters(self) -> bytes:
        """Gets the current parameters of the network and encodes them into bytes"""
        buff = io.BytesIO()
        flattened_params = parameters_to_vector(self.parameters())
        torchsave(flattened_params, buff)
        buff.seek(0)
        return buff.read()

    def decode_implement_parameters(self, b: bytes):
        """Gets the encoded parameters of the networks in bytes, decodes them and pugs them into the net"""
        flattened_params = torchload(io.BytesIO(b))
        vector_to_parameters(flattened_params, self.parameters())

    @staticmethod
    def initialize_layers(layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, .0)