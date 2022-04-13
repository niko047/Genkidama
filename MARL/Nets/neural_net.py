import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import save as torchsave
from torch import load as torchload
import io

"""
IDEA FOR TRAINING: Log tree of machines, efficient communication
"""

# TODO - Make a more general network class of networks inheriting both from nn.Module and another blueprint
class ToyNet(nn.Module):

    def __init__(self):
        super().__init__()
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

    def encode_parameters(self) -> bytes:
        """Gets the current parameters of the network and encodes them into bytes"""
        buff = io.BytesIO()
        flattened_params = parameters_to_vector(self.parameters())
        torchsave(flattened_params, buff)
        buff.seek(0)
        r = buff.read()
        return r

    def decode_implement_parameters(self, b: bytes, alpha: float):
        """Gets the encoded parameters of the networks in bytes, decodes them and pugs them into the net"""
        try:
            assert 0 <= alpha <= 1
        except AssertionError:
            Exception("Alpha should be a parameter valued in [0,1]")
        flattened_new_params = torchload(io.BytesIO(b))
        flattened_old_params = parameters_to_vector(self.parameters())
        # Alpha determines the learning contribute of each worker at each gradient sent
        flat_weighted_avg = (1-alpha) * flattened_old_params + alpha * flattened_new_params
        vector_to_parameters(flat_weighted_avg, self.parameters())

    @staticmethod
    def initialize_layers(layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, .0)

"""a * old params + b * new_params for update, 
not copy fully """