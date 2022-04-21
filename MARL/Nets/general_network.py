from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
from torch import save as torchsave
from torch import load as torchload

import io

class GeneralNeuralNet(object):

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
