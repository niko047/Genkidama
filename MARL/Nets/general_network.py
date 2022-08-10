import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
from torch import save as torchsave
from torch import load as torchload

import io

class GeneralNeuralNet(object):

    def encode_parameters(self) -> bytes:
        """Gets the current parameters of the network and encodes them into bytes"""
        buff = io.BytesIO()
        with torch.no_grad():
            state_dict = self.state_dict()
        torchsave(state_dict, buff)
        buff.seek(0)
        r = buff.read()
        return r

    def decode_implement_parameters(self, b: bytes, alpha: float):
        """Gets the encoded parameters of the networks in bytes, decodes them and pugs them into the net"""
        try:
            assert 0 <= alpha <= 1
        except AssertionError:
            Exception("Alpha should be a parameter valued in [0,1]")

        with torch.no_grad():
            flattened_new_params = torchload(io.BytesIO(b))
            flattened_old_params = parameters_to_vector(self.parameters())
            # Alpha determines the learning contribute of each worker at each gradient sent
            flat_weighted_avg = (1-alpha) * flattened_old_params + alpha * flattened_new_params
            vector_to_parameters(flat_weighted_avg, self.parameters())


    # TODO - Try instead the for loop approach
    @staticmethod
    def decode_implement_shared_parameters_(b: bytes, alpha: float, neural_net):
        with torch.no_grad():
            new_state_dict = torchload(io.BytesIO(b))

            SDnet = neural_net.state_dict()

            for key in SDnet:
                SDnet[key] = SDnet[key] * (1 - alpha) + alpha * new_state_dict[key]
            neural_net.load_state_dict(SDnet)


    @staticmethod
    def initialize_layers(layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, .0)
