import torch
import torch.multiprocessing as mp
import random


class Manager(object):

    @staticmethod
    def initialize_semaphor(num_workers):
        """Shared semaphor that when filled with 1's, allows network updates to begin"""
        s = torch.Tensor([0] * num_workers).to(torch.bool)
        s.share_memory_()
        return s

    @staticmethod
    def initialize_queue(len_queue=None):
        """Initializes a queue object shared amongst cpu processes"""
        return mp.Queue(len_queue) if len_queue is not None else mp.Queue()

    # NOTE: change the code below according to your needs

    @staticmethod
    def f_true(x, y): return x ** 2 + y ** 2

    @staticmethod
    def generate_input(): return [(random.random() - .5) * 10, (random.random() - .5) * 10]

    @staticmethod
    def data_generative_mechanism():
        """Samples a new instance from the data generative mechanism"""

        # Generate our data, where last column is output
        inputs = Manager.generate_input()
        output = Manager.f_true(*inputs)

        return torch.Tensor([*inputs, output]).to(torch.float32)

    @staticmethod
    def wait_for_green_light(semaphor: mp.Array, cpu_id: int):
        semaphor[cpu_id] = True
        while not torch.all(semaphor):
            pass

    @staticmethod
    def turn_off_semaphor_lights(semaphor: mp.Array):
        semaphor[:] = False


    @staticmethod
    def weighted_avg_net_parameters(p1: torch.Tensor, p2: torch.Tensor, alpha: float):
        """
        Takes the Weighted average between two flattened tensors of parameters
        :param p1: First flattened set of parameters
        :param p2: Second flattened set of parameters
        :param alpha: Relevance of the second vector of parameters with respect to the first one
        :return: Weighted average between two flattened tensors of parameters
        """
        return (1 - alpha) * p1 + alpha * p2



