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
    def initialize_queue(len_queue):
        """Initializes a first-in first-out queue of batches updates of the shared network"""
        return mp.Queue(len_queue)

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
