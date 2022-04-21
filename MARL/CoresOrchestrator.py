import torch.multiprocessing as mp
import torch
import math

from .SingleCore import SingleCore
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager

# TODO - Allow for the possibility of connection to a shared database in the future
# TODO - Get the parameters from an input json somewhere

class CoresOrchestrator(object):

    def __init__(self,
                 neural_net,
                 shared_optimizer,
                 shared_optimizer_kwargs,
                 len_interaction_X,
                 len_interaction_Y,
                 batch_size,
                 num_iters,
                 replacement,
                 sample_from_shared_memory,
                 cpu_capacity,
                 num_steps,
                 num_episodes):
        """
        :param neural_net: Blueprint of the neural net to be used
        :param shared_optimizer: Blueprint of the shared optimizer to be used
        :param shared_optimizer_kwargs: Kwargs of the chosen optimizer
        :param len_interaction_X: Length of input (int)
        :param len_interaction_Y: Length of output (int)
        :param batch_size: How big should the batch sampled from the replay buffer be
        :param num_iters: Number of iterations that can be stored inside the buffer before it's full
        :param replacement: Whether sampling from the buffer happens with or without replacement
        :param sample_from_shared_memory: (bool)
            - True -> Each cpu core samples from the replay memory shared across all cores
            - False -> Each cpu core samples from it's own memory of the past
        :param cpu_capacity: Number between 0 and 1 deciding the % of the cpu dedicated for the training
        :param num_steps: Max number of steps to be performed within each episode
        :param num_episodes: Number of episode that each agent in each CPU must "live"
        """
        self.neural_net = neural_net

        # Defines a precise object orchestrator neural net from the blueprint and shares its memory
        self.orchestrator_neural_net = self.neural_net()
        self.orchestrator_neural_net.share_memory()

        self.shared_optimizer = shared_optimizer(self.orchestrator_neural_net.parameters(), **shared_optimizer_kwargs)
        self.len_interaction_X = len_interaction_X
        self.len_interaction_Y = len_interaction_Y
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.replacement = replacement
        self.sample_from_shared_memory = sample_from_shared_memory

        self.n_cores = mp.cpu_count()
        self.n_available_cores = math.floor(self.n_cores * cpu_capacity)

        self.replay_buffer = ReplayBuffers.init_global_buffer(len_interaction=len_interaction_X + len_interaction_Y,
                                                              # 2 inputs + 1 output
                                                              num_iters=num_iters,
                                                              tot_num_cpus=self.n_available_cores,
                                                              dtype=torch.float32)

        self.num_steps = num_steps
        self.num_episodes = num_episodes


    def run_procs(self):

        # Define a semaphor here
        semaphor = Manager.initialize_semaphor(self.n_available_cores)

        # Define a queue here for storing ongoing results
        res_queue = Manager.initialize_queue()

        # Define the processes
        procs = [SingleCore(
            single_core_neural_net=self.neural_net,
            cores_orchestrator_neural_net=self.orchestrator_neural_net,
            semaphor=semaphor,
            optimizer=self.shared_optimizer,
            buffer=self.replay_buffer,
            cpu_id=cpu_id,
            len_interaction_X=self.len_interaction_X,
            len_interaction_Y=self.len_interaction_Y,
            batch_size=self.batch_size,
            num_iters=self.num_iters,
            tot_num_active_cpus=self.n_available_cores,
            replacement=self.replacement,
            sample_from_shared_memory=self.sample_from_shared_memory,
            res_queue=res_queue,
            num_episodes=self.num_episodes,
            num_steps=self.num_steps
        ) for cpu_id in
            range(self.n_available_cores)]

        # Start the processes
        [p.start() for p in procs]
        # Join the processes (terminate them)
        [p.join() for p in procs]
