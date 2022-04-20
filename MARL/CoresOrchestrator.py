import torch.multiprocessing as mp
import torch
import math

from MARL.ReplayBuffer.buffer import ReplayBuffers
from M


# TODO - Allow for the possibility of connection to a shared database in the future

# TODO :
# 1. Get the configuration info of the algorithm from somewhere like a Json
LEN_INPUTS_X = 10
LEN_OUTPUTS_Y = 5
LEN_ITERATIONS = 10
NUM_CPUS = mp.cpu_count()


class CoresOrchestrator(object):

    def __init__(self,
                 orchestrator_neural_net,
                 shared_optimizer,
                 results_queue,
                 cpu_capacity,
                 replay_buffer,
                 starting_semaphor):
        self.orchestrator_neural_net = orchestrator_neural_net
        self.shared_optimizer = shared_optimizer
        self.results_queue = results_queue
        self.cpu_capacity = cpu_capacity
        self.replay_buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_INPUTS_X + LEN_OUTPUTS_Y,  # 2 inputs + 1 output
                                  num_iters=LEN_ITERATIONS,
                                  tot_num_cpus=NUM_CPUS, #TODO - Change this to the selected % of cpus
                                  dtype=torch.float32)
        self.starting_semaphor = starting_semaphor

    def run_procs(self):
        n_cores = mp.cpu_count()

        # Define the processes
        procs = [mp.Process(args=(self.orchestrator_neural_net,
                                  self.shared_optimizer,
                                  self.replay_buffer,
                                  i,
                                  self.starting_semaphor,
                                  self.results_queue)) for i in
                 range(math.floor(n_cores*self.cpu_capacity))]

        # Start the processes
        [p.start() for p in procs]
        # Join the processes (terminate them)
        [p.join() for p in procs]

