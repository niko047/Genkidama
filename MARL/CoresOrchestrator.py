import torch.multiprocessing as mp
import torch
import math

from .SingleCore import SingleCore
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from MARL.Optims.shared_optims import SharedAdam
from MARL.Nets.neural_net import ToyNet

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
        self.replay_buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_INPUTS_X + LEN_OUTPUTS_Y,
                                                              # 2 inputs + 1 output
                                                              num_iters=LEN_ITERATIONS,
                                                              tot_num_cpus=NUM_CPUS,
                                                              # TODO - Change this to the selected % of cpus
                                                              dtype=torch.float32)
        self.starting_semaphor = starting_semaphor
        self.n_cores = mp.cpu_count()
        self.n_available_cores = math.floor(self.n_cores * cpu_capacity)

    def run_procs(self):

        # TODO - Define the orchestrator net here and share its memory

        # TODO - Define a semaphor here
        semaphor = Manager.initialize_semaphor(self.n_available_cores)

        # TODO - Define a queue here for storing ongoing results

        # TODO -
        # Define the processes
        procs = [SingleCore(
            single_core_neural_net=None,
            cores_orchestrator_neural_net=None,
            semaphor=semaphor,
            optimizer=None,
            buffer=None,
            cpu_id=None,
            len_interaction_X=None,
            len_interaction_Y=None,
            batch_size=None,
            num_iters=None,
            tot_num_cpus=None,
            replacement=None,
            sample_from_shared_memory=None,
            res_queue=None
        ) for i in
            range(self.n_available_cores)]

        # Start the processes
        [p.start() for p in procs]
        # Join the processes (terminate them)
        [p.join() for p in procs]
