import torch.multiprocessing as mp
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from MARL.Sockets.child import Client
import torch.nn.functional as F
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
from torch.optim import Adam

class SingleCoreProcess(mp.Process):
    """Class defining the behavior of each process running in each CPU core in the machine"""

    def __init__(self,
                 single_core_neural_net,
                 cores_orchestrator_neural_net,
                 starting_semaphor,
                 cores_waiting_semaphor,
                 ending_semaphor,
                 optimizer,
                 buffer,
                 cpu_id,
                 len_interaction_X,
                 len_interaction_Y,
                 batch_size,
                 num_iters,
                 tot_num_active_cpus,
                 replacement,
                 sample_from_shared_memory,
                 res_queue,
                 num_episodes,
                 num_steps,
                 socket_connection,
                 address
                 ):
        super(SingleCoreProcess, self).__init__()
        self.single_core_neural_net = single_core_neural_net()
        self.cores_orchestrator_neural_net = cores_orchestrator_neural_net
        self.starting_semaphor = starting_semaphor
        self.cores_waiting_semaphor = cores_waiting_semaphor
        self.ending_semaphor = ending_semaphor
        self.optimizer = optimizer
        self.b = ReplayBuffers(
            shared_replay_buffer=buffer,
            cpu_id=cpu_id,
            len_interaction=len_interaction_X + len_interaction_Y,
            batch_size=batch_size,  # If increased it's crap
            num_iters=num_iters,
            tot_num_cpus=tot_num_active_cpus,
            replacement=replacement,
            sample_from_shared_memory=sample_from_shared_memory)
        self.cpu_id = cpu_id
        self.len_interaction_X = len_interaction_X
        self.len_interaction_Y = len_interaction_Y
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.tot_num_active_cpus = tot_num_active_cpus
        self.replacement = replacement
        self.sample_from_shared_memory = sample_from_shared_memory
        self.res_queue = res_queue
        self.num_episodes = num_episodes
        self.num_steps = num_steps

        self.socket_connection = socket_connection
        self.address = address
        self.is_designated_core = True if not self.cpu_id else False

        #TODO - Change
        self.local_optimizer = Adam(self.single_core_neural_net.parameters(), betas=(0.9, 0.99), eps=1e-3)

    def run(self):
        # TODO - Initialize the connection here to the designated cpu
        print(f'[CORE {self.cpu_id}] About to run')
        if self.is_designated_core:
            old_weights_bytes = self.single_core_neural_net.encode_parameters()
            len_msg_bytes = len(old_weights_bytes)
            print(f'Length of weights is {len_msg_bytes}')
            start_end_msg = b' ' * len_msg_bytes
            Client.handshake(
                conn_to_parent=self.socket_connection,
                has_handshaked=False,
                len_msg_bytes=len_msg_bytes,
                start_end_msg=start_end_msg
            )

            # Wait for weights to be received
            print(f"Receiving bytes from parent")
            recv_weights_bytes = b''
            while len(recv_weights_bytes) < len_msg_bytes:
                recv_weights_bytes += self.socket_connection.recv(len_msg_bytes)

            print(f"Implementing the parameters received to the local core net")

            self.single_core_neural_net.decode_implement_parameters(recv_weights_bytes, alpha=1)

        print(f'[CORE {self.cpu_id}] About to enter loop')

        for i in range(self.num_episodes):
            # Generate training data and update buffer

            for j in range(self.num_steps):
                # Generates some data according to the data generative mechanism
                tensor_tuple = Manager.data_generative_mechanism()

                # Records the interaction inside the shared Tensor buffer
                self.b.record_interaction(tensor_tuple)

                # Every once in a while, define better this condition
                if (j + 1) % 3 == 0:  # todo 5 gradients step for eGSD and change it to be configurable
                    # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                    if not self.num_episodes:
                        # Do this only for the first absolute run
                        print(f'Waiting for starting green light')
                        Manager.wait_for_green_light(semaphor=self.starting_semaphor, cpu_id=self.cpu_id)

                    # Random samples a batch
                    sampled_batch = self.b.random_sample_batch()

                    # Forward pass of the neural net, until the output columns, in this case last one
                    loc_output = self.single_core_neural_net.forward(sampled_batch[:, :-1])

                    # Calculates the loss between target and predict
                    loss = F.mse_loss(loc_output, torch.Tensor(sampled_batch[:, -1]).reshape(-1, 1))

                    # Averages the loss if using batches, else only the single value
                    self.res_queue.put(loss.item())

                    # Zeroes the gradients out
                    self.optimizer.zero_grad()

                    # Performs calculation of the gradients
                    loss.backward()

                    # Performs backpropagation with the gradients computed
                    self.local_optimizer.step()

                    if (j+1) % 20 == 0:

                        # Get the current flat weights of the local net and global one
                        flat_orch_params = parameters_to_vector(self.cores_orchestrator_neural_net.parameters())
                        flat_core_params = parameters_to_vector(self.single_core_neural_net.parameters())

                        # Compute the new weighted params
                        new_orch_params = Manager.weighted_avg_net_parameters(p1=flat_orch_params,
                                                                              p2=flat_core_params,
                                                                              alpha=0.25)#TODO - Change it to a param

                        # Update the parameters of the orchestrator with the new ones
                        vector_to_parameters(new_orch_params, self.cores_orchestrator_neural_net.parameters())

                        self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())




                    print(f'[CORE {self.cpu_id}] EPISODE {i} STEP {j + 1} -> Loss is: {loss}')


            # Wait for the green light to avoid overwriting
            print(f'[CORE {self.cpu_id}] Semaphor is currently {self.cores_waiting_semaphor}')

            # TODO - If it's the deisgnated core, then wait and check that all the others are true
            # TODO - Also when done assign false to everybody

            if self.is_designated_core:
                while not torch.all(torch.logical_or(self.cores_waiting_semaphor[1:], self.ending_semaphor[1:])):
                    pass

                # TODO - They're sleeping now, perform updates

                # Send the old data to the global network
                Client.prepare_send(
                    conn_to_parent=self.socket_connection,
                    neural_net=self.cores_orchestrator_neural_net
                )

                # Wait for response and update current
                Client.wait_receive_update(
                    conn_to_parent=self.socket_connection,
                    len_msg_bytes=len_msg_bytes,
                    neural_net=self.cores_orchestrator_neural_net)

                # TODO - Set them free to read the news and update themselves
                self.cores_waiting_semaphor[1:] = False

            # TODO - If not the designated core, sign yourself as true, and wait till you're false
            else:
                self.cores_waiting_semaphor[self.cpu_id] = True
                while self.cores_waiting_semaphor[self.cpu_id]:
                    pass

            # Pull parameters from orchestrator to each single node
            self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())

        self.ending_semaphor[self.cpu_id] = True
        self.res_queue.put(None)
