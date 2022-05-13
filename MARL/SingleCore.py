import torch.multiprocessing as mp
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from MARL.Sockets.child import Client
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import gym


# TODO - New steps to be carried out:
# TODO 1. Fix the replay buffer to accept (state, action, reward) and not anymore only X and Y
# DONE, now the sampling returns the following -> s, a, r = buff.random_sample_batch()
# TODO 2. Set up the environment and connect it at every step of the process
# TODO 3. Set up the mechanism of going back to actualize the rewards BEFORE storing them in the buffer
# TODO 4 (?). Implement a buffer that is emptied when a row is smapled (? would take away efficiency in parallel access)

class SingleCoreProcess(mp.Process):
    """Class defining the behavior of each process running in each CPU core in the machine"""

    def __init__(self,
                 single_core_neural_net,
                 cores_orchestrator_neural_net,
                 gym_rl_env_str,
                 starting_semaphor,
                 cores_waiting_semaphor,
                 ending_semaphor,
                 optimizer,
                 buffer,
                 cpu_id,
                 len_state,
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
        self.single_core_neural_net = single_core_neural_net(s_dim=4, a_dim=2)
        self.cores_orchestrator_neural_net = cores_orchestrator_neural_net

        self.env = gym.make(gym_rl_env_str)
        # TODO - Define what has to be saved in the buffer
        # (*states, *actions, *actualized_rewards)
        self.len_state = len_state

        self.starting_semaphor = starting_semaphor
        self.cores_waiting_semaphor = cores_waiting_semaphor
        self.ending_semaphor = ending_semaphor
        self.optimizer = optimizer
        self.b = ReplayBuffers(
            shared_replay_buffer=buffer,
            cpu_id=cpu_id,
            len_interaction=len_state + 1 + 1,
            batch_size=batch_size,  # If increased it's crap
            num_iters=num_iters,
            tot_num_cpus=tot_num_active_cpus,
            replacement=replacement,
            sample_from_shared_memory=sample_from_shared_memory,
            len_state=self.len_state,
            len_action= 1,  # Change in case of a problem with multiple actions necessary
            len_reward= 1  # Change in case of a problem with multiple rewards necessary
        )
        self.cpu_id = cpu_id
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

        self.GAMMA = .9 # TODO - Change this and make it a parameter

    def run(self):
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



        # Sets up a temporary buffer
        temporary_buffer = []

        for i in range(self.num_episodes):
            # Resets the environment
            state = self.env.reset()

            # Generate training data and update buffer
            for j in range(self.num_steps):
                if not self.is_designated_core:
                    # TODO - Change it with an interaction with the environment
                    # Generates some data according to the data generative mechanism
                    # tensor_tuple = Manager.data_generative_mechanism()

                    # Transforms the numpy array state into a tensor object of float32
                    state_tensor = torch.Tensor(state).to(torch.float32)

                    # Choose an action using the network, using the current state as input
                    action_chosen = self.single_core_neural_net.choose_action(state_tensor)

                    # Prepares a list containing all the objects above
                    tensor_tuple = [*state, action_chosen]

                    # Note that this state is the next one observed, it will be used in the next iteration
                    state, reward, done, _ = self.env.step(action_chosen)
                    if done: reward = -1

                    # Adds the reward experienced to the current episode reward
                    # cum_reward += reward

                    # Adds the reward and a placeholder for the discounted reward to be calculated
                    tensor_tuple.append(reward)

                    # Appends (state, action, reward, reward_observed) tensor object
                    temporary_buffer.append(torch.Tensor(tensor_tuple))

                    # Records the interaction inside the shared Tensor buffer
                    # self.b.record_interaction(tensor_tuple)

                    # Every once in a while, define better this condition
                    if (j + 1) % 5 == 0:
                        # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                        if i == 0:
                            # Do this only for the first absolute run
                            self.starting_semaphor[self.cpu_id] = True
                            while not torch.all(self.starting_semaphor[1:]):
                                pass

                        # Reverses the temporal order of tuples, because of ease in discounting rewards
                        temporary_buffer.reverse()
                        if done:
                            R = 0
                        else:
                            _, output = self.single_core_neural_net.forward(temporary_buffer[-1][:self.len_state])
                            # Output in this case is the estimation of the value coming from the state
                            R = output.item()

                        for idx, interaction in enumerate(temporary_buffer):
                            # Take the true experienced reward from that session and the action taken in that step
                            r = interaction[-1]
                            a = interaction[-2]

                            R = r + self.GAMMA * R

                            # Append this tuple to the memory buffer, with the discounted reward
                            self.b.record_interaction(torch.Tensor([*interaction[:self.len_state], a, R]) \
                                                 .to(torch.float32))

                        temporary_buffer = []

                        if (j + 1) % 5 == 0:
                            # Here update the local network

                            # Random samples a batch
                            state_samples, action_samples, rewards_samples = self.b.random_sample_batch()

                            # Calculates the loss between target and predict
                            loss = self.single_core_neural_net.loss_func(
                                s=state_samples,
                                a=action_samples,
                                v_t=rewards_samples
                            )

                            # Zeroes the gradients out
                            self.opt.zero_grad()
                            # Performs calculation of the backward pass
                            loss.backward()
                            # Performs step of the optimizer
                            # opt.step()

                            for lp, gp in zip(
                                    self.single_core_neural_net.parameters(),
                                    self.cores_orchestrator_neural_net.parameters()
                            ):
                                gp._grad = lp.grad
                            self.opt.step()

                        if (j + 1) % 15 == 0:
                            self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())
                            print(f'EPISODE {i} STEP {j + 1} -> Loss for cpu {b.cpu_id} is: {loss}')

            # They're sleeping now, perform updates
            if self.is_designated_core:
                while not torch.all(torch.logical_or(self.cores_waiting_semaphor[1:], self.ending_semaphor[1:])):
                    pass

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

                # Wake up the other cpu cores that were sleeping
                self.cores_waiting_semaphor[1:] = False

            # Sleeping pill for all cores except the designated one
            else:
                self.cores_waiting_semaphor[self.cpu_id] = True
                while self.cores_waiting_semaphor[self.cpu_id]:
                    pass

            # Pull parameters from orchestrator to each single node
            self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())

        # Writes down that this cpu core has finished its job
        self.ending_semaphor[self.cpu_id] = True

        # The designated core can then close the connection with the parent
        if self.is_designated_core:
            Client.close_connection(conn_to_parent=self.socket_connection, start_end_msg=start_end_msg)

        # Signals the outer process that it will not be receiving any more information
        self.res_queue.put(None)
