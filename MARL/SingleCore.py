import torch.multiprocessing as mp
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Sockets.child import Client
from MARL.RL_Algorithms.ActorCritic import ActorCritic
import torch
import gym
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.set_printoptions(profile="full")

"""
TODO:
+ Add tdqm to keep track of at which iteration the algorithm is
+ Apply inline changes of average rewards and print it in a prettier way 

FIX:
/home/nicco047/Projects/thesis/MARL/SingleCore.py:147: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
(Triggered internally at  /root/pytorch/torch/csrc/utils/tensor_new.cpp:207.)

ANALYSIS OF THE CODE:
1. First part of synchronization coming from the designated node
2. Interaction of agents with the environment (algorithm independent)
3. Discounting of rewards (by default one step)
4. Storage of the interaction of the environment either in ReplayBuffer or TemporaryBuffer (algorithm dependent)
5. Computation of gradients and update of the network
"""


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
                 address,
                 gamma,
                 empty_net_trial
                 ):
        super(SingleCoreProcess, self).__init__()
        self.single_core_neural_net = single_core_neural_net(s_dim=8, a_dim=4) # TODO - pass these as params
        self.cores_orchestrator_neural_net = cores_orchestrator_neural_net
        self.empty_net_trial = empty_net_trial

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
            time_ordered_sampling=True,
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

        self.gamma = gamma

        self.results = []
        self.cum_grads_list = []

    def reset_environment(self):
        temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))
        state = self.env.reset()
        ep_reward = 0
        temporary_buffer_idx = 0
        return temporary_buffer, temporary_buffer_idx, state, ep_reward


    def run(self):
        # TODO - Put all of this inside a function
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
            print(f"Designated Core Receiving bytes from parent")
            recv_weights_bytes = b''
            while len(recv_weights_bytes) < len_msg_bytes:
                recv_weights_bytes += self.socket_connection.recv(len_msg_bytes)
            print(f"Designated Core Finished to receive bytes from parent")
            print(f"STARTING NOW TO MANAGE THE PROCESSES")

        for i in range(self.num_episodes):
            # Creates temporary buffer and resets the environment

            temporary_buffer, temporary_buffer_idx, state, ep_reward = self.reset_environment()

            # Generate training data and update buffer
            for j in range(self.num_steps):

                if not self.is_designated_core:
                    # TODO - Put all of this inside a function

                    # Transforms the numpy array state into a tensor object of float32
                    state_tensor = torch.Tensor(state).to(torch.float32)

                    # Choose an action using the network, using the current state as input
                    action_chosen = self.single_core_neural_net.choose_action(state_tensor)

                    # Prepares a list containing all the objects above
                    tensor_tuple = torch.Tensor([*state, action_chosen, 0])

                    # Note that this state is the next one observed, it will be used in the next iteration
                    state, reward, done, _ = self.env.step(action_chosen)

                    if not done:
                        ep_reward += reward

                        tensor_tuple[-1] = reward

                        # print(f"Trying to access with j {j} \n index {temporary_buffer_idx}\n the memory {temporary_buffer}")

                        temporary_buffer[temporary_buffer_idx, :] = tensor_tuple

                        temporary_buffer_idx += 1

                    # Every once in a while, define better this condition
                    if (j + 1) % 5 == 0:

                        temporary_buffer_idx = 0
                        # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                        if i == 0:
                            # Do this only for the first absolute run
                            self.starting_semaphor[self.cpu_id] = True
                            while not torch.all(self.starting_semaphor[1:]):
                                pass

                        # TODO - Make this into a function, it is just masking the temporary rewards in case it did not reach 5 interactions
                        if done:
                            zero_row = torch.zeros(size=(self.len_state + 2,))
                            mask = ~(temporary_buffer == zero_row)[:, 0]
                            # Masks the array for valid rows
                            temporary_buffer = temporary_buffer[mask]

                        # TODO - Make these flipping and discounting operation into a function
                        temporary_buffer_flipped = torch.flip(temporary_buffer, dims=(0,))

                        if done:
                            R = 0
                        else :
                            _, output = self.single_core_neural_net.forward(temporary_buffer_flipped[0, :self.len_state])

                            R = output.item()

                        for idx, interaction in enumerate(temporary_buffer_flipped):
                            r = interaction[-1]

                            R = r + self.gamma * R

                            temporary_buffer_flipped[idx, -1] = R

                        temporary_buffer = torch.flip(temporary_buffer_flipped, dims=(0,))

                        state_samples = temporary_buffer[:, :self.len_state]
                        action_samples = temporary_buffer[:, -2]
                        rewards_samples = temporary_buffer[:, -1]

                        # Calculates the loss between target and predict
                        loss = self.single_core_neural_net.loss_func(
                            s=state_samples,
                            a=action_samples,
                            v_t=rewards_samples
                        )

                        # Zeroes the gradients out
                        self.optimizer.zero_grad()
                        # Performs calculation of the backward pass
                        loss.backward()

                        for idx, (lp, gp) in enumerate(zip(
                                self.single_core_neural_net.parameters(),
                                self.cores_orchestrator_neural_net.parameters()
                        )):
                            gp._grad = lp.grad

                        self.optimizer.step()

                        self.optimizer.zero_grad()

                        temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))

                    if (j + 1) % 20 == 0:
                        # TODO - Make this into a function
                        with torch.no_grad():
                            orchestrator_params = parameters_to_vector(self.cores_orchestrator_neural_net.parameters())
                            vector_to_parameters(
                                orchestrator_params, self.single_core_neural_net.parameters()
                            )
                            # self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())

                    if done:
                        break

            # Appends the current reward to the list of rewards
            if not self.is_designated_core: self.results.append(ep_reward)

            print(f'EPISODE {i} -> EP Reward for cpu {self.b.cpu_id} is: {ep_reward}') if self.b.cpu_id else None

            # Update here the local network sending the updates
            if self.is_designated_core:
                # TODO - Make this into a function
                while not torch.all(
                        torch.logical_or(self.cores_waiting_semaphor[1:], self.ending_semaphor[1:])):
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
            with torch.no_grad():
                orchestrator_params = parameters_to_vector(
                    self.cores_orchestrator_neural_net.parameters())
                vector_to_parameters(
                    orchestrator_params, self.single_core_neural_net.parameters()
                )

        # Writes down that this cpu core has finished its job
        self.ending_semaphor[self.cpu_id] = True

        # The designated core can then close the connection with the parent
        if self.is_designated_core:
            Client.close_connection(conn_to_parent=self.socket_connection, start_end_msg=start_end_msg)

        # Print here the results of the algorithm
        # plt.plot(range(self.num_episodes), self.results)
        # plt.waitforbuttonpress()

        # Signals the outer process that it will not be receiving any more information
        self.res_queue.put(None)