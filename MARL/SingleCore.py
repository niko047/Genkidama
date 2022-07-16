import torch.multiprocessing as mp
import numpy as np
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Sockets.child import Client
from MARL.RL_Algorithms.ActorCritic import ActorCritic
import torch
import gym
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters


"""
TODO:
+ Add tdqm to keep track of at which iteration the algorithm is
+ Apply inline changes of average rewards and print it in a prettier way 

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
        self.start_end_msg = None

        self.results = []
        self.cum_grads_list = []


    def reset_environment(self):
        """Resets the environment at episode start"""
        temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))
        state = self.env.reset()
        ep_reward = 0
        temporary_buffer_idx = 0
        return temporary_buffer, temporary_buffer_idx, state, ep_reward

    def environment_interaction(self, state):
        """Interacts with the environment according to the current current core policy (NN)"""
        # Transforms the numpy array state into a tensor object of float32
        state_tensor = torch.Tensor(state).to(torch.float32)

        # Choose an action using the network, using the current state as input
        action_chosen = self.single_core_neural_net.choose_action(state_tensor)

        # Prepares a tensor containing all the objects above
        tensor_sar = torch.Tensor(np.array([*state, action_chosen, 0]))

        # Note that this state is the next one observed, it will be used in the next iteration
        state, reward, done, _ = self.env.step(action_chosen)

        return state, reward, done, tensor_sar

    def return_sar_discounted(self, temporary_buffer, done):
        """
        Takes the current buffer and performs a one-step discounting on the rewards,
        based on the actual future rewards it has seen
        """
        temporary_buffer_flipped = torch.flip(temporary_buffer, dims=(0,))

        if done:
            R = 0
        else:
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
        return state_samples, action_samples, rewards_samples

    def designated_core_handshake(self):
        """Handshakes the parent checking for agreement"""
        old_weights_bytes = self.single_core_neural_net.encode_parameters()
        len_msg_bytes = len(old_weights_bytes)
        self.len_msg_bytes = len_msg_bytes
        print(f"########## STARTING INITIALIZATION OF NODE, HANDSHAKING ##########")
        print(f'=> Length of the weights is {len_msg_bytes} bytes.')
        start_end_msg = b' ' * len_msg_bytes
        self.start_end_msg = start_end_msg
        print(f'=> Starting the SYN with ADDRESS: {self.address}')
        Client.handshake(
            conn_to_parent=self.socket_connection,
            has_handshaked=False,
            len_msg_bytes=len_msg_bytes,
            start_end_msg=start_end_msg
        )
        print(f'=> Ending the SYN with ADDRESS: {self.address}')
        print(f'=> Starting the ACK with ADDRESS: {self.address}')
        recv_weights_bytes = b''
        while len(recv_weights_bytes) < len_msg_bytes:
            recv_weights_bytes += self.socket_connection.recv(len_msg_bytes)
        print(f'=> Ending the ACK with ADDRESS: {self.address}')
        print(f"########## INITIALIZATION OF NODE DONE, STARTING COMPUTATIONS ##########")


    def push_gradients_to_orchestrator(self):
        """Push the accumulated gradients from the single core to the orchestrator"""
        for idx, (lp, gp) in enumerate(zip(
                self.single_core_neural_net.parameters(),
                self.cores_orchestrator_neural_net.parameters()
        )):
            gp._grad = lp.grad

    def pull_parameters_to_single_core(self):
        """Takes the parameter of the orchestrator net and with those replaces the single net parameters"""
        with torch.no_grad():
            orchestrator_params = parameters_to_vector(self.cores_orchestrator_neural_net.parameters())
            vector_to_parameters(
                orchestrator_params, self.single_core_neural_net.parameters()
            )

    def run(self):
        """Main function, starts the whole process, the underlying algorithm is this function itself"""
        if self.is_designated_core:
            self.designated_core_handshake()

        for i in range(self.num_episodes):
            # Creates temporary buffer and resets the environment
            temporary_buffer, temporary_buffer_idx, state, ep_reward = self.reset_environment()

            for j in range(self.num_steps):

                if not self.is_designated_core:
                    # Interacts with the environment and return results
                    state, reward, done, tensor_sar = self.environment_interaction(state)

                    if not done:
                        # Add the reward to the cumulative reward bucket
                        ep_reward += reward

                        # Store current interaction in the tensor S,A,R, of this step of the episode
                        tensor_sar[-1] = reward
                        temporary_buffer[temporary_buffer_idx, :] = tensor_sar
                        temporary_buffer_idx += 1

                    # TODO - Every once in a while, define better this condition
                    if (j + 1) % 5 == 0:
                        # Resets the index of the temporary buffer because we are about to reset the buffer itself
                        temporary_buffer_idx = 0

                        # Do this only for the first absolute run
                        if i == 0:
                            # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                            self.starting_semaphor[self.cpu_id] = True
                            while not torch.all(self.starting_semaphor[1:]):
                                pass

                        # TODO - Make this into a function, it is just masking the temporary rewards in case it did not reach 5 interactions
                        if done:
                            zero_row = torch.zeros(size=(self.len_state + 2,))
                            mask = ~(temporary_buffer == zero_row)[:, 0]
                            # Masks the array for valid rows (non-zero rows) in case the agent died halfway
                            temporary_buffer = temporary_buffer[mask]

                        # Takes the data it has experienced over the last 5 <= steps and discount the rewards
                        state_samples, action_samples, rewards_samples = self.return_sar_discounted(
                            temporary_buffer,
                            done
                        )
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

                        # Pushes the gradients accumulated from core to the orchestrator net
                        self.push_gradients_to_orchestrator()
                        # Performs backprop
                        self.optimizer.step()
                        # Zeroes out the gradients
                        self.optimizer.zero_grad()
                        # Empties out the temporary buffer for the next 5 iterations
                        temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))

                    if (j + 1) % 15 == 0:
                        # Syncs the parameter of this cpu core to the one of the orchestrator
                        self.pull_parameters_to_single_core()

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
                    len_msg_bytes=self.len_msg_bytes,
                    neural_net=self.cores_orchestrator_neural_net)

                # Wake up the other cpu cores that were sleeping
                self.cores_waiting_semaphor[1:] = False

            # Sleeping pill for all cores except the designated one
            else:
                self.cores_waiting_semaphor[self.cpu_id] = True
                while self.cores_waiting_semaphor[self.cpu_id]:
                    pass

            # Pull parameters from orchestrator to each single node
            self.pull_parameters_to_single_core()

        # Writes down that this cpu core has finished its job
        self.ending_semaphor[self.cpu_id] = True

        # The designated core can then close the connection with the parent
        if self.is_designated_core:
            Client.close_connection(conn_to_parent=self.socket_connection, start_end_msg=self.start_end_msg)

        # Print here the results of the algorithm
        # plt.plot(range(self.num_episodes), self.results)
        # plt.waitforbuttonpress()

        # Signals the outer process that it will not be receiving any more information
        if self.is_designated_core:
            torch.save(self.cores_orchestrator_neural_net, 'lunar_lander_a4c.pt')
        self.res_queue.put(None)