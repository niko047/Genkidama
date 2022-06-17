import torch.multiprocessing as mp
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Sockets.child import Client
from MARL.RL_Algorithms.ActorCritic import ActorCritic
import torch
import gym
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters

torch.set_printoptions(profile="full")

# TODO - Important, optimize the storage of information and the handling of temporary buffers, now it is inefficient
# TODO - Read papers on stochastic weight averaging


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
        self.single_core_neural_net = single_core_neural_net(s_dim=4, a_dim=2)
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

        i = 0
        #for i in range(self.num_episodes):
        while True:
            # Resets the environment
            temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))
            state = self.env.reset()
            ep_reward = 0
            #DEC temporary_buffer_idx = 0
            done = False
            temporary_buffer_idx = 0
            j=0
            # Generate training data and update buffer
            while not done:

                if not self.is_designated_core:
                    # Chooses an action and takes it, modifies inplace the temporary buffer
                    # state, reward, done = ActorCritic.agent_step(
                    #     neural_net=self.single_core_neural_net,
                    #     env=self.env,
                    #     state=state,
                    #     temporary_buffer=temporary_buffer
                    # )
                    # ep_reward += reward

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

                        temporary_buffer[temporary_buffer_idx, :] = tensor_tuple

                        temporary_buffer_idx += 1

                    # Every once in a while, define better this condition
                    if (j + 1) % 5 == 0:
                        # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                        if i == 0:
                            # Do this only for the first absolute run
                            self.starting_semaphor[self.cpu_id] = True
                            while not torch.all(self.starting_semaphor[1:]):
                                pass

                        if done:
                            zero_row = torch.zeros(size=(self.len_state + 2,))
                            mask = ~(temporary_buffer == zero_row)[:, 0]
                            # Masks the array for valid rows
                            temporary_buffer = temporary_buffer[mask]

                        # ActorCritic.discount_rewards(
                        #     neural_net=self.single_core_neural_net,
                        #     shared_memory_buffer=self.b,
                        #     temporary_buffer=temporary_buffer,
                        #     len_state=self.len_state,
                        #     gamma=self.gamma,
                        #     done=done
                        # )
                        temporary_buffer_flipped = torch.flip(temporary_buffer, dims=(0,))

                        if done:
                            R = 0
                        else :
                            _, output = self.single_core_neural_net.forward(temporary_buffer_flipped[0:self.len_state])

                            R = output.item()

                        for idx, interaction in enumerate(temporary_buffer_flipped):
                            r = interaction[-1]
                            a = interaction[-2]

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

                        temporary_buffer = torch.zeros(size=(self.num_iters, self.len_state + 2))

                        # Zeroes the gradients out
                        self.optimizer.zero_grad()
                        # Performs calculation of the backward pass
                        loss.backward()

                        for idx, (lp, gp) in enumerate(zip(
                                self.single_core_neural_net.parameters(),
                                self.cores_orchestrator_neural_net.parameters()
                        )):
                            # print(f'GRADIENT IS {lp.grad} OF TYPE {type(lp.grad)}')
                            gp._grad = lp.grad
                            # Here also copy them to the other episode-wise gradient bucket, without optimizing
                        self.optimizer.step()

                        self.optimizer.zero_grad()

                    if (j + 1) % 20 == 0:
                        with torch.no_grad():
                            orchestrator_params = parameters_to_vector(self.cores_orchestrator_neural_net.parameters())
                            vector_to_parameters(
                                orchestrator_params, self.single_core_neural_net.parameters()
                            )
                            # self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())

                    if done:
                        break

            self.results.append(ep_reward)

            i += 1
            print(f'EPISODE {i} -> EP Reward for cpu {self.b.cpu_id} is: {ep_reward}') if self.b.cpu_id else None

            # Update here the local network sending the updates
            if self.is_designated_core:
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
                # print(f'[CHILD] New weights just implemented are {parameters_to_vector(self.cores_orchestrator_neural_net.parameters())}')

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
            # self.single_core_neural_net.load_state_dict(self.cores_orchestrator_neural_net.state_dict())

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



"""
Observations from using the model and adjustments:
1. Faster CPUs (or the ones that perform worse, by finishing episodes early) update the network many more times
than the good ones, thus if they all have the same weight, they slow down the process.
2. Weight the updates based on the overall episode performance
"""