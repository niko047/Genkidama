import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from MARL.Optims.shared_optims import SharedAdam
from torch.optim import SGD
from MARL.Nets.neural_net import ToyNet
from MARL.Nets.CartPoleNet import CartPoleNet
from torch.nn.utils import parameters_to_vector, vector_to_parameters

mp.set_start_method('spawn', force=True)

len_state = 4
len_actions = 1
len_reward = 2

LEN_ITERATIONS: int = 20
NUM_CPUS: int = mp.cpu_count()
NUM_EPISODES: int = 100
NUM_STEPS: int = 200
BATCH_SIZE: int = 5
SAMPLE_FROM_SHARED_MEMORY: bool = True
SAMPLE_WITH_REPLACEMENT: bool = False
GAMMA = .9  # TODO - Put gamma inside the functions

""" 
TO-DO
1. Obtain the parameters
2. Flatten them all with a = torch.nn.utils.parameters_to_vector(glob_net.parameters())
3. Encode the data for it to be sent over the network with BytesIO
4. Decode the data once received back to a flat tensor
5. Assign that flat tensor to the model's weights with torch.nn.utils.vector_to_parameters(a*1e5, glob_net.parameters())
"""


def train_model(glob_net, opt, buffer, cpu_id, semaphor, res_queue):
    loc_net = CartPoleNet(s_dim=4, a_dim=1)

    opt = SGD(loc_net.parameters(), lr=1e-5, momentum=0.9)

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=cpu_id,
                      len_interaction=len_state + len_actions + len_reward,
                      batch_size=BATCH_SIZE,  # If increased it's crap
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=SAMPLE_WITH_REPLACEMENT,
                      sample_from_shared_memory=SAMPLE_FROM_SHARED_MEMORY,
                      len_state=len_state,
                      len_action=1,
                      len_reward=len_reward)

    for i in range(NUM_EPISODES):

        # TODO - Change values here below to automatically adjust
        temporary_buffer = []

        for j in range(NUM_STEPS):
            # Generates some data according to the data generative mechanism
            # tensor_tuple = Manager.data_generative_mechanism()
            # TODO - Change this one with a data generative mechanims
            tensor_tuple = torch.randn(7, dtype=torch.float32)
            tensor_tuple[4] = 0 if tensor_tuple[4] < .5 else 1

            # Records the interaction inside the shared Tensor buffer
            # b.record_interaction(tensor_tuple)

            # TODO - Choose an action right here with the neural network (CHANGE)
            state = tensor_tuple[:4]
            action = tensor_tuple[-3]

            # TODO - Take the action and observe the consequent reward
            reward = tensor_tuple[-2]

            # TODO - Append (state, action, reward_observed)
            temporary_buffer.append(tensor_tuple)

            # Every once in a while
            if (j + 1) % 5 == 0:
                # TODO - Here update the rewards by discounting their future value

                # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                if not NUM_EPISODES:
                    # Do this only for the first absolute run
                    Manager.wait_for_green_light(semaphor=semaphor, cpu_id=i)

                temporary_buffer.reverse()

                # TODO - Check whether the algorithm is done or not
                done = False
                if done:
                    R = 0
                else:
                    # TODO - Change this into a true prediction coming from the neural net
                    _, output = loc_net.forward(temporary_buffer[-1][:4])

                    # Remember that the output is a tuple, select only the value of the state
                    R = output.item()

                for idx, interaction in enumerate(temporary_buffer):
                    reward = interaction[-2]
                    action = interaction[-3]
                    state = interaction[:len_state]

                    discounted_reward = reward + GAMMA * R

                    # Append this tuple to the memory buffer, with the discounted reward
                    b.record_interaction(torch.Tensor([*state, action, reward, discounted_reward]).to(torch.float32))

                temporary_buffer = []

            if (j + 1) % 20 == 0:
                # TODO - Here update the local network

                # Random samples a batch
                state, action, rewards = b.random_sample_batch()

                # TODO - Change the following to accomodate a neural network with s, a, r

                # Forward pass of the neural net, until the output columns, in this case last one
                # loc_output = loc_net.forward(state[:, :-1])

                # Calculates the loss between target and predict
                loss = loc_net.loss_func(
                    s=state,
                    a=action,
                    v_t=rewards[:, -1]
                )
                # loss = F.mse_loss(loc_output, torch.Tensor(sampled_batch[:, -1]).reshape(-1, 1))
                # Averages the loss if using batches, else only the single value
                res_queue.put(loss.item())
                # Zeroes the gradients out
                opt.zero_grad()
                # Performs calculation of the backward pass
                loss.backward()
                # Performs step of the optimizer
                opt.step()

                print(f'EPISODE {i} STEP {j + 1} -> Loss for cpu {b.cpu_id} is: {loss}')

            if (j + 1) % 40 == 0:
                # Get the current flat weights of the local net and global one
                flat_orch_params = parameters_to_vector(glob_net.parameters())
                flat_core_params = parameters_to_vector(loc_net.parameters())

                # Compute the new weighted params
                new_orch_params = Manager.weighted_avg_net_parameters(p1=flat_orch_params,
                                                                      p2=flat_core_params,
                                                                      alpha=.7)  # TODO - Change it to a param

                # Update the parameters of the orchestrator with the new ones
                vector_to_parameters(new_orch_params, glob_net.parameters())

                loc_net.load_state_dict(glob_net.state_dict())


    res_queue.put(None)


if __name__ == '__main__':
    glob_net = CartPoleNet(a_dim=1, s_dim=4)
    glob_net.share_memory()

    # opt = SharedAdam(glob_net.parameters(), lr=1e-3, betas=(0.92, 0.999))  # global optimizer

    # Queue used to store the history of rewards while training
    res_queue = mp.Queue()

    # Initializes the global buffer, where interaction with the environment are stored
    buffer = ReplayBuffers.init_global_buffer(len_interaction=len_state + len_actions + len_reward,
                                              # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    # Creates a starting semaphor
    semaphor = Manager.initialize_semaphor(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(glob_net, None, buffer, i, semaphor, res_queue)) for i in
             range(NUM_CPUS)]

    [p.start() for p in procs]

    # Stuff to pop the rewards from the queue continuously until all agents are done and append None
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [p.join() for p in procs]

    # Code for plotting the rewards

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Loss')
    plt.xlabel('Step of the NN')
    plt.show()
