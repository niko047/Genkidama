import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import gym
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from MARL.Optims.shared_optims import SharedAdam
from torch.optim import SGD
from MARL.Nets.neural_net import ToyNet
from MARL.Nets.CartPoleNet import CartPoleNet
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt

mp.set_start_method('spawn', force=True)

len_state = 4
len_actions = 2
len_reward = 2

LEN_ITERATIONS: int = 50
NUM_CPUS: int = mp.cpu_count()
NUM_EPISODES: int = 500
NUM_STEPS: int = 300
BATCH_SIZE: int = 5
SAMPLE_FROM_SHARED_MEMORY: bool = False
SAMPLE_WITH_REPLACEMENT: bool = False
GAMMA = .9

env = gym.make('CartPole-v1')


def train_model(glob_net, opt, buffer, cpu_id, semaphor, res_queue):
    loc_net = CartPoleNet(s_dim=4, a_dim=2)

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=cpu_id,
                      len_interaction=len_state + 1 + len_reward,
                      batch_size=BATCH_SIZE,  # If increased it's crap
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=SAMPLE_WITH_REPLACEMENT,
                      sample_from_shared_memory=SAMPLE_FROM_SHARED_MEMORY,
                      len_state=len_state,
                      len_action=1,
                      len_reward=len_reward)

    rewards_list = []

    for i in range(NUM_EPISODES):

        # TODO - Change values here below to automatically adjust
        temporary_buffer = []
        state = env.reset()

        cum_reward = 0
        done = False
        j = 0

        while not done:
            # Transforms the numpy array state into a tensor object of float32
            state_tensor = torch.Tensor(state).to(torch.float32)

            # Choose an action using the network, using the current state as input
            action_chosen = loc_net.choose_action(state_tensor)

            # Prepares a list containing all the objects above
            tensor_tuple = [*state, action_chosen]

            # Note that this state is the next one observed, it will be used in the next iteration
            state, reward, done, _ = env.step(action_chosen)
            if done: reward = -1

            # Adds the reward experienced to the current episode reward
            cum_reward += reward

            # Adds the reward and a placeholder for the discounted reward to be calculated
            tensor_tuple.append(reward)

            # Appends (state, action, reward, reward_observed) tensor object
            temporary_buffer.append(torch.Tensor(tensor_tuple))

            # Every once in a while
            if (j + 1) % BATCH_SIZE == 0 or done:

                # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                if not NUM_EPISODES:
                    # Do this only for the first absolute run
                    Manager.wait_for_green_light(semaphor=semaphor, cpu_id=cpu_id)

                # Reverses the temporal order of tuples, because of ease in discounting rewards
                temporary_buffer.reverse()

                if done:
                    R = 0
                else:
                    _, output = loc_net.forward(temporary_buffer[-1][:len_state])

                    # Output in this case is the estimation of the value coming from the state
                    R = output.item()

                for idx, interaction in enumerate(temporary_buffer):
                    # Take the true experienced reward from that session and the action taken in that step
                    r = interaction[-1]
                    a = interaction[-2]

                    R = r + GAMMA * R

                    # Append this tuple to the memory buffer, with the discounted reward
                    b.record_interaction(torch.Tensor([*interaction[:len_state], a, r, R])\
                                         .to(torch.float32))

                temporary_buffer = []

            if (j + 1) % BATCH_SIZE == 0:
                # Here update the local network

                # Random samples a batch
                state_samples, action_samples, rewards_samples = b.random_sample_batch()

                # Calculates the loss between target and predict
                loss = loc_net.loss_func(
                    s=state_samples,
                    a=action_samples,
                    v_t=rewards_samples[:, -1]
                )

                # Zeroes the gradients out
                opt.zero_grad()
                # Performs calculation of the backward pass
                loss.backward()
                # Performs step of the optimizer
                # opt.step()

                for lp, gp in zip(loc_net.parameters(), glob_net.parameters()):
                    gp._grad = lp.grad
                opt.step()

            if (j + 1) % 25 == 0:
                loc_net.load_state_dict(glob_net.state_dict())
                print(f'EPISODE {i} STEP {j + 1} -> CumReward for cpu {b.cpu_id} is: {cum_reward}')

            j += 1

            if done :
                rewards_list.append(cum_reward)
                break

    plt.plot(range(NUM_EPISODES), rewards_list)
    plt.waitforbuttonpress()

    res_queue.put(None)


if __name__ == '__main__':
    glob_net = CartPoleNet(a_dim=2, s_dim=4)
    glob_net.share_memory()

    # opt = SharedAdam(glob_net.parameters(), lr=1e-3, betas=(0.92, 0.999))  # global optimizer
    opt = SGD(glob_net.parameters(), lr=1e-4, momentum=0.9)

    # Queue used to store the history of rewards while training
    res_queue = mp.Queue()

    # Initializes the global buffer, where interaction with the environment are stored
    buffer = ReplayBuffers.init_global_buffer(len_interaction=len_state + 1 + len_reward,
                                              # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    # Creates a starting semaphor
    semaphor = Manager.initialize_semaphor(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(glob_net, opt, buffer, i, semaphor, res_queue)) for i in
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



    torch.save(glob_net, 'cart_pole_model_a3c.pt')

    plt.plot(res)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()
