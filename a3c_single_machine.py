import torch
import torch.multiprocessing as mp
import gym
from MARL.ReplayBuffer.buffer import ReplayBuffers
from MARL.Manager.manager import Manager
from torch.optim import SGD, Adam
from MARL.Nets.SmallNet import SmallNet
import matplotlib.pyplot as plt
import random
import pandas as pd


torch.manual_seed(0)

mp.set_start_method('spawn', force=True)

len_state = 8
len_actions = 4
len_reward = 1

LEN_ITERATIONS: int = 120
NUM_CPUS: int = mp.cpu_count()
NUM_EPISODES: int = 2000 # Try increasing this to 2000+
NUM_STEPS: int = 2000
BATCH_SIZE: int = 120
SAMPLE_FROM_SHARED_MEMORY: bool = False
SAMPLE_WITH_REPLACEMENT: bool = False
GAMMA = .999

env = gym.make('LunarLander-v2')

import torch


def train_model(glob_net, opt, buffer, cpu_id, semaphor, res_queue):
    loc_net = SmallNet(s_dim=len_state, a_dim=len_actions)

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=cpu_id,
                      len_interaction=len_state + 1 + 1,
                      batch_size=BATCH_SIZE,  # If increased it's crap
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=SAMPLE_WITH_REPLACEMENT,
                      sample_from_shared_memory=SAMPLE_FROM_SHARED_MEMORY,
                      time_ordered_sampling=True,
                      len_state=len_state)

    rewards_list = []

    for i in range(NUM_EPISODES):

        # TODO - Change values here below to automatically adjust
        temporary_buffer = torch.zeros(size=(LEN_ITERATIONS, len_state + 2))
        state = env.reset()

        cum_reward = 0
        done = False
        j = 0
        temporary_buffer_idx = 0

        while not done:
            # Transforms the numpy array state into a tensor object of float32
            state_tensor = torch.Tensor(state).to(torch.float32)

            # Choose an action using the network, using the current state as input
            
            if random.random() < 0.01:
                action_chosen = random.choice(list(range(len_actions)))

            else:
                action_chosen = loc_net.choose_action(state_tensor)

            # Prepares a list containing all the objects above
            tensor_tuple = torch.Tensor([*state, action_chosen, 0])

            # Note that this state is the next one observed, it will be used in the next iteration
            state, reward, done, _ = env.step(action_chosen)

            if not done:

                # Adds the reward experienced to the current episode reward
                cum_reward += reward

                # Adds the reward and a placeholder for the discounted reward to be calculated
                tensor_tuple[-1] = reward

                # Appends (state, action, reward, reward_observed) tensor object
                temporary_buffer[temporary_buffer_idx, :] = tensor_tuple

                temporary_buffer_idx += 1

            # Every once in a while
            if (j + 1) % BATCH_SIZE == 0 or done:
                # If done, mask the rows that are 0 and perform the update with what's remaining before death of agent
                if done:
                    zero_row = torch.zeros(size=(len_state + 2,))
                    mask = ~(temporary_buffer == zero_row)[:, 0]
                    # Masks the array for valid rows
                    temporary_buffer = temporary_buffer[mask]

                # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                if not NUM_EPISODES:
                    # Do this only for the first absolute run
                    Manager.wait_for_green_light(semaphor=semaphor, cpu_id=cpu_id)

                # print(f'Temporary buffer before processing is {temporary_buffer}')

                # Reverses the temporal order of tuples, because of ease in discounting rewards
                # temporary_buffer.reverse()
                temporary_buffer_flipped = torch.flip(temporary_buffer, dims=(0,))

                if done:
                    R = 0
                else:
                    _, output = loc_net.forward(temporary_buffer_flipped[0, :len_state])

                    # Output in this case is the estimation of the value coming from the state
                    R = output.item()

                for idx, interaction in enumerate(temporary_buffer_flipped):
                    # Take the true experienced reward from that session and the action taken in that step
                    r = interaction[-1]
                    a = interaction[-2]

                    R = r + GAMMA * R

                    # Append this tuple to the memory buffer, with the discounted reward
                    # b.record_interaction(torch.Tensor([*interaction[:len_state], a, R]).to(torch.float32))
                    temporary_buffer_flipped[idx, -1] = R

                temporary_buffer = torch.flip(temporary_buffer_flipped, dims=(0,))

                # Random samples a batch
                state_samples = temporary_buffer[:, :len_state]
                action_samples = temporary_buffer[:, -2]
                rewards_samples = temporary_buffer[:, -1]

                # Calculates the loss between target and predict
                loss = loc_net.loss_func(
                    s=state_samples,
                    a=action_samples,
                    v_t=rewards_samples
                )

                # Zeroes the gradients out
                opt.zero_grad()
                # Performs calculation of the backward pass
                loss.backward()

                for lp, gp in zip(loc_net.parameters(), glob_net.parameters()):
                    gp._grad = lp.grad

                # Performs step of the optimizer
                opt.step()

                temporary_buffer = torch.zeros(size=(LEN_ITERATIONS, len_state + 2))
                temporary_buffer_idx = 0

            if (j + 1) % (BATCH_SIZE*2) == 0 or done:

                # Loads the state dict locally after the global optimization step
                loc_net.load_state_dict(glob_net.state_dict())

                print(f'EPISODE {i} STEP {j + 1} -> CumReward for cpu {b.cpu_id} is: {cum_reward}')

            j += 1

            if done:
                rewards_list.append(cum_reward)
                # print(f'EPISODE {i} STEP {j + 1} -> CumReward for cpu {b.cpu_id} is: {cum_reward}')
                break

        # loc_net.load_state_dict(glob_net.state_dict())

    plt.plot(range(NUM_EPISODES), rewards_list)
    results_path = f'runs/A3C/{cpu_id}_history.csv'
    df_res = pd.DataFrame({'rewards': rewards_list})
    df_res.to_csv(results_path)

    plt.waitforbuttonpress()

    res_queue.put(None)


if __name__ == '__main__':
    glob_net = SmallNet(a_dim=len_actions, s_dim=len_state)
    glob_net.share_memory()

    # opt = SharedAdam(glob_net.parameters(), lr=3e-4, betas=(0.92, 0.999))  # global optimizer
    opt = SGD(glob_net.parameters(), lr= 1e-4, momentum=0.9, weight_decay=.001)

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
             range(4)]

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
    torch.save(glob_net, 'SolvedLunarLander/2k_steps/lunar_lander_a3c_1500_solved.pt')

    plt.plot(res)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()
