import torch
import gym
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.multiprocessing as mp
# os.chdir('Tests')
'''
2000 steps -> 150 MoR
'''


from multiprocessing import Process, Manager

def run_episodes(file, rewards, i):  # the managed list `L` passed explicitly.
    env = gym.make("LunarLander-v2")
    model = torch.load(file)
    num_episodes = 50
    for j in range(num_episodes):
        if j % 10 == 0 and j:
            print(f"Iteration {j} complete")
        #     print(f'Current mean of rewards is {round(sum(rewards) / len(rewards), 2)}')
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = model.choose_action(torch.Tensor(state))
            state, r, done, _ = env.step(action)
            # env.render()

            episode_reward += r
            # print(f'{state[:]}')
            # print(f'Current episode reward: {episode_reward}')
        rewards.append(episode_reward)


if __name__ == "__main__":
    for file in os.listdir():
        if not file.endswith('.pt'):
            continue
        with Manager() as manager:
            rewards = manager.list()

            processes = []
            for i in range(10):
                p = Process(target=run_episodes, args=(file, rewards, i))  # Passing the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            # Save here the results
            rewards = list(rewards)
            print(f'Finished running the episodes, length results is {len(rewards)}')
            print(f'Current mean of rewards is {round(sum(rewards) / len(rewards), 2)}')

            df = pd.DataFrame(rewards)
            df.to_csv(f'{file[:-3]}.csv', index=False)
