import torch
import gym
import matplotlib.pyplot as plt
import os

# os.chdir('Tests')
'''
2000 steps -> 150 MoR
'''

for file in os.listdir():
    if not file.endswith('lunar_lander_1999_999.pt'):
        continue
    env = gym.make("LunarLander-v2")
    model = torch.load(file)
    num_episodes = 500

    rewards = []

    for j in range(num_episodes):
        if j % 20 == 0 and j:
            print(j)
            print(f'Current mean of rewards is {round(sum(rewards)/len(rewards),2)}')
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = model.choose_action(torch.Tensor(state))
            state, r, done, _ = env.step(action)
            #env.render()

            episode_reward += r
            # print(f'{state[:]}')
            # print(f'Current episode reward: {episode_reward}')
        rewards.append(episode_reward)

    plt.hist(rewards, bins=100)

    plt.savefig(file[:-3])
    plt.show()



