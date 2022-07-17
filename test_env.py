import torch
import gym
import matplotlib.pyplot as plt
import os


env = gym.make("LunarLander-v2")
model = torch.load('A4C/episode_990_lunar_lander_a4c.pt')
num_episodes = 500

rewards = []

for j in range(num_episodes):
    if j % 20 == 0:
        print(j)
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = model.choose_action(torch.Tensor(state))
        state, r, done, _ = env.step(action)
        episode_reward += r
    rewards.append(episode_reward)

plt.hist(rewards, bins=100)
path = 'runs'
os.chdir('runs')
files = os.listdir()
filename = 'episode_495_lunar_lander_a4c'


# if not files:
#     filename = 'run_1'
# else:
#     run_n = max([int(a[a.find('_')+1:-4]) for a in files]) + 1
#     filename = f'run_{run_n}'

plt.savefig(filename)
plt.show()



