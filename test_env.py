import torch
import gym
import matplotlib.pyplot as plt
import os


env = gym.make("CartPole-v1")
model = torch.load('cart_pole_model_a4c.pt')
num_episodes = 1000

rewards = []

for j in range(num_episodes):
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
if not files:
    filename = 'run_1'
else:
    run_n = max([int(a[a.find('_')+1:-4]) for a in files]) + 1
    filename = f'run_{run_n}'

plt.savefig(filename)
plt.show()



