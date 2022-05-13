import torch
import gym
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")
model = torch.load('cart_pole_model.pt')
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
plt.show()



