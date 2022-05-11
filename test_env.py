import torch
import gym
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")
model = torch.load('cart_pole_model.pt')
n_steps = 300
num_episodes = 1000

rewards = []

for j in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for s in range(n_steps):
        action = model.choose_action(torch.Tensor(state))
        state, r, done, _ = env.step(action)
        if done: break
        episode_reward += r
    rewards.append(episode_reward)

plt.hist(rewards)
plt.show()



