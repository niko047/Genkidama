import torch

from MARL.Sockets.child import Client
from MARL.CoresOrchestrator import CoresOrchestrator
from MARL.Nets.neural_net import ToyNet
from MARL.Optims.shared_optims import SharedAdam
from MARL.Nets.SmallNet import SmallNet
from torch.optim import SGD
import argparse
import gym

my_parser = argparse.ArgumentParser(description='runs the child socket servers')

my_parser.add_argument('-ip',
                       '--ipaddress',
                       type=str,
                       help='ip on which to run socket',
                       default='127.0.0.1')
my_parser.add_argument('-p',
                       '--port',
                       type=int,
                       help='port on which to run socket',
                       default=5050)
args = my_parser.parse_args()

PORT = args.port
ADDRESS = args.ipaddress

# TODO - Set up a different environment for these ones
neural_net = SmallNet
gym_rl_env_str = "LunarLander-v2"

# Initialize environment just to retrieve informations and then close it
env = gym.make(gym_rl_env_str)

len_state = env.observation_space.shape[0]
len_actions = env.action_space.n
len_reward = 1


shared_optimizer = SGD
shared_opt_kwargs = {
    "lr": 1e-4,
    "momentum": 0.9
}

batch_size = 5
num_iters = 5
replacement = False
sample_from_shared_memory = True
cpu_capacity = 1  # 80%
num_episodes = 500
num_steps = 2000


# Alpha is the parameter determining the importance of the individual cores when sending weights to parent net
# TODO - Insert alpha inside the function
alpha = 1
gamma=.99


def start_child():
    cores_orchestrator = CoresOrchestrator(
        neural_net=neural_net,
        gym_rl_env_str=gym_rl_env_str,
        shared_optimizer=shared_optimizer,
        shared_optimizer_kwargs=shared_opt_kwargs,
        len_state=len_state,
        len_actions=len_actions,
        len_reward=1,
        batch_size=batch_size,
        num_iters=num_iters,
        replacement=replacement,
        sample_from_shared_memory=sample_from_shared_memory,
        cpu_capacity=cpu_capacity,
        num_steps=num_steps,
        num_episodes=num_episodes,
        gamma=gamma
    )
    c = Client(address=ADDRESS, port=PORT, cores_orchestrator=cores_orchestrator)
    c.start_worker()


if __name__ == '__main__':
    start_child()
