from MARL.Sockets.child import Client
from MARL.CoresOrchestrator import CoresOrchestrator
from MARL.Nets.neural_net import ToyNet
from MARL.Optims.shared_optims import SharedAdam
from torch.optim import SGD
import argparse

my_parser = argparse.ArgumentParser(description='Runs the child socket servers')

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
neural_net = ToyNet
shared_optimizer = SGD
shared_opt_kwargs = {
    "lr": 1e-5,
    "momentum": 0.9
}
len_interaction_X = 2
len_interaction_Y = 1
batch_size = 1
num_iters = 10
replacement = False
sample_from_shared_memory = True
cpu_capacity = 1  # 80%
num_steps = 50
num_episodes = 50

# Alpha is the parameter determining the importance of the individual cores when sending weights to parent net
# TODO - Insert alpha inside the function
alpha = 1


def start_child():
    cores_orchestrator = CoresOrchestrator(
        neural_net=neural_net,
        shared_optimizer=shared_optimizer,
        shared_optimizer_kwargs=shared_opt_kwargs,
        len_interaction_X=len_interaction_X,
        len_interaction_Y=len_interaction_Y,
        batch_size=batch_size,
        num_iters=num_iters,
        replacement=replacement,
        sample_from_shared_memory=sample_from_shared_memory,
        cpu_capacity=cpu_capacity,
        num_steps=num_steps,
        num_episodes=num_episodes
    )
    c = Client(address=ADDRESS, port=PORT, cores_orchestrator=cores_orchestrator)
    c.start_worker()


if __name__ == '__main__':
    start_child()
