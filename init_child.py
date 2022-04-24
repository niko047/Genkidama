from MARL.Nets.neural_net import ToyNet
from MARL.Sockets.child import Client

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


def start_child():
    c = Client(address=ADDRESS, port=PORT, network_blueprint=ToyNet)
    c.start_worker()


if __name__ == '__main__':
    start_child()
