import threading

from MARL.Sockets.parent import Parent
from MARL.Nets.neural_net import ToyNet
from timeit import default_timer as timer

import argparse


# TODO - First version is going to use the lock threads, therefore it is not going to be a parent Hogwild
my_parser = argparse.ArgumentParser(description='Runs the parent socket servers')

my_parser.add_argument('-ip',
                       '--ipaddress',
                       nargs='+',
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
ADDRESSES = args.ipaddress


def start_parent():
    for address in ADDRESSES:
        s = Parent(child_address=address, port=PORT, network_blueprint=ToyNet)
        s.run()

if __name__ == '__main__':
    start = timer()
    start_parent()
    end = timer()
    print(f'{end - start} seconds elapsed')