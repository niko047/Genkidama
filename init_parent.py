from MARL.Sockets.parent import Parent
from timeit import default_timer as timer
from MARL.Nets.SmallNet import SmallNet
import argparse


# TODO - First version is going to use the lock threads, therefore it is not going to be a parent Hogwild
my_parser = argparse.ArgumentParser(description='runs the parent socket servers')

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
print(f'ADDRESSES ARE {ADDRESSES}')
print(len(ADDRESSES))

def start_parent():
    net = SmallNet(s_dim=8, a_dim=4)
    s = Parent(addresses=ADDRESSES, port=PORT, network_blueprint=net)
    s.run()

if __name__ == '__main__':
    start_parent()