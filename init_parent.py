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

def start_parent():
    net = SmallNet(8, 4)
    for idx, address in enumerate(ADDRESSES):
        s = Parent(child_address=address, port=PORT, network_blueprint=net)
        s.run()
        print(f'Process number {idx} is running on {address}')
    # torch.save(net, 'cart_pole_model_a4c.pt')

if __name__ == '__main__':
    start = timer()
    start_parent()
    end = timer()

    print(f'{end - start} seconds elapsed')