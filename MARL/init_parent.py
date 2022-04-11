from Sockets.parent import Parent
from Nets.neural_net import ToyNet
from timeit import default_timer as timer

PORT = 5050
ADDRESS = '172.16.3.26'
LOCAL_ADDRESS = '127.0.0.1'
WLAN_SELF_ADDRESS = '10.50.73.194'


def start_parent():
    s = Parent(child_address=WLAN_SELF_ADDRESS, port=PORT, parent_net=ToyNet)
    s.handle_worker()

if __name__ == '__main__':
    start = timer()
    start_parent()
    end = timer()
    print(f'{end - start} seconds elapsed')