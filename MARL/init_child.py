from Nets.neural_net import ToyNet
from Sockets.child import Client

PORT = 5050
LOCAL_ADDRESS = '127.0.0.1'
WLAN_SELF_ADDRESS = '10.50.73.194'
ADDRESS = '172.16.3.26'


def start_child():
    c = Client(address=WLAN_SELF_ADDRESS, port=PORT, child_net=ToyNet)
    c.start_worker()


if __name__ == '__main__':
    start_child()
