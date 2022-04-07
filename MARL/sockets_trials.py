"""

One central CPU core (1/4) is going to be responsible for the updates of all the other worker cores.
All of the nodes (even the workers) at first are going to be initialized as websocket servers)
Sketch of the algorithm:
1. The central server is going to trigger the start of the algorithm by sending a message to all raspberrys.
2. The workers are going to drop their listening server and from now on they become clients.
3. At episode end (within same Pi) compute gradients and update the central node through websockets
4. Repeat 3 until convergence

"""

import socket

from server_trials import Client

HEADER = 64
PORT = 5050
#Remember to put the pi address
ADDRESS = '172.16.3.26'
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "disconnect"
UPDATE_EVERY_X_CONTACTS = 5


class Parent(object):

    def __init__(self, child_address, port, init_header, parent_net):
        self.child_address = child_address
        self.port = port
        self.init_header = init_header
        self.parent_net = parent_net

        # Initialize the socket obj
        self.parent_init()

    def parent_init(self):
        self.parent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.parent.connect((self.child_address, PORT))

    def handle_worker(self):
        connected, handshake = True, False
        len_msg_bytes = 64
        # Start signal is just an empty set of bytes
        start_end_message = b' ' * len_msg_bytes
        # Every how many contacts client -> server have happened
        interaction_count = 1
        while connected:
            # At the first handshake
            if not handshake:
                print(f'[PARENT] Sending handshake message')
                self.parent.send(start_end_message)
                handshake = True
                print(f'[PARENT] Hansdhake done')
                continue
            # Now wait for them to start the process

            weights_received: bytes = self.parent.recv(len_msg_bytes)
            print(f'Received weights at interaction_count {interaction_count} : {weights_received}')
            if weights_received:
                if weights_received == start_end_message:
                    break
                else:
                    # Updates the parameters to the global net
                    #self.parent_net.decode_implement_parameters(weights_received)
                    interaction_count += 1
                    if interaction_count % UPDATE_EVERY_X_CONTACTS == 0:
                        # Gets the parameters encoded in a bytes form
                        #encoded_params = self.parent_net.encode_parameters()
                        # Give back the weights to the contacting node
                        encoded_params = b'1' * len_msg_bytes
                        self.parent.send(encoded_params)

        self.parent.close()


def start_parent():
    s = Parent(child_address=ADDRESS, port=PORT, init_header=HEADER, parent_net=None)
    s.handle_worker()

def start_child():
    c = Client(address=ADDRESS, port=PORT, init_header=HEADER, len_header=HEADER, child_net=None)
    c.start_worker()

if __name__ == '__main__':
    start_parent()
