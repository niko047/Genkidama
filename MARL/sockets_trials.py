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
        continue_msg = b'1' * len_msg_bytes
        # Every how many contacts client -> server have happened
        interaction_count = 1
        while connected:
            # At the first handshake
            if not handshake:
                print(f'[PARENT] Sending handshake message')
                self.parent.send(start_end_message)
                handshake_msg = self.parent.recv(len_msg_bytes)
                if handshake_msg == start_end_message:
                    handshake = True
                    print(f'[PARENT] Hansdhake done')

            # TODO - Takes the weights out of the network and sends them over, change this fake msg
            old_weights = continue_msg
            print(f'[PARENT] Sending old weights at iteration {interaction_count}')

            # Sending a copy of the global net parameters to the child
            self.parent.send(old_weights)

            # Receiving the new weights coming from the child
            new_weights = self.parent.recv(len_msg_bytes)
            print(f'[PARENT] Received new weights at iteration {interaction_count}')

            # Simple count of the number of interactions
            interaction_count += 1
            if new_weights == start_end_message:
                print(f'[PARENT] About to close the connection on the parent side')
                connected=False

            # Updates the network parameters

        self.parent.close()


def start_parent():
    s = Parent(child_address=ADDRESS, port=PORT, init_header=HEADER, parent_net=None)
    s.handle_worker()

def start_child():
    c = Client(address=ADDRESS, port=PORT, init_header=HEADER, len_header=HEADER, child_net=None)
    c.start_worker()

if __name__ == '__main__':
    start_parent()
