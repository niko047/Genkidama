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

from neural_net import ToyNet

HEADER = 64
PORT = 5050
#Remember to put the pi address
ADDRESS = '172.16.3.26'
LOCAL_ADDRESS = '127.0.0.1'
WLAN_SELF_ADDRESS = '172.16.4.209'
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "disconnect"
UPDATE_EVERY_X_CONTACTS = 5


class Parent(object):

    def __init__(self, child_address, port, init_header, parent_net):
        self.child_address = child_address
        self.port = port
        self.init_header = init_header
        # Initializes an object from the class parent_net
        self.parent_net = parent_net()

        # Initialize the socket obj
        self.parent_init()

    def parent_init(self):
        self.parent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.parent.connect((self.child_address, PORT))

    def handle_worker(self):
        with self.parent as parent:
            connected, handshake = True, False

            old_weights_bytes = self.parent_net.encode_parameters()
            #print(f'Old weights bytes are : {old_weights_bytes}')
            len_msg_bytes = len(old_weights_bytes)
            print(f'Length of weights is {len_msg_bytes}')
            # Start signal is just an empty set of bytes
            start_end_message = b' ' * len_msg_bytes
            # Every how many contacts client -> server have happened
            interaction_count = 1
            while connected:
                # At the first handshake
                if not handshake:
                    print(f'[PARENT] Sending handshake message')
                    parent.send(start_end_message)
                    print(f'[PARENT] Length of msg being sent at handshake is len : {len(start_end_message)}')
                    handshake_msg = b''
                    while len(handshake_msg) < len_msg_bytes:
                        handshake_msg += parent.recv(len_msg_bytes)
                    if handshake_msg == start_end_message:
                        handshake = True
                        print(f'[PARENT] Hansdhake done')
                    else:
                        print(f'[PARENT] Handshake has failed, length of handshake message is : {len(handshake_msg)}')

                # Gets the current weights of the network
                print(f'[PARENT] Sending old weights at iteration {interaction_count}')

                # Sending a copy of the global net parameters to the child
                parent.send(old_weights_bytes)

                # Receiving the new weights coming from the child
                new_weights_bytes = b''
                while len(new_weights_bytes) < len_msg_bytes:
                    new_weights_bytes += parent.recv(len_msg_bytes)
                    print('Not full yet')
                print(f'[PARENT] Received new weights at iteration {interaction_count}, length: {len(new_weights_bytes)}')

                if new_weights_bytes == start_end_message:
                    print(f'[PARENT] About to close the connection on the parent side')
                    connected = False
                    break

                #Upload the new weights to the network
                self.parent_net.decode_implement_parameters(new_weights_bytes)

                # Simple count of the number of interactions
                interaction_count += 1


                    # Updates the network parameters
            print(f'[PARENT] Correctly closing parent')
            parent.close()


def start_parent():
    s = Parent(child_address=WLAN_SELF_ADDRESS, port=PORT, init_header=HEADER, parent_net=ToyNet)
    s.handle_worker()

if __name__ == '__main__':
    from timeit import default_timer as timer

    start = timer()

    start_parent()
    end = timer()
    print(end - start)
