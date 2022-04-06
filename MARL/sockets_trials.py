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
import threading
import torch.multiprocessing as mp
import numpy.distutils.cpuinfo
import torch

from server_trials import Client

HEADER = 64
PORT = 5050
ADDRESS = '172.16.4.209'
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "disconnect"
UPDATE_EVERY_X_CONTACTS = 5


class Server(object):

    def __init__(self, address, port, init_header, parent_net):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.parent_net = parent_net

        # Initialize the socket obj
        self.socket_init()

    def socket_init(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.address, self.port))

    def handle_client(self, new_conn_obj, new_conn_addr):
        print(f"[NEW CONNECTION] {new_conn_addr} just connected.")
        connected, handshake = True, False
        len_msg_bytes = 64
        # Start signal is just an empty set of bytes
        start_end_message = b' ' * len_msg_bytes
        # Every how many contacts client -> server have happened
        interaction_count = 0
        while connected:
            # At the first handshake
            if not handshake:
                new_conn_obj.send(start_end_message)
                handshake = True
                continue
            # Now wait for them to start the process

            weights_received: bytes = new_conn_obj.recv(len_msg_bytes)
            if weights_received:
                if weights_received == start_end_message:
                    connected = False
                else:
                    # Updates the parameters to the global net
                    #self.parent_net.decode_implement_parameters(weights_received)
                    print(f'[{new_conn_addr}] : {weights_received}')
                    interaction_count += 1
                    if interaction_count % UPDATE_EVERY_X_CONTACTS == 0:
                        # Gets the parameters encoded in a bytes form
                        #encoded_params = self.parent_net.encode_parameters()
                        # Give back the weights to the contacting node
                        encoded_params = b'1'*len_msg_bytes
                        new_conn_obj.send(encoded_params)

        new_conn_obj.close()

    def start_server(self, semaphor):
        self.server.listen()
        print(f'[LISTENING] Server is listening on {self.address}:{self.port}')
        semaphor[0] = 1
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


def strt(i, semaphor):
    if i == 0:
        s = Server(address=ADDRESS, port=PORT, init_header=HEADER, parent_net=None)
        s.start_server(semaphor)

    else:
        while semaphor[0] != 1:
            pass
        c = Client(address=ADDRESS, port=PORT, init_header=HEADER, len_header=HEADER, cpu_id=i, child_net=None)
        c.server_interact()


if __name__ == '__main__':
    semaphor = torch.Tensor([0])
    semaphor.share_memory_()
    procs = [mp.Process(target=strt, args=(i, semaphor)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]
