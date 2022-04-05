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


class Server(object):

    def __init__(self, address, port, init_header):
        self.address = address
        self.port = port
        self.init_header = init_header

        # Initialize the socket obj
        self.socket_init()

    def socket_init(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.address, self.port))

    def handle_client(self, new_conn_obj, new_conn_addr):
        print(f"[NEW CONNECTION] {new_conn_addr} just connected.")
        connected = True
        while connected:
            # Args is how many bytes we expect to receive from the client
            msg_received = new_conn_obj.recv(64).decode(FORMAT)
            if msg_received:

                print(f'[SERVER] Message received, is {msg_received}')
                print('[SERVER] Sending one to client')
                #message_to_be_sent = msg_received.encode(FORMAT)

                #msg_length = len(message)
                #send_length = str(msg_length).encode(FORMAT)
                #final_msg_length = send_length + b' ' * (self.len_header - len(send_length))
                # Apply padding to the message
                #send_length = 64
                #new_conn_obj.send()
                new_conn_obj.send(msg_received.encode(FORMAT))
                print('[SERVER] Message sent')

                if msg_received == DISCONNECT_MESSAGE:
                    connected = False
                print(f'[{new_conn_addr}] : {msg_received}')
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
    if i == 0 :
        s = Server(address=ADDRESS, port=PORT, init_header=HEADER)
        s.start_server(semaphor)

    else:
        while semaphor[0] != 1:
            pass
        c = Client(address=ADDRESS, port=PORT, init_header=HEADER, format=FORMAT, len_header=HEADER, cpu_id=i)
        c.server_interact()

if __name__ == '__main__':
    semaphor = torch.Tensor([0])
    semaphor.share_memory_()
    procs = [mp.Process(target=strt, args=(i, semaphor)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]

