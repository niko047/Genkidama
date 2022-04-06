import socket
import torch.multiprocessing as mp
import time
import threading


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = 'disconnect'
ADDRESS = '172.16.3.26'


class Client(object):

    def __init__(self, address, port, init_header, len_header, cpu_id, child_net):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.len_header = len_header
        self.cpu_id = cpu_id
        self.child_net = child_net

        # Initializes the client socket
        self.worker_init()

    '''
    def socket_init(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.address, self.port))
    '''

    def worker_init(self):
        self.worker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.worker.bind((self.address, self.port))

    def get_message(self):
        return 'Hello server, im the client!'

    def start_worker(self):
        self.worker.listen()
        print(f'[LISTENING] Server is listening on {self.address}:{self.port}')
        while True:
            conn, addr = self.worker.accept()
            thread = threading.Thread(target=self.worker_interact, args=(conn, addr))
            thread.start()

    def worker_interact(self, conn_to_parent, addr_of_parent):
        connected, handshake = True, False
        len_msg_bytes = 64
        start_end_msg = b' ' * len_msg_bytes
        num_interactions = 0
        while connected:
            while not handshake:
                start_msg = conn_to_parent.recv(len_msg_bytes)
                if start_msg == start_end_msg:
                    handshake = True

            # Start working now
            if not num_interactions % 5 == 0:
                conn_to_parent.send(b'1' * num_interactions)
            else:
                conn_to_parent.send(start_end_msg)
        conn_to_parent.close()

