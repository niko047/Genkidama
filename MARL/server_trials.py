import socket
import threading


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = 'disconnect'
ADDRESS = '172.16.3.26'


class Client(object):

    def __init__(self, address, port, init_header, len_header, child_net):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.len_header = len_header
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
        with self.worker as worker:
            worker.listen()
            print(f'[LISTENING] Server is listening on {self.address}:{self.port}')
            while True:
                conn, addr = worker.accept()
                thread = threading.Thread(target=self.worker_interact, args=(conn, addr))
                thread.start()

    def worker_interact(self, conn_to_parent, addr_of_parent):
        connected, handshake = True, False
        len_msg_bytes = 64
        start_end_msg = b' ' * len_msg_bytes
        continue_msg = b'1' * len_msg_bytes
        num_interactions = 1
        while connected:
            while not handshake:
                # Waits for some parent to send a handshake
                start_msg = conn_to_parent.recv(len_msg_bytes)
                # If the handshake is accepted
                if start_msg == start_end_msg:
                    # Makes contact and sends confirmation
                    conn_to_parent.send(start_end_msg)
                    handshake = True
                    print(f'[CHILD] Handshake done')

            # Wait for weights to be received
            recv_weights = conn_to_parent.recv(len_msg_bytes)

            # TODO - Updates the network
            # Does some calculations
            new_weights = continue_msg

            num_interactions += 1
            if num_interactions != 20:
                # Sends the new weights over the network to the parent
                print(f'[CHILD] Sending data at iteration {num_interactions}')
                conn_to_parent.send(new_weights)
            else:
                conn_to_parent.send(start_end_msg)
                print(f'[CHILD] Closing the connection with the parent')
                connected= False
        conn_to_parent.close()