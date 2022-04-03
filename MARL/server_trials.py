import socket
import torch.multiprocessing as mp
import time


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = 'disconnect'
ADDRESS = '172.16.4.209'


class Client(object):

    def __init__(self, address, port, init_header, format, len_header):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.format = format
        self.len_header = len_header

        # Initializes the client socket
        self.socket_init()

    def socket_init(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.address, self.port))

    def get_message(self):
        return 'Hello server, im the client!'

    def server_interact(self):
        message = self.get_message()
        msg_length = len(message)
        send_length = str(msg_length).encode(self.format)
        # Apply padding to the message, later the below will tell the length of the weights
        final_msg_length = send_length + b' ' * (self.len_header - len(send_length))
        # Number of one-way interactions
        n_inter = 0
        # Begins an uninterrupted communication with the server
        while True:
            # In the first handshake we must convey the length of the objects we are exchanging
            print(f'[CLIENT] Sending the message {message}')
            if not(n_inter):
                self.client.send(msg_length)
            self.client.send(message)
            print(f'[CLIENT] The message has been sent')
            # Update the count of the messages that have been sent
            n_inter += 1
            # Now that it's sent, we need to wait for the response
            while True:
                print('[CLIENT] Waiting to receive the message')
                receive = self.client.recv(final_msg_length)
                if receive is not None:
                    print(f'[CLIENT] Message received, is {receive}')
                    print('[CLIENT] Sleeping for two seconds')
                    time.sleep(2)
                    break
