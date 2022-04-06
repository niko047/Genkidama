import socket
import torch.multiprocessing as mp
import time


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = 'disconnect'
ADDRESS = '172.16.4.209'


class Client(object):

    def __init__(self, address, port, init_header, format, len_header, cpu_id):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.format = format
        self.len_header = len_header
        self.cpu_id = cpu_id

        # Initializes the client socket
        self.socket_init()

    def socket_init(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.address, self.port))

    def get_message(self):
        return 'Hello server, im the client!'

    def server_interact(self):
        # TODO - Right now I am assuming that both endpoints know the length of the message being sent and received
        #message = self.get_message()
        # Encode the message
        encoded_message = message.encode(self.format)
        # Pad the encode message to make it fit within the allowed maximum size
        padded_encoded_message = encoded_message + b' ' * (64 - len(encoded_message))

        # Raw length of the message
        #msg_length = len(message)
        # Encoded length of the message
        #send_length = str(msg_length).encode(self.format)
        # Apply padding to the message, later the below will tell the length of the weights
        #final_msg_length = send_length + b' ' * (self.len_header - len(send_length))

        len_msg_bytes = 64
        # Begins an uninterrupted communication with the server
        while True:
            receive = self.client.recv(len_msg_bytes)
            if receive == 'disconnect':


            # Decode the data that comes through

            # Do the calculations here to get the data

            # Now get the data ready to be sent
            self.client.send(padded_encoded_message)

            # Now that it's sent, we need to wait for the response
            while True:
                print(f'[CLIENT {self.cpu_id}] Waiting to receive the message')
                receive = self.client.recv(64)
                if receive is not None:
                    print(f'[CLIENT {self.cpu_id}] Message received, is {receive}')
                    print(f'[CLIENT {self.cpu_id}] Sleeping for two seconds')
                    time.sleep(2)
                    break
