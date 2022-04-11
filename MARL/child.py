import socket
import threading
from Nets.neural_net import ToyNet


HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = 'disconnect'
LOCAL_ADDRESS = '127.0.0.1'
WLAN_SELF_ADDRESS = '10.50.73.194'
ADDRESS = '172.16.3.26'


class Client(object):

    def __init__(self, address, port, init_header, len_header, child_net):
        self.address = address
        self.port = port
        self.init_header = init_header
        self.len_header = len_header
        self.child_net = child_net()

        # Initializes the client socket
        self.worker_init()

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
        old_weights_bytes = self.child_net.encode_parameters()
        #print(f'Old weights are {old_weights_bytes}')
        len_msg_bytes = len(old_weights_bytes)
        print(f'Length of weights is {len_msg_bytes}')
        start_end_msg = b' ' * len_msg_bytes
        num_interactions = 1
        while connected:
            while not handshake:
                # Waits for some parent to send a handshake
                start_msg_bytes = b''
                while len(start_msg_bytes) < len_msg_bytes:
                    start_msg_bytes += conn_to_parent.recv(len_msg_bytes)
                # If the handshake is accepted
                if start_msg_bytes == start_end_msg:
                    # Makes contact and sends confirmation
                    conn_to_parent.send(start_end_msg)
                    handshake = True
                    print(f'[CHILD] Handshake done')
                else:
                    print(f'[CHILD] Error during hansdhake')

            # Wait for weights to be received
            recv_weights_bytes = b''
            while len(recv_weights_bytes) < len_msg_bytes:
                print(f'Not full yet')
                recv_weights_bytes += conn_to_parent.recv(len_msg_bytes)


            self.child_net.decode_implement_parameters(recv_weights_bytes)

            # Does some calculations, change this fake like to interaction between the agent and the environment
            # TODO

            # Encodes the new parameters that have changed after the calculations
            new_weights_bytes = self.child_net.encode_parameters()
            # Just a counter to keep track of the number of interactions
            num_interactions += 1
            # Keep on going until a certain stopping condition is met
            if num_interactions != 20:
                # Sends the new weights over the network to the parent
                print(f'[CHILD] Sending data at iteration {num_interactions}, length: {len_msg_bytes}')
                conn_to_parent.send(new_weights_bytes)
            else:
                conn_to_parent.send(start_end_msg)
                print(f'[CHILD] Closing the connection with the parent')
                connected = False
                break
        # Close the connection and detach parent from child
        conn_to_parent.close()


def start_child():
    c = Client(address=WLAN_SELF_ADDRESS, port=PORT, init_header=HEADER, len_header=HEADER, child_net=ToyNet)
    c.start_worker()

if __name__ == '__main__':
    start_child()