import socket
import threading

import torch

from .general_socket import GeneralSocket


class Client(GeneralSocket):

    def __init__(self, address, port, cores_orchestrator):

        # Inside here the blueprint is made into an object
        super().__init__(port=port)
        self.cores_orchestrator = cores_orchestrator
        self.address = address

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
                self.cores_orchestrator.run_procs(conn, addr)

    @staticmethod
    def handshake(conn_to_parent, has_handshaked, len_msg_bytes, start_end_msg):
        while not has_handshaked:
            # Waits for some parent to send a handshake
            start_msg_bytes = b''
            while len(start_msg_bytes) < len_msg_bytes:
                start_msg_bytes += conn_to_parent.recv(len_msg_bytes)
            # If the handshake is accepted
            if start_msg_bytes == start_end_msg:
                # Makes contact and sends confirmation
                conn_to_parent.send(start_end_msg)
                print(f'[CHILD] Handshake done')
                return True
            else:
                print(f'[CHILD] Error during hansdhake')
                return False

    @staticmethod
    def wait_receive_update(conn_to_parent, len_msg_bytes, neural_net):
        # Wait for weights to be received
        recv_weights_bytes = b''
        while len(recv_weights_bytes) < len_msg_bytes:
            recv_weights_bytes += conn_to_parent.recv(len_msg_bytes)

        # Alpha = 1 means it's going to completely overwrite the child params with the parent ones
        neural_net.decode_implement_parameters(recv_weights_bytes, alpha=1)

    @staticmethod
    def prepare_send(conn_to_parent, neural_net):
        # Encodes the new parameters that have changed after the calculations
        new_weights_bytes = neural_net.encode_parameters()
        # Sends the new weights over the network to the parent
        print(f'[CHILD] Sending data over')
        conn_to_parent.send(new_weights_bytes)

    @staticmethod
    def close_connection(conn_to_parent, start_end_msg):
        conn_to_parent.send(start_end_msg)







'''
    def child_interact(self, conn_to_parent, addr_of_parent):
        connected, handshake = True, False
        old_weights_bytes = self.neural_net.encode_parameters()
        # print(f'Old weights are {old_weights_bytes}')
        len_msg_bytes = len(old_weights_bytes)
        print(f'Length of weights is {len_msg_bytes}')
        start_end_msg = b' ' * len_msg_bytes
        num_interactions = 1
        while connected:
            if not handshake:
                handshake = Client.handshake(
                    conn_to_parent=conn_to_parent,
                    has_handshaked=handshake,
                    len_msg_bytes=len_msg_bytes,
                    start_end_msg=start_end_msg
                )

            # Wait for weights to be received
            recv_weights_bytes = b''
            while len(recv_weights_bytes) < len_msg_bytes:
                recv_weights_bytes += conn_to_parent.recv(len_msg_bytes)

            # Alpha = 1 means it's going to completely overwrite the child params with the parent ones
            self.neural_net.decode_implement_parameters(recv_weights_bytes, alpha=1)

            # Does some calculations, change this fake like to interaction between the agent and the environment

            # Encodes the new parameters that have changed after the calculations
            new_weights_bytes = self.neural_net.encode_parameters()
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
'''