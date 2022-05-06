"""

Sketch of the algorithm:
0. Initialize a child in every raspberry (a websocket server listening)
1. The central server is going to trigger the start of the algorithm by sending a message to all raspberrys.
2. Every X steps (within same Pi) compute gradients and update local machine network
3. Every Y>=X steps, gather the gradients and send them over to the parent socket for general update
4. Pull gradients from parent socket and send them back to worker for diffusion
5. Repeat until covergence

"""

import socket
from .general_socket import GeneralSocket


class Parent(GeneralSocket):

    def __init__(self, child_address, port, network_blueprint):
        super().__init__(address=child_address, port=port)

        self.neural_net = network_blueprint()

        # Initialize the socket obj
        self.parent_init()

    def parent_init(self):
        self.parent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.parent.connect((self.address, self.port))

    def get_start_end_msg(self):
        encoded_params = self.neural_net.encode_parameters()
        len_msg_bytes = len(encoded_params)
        print(f"Weights of the network are {len_msg_bytes} Bytes.")
        start_end_msg = b' ' * len_msg_bytes
        return len_msg_bytes, start_end_msg, encoded_params

    def check_handshake(self, parent, start_end_msg, len_msg_bytes):
        """Checks for the first interaction between child and parent"""
        print(f'[PARENT] Sending handshake message')
        parent.send(start_end_msg)
        print(f'[PARENT] Length of msg being sent at handshake is len : {len(start_end_msg)}')
        handshake_msg = b''
        while len(handshake_msg) < len_msg_bytes:
            handshake_msg += parent.recv(len_msg_bytes)
        return True

    def connection_interaction(self, parent, start_end_msg, len_msg_bytes, old_weights_bytes):
        """Handles what's up until the connection is alive:
        - Handshake with the parent
        - While stopping condition is met
            - While all cpu cores are not done
                - Generate data and let the agent in the environment for each CPU
            - Gather gradients and update the local network
            - Send gradients to the central node and wait for response
            - Update all cores with the new parameters
        - Close connection
        """
        interaction_count = 1
        has_handshake_happened = False

        # Until it receives the ending message from the child
        while True:
            if not has_handshake_happened:
                has_handshake_happened = self.check_handshake(parent, start_end_msg, len_msg_bytes)

            print(f'[PARENT] Sending old weights at iteration {interaction_count}')

            # TODO - Here interact with the environment
            # handle_all_cpu_cores

            # Sending a copy of the global net parameters to the child
            parent.send(old_weights_bytes)

            print(f"Weights sent, now waiting to receive the others")


            # Receiving the new weights coming from the child
            new_weights_bytes = GeneralSocket.wait_msg_received(len_true_msg=len_msg_bytes,
                                                                gsocket=parent)

            print(f"Received weights, about to implement them")

            # If the message received from the child is the one signaling the end, then close the connection
            if new_weights_bytes == start_end_msg:
                break

            # Upload the new weights to the network
            self.neural_net.decode_implement_parameters(new_weights_bytes, alpha=1)

            print("Implemented, going forward")

            # Simple count of the number of interactions
            interaction_count += 1

    def handle_worker(self):
        """Handles the worker, all the functionality is inside here"""
        with self.parent as parent:
            # Gets some starting information to initialize the connection
            len_msg_bytes, start_end_msg, old_weights_bytes = self.get_start_end_msg()

            # Interaction has started here, all the talking is done inside this function
            self.connection_interaction(parent, start_end_msg, len_msg_bytes, old_weights_bytes)

            # Interaction has been truncated, close connection
            print(f'[PARENT] Correctly closing parent')
            parent.close()
