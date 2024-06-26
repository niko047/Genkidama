"""

Sketch of the algorithm:
0. Initialize a child in every raspberry (a websocket server listening)
1. The central server is going to trigger the start of the algorithm by sending a message to all raspberrys.
2. Every X steps (within same Pi) compute gradients and update local machine network
3. Every Y>=X steps, gather the gradients and send them over to the parent socket for general update
4. Pull gradients from parent socket and send them back to worker for diffusion
5. Repeat until covergence

"""
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
from torch import save as torchsave
from torch import load as torchload
import os
import io
import numpy as np
import socket
import os


import pandas as pd

from .general_socket import GeneralSocket
import threading
import gym
import torch
from torch.nn.utils import parameters_to_vector

torch.set_printoptions(profile='full')

class Parent(GeneralSocket):

    def __init__(self, addresses, port, network_blueprint):
        super().__init__(port=port)

        self.addresses = addresses
        self.neural_net = network_blueprint

        self.rewards = []
        self.storage_current = []
        self.storage_received = []

        self.connections = [0 for i in range(len(addresses))]

        self.global_iter_count = 0



    def parent_init(self, address, port, pid):

        self.connections[pid] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connections[pid].connect((address, port))

        # self.parent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.parent.connect((address, port))

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

    def connection_interaction(self, parent, start_end_msg, len_msg_bytes, pid):
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

                #print(f'[PARENT] Sending old weights at iteration {interaction_count} to {self.address}')

                # Sending a copy of the global net parameters to the child
                current_encoded_weights = self.neural_net.encode_parameters()
                parent.send(current_encoded_weights)

            # Receiving the new weights coming from the child
            new_weights_bytes = GeneralSocket.wait_msg_received(len_true_msg=len_msg_bytes,
                                                                gsocket=parent)

            # If the message received from the child is the one signaling the end, then close the connection
            if new_weights_bytes == start_end_msg:
                break

            # with torch.no_grad():
            #     flattened_new_params = torchload(io.BytesIO(new_weights_bytes))
            #     self.storage_received.append(flattened_new_params.detach().numpy())

            # Upload the new weights to the network
            # self.neural_net.decode_implement_parameters(new_weights_bytes, alpha=.9)
            # self.decode_implement_shared_parameters_(new_weights_bytes, alpha=.7, neural_net=self.neural_net)

            alpha = .9

            with torch.no_grad():
                new_state_dict = torchload(io.BytesIO(new_weights_bytes))

            SDnet = self.neural_net.state_dict()


            for key in SDnet:
                # if key == 'v2.weight':
                    # print(f"Old param is {SDnet[key]}")
                    # print(f"Incoming param is {new_state_dict[key]}")
                SDnet[key] = SDnet[key] * (1 - alpha) + alpha * new_state_dict[key]
                self.neural_net.load_state_dict(SDnet)
                # if key == 'v2.weight':
                    # print(f'New params after weighted are {self.neural_net.state_dict()[key]}')

            current_encoded_weights = self.neural_net.encode_parameters()
            parent.send(current_encoded_weights)
            
            # Simple count of the number of interactions
            self.global_iter_count += 1
            interaction_count += 1
            if self.global_iter_count % 200 == 0:
                if f'lunar_lander_a4c_{self.global_iter_count//2}.pt' not in os.listdir('Tests'):
                    torch.save(self.neural_net, f'Tests/lunar_lander_a4c_{self.global_iter_count//2}.pt')


    def handle_client(self, addr, pid):
        """Handles the worker, all the functionality is inside here"""
        self.parent_init(address=addr, port=self.port, pid=pid)

        with self.connections[pid] as parent:
            # Gets some starting information to initialize the connection
            len_msg_bytes, start_end_msg, old_weights_bytes = self.get_start_end_msg()

            # Interaction has started here, all the talking is done inside this function
            self.connection_interaction(parent, start_end_msg, len_msg_bytes, pid)

            # Interaction has been truncated, close connection
            print(f'[PARENT] Correctly closing parent')
            parent.close()

    def run(self):
        for i, addr in enumerate(self.addresses):
            t = threading.Thread(target=self.handle_client, args=(addr, i,))
            t.start()
        # print(self.rewards)


    def run_episode(self):
        env = gym.make("LunarLander-v2")
        state = env.reset()
        done = False
        reward = 0
        while not done:
            state_tensor = torch.Tensor(state).to(torch.float32)

            # Choose an action using the network, using the current state as input
            action_chosen = self.neural_net.choose_action(state_tensor)

            state, r, done, _ = env.step(action_chosen)
            reward += r
        print(f'CURRENT REWARD FROM EPISODE IS : {reward}')

        return reward
