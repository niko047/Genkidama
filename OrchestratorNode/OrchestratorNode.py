import json
from channels.generic.websocket import AsyncWebsocketConsumer
from bayes_opt import BayesianOptimization


class OrchestratorNode(AsyncWebsocketConsumer):

    #groups = ["broadcast"]

    async def connect(self):
        # Called on connection.
        # To accept the connection call:
        self.group_name = "computational_net"

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        #1. Define a route which starts and initially splits the hyperparams
        #2. Define another route which then does the update and returns the new points

        #Receives the payload containing the result from the previous task it had to solve
        #TODO - Unpack the incoming data


        #TODO - Update the gaussian process approximation, changin the redis cache

        #TODO - Send back the latest point that needs to be re-evaluated



        #Should access to redis to get f and bounds
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            random_state=42,
        )
        suggestion = optimizer.suggest(utility)



        await self.send(json.dumps({
            "data":"data"
        }))

    async def disconnect(self, close_code=400):
        # Called when the socket closes
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )