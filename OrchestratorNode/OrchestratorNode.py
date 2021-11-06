import json
from channels.generic.websocket import AsyncWebsocketConsumer


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

        await self.send(json.dumps({
            "data":"data"
        }))

    async def disconnect(self, close_code=400):
        # Called when the socket closes
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )