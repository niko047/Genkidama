import json
from channels.generic.websocket import AsyncWebsocketConsumer

class NodeConsumer(AsyncWebsocketConsumer):

    # groups = ["broadcast"]

    async def connect(self):
        #Accepts only requests from itself (or very trusted nodes)
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):

        #Pings the networks and checks for pingable IPs


        #Takes the list of pingable IPs and tries to connect to them checking if they're Genkisockets


        #Saves the result to the database


        #Returns a status 200 to the data scientist requiring the scan
        await self.send(json.dumps({
            "status": 200
        }))

        #Returns something else if the process fails


