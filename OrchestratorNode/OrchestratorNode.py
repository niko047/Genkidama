import json
from channels.generic.websocket import AsyncWebsocketConsumer
from ast import literal_eval
from .Utils.general import is_valid_input_payload, str_bytes_to_pandas_df
from .Utils.algorithm_switcher import algorithm_switch
import pandas as pd
from types import SimpleNamespace
from sklearn.model_selection import GridSearchCV
from .models import NodeLog
from django.utils import timezone
import hashlib
from channels.db import database_sync_to_async
import datetime

class NodeConsumer(AsyncWebsocketConsumer):

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