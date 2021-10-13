from django.urls import path
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from .ComputationalNode import ComputationalNode

ws_pattern = [
    path('scan_computational_nodes/', ComputationalNode.as_asgi()),
]

application = ProtocolTypeRouter(
    {
        'websocket':AuthMiddlewareStack(URLRouter(ws_pattern))
    }
)