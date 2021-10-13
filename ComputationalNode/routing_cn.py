from django.urls import path
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from .ComputationalNode import ComputationalNode

ws_pattern = [
    path('compute/', ComputationalNode.as_asgi()),
]

application = ProtocolTypeRouter(
    {
        'websocket':AuthMiddlewareStack(URLRouter(ws_pattern))
    }
)