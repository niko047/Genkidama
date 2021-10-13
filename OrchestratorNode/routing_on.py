from django.urls import path
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from .ScanComputationalNodes import ScanComputationalNodes

ws_pattern = [
    path('scan_computational_nodes/', ScanComputationalNodes.as_asgi()),
]

application = ProtocolTypeRouter(
    {
        'websocket':AuthMiddlewareStack(URLRouter(ws_pattern))
    }
)