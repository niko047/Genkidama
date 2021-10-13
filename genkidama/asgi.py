"""
ASGI config for genkidama project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from ComputationalNode import routing_cn
from OrchestratorNode import routing_on

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "genkidama.settings")

application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            routing_cn.ws_pattern + routing_on.ws_pattern)
    ),
})