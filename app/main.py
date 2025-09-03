"""
A script that initializes the Flask app and the Dramatiq broker.
"""
import json

from dramatiq import JSONEncoder
from dramatiq.encoder import MessageData
from dramatiq.message import set_encoder

from . import config
from flask import Flask

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq_abort import Abortable, backends
from dramatiq.middleware import CurrentMessage
from dramatiq.results.backends import RedisBackend
from .shared.utils.logging import LoggedResults, serializer
from .shared.utils.modular import auto_import_apps

# Flask setup
app = Flask(__name__)
app.config.from_object(config.FLASK_CONFIG)


class CustomEncoder(JSONEncoder):
    def encode(self, data: MessageData) -> bytes:
        return json.dumps(data, default=serializer, separators=(",", ":")).encode(
            "utf-8"
        )


set_encoder(CustomEncoder())

# Dramatiq setup
broker = RedisBroker(url=config.BROKER_URL)

event_backend = backends.RedisBackend(client=broker.client)
abortable = Abortable(backend=event_backend)

result_backend = RedisBackend(client=broker.client)
results = LoggedResults(backend=result_backend)

broker.add_middleware(abortable)
broker.add_middleware(CurrentMessage())
broker.add_middleware(results)

dramatiq.set_broker(broker)

# Import routes and tasks
auto_import_apps(app, ["shared.dataset"], __package__)  # import shared dataset routes
auto_import_apps(app, config.INSTALLED_APPS, __package__)
