from .base import *

USE_NGINX_XACCEL = False

BASE_URL = ENV("PROD_URL", default=f"http://localhost:{ENV('API_PORT')}")
