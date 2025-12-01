from .base import *

USE_NGINX_XACCEL = True

BASE_URL = ENV("PROD_API_URL", default=f"http://localhost:{ENV('API_PORT')}")
