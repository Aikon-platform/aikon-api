from .base import *

USE_NGINX_XACCEL = False

prod_url = ENV("PROD_URL", default=f"http://localhost:{ENV('API_PORT')}")
BASE_URL = f"https://{prod_url}" if "http" not in prod_url else prod_url
