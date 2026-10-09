from .base import *


if ENV("MODE")=="local" and ENV("BUNDLED")=="aikon-demo":
    if api_url_from_docker := ENV("API_URL_FROM_DOCKER", default=None):
        BASE_URL = api_url_from_docker
    else:
        raise EnvironmentError(".env variable 'API_URL_FROM_DOCKER' is undefined but required with `mode==local`")
else:
    BASE_URL = f"http://localhost:{ENV('API_PORT', default=5000)}"
