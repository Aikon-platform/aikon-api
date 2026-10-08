from .base import *


if ENV("MODE")=="local" and ENV("BUNDLED")=="aikon-demo":
    BASE_URL = f"http://api:{ENV('API_PORT', default=5000)}"
else:
    BASE_URL = f"http://localhost:{ENV('API_PORT', default=5000)}"
# BASE_URL = f"http://localhost:{ENV('API_PORT', default=5000)}"
