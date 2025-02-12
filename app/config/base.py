from pathlib import Path
from environ import Env

# Path to api/ folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV = Env()
Env.read_env(env_file=f"{BASE_DIR}/.env")

INSTALLED_APPS = ENV("INSTALLED_APPS").split(",")

data_path = Path(ENV("API_DATA_FOLDER", default=f"{BASE_DIR}/data"))
API_DATA_FOLDER = (
    data_path if data_path.is_absolute() else (BASE_DIR / data_path).resolve()
)

# prefix url for exposing results: each app has => /<prefix>/<app_name> (must match docker-confs/nginx.conf)
XACCEL_PREFIX = ENV("XACCEL_PREFIX", default="/media")

# time limit for dramatiq
TIME_LIMIT = 1000 * 60 * 60 * 24  # 24h


class FLASK_CONFIG:
    pass


BROKER_URL = f"redis:///{ENV('REDIS_DB_INDEX', default=0)}"
# BROKER_URL = f"redis://:{ENV('REDIS_PASSWORD')}@localhost:6379/1"

USE_NGINX_XACCEL = False
