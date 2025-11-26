from pathlib import Path
from environ import Env
import torch

# Path to api/ folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV = Env()
Env.read_env(env_file=f"{BASE_DIR}/.env")

IS_CUDA = torch.cuda.is_available()

INSTALLED_APPS = ENV("INSTALLED_APPS").split(",")
if not IS_CUDA:
    cuda_apps = {"vectorization", "dticlustering"}
    removed = cuda_apps & set(INSTALLED_APPS)
    INSTALLED_APPS = [app for app in INSTALLED_APPS if app not in cuda_apps]
    for app in removed:
        print(
            f"Warning: {app} removed from INSTALLED_APPS because CUDA is not available."
        )

data_path = Path(ENV("API_DATA_FOLDER", default=f"{BASE_DIR}/data"))
API_DATA_FOLDER = (
    data_path if data_path.is_absolute() else (BASE_DIR / data_path).resolve()
)

# prefix url for exposing results: each app has => /<prefix>/<app_name> (must match docker/nginx.conf)
XACCEL_PREFIX = "/media"

# time limit for dramatiq
TIME_LIMIT = 1000 * 60 * 60 * 24  # 24h


class FLASK_CONFIG:
    pass


BROKER_URL = f"redis:///{ENV('REDIS_DB_INDEX', default=0)}"
# BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{ENV('REDIS_DB_INDEX', default=0)}"

USE_NGINX_XACCEL = False
