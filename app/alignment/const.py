from ..shared.utils.fileutils import create_dirs_if_not
from ..config.base import BASE_DIR, XACCEL_PREFIX, API_DATA_FOLDER

DEMO_NAME = "alignment"

# Path to alignment/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME

ALIGN_QUEUE = "queue1"  # see docker-confs/supervisord.conf

ALIGN_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
ALIGN_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"
ALIGN_RESULTS_PATH = ALIGN_DATA_FOLDER / "results"

# Available alignment methods
ALIGN_METHODS = ["sift+ransac"]
DEFAULT_METHOD = "sift+ransac"

create_dirs_if_not([ALIGN_RESULTS_PATH])
