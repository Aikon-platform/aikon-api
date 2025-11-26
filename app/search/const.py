from ..shared.utils.fileutils import create_dirs_if_not
from ..config.base import BASE_DIR, XACCEL_PREFIX, API_DATA_FOLDER
from ..similarity.const import MODEL_PATH as SIM_MODEL_PATH

DEMO_NAME = "search"

# Path to search/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

SEARCH_QUEUE = "queue5"  # see docker-confs/supervisord.conf

SEARCH_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
SEARCH_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"
SEARCH_QUERY_PATH = SEARCH_DATA_FOLDER / "query"
SEARCH_INDEX_PATH = SEARCH_DATA_FOLDER / "index"

MODEL_PATH = SIM_MODEL_PATH

create_dirs_if_not([MODEL_PATH, SEARCH_INDEX_PATH])
