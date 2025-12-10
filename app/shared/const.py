"""
Various constants and paths, derived from the config
"""

from .utils.fileutils import create_dirs_if_not
from ..config import BASE_DIR, API_DATA_FOLDER, XACCEL_PREFIX

DIR_NAME = "shared"
DEMO_NAME = DIR_NAME

# Path to shared/ folder
SHARED_DIR = BASE_DIR / "app" / DIR_NAME
SHARED_DATA_FOLDER = API_DATA_FOLDER / DIR_NAME

SHARED_XACCEL_PREFIX = f"{XACCEL_PREFIX}/shared"

UTILS_DIR = SHARED_DIR / "utils"

DATASETS_PATH = SHARED_DATA_FOLDER / "datasets"
DOCUMENTS_PATH = SHARED_DATA_FOLDER / "documents"


create_dirs_if_not([DATASETS_PATH, DOCUMENTS_PATH])
