"""
Module-specific constants for the paleography app
"""
from ..config.base import BASE_DIR, API_DATA_FOLDER, XACCEL_PREFIX

DEMO_NAME = "paleography"

DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"
M4M_PATH = LIB_PATH / "m4m"  # future git submodule (not added yet)

# dedicated queue (queue0-4 are taken, see docker/supervisord.conf.template): requires NB_PROCS >= 6
PALEO_QUEUE = "queue5"

PALEO_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
PALEO_RESULTS_PATH = PALEO_DATA_FOLDER / "results"  # <doc_uid>.zip (same pattern as vectorization)
PALEO_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"
