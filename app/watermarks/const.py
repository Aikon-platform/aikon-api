from pathlib import Path

from ..config.base import ENV, BASE_DIR, API_DATA_FOLDER, XACCEL_PREFIX
from ..region_extraction.const import EXT_DATA_FOLDER
from ..similarity.const import SIM_DATA_FOLDER

DEMO_NAME = "watermarks"

# Path to DEMO_NAME/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME

WATERMARKS_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
WATERMARKS_SOURCES_FOLDER = WATERMARKS_DATA_FOLDER / "sources"
WATERMARKS_RESULTS_FOLDER = WATERMARKS_DATA_FOLDER / "results"
WATERMARKS_TMP_FOLDER = WATERMARKS_DATA_FOLDER / "tmp_queries"
WATERMARKS_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"

MODEL_PATHS = {
    "detection": EXT_DATA_FOLDER / "models" / "fasterrcnn_watermark_extraction.pth",
    "features": SIM_DATA_FOLDER / "models" / "resnet18_watermarks.pth",
}

DEVICE = "cpu"
WATERMARKS_QUEUE = "queue3"  # see docker-confs/supervisord.conf
