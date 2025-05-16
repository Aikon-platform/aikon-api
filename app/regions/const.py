"""
Module-specific constants for the regions app
"""
from pathlib import Path

from ..shared.utils.fileutils import create_dirs_if_not, download_model_if_not
from ..config.base import ENV, BASE_DIR, API_DATA_FOLDER

DEMO_NAME = "regions"

# Path to regions/ folder
DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"

EXT_QUEUE = "queue0"  # see docker-confs/supervisord.conf

EXT_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
# EXT_XACCEL_PREFIX = Path(ENV("EXT_XACCEL_PREFIX", default="/media/regions"))
# SHARED_XACCEL_PREFIX is used for exposing annotations

MODEL_PATH = EXT_DATA_FOLDER / "models"

create_dirs_if_not([MODEL_PATH])

# TODO retrieve available model instead
download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Illustration-Extraction",
        "filename": "illustration_extraction.pt",
    },
    MODEL_PATH / "illustration_extraction.pt",
)
download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Illustration-Extraction",
        "filename": "diagram_extraction.pt",
    },
    MODEL_PATH / "diagram_extraction.pt",
)
download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Illustration-Extraction",
        "filename": "line_extraction.pth",
    },
    MODEL_PATH / "line_extraction.pth",
)
download_model_if_not(
    {
        "repo_id": "seglinglin/Historical-Illustration-Extraction",
        "filename": "fasterrcnn_watermark_extraction.pth",
    },
    MODEL_PATH / "fasterrcnn_watermark_extraction.pth",
)
# download_model_if_not(
#     {
#         "repo_id": "seglinglin/Historical-Illustration-Extraction",
#         "filename": "character_line_extraction.pth",
#     },
#     MODEL_PATH / "character_line_extraction.pth",
# )
DEFAULT_MODEL = "illustration_extraction.pt"

DEFAULT_MODEL_INFOS = {
    "illustration_extraction": {
        "name": "Illustration extraction",
        "model": "illustration_extraction",
        "desc": "YOLO model fine-tuned on historical illustrations.",
    },
    "diagram_extraction": {
        "name": "Diagram extraction",
        "model": "diagram_extraction",
        "desc": "YOLO model fine-tuned on historical diagrams.",
    },
    "fasterrcnn_watermark_extraction": {
        "name": "Watermark extraction",
        "model": "fasterrcnn_watermark_extraction",
        "desc": "Faster-RCNN model trained to detect watermark in historical documents.",
    },
    "line_extraction": {
        "name": "Line extraction",
        "model": "line_extraction",
        "desc": "DINO-DETR model trained to extract line from historical documents.",
    },
    # "character_line_extraction": {
    #     "name": "Character line extraction",
    #     "model": "character_line_extraction",
    #     "desc": "⚠️ Will work only if used with cropped lines as input (from Line extraction model).",
    # },
}
