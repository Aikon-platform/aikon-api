"""
Module-specific constants for the paleography app
"""
from ..config.base import BASE_DIR, API_DATA_FOLDER, XACCEL_PREFIX
from ..shared.utils.fileutils import create_dirs_if_not, download_model_if_not

DEMO_NAME = "paleography"

DEMO_DIR = BASE_DIR / "app" / DEMO_NAME
LIB_PATH = DEMO_DIR / "lib"
M4M_PATH = LIB_PATH / "m4m"  # future git submodule (not added yet)
CONVERTER_PATH = LIB_PATH / "convert_d1_to_d2.py"  # D1 -> M4M dataset (from m4m_d2_tools 3e5b3f6)

# dedicated queue (queue0-4 are taken, see docker/supervisord.conf.template): requires NB_PROCS >= 6
PALEO_QUEUE = "queue5"

PALEO_DATA_FOLDER = API_DATA_FOLDER / DEMO_NAME
PALEO_RESULTS_PATH = PALEO_DATA_FOLDER / "results"  # <doc_uid>.zip (same pattern as vectorization)
PALEO_XACCEL_PREFIX = f"{XACCEL_PREFIX}/{DEMO_NAME}"
PALEO_MODEL_PATH = PALEO_DATA_FOLDER / "models"
PALEO_LOGS_PATH = PALEO_DATA_FOLDER / "logs"  # M4M outputs: lib/m4m/logs_reconstruction links here
create_dirs_if_not([PALEO_RESULTS_PATH, PALEO_MODEL_PATH, PALEO_LOGS_PATH])

# pretrained checkpoint for step 0 (--init), same mechanism as vectorization
M4M_CHECKPOINT = PALEO_MODEL_PATH / "checkpoint.pth"
download_model_if_not(
    {"repo_id": "RaphaelBfr/morphology4metrology", "filename": "checkpoint.pth"},
    M4M_CHECKPOINT,
)
