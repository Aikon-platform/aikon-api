"""
Tasks for DTI clustering

**train_dti**
This function is a Dramatiq actor that handles the training of a DTI model.
It downloads the dataset, prepares it, and runs the training process based on the provided parameters.

- experiment_id (str): The ID of the clustering task.
- dataset_id (str): The ID of the dataset.
- dataset_url (str): The URL of the zipped dataset to be downloaded.
- parameters (Optional[dict]): A JSON object containing the training parameters.
- logger (TLogger): A logger object.
- notifier: An optional notifier object.

Returns:

- dict: A dictionary containing the result URL of the trained model.
"""
import json
import os
from pathlib import Path

import dramatiq
from typing import Optional
from zipfile import ZipFile
from PIL import Image

from ..config import BASE_URL, TIME_LIMIT
from .const import DATASETS_PATH, DTI_RESULTS_PATH, DTI_QUEUE
from .training import (
    run_kmeans_training,
    run_sprites_training,
)
from ..shared.dataset import Dataset
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


def symlink_dataset(dataset: Dataset, dti_dataset_path: Path):
    """
    Create symbolic links from the shared document directory
    to the expected dataset path for the DTI submodule
    """
    dti_dataset_path.mkdir(parents=True, exist_ok=True)
    ready_file = dti_dataset_path / "ready.meta"

    train_dir = dti_dataset_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    for i, document in enumerate(dataset.documents):
        document.download()
        images = document.list_images()

        for j, image in enumerate(images):
            img_name = str(image.path.relative_to(document.path / "images")).replace(
                "/", "+"
            )
            target_path = train_dir / img_name
            # target_path = train_dir / f"doc_{document.uid}_{j:04d}{image.path.suffix}"

            if target_path.exists():
                target_path.unlink()  # Remove existing link if any

            rel_path = os.path.relpath(image.path, target_path.parent)
            os.symlink(rel_path, target_path)

    ready_file.touch()
    return dti_dataset_path


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=DTI_QUEUE
)
@notifying
def train_dti(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[str] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs,
):
    # TODO add notify_url in arguments
    """
    Train a DTI model

    Parameters:
    - experiment_id: the ID of the clustering task
    - dataset_uid: the ID of the dataset
    - dataset_url: the URL of the zipped dataset to be downloaded
    - parameters: a JSON object containing the training parameters
    - notify_url: an optional URL to notify when the task is complete
    - logger: a logger object
    - notifier: an optional notifier object
    """

    # current_task = CurrentMessage.get_current_message()
    # current_task_id = current_task.message_id

    result_file = DTI_RESULTS_PATH / f"{experiment_id}.zip"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(parameters)

    dataset = Dataset(dataset_uid, load=True)
    dti_dataset_path = DATASETS_PATH / "generic" / dataset.uid

    if not (dti_dataset_path / "ready.meta").exists():
        symlink_dataset(dataset, dti_dataset_path)
    else:
        print("Dataset already ready")
        # (dti_dataset_path / "ready.meta").touch()

    # Start training for dataset_name = generic
    if parameters.get("background_option", {}).get("use_sprites", False):
        # Use DTI sprites (1_learn_bg / 2_const_bg / 3_learn_fg)
        output_path = run_sprites_training(
            experiment_id, dataset_uid, parameters, logger
        )
    else:
        # Use DTI clustering
        output_path = run_kmeans_training(
            experiment_id, dataset_uid, parameters, logger
        )

    # zip results to DTI_RESULTS_PATH
    with ZipFile(result_file, "w") as zipObj:
        for file in output_path.glob("**/*"):
            if file.suffix == ".pkl":  # Don't include the model
                continue

            if file.suffix == ".png":  # Convert to jpg if not transparent
                img = Image.open(file)
                if img.mode != "RGBA" and img.format != "JPEG":
                    img.save(file, "JPEG", quality=85)

            zipObj.write(file, file.relative_to(output_path))

    return {
        "result_url": f"{BASE_URL}/clustering/{experiment_id}/result",
    }
