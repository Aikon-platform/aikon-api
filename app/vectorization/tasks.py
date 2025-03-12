import dramatiq
from typing import Optional
from datetime import datetime, timedelta
import os
import shutil

from ..config import TIME_LIMIT
from .const import VEC_QUEUE
from .vectorization import ComputeVectorization
from ..shared.dataset import Dataset
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=VEC_QUEUE
)
@notifying
def compute_vectorization(
    dataset_uid: str,
    model: str,
    experiment_id: str,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
):
    """
    Run vecto task on list of URL

    Parameters:
    - dataset_uid: id of the dataset to be vectorized
    - model: model used to vectorize dataset
    """
    dataset = Dataset(dataset_uid, load=True)
    vectorization_task = ComputeVectorization(
        dataset=dataset,
        model=model,
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        notifier=notifier,
    )
    success = vectorization_task.run_task()
    if success:
        return {
            "dataset_url": dataset.get_absolute_url(),
            "results_url": vectorization_task.results_url,
            "error": vectorization_task.error_list,
        }
    return {
        "error": vectorization_task.error_list
    }


@dramatiq.actor
def delete_images():
    # Function to delete images after a week
    week_ago = datetime.now() - timedelta(days=7)

    # TODO delete images associated with a vectorization

    # for vec_dir in os.listdir(IMG_PATH):
    #     dir_path = os.path.join(IMG_PATH, vec_dir)
    #     if os.path.isdir(dir_path):
    #         dir_modified_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
    #         if dir_modified_time < week_ago:
    #             shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
