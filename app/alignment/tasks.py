import dramatiq
from typing import Optional

from .alignment import ComputeAlignment, DEFAULT_TRANSFORM
from .const import ALIGN_QUEUE, DEFAULT_METHOD
from ..config import TIME_LIMIT
from ..shared.utils.logging import notifying, TLogger, LoggerHelper, console
from ..shared.dataset import Dataset

from .lib.const import ALIGN_METHODS, AVAILABLE_TRANSFORMS


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=ALIGN_QUEUE
)
@notifying
def compute_alignment(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Run alignment task on a dataset of images

    Args:
    - experiment_id: the ID of the alignment task
    - dataset_uid: the ID of the dataset to use
    - parameters: a JSON object containing the task parameters (method)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    """
    dataset = Dataset(uid=dataset_uid, load=True)

    console(parameters, color="yellow")

    # Extract and validate method parameter
    method = parameters.get("method", DEFAULT_METHOD)
    if method not in ALIGN_METHODS:
        return {
            "error": f"Invalid method: {method}. Available methods: {ALIGN_METHODS}",
            "status": "ERROR",
        }

    target_transform = parameters.get("target_transform", DEFAULT_TRANSFORM)
    if target_transform not in AVAILABLE_TRANSFORMS:
        return {
            "error": f"Invalid target transform: {target_transform}. Available transforms: {AVAILABLE_TRANSFORMS}",
            "status": "ERROR",
        }

    alignment_task = ComputeAlignment(
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        logger=logger,
        notifier=notifier,
    )
    success = alignment_task.run_task()

    if success:
        return {
            "dataset_url": dataset.get_absolute_url(),
            "results_url": alignment_task.results_url,
            "error": alignment_task.error_list,
        }

    return {
        "error": alignment_task.error_list,
    }
