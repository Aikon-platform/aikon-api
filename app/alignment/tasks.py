import dramatiq
from typing import Optional

from .const import ALIGN_QUEUE, DEMO_NAME, ALIGN_METHODS, DEFAULT_METHOD
from ..config import TIME_LIMIT
from ..shared.dataset.document import get_file_url
from ..shared.utils.logging import notifying, TLogger, LoggerHelper, console
from ..shared.dataset import Dataset


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

    # TODO: Implement alignment logic here
    # For now, just return a placeholder
    alignment_task = None  # Will be implemented later
    
    # Placeholder for actual implementation
    # alignment_task = ComputeAlignment(
    #     experiment_id=experiment_id,
    #     dataset=dataset,
    #     parameters=parameters,
    #     notify_url=notify_url,
    #     logger=logger,
    #     notifier=notifier,
    # )
    # success = alignment_task.run_task()

    # For now, return a placeholder success
    return {
        "dataset_url": dataset.get_absolute_url(),
        "results_url": None,  # Will be updated when implementation is ready
        "method": method,
        "status": "SUCCESS",
        "message": "Alignment task placeholder - implementation pending",
    }
