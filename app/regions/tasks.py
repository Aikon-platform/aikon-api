"""
Region extraction tasks
"""
import dramatiq
from typing import Optional

from ..config import TIME_LIMIT
from .const import EXT_QUEUE
from .regions import ExtractRegions
from ..shared.utils.logging import notifying, TLogger, LoggerHelper
from ..shared.dataset import Dataset


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, queue_name=EXT_QUEUE, store_results=True
)
@notifying
def extract_objects(
    experiment_id: str,
    dataset_uid: str,
    model: Optional[str] = None,
    postprocess: Optional[str] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Extract objects from a dataset using a model

    :param experiment_id: the experiment id
    :param dataset_uid: the dataset UID to process
    :param model: the model to use for extraction
    :param postprocess: the postprocess mode to use
    :param notify_url: the URL to notify the frontend
    :param notifier: the function allowing to send update to the frontend
    :param logger: the logger to use
    """
    dataset = Dataset(dataset_uid, load=True)

    regions_extraction_task = ExtractRegions(
        model=model,
        postprocess=postprocess,
        dataset=dataset,
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        notifier=notifier,
        **kwargs
    )
    success = regions_extraction_task.run_task()
    if success:
        # json to be dispatch to frontend with @notifying
        return {
            "dataset_url": dataset.get_absolute_url(),
            "results_url": regions_extraction_task.results_url,
            "error": regions_extraction_task.error_list,
        }

    # json to be dispatch to frontend with @notifying
    return {"error": regions_extraction_task.error_list}
