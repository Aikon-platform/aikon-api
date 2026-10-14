"""
Paleography tasks
"""
import dramatiq
from typing import Optional

from ..config import TIME_LIMIT
from .const import PALEO_QUEUE
from .paleography import Paleography
from ..shared.utils.logging import notifying, TLogger, LoggerHelper
from ..shared.dataset import Dataset


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, queue_name=PALEO_QUEUE, store_results=True
)
@notifying
def run_paleography(
    experiment_id: str,
    dataset_uid: str,
    params: Optional[dict] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    dataset = Dataset(dataset_uid, load=True)
    task = Paleography(
        dataset=dataset,
        params=params,
        logger=logger,
        experiment_id=experiment_id,
        notify_url=notify_url,
        notifier=notifier,
        **kwargs
    )
    if task.run_task():
        return {
            "dataset_url": dataset.get_absolute_url(),
            "results_url": task.results_url,
            "error": task.error_list,
        }
    return {"error": task.error_list}
