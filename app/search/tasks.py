import dramatiq
from typing import Optional

from .const import SEARCH_QUEUE
from ..config import TIME_LIMIT
from ..shared.utils.logging import notifying, TLogger, LoggerHelper, console
from ..shared.dataset import Dataset


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=SEARCH_QUEUE
)
@notifying
def index_dataset(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Create the feature index for a dataset of images

    Args:
    - experiment_id: the ID of the indexation task
    - dataset_uid: the ID of the dataset to use
    - parameters: a JSON object containing the task parameters (model)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    """
    dataset = Dataset(uid=dataset_uid, load=True)

    console(parameters, color="yellow")

    index_task = IndexDataset(
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        logger=logger,
        notifier=notifier,
    )
    success = index_task.run_task()

    if success:
        if (
            isinstance(success, dict)
            and success.get("dataset_url", False)
            and success.get("results_url", False)
        ):
            return success

        return {
            "dataset_url": dataset.get_absolute_url(),
            # TODO change to use only results_url (and remove annotations)
            "results_url": index_task.results_url,
            "error": index_task.error_list,
        }

    return {
        "error": index_task.error_list,
    }

@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=SEARCH_QUEUE
)
@notifying
def query_index(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Query an existing index with one or several images

    Args:
    - experiment_id: the ID of the query task
    - dataset_uid: the ID of the dataset to use
    - parameters: a JSON object containing the task parameters (index_id, query_images)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    """
    dataset = Dataset(uid=dataset_uid, load=True)

    console(parameters, color="yellow")

    query_task = QueryIndex(
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        logger=logger,
        notifier=notifier,
    )
    success = query_task.run_task()

    if success:
        if (
            isinstance(success, dict)
            and success.get("dataset_url", False)
            and success.get("results_url", False)
        ):
            return success

        return {
            "dataset_url": dataset.get_absolute_url(),
            # TODO change to use only results_url (and remove annotations)
            "results_url": query_task.results_url,
            "error": query_task.error_list,
        }

    return {
        "error": query_task.error_list,
    }