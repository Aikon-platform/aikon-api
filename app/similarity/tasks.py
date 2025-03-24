import dramatiq
from typing import Optional

from .const import SIM_QUEUE, DEMO_NAME
from .similarity import ComputeSimilarity
from ..config import TIME_LIMIT
from ..shared.dataset.document import get_file_url
from ..shared.utils.logging import notifying, TLogger, LoggerHelper, console
from ..shared.dataset import Dataset


@dramatiq.actor(
    time_limit=TIME_LIMIT, max_retries=0, store_results=True, queue_name=SIM_QUEUE
)
@notifying
def compute_similarity(
    experiment_id: str,
    dataset_uid: str,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs
):
    """
    Run similarity retrieval task on all pairs of documents

    Args:
    - experiment_id: the ID of the similarity retrieval task
    - dataset_uid: the ID of the dataset to use
    - parameters: a JSON object containing the task parameters (model)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object
    """
    dataset = Dataset(uid=dataset_uid, load=True)

    console(parameters, color="yellow")

    similarity_task = ComputeSimilarity(
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        logger=logger,
        notifier=notifier,
    )
    success = similarity_task.run_task()

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
            "results_url": similarity_task.results_url,
            "error": similarity_task.error_list,
        }

    return {
        "error": similarity_task.error_list,
    }
