"""
Routes for the search API.

Routes:

- POST ``/search/indexing/start``:
    Starts the search indexation process for a dataset.

    - Parameters: see start_indexing
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/search/indexing/<tracking_id>/cancel``:
    Cancel a search indexation task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/search/indexing/<tracking_id>/status``:
    Get the status of a search indexation task.

    - Response: JSON object containing the status of the task.

- GET ``/search/indexing/<tracking_id>/result``:
    Returns the index ID and metadata

    - Response: JSON object containing the result of the task.

- POST ``/search/query/start``:
    Starts the search query process for a dataset.

    - Parameters: see start_query
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/search/query/<tracking_id>/cancel``:
    Cancel a search query task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/search/query/<tracking_id>/status``:
    Get the status of a search query task.

    - Response: JSON object containing the status of the task.

- GET ``/search/query/<tracking_id>/result``:
    Sends the search results file for a given document

    - Response: JSON object containing the result of the task.

- GET ``/search/qsizes``:
    List the queues of the broker and the number of tasks in each queue.

    - Response: JSON object containing the queue sizes.

- GET ``/search/monitor``:
    Monitor the tasks of the broker.

    - Response: JSON object containing the monitoring information.

- GET ``/search/models``:
    Get the list of available models.

    - Response: JSON object containing the models and their modification dates.

- POST ``/search/clear``:
    Clear the images of a dataset.

    - Parameters:
        - ``dataset_id``: The ID of the dataset.
    - Response: JSON object indicating the number of cleared image directories.

"""

from flask import request, Blueprint, jsonify

import orjson
from safetensors.torch import safe_open as sft_safe_open, save_file as sft_save_file

from .tasks import index_dataset, query_index
from ..shared import routes as shared_routes
from ..shared.const import DATASETS_PATH
from ..shared.utils.fileutils import clear_dir, delete_path, delete_empty_dirs
from .const import (
    SEARCH_RESULTS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_XACCEL_PREFIX,
    MODEL_PATH,
)

from ..similarity.lib.const import FEAT_NET
from ..similarity.lib.models import DEFAULT_MODEL_INFOS

blueprint = Blueprint("similarity", __name__, url_prefix="/similarity")


@blueprint.route("indexing/start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_indexing(client_id):
    """
    Create the index for a dataset of images.

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "parameters": {
                "feat_net": "model.pt",
                "transpositions": ["none"]
            }
        }
    """

    if not request.is_json:
        return "No JSON in request: Indexing task aborted!"

    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request)

    parameters = {
        "feat_net": param.get("feat_net", FEAT_NET),
        "transpositions": param.get("transpositions", ["none"]),
        "client_id": client_id,
    }

    return shared_routes.start_task(
        index_dataset,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "parameters": parameters,
            "notify_url": notify_url,
        },
    )

@blueprint.route("query/start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_query(client_id):
    """
    Queries an existing index with one or several images

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "parameters": {
                "index_id": "a6b7c8d9e0f",
                "query_images": ["url/to/image1.jpg", "url/to/image2.jpg"],
                "transpositions": ["none"]
            }
        }
    """

    if not request.is_json:
        return "No JSON in request: Query task aborted!"

    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request)

    parameters = {
        "index_id": param.get("index_id"),
        "transpositions": param.get("transpositions", ["none"]),
        "client_id": client_id,
    }

    if not parameters["index_id"]:
        return "Missing index_id: Query task aborted!"

    return shared_routes.start_task(
        query_index,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "parameters": parameters,
            "notify_url": notify_url,
        },
    )

@blueprint.route("indexing/<tracking_id>/result", methods=["GET"])
def result_index(tracking_id: str):
    """
    Sends the index results file for a given document pair
    """
    return shared_routes.result(tracking_id, SEARCH_RESULTS_PATH, SEARCH_XACCEL_PREFIX, "json")


@blueprint.route("query/<tracking_id>/result", methods=["GET"])
def result_query(tracking_id: str):
    """
    Sends the query results file for a given document pair
    """
    return shared_routes.result(tracking_id, SEARCH_RESULTS_PATH, SEARCH_XACCEL_PREFIX, "json")


@blueprint.route("indexing/<tracking_id>/cancel", methods=["POST"])
@blueprint.route("query/<tracking_id>/cancel", methods=["POST"])
def cancel_task(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("indexing/<tracking_id>/status", methods=["GET"])
@blueprint.route("query/<tracking_id>/status", methods=["GET"])
def status_task(tracking_id: str):
    return shared_routes.status(tracking_id, index_dataset)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_search():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    assert index_dataset.broker == query_index.broker
    return shared_routes.qsizes(index_dataset.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_search():
    return shared_routes.monitor(SEARCH_RESULTS_PATH, index_dataset.broker)


@blueprint.route("monitor/clear/", methods=["POST"])
def clear_old_search():
    # TODO clear features associated with an old index or query
    return {
        "cleared_index": clear_dir(SEARCH_INDEX_PATH, path_to_clear="*.safetensors"),
        "cleared_results": clear_dir(SEARCH_RESULTS_PATH, path_to_clear="*.safetensors"),
    }


@blueprint.route("monitor/clear/index/<tracking_id>/", methods=["POST"])
def clear_index(tracking_id: str):
    """
    Clear all images, features and scores related to a given task
    """
    from .search import DatasetIndex, IndexDataset

    experiment_file = IndexDataset.path_for_task(tracking_id)
    if not experiment_file.exists():
        return {"error": "Experiment not found"}
    
    with open(experiment_file, "r") as f:
        experiment = orjson.loads(f.read())
    
    index_id = experiment["metadata"]["index_id"]
    index_path = DatasetIndex.path_for_id(index_id)

    if index_path.exists():
        with sft_safe_open(index_path, "torch") as f:
            metadata = f.metadata()
        
        # we check that the index is associated with the experiment
        if metadata["from_experiment"] != tracking_id:
            return {"error": "Index was not created by this experiment"}

    delete_path(index_path)
    delete_path(experiment_file)

    return {
        "cleared_index": 1,
        "cleared_experiment": 1,
    }

@blueprint.route("models", methods=["GET"])
def get_models():
    return shared_routes.models(MODEL_PATH, DEFAULT_MODEL_INFOS)
