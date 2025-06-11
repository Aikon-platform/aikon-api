"""
Routes for the similarity API.

Routes:

- POST ``/similarity/start``:
    Starts the similarity process for a dataset.

    - Parameters: see start_similarity
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/similarity/<tracking_id>/cancel``:
    Cancel a similarity task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/similarity/<tracking_id>/status``:
    Get the status of a similarity task.

    - Response: JSON object containing the status of the task.

- GET ``/similarity/qsizes``:
    List the queues of the broker and the number of tasks in each queue.

    - Response: JSON object containing the queue sizes.

- GET ``/similarity/monitor``:
    Monitor the tasks of the broker.

    - Response: JSON object containing the monitoring information.

- GET ``/similarity/models``:
    Get the list of available models.

    - Response: JSON object containing the models and their modification dates.

- POST ``/similarity/clear``:
    Clear the images of a dataset.

    - Parameters:
        - ``dataset_id``: The ID of the dataset.
    - Response: JSON object indicating the number of cleared image directories.


"""

from flask import request, Blueprint, jsonify

from .tasks import compute_similarity
from ..shared import routes as shared_routes
from ..shared.const import DATASETS_PATH
from ..shared.utils.fileutils import clear_dir, delete_path, delete_empty_dirs
from .const import (
    SIM_RESULTS_PATH,
    SIM_XACCEL_PREFIX,
    MODEL_PATH,
)

from .lib.const import FEAT_NET
from .lib.models import DEFAULT_MODEL_INFOS

blueprint = Blueprint("similarity", __name__, url_prefix="/similarity")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_similarity(client_id):
    """
    Compute similarity for a dataset of images.

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "parameters": {
                "algorithm": "algorithm",
                "feat_net": "model.pt",
                "feat_set": "set",
                "feat_layer": "layer",
                "segswap_prefilter": true, # if algorithm is "segswap"
                "segswap_n": 0, # if algorithm is "segswap"
                "transpositions": ["none", "rot90"]
            }
        }
    """

    if not request.is_json:
        return "No JSON in request: Similarity task aborted!"

    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request)

    parameters = {
        "feat_net": param.get("feat_net", FEAT_NET),
        "algorithm": param.get("algorithm", "cosine"),
        "cosine_n_filter": param.get("cosine_n_filter", 10),
        "segswap_prefilter": param.get("segswap_prefilter", True),
        "segswap_n": param.get("segswap_n", 10),
        "transpositions": param.get("transpositions", ["none"]),
        "client_id": client_id,
    }

    return shared_routes.start_task(
        compute_similarity,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "parameters": parameters,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_similarity(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_similarity(tracking_id: str):
    return shared_routes.status(tracking_id, compute_similarity)


@blueprint.route("<doc_pair>/result", methods=["GET"])
def result_similarity(doc_pair: str):
    """
    Sends the similarity results file for a given document pair
    """
    return shared_routes.result(doc_pair, SIM_RESULTS_PATH, SIM_XACCEL_PREFIX, "json")


@blueprint.route("qsizes", methods=["GET"])
def qsizes_similarity():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(compute_similarity.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_similarity():
    return shared_routes.monitor(SIM_RESULTS_PATH, compute_similarity.broker)


@blueprint.route("monitor/clear/", methods=["POST"])
def clear_old_similarity():
    # TODO clear images and features associated with an old similarity
    return {
        # "cleared_img_dir": clear_dir(IMG_PATH),
        # "cleared features": clear_dir(FEATS_PATH, path_to_clear="*.pt"),
        "cleared_results": clear_dir(SIM_RESULTS_PATH, path_to_clear="*.npy"),
    }


@blueprint.route("monitor/clear/<doc_id>/", methods=["POST"])
def clear_doc(doc_id: str):
    """
    Clear all images, features and scores related to a given document
    """

    # doc_id = "{doc_id}"
    # TODO: delete images and features associated with similarity

    return {
        # "cleared_img_dir": clear_dir(
        #     IMG_PATH, path_to_clear=f"*{doc_id}*", delete_anyway=True
        # ),
        # "cleared features": clear_dir(
        #     FEATS_PATH, path_to_clear=f"*{doc_id}*.pt", delete_anyway=True
        # ),
        "cleared_results": clear_dir(
            SIM_RESULTS_PATH, path_to_clear=f"*{doc_id}*.npy", delete_anyway=True
        ),
    }


@blueprint.route("<doc_id>/delete", methods=["POST"])
def delete(doc_id: str):
    algorithm = request.args.get("algorithm")
    feat_net = request.args.get("feat_net")

    doc_dir, dataset_id = shared_routes.delete(doc_id)
    if not doc_dir:
        return {
            "error": f"Document {doc_id} not found",
        }

    res_to_clear = (
        f"*/{algorithm}*{doc_id}*.json" if algorithm else f"*/*{doc_id}*.json"
    )
    cleared_results = clear_dir(SIM_RESULTS_PATH, res_to_clear, delete_anyway=True)
    # delete empty directory of results
    cleared_res_dir = delete_empty_dirs(SIM_RESULTS_PATH)

    cleared_imgs = clear_dir(doc_dir / "images", delete_anyway=True)
    delete_path(doc_dir / "images.json")
    delete_path(doc_dir / "metadata.json")

    if not dataset_id:
        return {
            "cleared_images": cleared_imgs,
            "cleared_results": cleared_results,
            "cleared_result_directories": cleared_res_dir,
        }

    cleared_results += clear_dir(
        SIM_RESULTS_PATH, f"*/{dataset_id}-scores.json", delete_anyway=True
    )
    feat_to_clear = f"{dataset_id}*{feat_net}.pt" if feat_net else "*"
    cleared_feats = clear_dir(
        DATASETS_PATH / dataset_id / "features", feat_to_clear, delete_anyway=True
    )

    return {
        "cleared_images": cleared_imgs,
        "cleared_results": cleared_results,
        "cleared_result_directories": cleared_res_dir,
        "cleared_features": cleared_feats,
    }


@blueprint.route("models", methods=["GET"])
def get_models():
    return shared_routes.models(MODEL_PATH, DEFAULT_MODEL_INFOS)
