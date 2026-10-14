"""
Routes for the alignment API.

Routes:

- POST ``/alignment/start``:
    Starts the alignment process for a dataset.

    - Parameters: see start_alignment
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/alignment/<tracking_id>/cancel``:
    Cancel an alignment task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/alignment/<tracking_id>/status``:
    Get the status of an alignment task.

    - Response: JSON object containing the status of the task.

- GET ``/alignment/models``:
    List the features available as ``ransac_feature`` (SIFT and the similarity models).

    - Response: JSON object mapping each feature to its ``model``, ``name`` and ``desc``.

- GET ``/alignment/qsizes``:
    List the queues of the broker and the number of tasks in each queue.

    - Response: JSON object containing the queue sizes.

- GET ``/alignment/monitor``:
    Monitor the tasks of the broker.

    - Response: JSON object containing the monitoring information.

- POST ``/alignment/<doc_id>/delete``:
    Delete all images, features and alignment results related to a given document.

    - Parameters:
        - ``doc_id``: The document ID.
        - ``method``: The alignment method (optional, defaults to "*").
    - Response: JSON object indicating the number of cleared files.

"""

from flask import request, Blueprint, jsonify

from .tasks import compute_alignment
from ..shared import routes as shared_routes
from ..shared.const import DATASETS_PATH
from ..shared.utils.fileutils import clear_dir, delete_path, delete_empty_dirs
from ..shared.utils.fileutils import list_known_models
from ..similarity.const import MODEL_PATH
from ..similarity.lib.models import DEFAULT_MODEL_INFOS, MODELS
from .const import (
    ALIGN_RESULTS_PATH,
    ALIGN_XACCEL_PREFIX,
    DEFAULT_METHOD,
    DEFAULT_RANSAC_FEATURE,
)
from .lib.const import ALIGN_METHODS, AVAILABLE_TRANSFORMS, DENSE_ARCHS, SIFT_FEATURE

blueprint = Blueprint("alignment", __name__, url_prefix="/alignment")


def available_features() -> dict:
    """SIFT plus the feature extraction networks of the similarity app giving a feature map"""
    return {
        SIFT_FEATURE: {
            "model": SIFT_FEATURE,
            "name": SIFT_FEATURE,
            "desc": "Classical SIFT keypoints and descriptors (no network)",
        },
        **{
            name: info
            for name, info in list_known_models(MODEL_PATH, DEFAULT_MODEL_INFOS).items()
            if MODELS.get(name, {}).get("arch") in DENSE_ARCHS
        },
    }


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_alignment(client_id):
    """
    Compute alignment for a dataset of images.

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "parameters": {
                "method": "ransac",
                "ransac_feature": "SIFT",
                "target_transform": "homographic",
                "reference_image": null
            }
        }
    """

    if not request.is_json:
        return "No JSON in request: Alignment task aborted!"

    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request)

    parameters = {
        "method": param.get("method", DEFAULT_METHOD),
        "ransac_feature": param.get("ransac_feature", DEFAULT_RANSAC_FEATURE),
        "target_transform": param.get("target_transform", "homographic"),
        "reference_image": param.get("reference_image", None),
        "client_id": client_id,
    }

    # Validate method parameter
    method = parameters["method"]
    if method not in ALIGN_METHODS:
        return jsonify(
            {
                "error": f"Invalid method: {method}. Available methods: {ALIGN_METHODS}",
                "available_methods": ALIGN_METHODS,
            }
        )

    # Validate ransac_feature parameter
    features = available_features()
    ransac_feature = parameters["ransac_feature"]
    if ransac_feature not in features:
        return jsonify(
            {
                "error": f"Invalid ransac_feature: {ransac_feature}. Available features: {list(features)}",
                "available_features": list(features),
            }
        )

    # Validate target_transform parameter
    target_transform = parameters["target_transform"]
    if target_transform not in AVAILABLE_TRANSFORMS:
        return jsonify(
            {
                "error": f"Invalid target_transform: {target_transform}. Available transforms: {AVAILABLE_TRANSFORMS}",
                "available_transforms": AVAILABLE_TRANSFORMS,
            }
        )

    return shared_routes.start_task(
        compute_alignment,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "parameters": parameters,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_alignment(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_alignment(tracking_id: str):
    return shared_routes.status(tracking_id, compute_alignment)


@blueprint.route("<doc_pair>/result", methods=["GET"])
def result_alignment(doc_pair: str):
    """
    Sends the alignment results file for a given document pair
    """
    return shared_routes.result(
        doc_pair, ALIGN_RESULTS_PATH, ALIGN_XACCEL_PREFIX, "json"
    )


@blueprint.route("qsizes", methods=["GET"])
def qsizes_alignment():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(compute_alignment.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_alignment():
    return shared_routes.monitor(ALIGN_RESULTS_PATH, compute_alignment.broker)


@blueprint.route("monitor/clear/", methods=["POST"])
def clear_old_alignment():
    from ..shared.utils.fileutils import clear_dir

    return {
        "cleared_results": clear_dir(ALIGN_RESULTS_PATH, path_to_clear="*.json"),
    }


@blueprint.route("<doc_id>/delete", methods=["POST"])
def delete(doc_id: str):
    """
    Delete all images, features and alignment results related to a given document
    """
    method = request.args.get("method") or "*"

    if not (doc_dir := shared_routes.find_document(doc_id)):
        return {"error": f"Document {doc_id} not found"}
    dataset_ids = shared_routes.document_datasets(doc_id)

    # Clear alignment results files
    # Result files are named based on method and doc_ids
    cleared_results = sum(
        clear_dir(ALIGN_RESULTS_PATH, pattern, delete_anyway=True)
        for pattern in (
            f"*/{method}-{doc_id}-*.json",
            f"*/{method}-*-{doc_id}.json",
            *(f"*/{d}-alignment.json" for d in dataset_ids),
        )
    )

    # Clear images and metadata
    cleared_imgs = clear_dir(doc_dir / "images", delete_anyway=True)
    delete_path(doc_dir / "images.json")
    delete_path(doc_dir / "metadata.json")

    return {
        "cleared_images": cleared_imgs,
        "cleared_results": cleared_results,
        "cleared_result_directories": delete_empty_dirs(ALIGN_RESULTS_PATH),
    }


@blueprint.route("models", methods=["GET"])
def get_models():
    """
    Get the list of features usable as ``ransac_feature``: SIFT and the
    feature extraction networks of the similarity app.

    - Response: JSON object mapping each feature to its ``model``, ``name`` and ``desc``.
    """
    return jsonify(available_features())


@blueprint.route("methods", methods=["GET"])
def get_methods():
    """
    Get the list of available alignment methods.

    - Response: JSON object containing the available methods.
    """
    return jsonify(
        {
            "methods": ALIGN_METHODS,
            "default_method": DEFAULT_METHOD,
        }
    )
