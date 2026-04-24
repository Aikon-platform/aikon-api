"""
Routes for the region extraction API.

Routes:

- POST ``/region_extraction/start``:
    Starts the region extraction process for a dataset.

    - Parameters:
        - ``experiment_id``: The ID of the experiment.
        - ``notify_url``: The URL to notify when the task is done.
        - ``tracking_url``: The URL to track the task. TODO delete
        - ``dataset``: The dataset UID to process.
        - ``documents``: The documents to put into the dataset.
        - ``model``: The model to use for the extraction.
    - Response: JSON object containing the task ID and experiment ID.

- POST ``/region_extraction/<tracking_id>/cancel``:
    Cancel a region extraction task.

    - Parameters:
        - ``tracking_id``: The task ID.
    - Response: JSON object indicating the cancellation status.

- GET ``/region_extraction/<tracking_id>/status``:
    Get the status of a region extraction task.

    - Response: JSON object containing the status of the task.

- GET ``/region_extraction/qsizes``:
    List the queues of the broker and the number of tasks in each queue.

    - Response: JSON object containing the queue sizes.

- GET ``/region_extraction/monitor``:
    Monitor the tasks of the broker.

    - Response: JSON object containing the monitoring information.

- GET ``/region_extraction/models``:
    Get the list of available models.

    - Response: JSON object containing the models and their modification dates.

- POST ``/region_extraction/clear``:
    Clear the images of a dataset.

    - Parameters:
        - ``dataset_id``: The ID of the dataset.
    - Response: JSON object indicating the number of cleared image directories.


"""
from pathlib import Path

from flask import request, Blueprint

from app.region_extraction.tasks import extract_objects
from app.region_extraction.const import MODEL_PATH, DEFAULT_MODEL_INFOS
from app.shared.const import DOCUMENTS_PATH, SHARED_XACCEL_PREFIX
from app.shared.utils.fileutils import delete_path, clear_dir
from app.shared import routes as shared_routes

blueprint = Blueprint("region_extraction", __name__, url_prefix="/region_extraction")


@blueprint.route("start", methods=["POST"])
@shared_routes.error_wrapper
def start_region_extraction():
    """
    Extract regions for images from a list of IIIF URLs.

    Expected request format:

    .. code-block:: json

        {
            ...(tasking.routes.receive_task request)...
            "model": "model.pt",
            "postprocess": "none",  # string? (if empty string, no postprocessing. else, postprocessing type)
        }

    :return: The tracking_id of the task
    """
    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request, use_crops=False)

    model = param.get("model")
    postprocess = param.get("postprocess", "")

    return shared_routes.start_task(
        extract_objects,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "model": model,
            "postprocess": postprocess,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_region_extraction(tracking_id: str):
    """
    Cancel a regions extraction task
    """
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_region_extraction(tracking_id: str):
    """
    Get the status of a regions extraction task
    """
    return shared_routes.status(tracking_id, extract_objects)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_region_extraction():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(extract_objects.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_region_extraction():
    """
    Monitor the tasks of the broker
    """
    return shared_routes.monitor(DOCUMENTS_PATH, extract_objects.broker)


@blueprint.route("<tracking_id>/result", methods=["GET"])
def result_extraction(tracking_id: str):
    result_dir = DOCUMENTS_PATH
    # not correct, should be DOCUMENTS_PATH / dtype / uid / "annotations"
    return shared_routes.result(tracking_id, result_dir, SHARED_XACCEL_PREFIX, "json")


@blueprint.route("models", methods=["GET"])
def get_models():
    return shared_routes.models(MODEL_PATH, DEFAULT_MODEL_INFOS)


@blueprint.route("clear", methods=["POST"])
def clear_images():
    dataset_id = request.form["dataset_id"]
    # TODO change to use new dataset architecture
    # return {
    #     "cleared_img_dir": 1 if delete_path(IMG_PATH / dataset_id) else 0,
    # }
    return {
        "cleared_img_dir": 0,
    }


@blueprint.route("<doc_id>/delete", methods=["POST"])
def delete(doc_id: str):
    model_name = request.args.get("model_name")

    doc_dir, _ = shared_routes.delete(doc_id, to_delete=bool(model_name))
    if not doc_dir:
        return {
            "error": f"Document {doc_id} not found",
        }

    if not model_name:
        return {"cleared_document": 1}

    return {
        "cleared_annotations": clear_dir(
            doc_dir / "annotations", f"{model_name}*.json", delete_anyway=True
        ),
    }
