from flask import request, send_from_directory, Blueprint, jsonify
from slugify import slugify
import uuid

from .tasks import compute_vectorization
from ..shared import routes as shared_routes
from .const import VEC_RESULTS_PATH, VEC_XACCEL_PREFIX, MODEL_PATH, DEFAULT_MODEL_INFOS
from ..shared.utils.fileutils import clear_dir, delete_path

blueprint = Blueprint("vectorization", __name__, url_prefix="/vectorization")


@blueprint.route("start", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def start_vectorization(client_id):
    """
    Start the vectorization task.

    Expected parameters:

    .. code-block:: json

        {
            "experiment_id": "wit17_img17_anno17"
            "model": "checkpoint0045" # model file name stem
            "callback": "https://domain-name.com/receive-vecto",
            "tracking_url": "url for updates",
            "images": {
                "img_name": "https://domain-name.com/image_name.jpg",
                "img_name": "https://other-domain.com/image_name.jpg",
                "img_name": "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
                "img_name": "..."
            }
        }

    A list of images to download + information
    """
    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request, use_crops=False)
    model = param.get("model", None)

    return shared_routes.start_task(
        compute_vectorization,
        experiment_id,
        {
            "dataset_uid": dataset.uid,
            "model": model,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_vectorization(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_vectorization(tracking_id: str):
    return shared_routes.status(tracking_id, compute_vectorization)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_vectorization():
    """
    List the queues of the broker and the number of tasks in each queue
    """
    return shared_routes.qsizes(compute_vectorization.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_vectorization():
    return shared_routes.monitor(VEC_RESULTS_PATH, compute_vectorization.broker)


@blueprint.route("delete_and_relaunch", methods=["POST"])
@shared_routes.get_client_id
@shared_routes.error_wrapper
def delete_and_relaunch(client_id):
    if not request.is_json:
        return "No JSON in request: Vectorization task aborted!"

    if True:
        return jsonify({"Error": "Deprecated route, to be removed"})

    data = request.get_json()
    experiment_id = slugify(request.form.get("experiment_id", str(uuid.uuid4())))
    # dict of document ids with a URL containing a list of images
    dataset = data.get("images", {})
    # which url to send back the vectorization results and updates on the task
    notify_url = data.get("callback", None)
    doc_id = data.get("doc_id", None)
    model = data.get("model", None)

    # TODO delete images associated with vectorization
    # cleared_img_dir = delete_directory(f"{IMG_PATH}/{doc_id}")
    cleared_img_dir = True

    if cleared_img_dir:
        start_response = shared_routes.start_task(
            compute_vectorization,
            experiment_id,
            {
                "dataset": dataset,
                "notify_url": notify_url,
                "doc_id": doc_id,
                "model": model,
            },
        )
        return jsonify({"cleared_img_dir": 1, "start_vectorization": start_response})
    else:
        return jsonify(
            {
                "cleared_img_dir": 0,
                "start_vectorization": "Directory deletion failed, vectorization not started.",
            }
        )


@blueprint.route("<doc_id>/result", methods=["GET"])
def result_vectorization(doc_id: str):
    return shared_routes.result(
        doc_id, VEC_RESULTS_PATH / doc_id, VEC_XACCEL_PREFIX, "zip"
    )


@blueprint.route("models", methods=["GET"])
def get_models():
    return shared_routes.models(MODEL_PATH, DEFAULT_MODEL_INFOS)


@blueprint.route("<doc_id>/delete", methods=["POST"])
def delete(doc_id: str):
    doc_dir, dataset_id = shared_routes.delete(doc_id)
    if not doc_dir:
        return {"error": f"Document {doc_id} not found"}

    cleared_results = clear_dir(
        VEC_RESULTS_PATH, f"*{doc_id}*.svg", force_deletion=True
    )
    cleared_imgs = clear_dir(doc_dir / "images", force_deletion=True)
    delete_path(doc_dir / "images.json")

    # TODO check if everything is deleted

    return {
        "cleared_images": cleared_imgs,
        "cleared_results": cleared_results,
    }


# TODO add clear_doc + clear_old_vectorization routes (see similarity.routes)
