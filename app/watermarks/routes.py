from flask import (
    request,
    send_from_directory,
    jsonify,
    Blueprint,
)
from slugify import slugify
from PIL import Image, ImageOps
import json
import uuid

from ..shared.utils.fileutils import xaccel_send_from_directory, clear_dir
from ..shared import routes as shared_routes
from .. import config
from .const import (
    WATERMARKS_SOURCES_FOLDER,
    WATERMARKS_XACCEL_PREFIX,
    WATERMARKS_RESULTS_FOLDER,
    WATERMARKS_TMP_FOLDER,
)
from .tasks import pipeline
from .sources import WatermarkSource

MODELS = {}
DEVICE = "cpu"

blueprint = Blueprint("watermarks", __name__, url_prefix="/watermarks")


@blueprint.route("sources", methods=["GET"])
def sources():
    return jsonify(WatermarkSource.list_available())


def send_result(source, filename):
    if not source.data_folder.exists():
        r = jsonify({"error": f"Source {source.uid} not found"})
        r.status_code = 404
        return r

    f = source.data_folder / filename
    if not f.exists():
        r = jsonify({"error": f"Source {filename} {source.uid} not found"})
        r.status_code = 404
        return r

    f = f.relative_to(WATERMARKS_SOURCES_FOLDER)

    if not config.USE_NGINX_XACCEL:
        return send_from_directory(WATERMARKS_SOURCES_FOLDER, str(f))

    return xaccel_send_from_directory(
        WATERMARKS_SOURCES_FOLDER, WATERMARKS_XACCEL_PREFIX, str(f)
    )


@blueprint.route("sources/<source>/images.zip", methods=["GET"])
def images(source):
    source = WatermarkSource(source)
    return send_result(source, "images.zip")


@blueprint.route("sources/<source>/index.json", methods=["GET"])
def index(source):
    source = WatermarkSource(source)
    return send_result(source, "index.json")


@blueprint.route("start", methods=["POST"])
@shared_routes.error_wrapper
def start_task():
    source = request.form.get("compare_to", None)
    detect = request.form.get("detect", "true").lower() == "true"
    experiment_id = slugify(request.form.get("experiment_id", str(uuid.uuid4())))
    notify_url = request.form.get("notify_url", None)

    if source is not None:
        source = WatermarkSource(source)

        if not source.data_folder.exists():
            r = jsonify({"error": f"Source {source.uid} not found"})
            r.status_code = 404
            return r

    im = Image.open(request.files["image"])
    im = ImageOps.exif_transpose(im).convert("RGB")
    sv_dir = WATERMARKS_TMP_FOLDER
    sv_dir.mkdir(parents=True, exist_ok=True)
    im_file = sv_dir / f"{experiment_id}.jpg"
    im.save(im_file)

    return shared_routes.start_task(
        pipeline,
        experiment_id,
        {
            "image_path": str(im_file),
            "detect": detect,
            "compare_to": source.uid if source else None,
            "notify_url": notify_url,
        },
    )


@blueprint.route("<tracking_id>/status", methods=["GET"])
def task_status(tracking_id):
    return shared_routes.status(tracking_id, pipeline)


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def task_cancel(tracking_id):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("task/<tracking_id>/result", methods=["GET"])  # TODO is it used?
def task_result(tracking_id):
    out_file = WATERMARKS_RESULTS_FOLDER / f"{tracking_id}.json"
    if not out_file.exists():
        r = jsonify({"error": f"Result {tracking_id} not found"})
        r.status_code = 404
        return r

    with open(out_file, "r") as f:
        return jsonify(json.load(f))


@blueprint.route("monitor/clear/", methods=["POST"])
def clear_old_tasks():
    return {
        "cleared_tmp_queries": clear_dir(WATERMARKS_TMP_FOLDER),
        "cleared_results": clear_dir(WATERMARKS_RESULTS_FOLDER),
    }


@blueprint.route("monitor/clear/<tracking_id>/", methods=["POST"])
def clear_task(tracking_id: str):
    return {
        "cleared_results": clear_dir(
            WATERMARKS_RESULTS_FOLDER,
            file_to_check=f"{tracking_id}.json",
            delete_anyway=True,
        ),
    }
