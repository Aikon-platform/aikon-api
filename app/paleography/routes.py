"""
Routes for the paleography API (M4M)
"""
from flask import request, Blueprint

from .tasks import run_paleography
from ..shared import routes as shared_routes
from ..shared.const import DOCUMENTS_PATH
from .const import PALEO_RESULTS_PATH, PALEO_XACCEL_PREFIX

blueprint = Blueprint("paleography", __name__, url_prefix="/paleography")


@blueprint.route("start", methods=["POST"])
@shared_routes.error_wrapper
def start_paleography():
    (
        experiment_id,
        notify_url,
        dataset,
        param,
    ) = shared_routes.receive_task(request, use_crops=False)

    return shared_routes.start_task(
        run_paleography,
        experiment_id,
        {"dataset_uid": dataset.uid, "params": param, "notify_url": notify_url},
    )


@blueprint.route("<tracking_id>/cancel", methods=["POST"])
def cancel_paleography(tracking_id: str):
    return shared_routes.cancel_task(tracking_id)


@blueprint.route("<tracking_id>/status", methods=["GET"])
def status_paleography(tracking_id: str):
    return shared_routes.status(tracking_id, run_paleography)


@blueprint.route("qsizes", methods=["GET"])
def qsizes_paleography():
    return shared_routes.qsizes(run_paleography.broker)


@blueprint.route("monitor", methods=["GET"])
def monitor_paleography():
    return shared_routes.monitor(DOCUMENTS_PATH, run_paleography.broker)


@blueprint.route("<doc_id>/result", methods=["GET"])
def result_paleography(doc_id: str):
    return shared_routes.result(doc_id, PALEO_RESULTS_PATH, PALEO_XACCEL_PREFIX, "zip")
