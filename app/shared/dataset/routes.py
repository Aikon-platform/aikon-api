"""
Routes for handling datasets

Allows to fetch and download content from a dataset
(mostly to avoid downloading IIIF manifests multiple times)
"""
from flask import Blueprint, jsonify, stream_with_context, Response

from .. import routes as shared_routes
from ..const import DOCUMENTS_PATH, SHARED_XACCEL_PREFIX
from ..utils.fileutils import zip_on_the_fly, sanitize_str

from .dataset import Dataset
from .document import Document

blueprint = Blueprint("datasets", __name__, url_prefix="/datasets")


@blueprint.route("dataset/<uid>", methods=["GET"])
def dataset_info(uid):
    """
    Get the dataset information
    """
    dataset = Dataset(uid, load=True)
    return jsonify(dataset.to_dict(with_url=True))


@blueprint.route("document/<dtype>/<path:uid>", methods=["GET"])
def document_info(dtype, uid):
    """
    Get the document information
    """
    document = Document(uid, dtype)
    return jsonify(document.to_dict(with_url=True))


@blueprint.route("document/<dtype>/<path:uid>/download", methods=["GET"])
def document_download(dtype, uid):
    """
    Download the document
    """
    document = Document(uid, dtype)
    relpath = document.path
    files = [
        # relative path beginning with "images/..."
        (str(im.path.relative_to(relpath)), im.path)
        for im in document.list_images()
        if im.path.exists()
    ]

    if not files:
        return jsonify({"error": f"No images found for document {uid}"}), 404

    if document.images_info_path.exists():
        files.append(("images.json", document.images_info_path))

    return Response(
        stream_with_context(zip_on_the_fly(files)),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={sanitize_str(document.uid)}.zip"
        },
    )


@blueprint.route("document/<dtype>/<path:uid>/<anno_file>", methods=["GET"])
def annotation_file(dtype, uid, anno_file):
    """
    Expose the annotation json file
    """
    return shared_routes.result(
        filename=anno_file,
        results_dir=DOCUMENTS_PATH / dtype / uid / "annotations",
        xaccel_prefix=SHARED_XACCEL_PREFIX,
        extension="json",
    )
