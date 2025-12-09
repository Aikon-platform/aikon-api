"""
The Document class, which represents a document in the dataset
"""
from typing_extensions import NotRequired

import requests
import json
import httpx
from pathlib import Path
from PIL import Image as PImage
from stream_unzip import stream_unzip
from typing import List, Optional, TypedDict, Literal
from iiif_download import IIIFManifest

from ... import config
from ..const import DOCUMENTS_PATH
from ..utils.fileutils import sanitize_str, check_if_file
from ..utils.img import MAX_SIZE, download_image, get_img_paths
from ..utils.logging import console, serializer
from ..utils import get_json
from .utils import Image, pdf_to_img
from ...config import BASE_URL

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".json", ".tiff", ".pdf"}

DocType = Literal["zip", "pdf", "img", "url_list", "iiif"]


class DocDict(TypedDict):
    uid: str
    type: DocType
    src: str
    url: NotRequired[str]
    download: NotRequired[str]
    metadata: NotRequired[dict]


def get_file_url(demo_name, filename):
    # filename can be relpath from demo result dir
    filename = filename.replace("/", "~")
    return f"{BASE_URL}/{demo_name}/{filename}/result"


class Document:
    """
    A Document is a list of images that are part of a single document

    It can be :
    - downloaded from a single IIIF manifest
    - downloaded from a ZIP file
    - downloaded from a dictionary of single image URLs
    - downloaded from a PDF file

    :param uid: The unique identifier of the document
    :param path: The path to the document on disk (default: DOCUMENTS_PATH/uid)

    The document is saved to disk in the following structure:

    .. code-block:: none

        - path/
            - images/
                - ...jpg
            - cropped/
                - ...jpg
            - annotations/
                - ...json
            - images.json
            - metadata.json
    """

    def __init__(
        self,
        uid: str = None,
        dtype: DocType = "zip",
        path: Path | str = None,
        src: Optional[str] = None,
    ):
        self.uid = sanitize_str(uid or src)
        self.path = Path(
            path if path is not None else DOCUMENTS_PATH / dtype / self.uid
        )
        self.src = src
        self.dtype = dtype
        self._images = []

    @classmethod
    def from_dict(cls, doc_dict: dict) -> "Document":
        """
        Create a new Document from a dictionary
        doc_dict = {
            "type": "zip | pdf | img | url_list | iiif",
            "src": "documents_to_be_downloaded",
            ?"uid": "optional custom_id"
        }
        """
        return Document(
            doc_dict.get("uid", None), doc_dict["type"], src=doc_dict["src"]
        )

    def to_dict(self, with_url: bool = False, with_metadata: bool = False) -> DocDict:
        """
        Convert the document to a dictionary
        """
        ret: DocDict = {
            "uid": self.uid,
            "type": self.dtype,
            "src": str(self.src),
        }
        if with_url:
            ret["url"] = self.get_absolute_url()
            ret["download"] = self.get_download_url()
        if with_metadata:
            ret["metadata"] = self.load_metadata()
        return ret

    def get_absolute_url(self):
        """
        Get the absolute URL of the document
        """
        return f"{config.BASE_URL}/datasets/document/{self.dtype}/{self.uid}"
        # return url_for("datasets.document_info", dtype=self.dtype, uid=self.uid, _external=True)

    def get_download_url(self):
        """
        Get the URL to download the document
        """
        return f"{self.get_absolute_url()}/download"
        # return url_for("datasets.document_download", dtype=self.dtype, uid=self.uid, _external=True)

    def get_results_url(self, demo_name):
        return get_file_url(demo_name, self.uid)

    def get_annotations_url(self, filename=None):
        return f"{self.get_absolute_url()}/{filename}"

    @property
    def images_path(self):
        return self.path / "images"

    @property
    def cropped_images_path(self):
        return self.path / "cropped"

    @property
    def annotations_path(self):
        return self.path / "annotations"

    @property
    def images_info_path(self):
        return self.path / "images.json"

    @property
    def metadata_path(self):
        return self.path / "metadata.json"

    def save_metadata(self, metadata: dict):
        metadata = {**self.load_metadata(), **metadata}
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)
        self._metadata = metadata

    def load_metadata(self):
        if hasattr(self, "_metadata"):
            return self._metadata
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def create_path(path):
        path = Path(path).resolve()
        if path.parent.exists() and path.parent.is_file():
            console(f"Removing file folder creation: {path.parent}")
            path.parent.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def images(self):
        """
        List of images in the document
        """
        if len(self._images) == 0:
            self.load_images()
        return self._images

    def save_images(self, images: List[Image], remove_src:bool=False):
        """
        Save the images of the document
        if `remove_src`, set `src` to `null` on each Image. use when saving the `images.json` of pdfs and zips: the `src` generated by `self.list_images_from_path` is not a download URL, which is what we would want.
        """
        if remove_src:
            images = [ im.remove_src() for im in images ]
        self._images = images
        with open(self.images_info_path, "w") as f:
            json.dump([img.to_dict() for img in images], f, default=serializer)

    def load_images(self):
        """
        Load the images of the document
        """
        if not self.images_info_path.exists():
            # TODO save it? or regenerate on each load?
            self._images = self.list_images_from_path()
            return
        with open(self.images_info_path, "r") as f:
            self._images = [Image.from_dict(img, self) for img in json.load(f)]

    def _download_from_iiif(self, manifest_url: str):
        """
        Download images from a IIIF manifest
        """
        manifest = IIIFManifest(manifest_url)
        manifest.download(save_dir=self.images_path)
        self.save_metadata(
            {
                "title": manifest.get_meta("Title"),
            }
        )

        self.save_images(
            [
                Image(
                    id=iiif_image.img_name,
                    src=iiif_image.url,
                    path=iiif_image.img_path,
                    metadata={
                        "page": iiif_image.idx,
                        **(
                            {"label": iiif_image.resource["label"]}
                            if iiif_image.resource.get("label")
                            else {}
                        ),
                    },
                    document=self,
                )
                for iiif_image in manifest.get_images()
            ]
        )

    def _download_from_zip(self, zip_url: str):
        def zipped_chunks():
            with httpx.stream("GET", zip_url) as r:
                yield from r.iter_bytes(chunk_size=8192)

        def skip():
            for _ in unzipped_chunks:
                pass

        for file_name, file_size, unzipped_chunks in stream_unzip(zipped_chunks()):
            file_name = file_name.decode("utf-8")
            if "/." in "/" + file_name.replace("\\", "/"):  # hidden file
                skip()
                continue
            path = self.images_path / file_name
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                skip()
                continue

            self.create_path(path)

            with open(path, "wb") as f:
                for chunk in unzipped_chunks:
                    f.write(chunk)

    def _download_from_url_list(self, images_list_url: str):
        """
        Download images from a dictionary of URLs [img_stem -> img_url]
        """
        images_dict = get_json(images_list_url)
        images = []
        for img_stem, img_url in images_dict.items():
            download_image(img_url, self.images_path, img_stem)
            images.append(
                Image(
                    id=img_stem,
                    src=img_url,
                    path=self.images_path / f"{img_stem}.jpg",
                    document=self,
                )
            )
        self.save_images(images)

    def _download_from_pdf(self, pdf_url: str):
        """
        Download pdf, convert to images and save
        """
        pdf_path = self.path / pdf_url.split("/")[-1]
        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        pdf_to_img(pdf_path, self.images_path)
        pdf_path.unlink()
        self.save_images(self.list_images_from_path(), True)

    def _download_from_img(self, img_url: str):
        """
        Download image and save
        """
        # TODO make this work
        download_image(img_url, self.images_path, "image_name")

    def download(self) -> None:
        """
        Download a document from its source definition
        """
        if self.images_path.exists() and self.has_images():
            console(
                f"Document {self.uid} already exists in {self.images_path}, skipping download.",
                color="blue",
            )
            return

        console(
            f"Downloading [{self.dtype}] document {self.uid} in {self.images_path}...",
            color="blue",
        )

        self.images_path.mkdir(parents=True, exist_ok=True)
        self.save_metadata({"src": self.src})
        if self.dtype == "iiif":
            self._download_from_iiif(self.src)
        elif self.dtype == "zip":
            self._download_from_zip(self.src)
        elif self.dtype == "url_list":
            self._download_from_url_list(self.src)
        elif self.dtype == "pdf":
            self._download_from_pdf(self.src)
        elif self.dtype == "img":
            self._download_from_img(self.src)
        else:
            raise ValueError(f"Unknown document type: {self.dtype}")

    def list_images(self) -> List[Image]:
        """
        Iterate over the images in the document
        """
        return self.images

    def list_images_from_path(self) -> List[Image]:
        """
        Iterate over the images in the document's folder
        """
        return [
            Image(
                id=str(img_path.relative_to(self.images_path)),
                src=str(img_path.relative_to(self.path)),
                path=img_path,
                document=self,
            )
            for img_path in get_img_paths(self.images_path)
        ]

    def has_images(self) -> bool:
        return check_if_file(self.images_path, extensions=ALLOWED_EXTENSIONS)

    def prepare_crops(self, crops: List[dict]) -> List[Image]:
        """
        Prepare crops for the document

        Args:
            crops: A list of crops {document, source, source_info, crops: [{crop_id, relative: {x1, y1, w, h}}]}

        Returns:
            A list of Image objects
        """
        source = None
        im = None
        crop_list = []

        mapping = {im.id: im for im in self.images}

        for img in crops:
            if img["doc_uid"] != self.uid:
                continue
            source_info = (
                Image.from_dict(img["source_info"], self)
                if img.get("source_info")
                else None
            )

            for crop in img["crops"]:
                crop_path = self.cropped_images_path / f"{crop['crop_id']}.jpg"
                crop_sum = ",".join(
                    [f'{crop["relative"][k]:0.3f}' for k in ["x1", "y1", "x2", "y2"]]
                )

                crop_list.append(
                    Image(
                        id=crop["crop_id"],
                        src=getattr(source_info, "src", None) or crop_path.name,
                        path=crop_path,
                        metadata={
                            **(getattr(source_info, "metadata", None) or {}),
                            "crop": crop_sum,
                        },
                        document=self,
                    )
                )
                if crop_path.exists():
                    continue

                self.create_path(crop_path)

                if source != img["source"]:
                    source = img["source"]
                    if source in mapping:
                        im = mapping[source].path
                    else:
                        im = self.images_path / source
                    im = PImage.open(im).convert("RGB")

                box = crop["relative"]
                if "x1" in box:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                else:
                    x1, y1, w, h = box["x"], box["y"], box["width"], box["height"]
                    x2, y2 = x1 + w, y1 + h
                x1, y1, x2, y2 = (
                    int(x1 * im.width),
                    int(y1 * im.height),
                    int(x2 * im.width),
                    int(y2 * im.height),
                )
                if x2 - x1 == 0 or y2 - y1 == 0:
                    # use placeholder image
                    im_cropped = PImage.new("RGB", (MAX_SIZE, MAX_SIZE))
                else:
                    im_cropped = im.crop((x1, y1, x2, y2))
                im_cropped.save(crop_path)

        return crop_list
