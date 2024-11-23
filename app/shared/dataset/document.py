"""
The Document class, which represents a document in the dataset
"""

from flask import url_for
from pathlib import Path
import requests
from stream_unzip import stream_unzip
from zipfile import ZipFile
from urllib.parse import urlparse
from typing import List, Optional
import json
from PIL import Image as PImage

from ... import config
from ..const import DOCUMENTS_PATH
from ..utils.iiif import IIIFDownloader, get_json
from ..utils.fileutils import sanitize_str, has_content
from ..utils.img import download_images, MAX_SIZE, download_image, get_img_paths
from ..utils.logging import console
from .utils import Image

class Document:
    """
    A Document is a list of images that are part of a single document

    It can be :
    - downloaded from a single IIIF manifest
    - downloaded from a ZIP file
    - downloaded from a dictionary of single image URLs

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
            - mapping.json
    """

    def __init__(self, uid: str=None, dtype: str="zip", path: Path | str = None, src: Optional[str] = None):
        if uid is None:
            uid = src
        uid = sanitize_str(uid)
        self.uid = uid
        if path is None:
            path = DOCUMENTS_PATH / dtype / uid
        self.path = Path(path)
        self.src = src
        self.dtype = dtype
        self._mapping = None  # A mapping of filenames to their URLs

    def from_dict(doc_dict: dict) -> "Document":
        """
        Create a new Document from a dictionary
        """
        return Document(doc_dict.get("uid", None), doc_dict["type"], src=doc_dict["src"])
    
    def to_dict(self, with_url: bool=False) -> dict:
        """
        Convert the document to a dictionary
        """
        ret = {
            "uid": self.uid,
            "type": self.dtype,
            "src": self.src,
        }
        if with_url:
            ret["url"] = self.get_absolute_url()
            ret["download"] = self.get_download_url()
        return ret
    
    def get_absolute_url(self):
        """
        Get the absolute URL of the document
        """
        return f"{config.BASE_URL}/datasets/document/{self.dtype}/{self.uid}"
        return url_for("datasets.document_info", dtype=self.dtype, uid=self.uid, _external=True)
    
    def get_download_url(self):
        """
        Get the URL to download the document
        """
        return f"{self.get_absolute_url()}/download"
        return url_for("datasets.document_download", dtype=self.dtype, uid=self.uid, _external=True)

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
    def mapping_path(self):
        return self.path / "mapping.json"
    
    @property
    def mapping(self):
        """
        A mapping filenames -> UID/URL
        """
        if self._mapping is None:
            self._load_mapping()
        return self._mapping

    def _extend_mapping(self, mapping: dict):
        """
        Extend the mapping of the document with a new mapping
        """
        self.mapping.update(mapping)
        with open(self.path / "mapping.json", "w") as f:
            json.dump(self.mapping, f)

    def _load_mapping(self):
        """
        Load the mapping of the document from a JSON file
        """
        if not self.mapping_path.exists():
            self._mapping = {}
            return
        with open(self.mapping_path, "r") as f:
            self._mapping = json.load(f)

    def _download_from_iiif(self, manifest_url: str):
        """
        Download images from a IIIF manifest
        """
        self.images_path.mkdir(parents=True, exist_ok=True)
        downloader = IIIFDownloader(manifest_url, target_path=self.images_path)
        mapping = downloader.run()
        self._extend_mapping(mapping)
        console(f"Downloaded {len(mapping)} images from {manifest_url} to {self.images_path}", color="green")

    def _download_from_images_zip(self, zip_url: str):
        """
        Download images from a ZIP file
        """
        self.images_path.mkdir(parents=True, exist_ok=True)

        def zipped_chunks():
            with requests.get(zip_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    yield chunk

        for file_name, file_size, unzipped_chunks in stream_unzip(zipped_chunks()):
            file_name = file_name.decode("utf-8")
            path = self.images_path / file_name
            if "/." in "/" + file_name.replace("\\", "/"): # hidden file
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                for chunk in unzipped_chunks:
                    f.write(chunk)

    def _download_from_url_list(self, images_list_url: str):
        """
        Download images from a dictionary of URLs [img_stem -> img_url]
        """
        images_dict = get_json(images_list_url)
        self.images_path.mkdir(parents=True, exist_ok=True)
        for img_stem, img_url in images_dict.items():
            download_image(img_url, self.images_path, img_stem)
        self._extend_mapping(images_dict)

    def download(self) -> None:
        """
        Download a document from its source definition
        """
        console(f"Downloading [{self.dtype}] {self.uid}...", color="blue")

        if self.dtype == "iiif":
            self._download_from_iiif(self.src)
        elif self.dtype == "zip":
            self._download_from_images_zip(self.src)
        elif self.dtype == "url_list":
            self._download_from_url_list(self.src)
        else:
            raise ValueError(f"Unknown document type: {self.dtype}")

    def list_images(self) -> List[Image]:
        """
        Iterate over the images in the document
        """
        return [
            Image(id=img_path.name, src=self.mapping.get(img_path.name, img_path.relative_to(self.images_path)), path=img_path, document=self)
            for img_path in get_img_paths(self.images_path)
        ]

    def prepare_crops(self, crops: List[dict]) -> List[Image]:
        """
        Prepare crops for the document

        Args:
            crops: A list of crops {document, source, crops: [{crop_id, relative: {x1, y1, w, h}}]}

        Returns:
            A mapping of crop_id -> crop_path
        """
        source = None
        im = None
        rev_mapping = {v: k for k, v in self.mapping.items()}
        liste = []

        for img in crops:
            if img["doc_uid"] != self.uid:
                continue
            for crop in img["crops"]:
                crop_path = self.cropped_images_path / f"{crop['crop_id']}.jpg"
                print(crop_path)

                liste.append(Image(id=crop["crop_id"], src=crop_path.name, path=crop_path, document=self))
                if crop_path.exists():
                    continue

                crop_path.parent.mkdir(parents=True, exist_ok=True)

                if source != img["source"]:
                    source = img["source"]
                    im = PImage.open(self.images_path / rev_mapping.get(source, source)).convert("RGB")
                
                box = crop["relative"]
                if "x1" in box:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                else:
                    x1, y1, w, h = box["x"], box["y"], box["width"], box["height"]
                    x2, y2 = x1 + w, y1 + h
                x1, y1, x2, y2 = int(x1 * im.width), int(y1 * im.height), int(x2 * im.width), int(y2 * im.height)
                if x2-x1 == 0 or y2-y1 == 0:
                    # use placeholder image
                    im_cropped = PImage.new("RGB", (MAX_SIZE, MAX_SIZE))
                else:
                    im_cropped = im.crop((x1, y1, x2, y2))
                im_cropped.save(crop_path)

        return liste