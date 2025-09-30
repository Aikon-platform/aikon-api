"""
The Dataset class, which represents a dataset of documents
"""

from pathlib import Path
from typing import List, Union, Optional, Dict, TypedDict, __all__
from typing_extensions import NotRequired

import orjson

from ... import config
from ..const import DATASETS_PATH
from ..utils import hash_str

from .document import Document, DocDict
from .utils import Image


class DatasetDict(TypedDict):
    uid: str
    documents: List[DocDict]
    url: NotRequired[str]


class Dataset:
    """
    This class represents a dataset of documents

    :param uid: The unique identifier of the dataset (if not provided, generate one)
    :param path: The path to the dataset on disk (default: DATASETS_PATH/uid)
    :param documents: The list of documents in the dataset (if not provided, load from disk)
    :param load: Whether to load the dataset from disk (default: False)

    The dataset is saved to disk in the following structure:

    .. code-block:: none

        - path/
            - documents.json
            - results/
                - ...
    """

    def __init__(
        self,
        uid: str = None,
        path: Union[Path, str] = None,
        documents: Optional[List[dict]] = None,
        load: bool = False,
        crops: Optional[List[dict]] = None,
    ):
        """
        Create a new dataset
        """
        crops_uid = None
        if crops:
            crops_uid = hash_str(orjson.dumps(crops))

        self.crops = crops

        if uid is None:
            assert (
                documents is not None
            ), "Documents must be provided when no UID is provided"
            base_uid = hash_str("".join([doc["src"] for doc in documents]))
        else:
            assert (
                documents is None
            ), "Documents are not supported when providing a custom UID"
            assert crops is None, "Crops are not supported when providing a custom UID"
            if "@" in uid:
                base_uid, crops_uid = uid.split("@", 1)
            else:
                base_uid = uid

        self.crops_uid = crops_uid
        self.base_uid = base_uid

        if path is None:
            path = DATASETS_PATH / self.base_uid
        self.path = Path(path)

        # TODO Check consistency when a dataset is used twice

        if load:
            self.load()
        else:
            self.documents: Optional[List[Document]] = (
                [Document.from_dict(doc) for doc in documents] if documents else None
            )

    @property
    def uid(self) -> str:
        if self.crops_uid:
            return f"{self.base_uid}@{self.crops_uid}"
        return self.base_uid

    @property
    def doc_uid(self) -> list:
        if not self.documents:
            return []
        return [d.uid for d in self.documents]

    def to_dict(self, with_url: bool = False) -> DatasetDict:
        """
        Convert the dataset to a dictionary
        """
        ret = {
            "uid": self.uid,
            "documents": [doc.to_dict(with_url) for doc in self.documents],
        }
        if with_url:
            ret["url"] = self.get_absolute_url()
        return ret

    @property
    def results_path(self) -> Path:
        """
        The path to the results of the dataset
        """
        return self.path / "results"

    @property
    def infos_path(self) -> Path:
        """
        The path to the info file of the dataset
        """
        return self.path / "info.json"

    @property
    def crops_path(self) -> Optional[Path]:
        """
        The path to the crops file of the dataset
        """
        if not self.crops_uid:
            return None
        return self.path / f"crops_{self.crops_uid}.json"

    def get_absolute_url(self) -> str:
        """
        Get the absolute URL of the dataset
        """
        return f"{config.BASE_URL}/datasets/dataset/{self.uid}"

    def save(self) -> None:
        """
        Save the dataset to disk
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.infos_path, "wb") as f:
            # json.dump([doc.uid for doc in self.documents], f)
            f.write(orjson.dumps(self.to_dict(), f))
        if self.crops:
            with open(self.crops_path, "wb") as f:
                f.write(orjson.dumps(self.crops))

    def download(self) -> None:
        """
        Download the dataset from the URL
        """
        for document in self.documents:
            document.download()

        # TODO if failed cancel task

    def load(self) -> None:
        """
        Load the dataset from disk
        """
        with open(self.infos_path, "rb") as f:
            documents = orjson.loads(f.read())["documents"]

        self.documents = [Document.from_dict(doc) for doc in documents]

        if self.crops_uid:
            with open(self.crops_path, "rb") as f:
                self.crops = orjson.loads(f.read())

    def list_images(self) -> List[Image]:
        """
        List all images in the dataset, iterating over all documents

        :return: A list of image paths
        """
        images = []
        for document in self.documents:
            images.extend(document.list_images(self.crops))
        return images

    def prepare(self) -> List[Image]:
        """
        Prepare the dataset for processing

        Returns:
            A list of images or crops
        """
        im_list = []
        crop_list = []
        for document in self.documents:
            document.download()
            im_list.extend(document.list_images())

            if self.crops:
                crop_list.extend(document.prepare_crops(self.crops))

        if self.crops:
            return crop_list
        return im_list
