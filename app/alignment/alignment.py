from typing import TypedDict

import orjson

from .const import (
    ALIGN_RESULTS_PATH,
    DEMO_NAME,
    DEFAULT_METHOD,
    DEFAULT_RANSAC_FEATURE,
)
from .lib.align_ransac import align_ransac
from .lib.types import AlignedImage

from ..shared.dataset import Dataset
from ..shared.dataset.document import get_file_url
from ..shared.dataset.types import Image
from ..shared.tasks import LoggedTask
from ..shared.utils.logging import serializer

DEFAULT_TRANSFORM = "homographic"



class AlignParameters(TypedDict):
    method: str
    ransac_feature: str
    target_transform: str
    reference_image: str


class AlignmentResults(TypedDict):
    parameters: AlignParameters
    # Dataset index, images[i] describes index["images"][i]
    index: dict
    images: list[AlignedImage]


class ComputeAlignment(LoggedTask):
    """
    Align all images of a dataset onto a reference image
    """

    def __init__(
        self, dataset: Dataset, parameters: dict | None = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        parameters = parameters or {}
        self.dataset = dataset
        self.method: str = parameters.get("method", DEFAULT_METHOD)
        self.ransac_feature: str = parameters.get(
            "ransac_feature", DEFAULT_RANSAC_FEATURE
        )
        self.target_transform: str = parameters.get(
            "target_transform", DEFAULT_TRANSFORM
        )
        self.requested_reference: str | None = parameters.get("reference_image")
        self.results: AlignmentResults | dict = {}
        self.results_url = []

    @property
    def result_name(self) -> str:
        return f"{self.experiment_id}/{self.dataset.uid}-alignment"

    def _find_reference_index(self, images: list[Image]) -> int:
        """
        Index of the requested reference image (matched by id or file name),
        or of the first image
        """
        if self.requested_reference:
            for i, im in enumerate(images):
                if self.requested_reference in (im.id, im.path.name):
                    return i
            self.log(
                f"Reference image {self.requested_reference} not found, using the first image"
            )
        return 0

    def _compute_alignment(self) -> AlignmentResults:
        """
        Core alignment pipeline
        """
        images = self.dataset.prepare()
        if len(images) == 0:
            raise Exception(f"No images found for {self.dataset.uid}")

        ref_idx = self._find_reference_index(images)
        self.log(
            f"Aligning {len(images)} images onto {images[ref_idx].id} ({self.target_transform})"
        )

        if self.method == "ransac":
            self.log(f"Using RANSAC alignment with {self.ransac_feature} features")
            aligned = align_ransac(
                images=images,
                ref_idx=ref_idx,
                target_transform=self.target_transform,
                ransac_feature=self.ransac_feature,
                log=self.log,
            )
        else:
            raise Exception(f"Unknown alignment method {self.method}")

        return {
            "parameters": {
                "method": self.method,
                "ransac_feature": self.ransac_feature,
                "target_transform": self.target_transform,
                "reference_image": images[ref_idx].id,
            },
            "index": Dataset.serialize(images=images),
            "images": aligned,
        }

    def _store_results(self, results: AlignmentResults) -> None:
        """
        Make results accessible through the API
        """
        result_file = ALIGN_RESULTS_PATH / f"{self.result_name}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, "wb") as f:
            f.write(orjson.dumps(results, default=serializer))

        self.log(f"Stored alignment results inside {result_file}")
        self.results_url = [
            {
                "result_url": get_file_url(DEMO_NAME, self.result_name),
            }
        ]

    def run_task(self) -> bool:
        """
        Main entry point
        """
        if self.dataset is None or len(self.dataset.documents) == 0:
            self.task_update(
                "ERROR",
                "[API ERROR] No documents to align",
                exception=Exception("No images to align"),
            )
            return False

        self.task_update("STARTED")

        try:
            self.results = self._compute_alignment()
            self._store_results(self.results)
            self.log("Successfully computed alignment")
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                message=[f"Error during alignment task: {e}"] + self.error_list,
                exception=e,
            )
            return False
