import gc
import json
import os
import torch
from pathlib import Path
from typing import Optional

from .const import DEFAULT_MODEL, MODEL_PATH
from .lib.extract import YOLOExtractor, FasterRCNNExtractor, LineExtractor
from ..shared.tasks import LoggedTask
from ..shared.dataset import Document, Dataset, Image as DImage
from ..shared.utils.fileutils import get_model

EXTRACTOR_POSTPROCESS_KWARGS = {
    "watermarks": {
        "squarify": True,
        "margin": 0.05,
    },
}


class ExtractRegions(LoggedTask):
    """
    Task to extract regions from a dataset

    Args:
        dataset (Dataset): The dataset to process
        model (str, optional): The model file name stem to use for extraction (default: DEFAULT_MODEL)
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Optional[str] = None,
        postprocess: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self._model = model
        self._extraction_model: Optional[str] = None

        self.result_dir = Path()
        self.results_url = []
        self.annotations = {}
        self.extractor = None
        self.extractor_kwargs = EXTRACTOR_POSTPROCESS_KWARGS.get(postprocess, {})

    def initialize(self):
        """
        Initialize the extractor, based on the model's name
        """
        if "rcnn" in self.model:
            self.extractor = FasterRCNNExtractor(self.weights, **self.extractor_kwargs)
        elif "line" in self.model:
            self.extractor = LineExtractor(self.weights, **self.extractor_kwargs)
        else:
            self.extractor = YOLOExtractor(self.weights, **self.extractor_kwargs)

    def terminate(self):
        """
        Clear memory
        """
        # self.annotations = {}
        del self.extractor
        self.extractor = None
        torch.cuda.empty_cache()
        gc.collect()

    @property
    def model(self) -> str:
        return DEFAULT_MODEL if self._model is None else self._model

    @property
    def weights(self) -> Path:
        return get_model(self.model, MODEL_PATH)

    @property
    def extraction_model(self) -> str:
        if self._extraction_model is None:
            self._extraction_model = self.model.split(".")[0]
        return self._extraction_model

    def check_doc(self) -> bool:
        # TODO improve check regarding dataset content
        if not self.dataset.documents:
            return False
        return True

    def process_img(self, img: DImage, extraction_ref: str, doc_uid: str) -> bool:
        """
        Process a single image, appends the annotations to self.annotations[extraction_ref]
        """
        self.print_and_log(f"====> Processing {img.path.name} 🔍")
        anno = self.extractor.extract_one(img)
        anno["doc_uid"] = doc_uid
        self.annotations[extraction_ref].append(anno)
        return True

    def process_doc_imgs(self, doc: Document, extraction_ref: str) -> bool:
        """
        Process all images in a document, store the annotations in self.annotations[extraction_ref] (clears it first)
        """
        self.annotations[extraction_ref] = []
        for img in self.jlogger.iterate(doc.list_images(), "Analyzing images"):
            self.process_img(img, extraction_ref, doc.uid)
        return True

    def store(self, doc, extraction_ref):
        with open(self.result_dir / f"{extraction_ref}.json", "w") as f:
            json.dump(self.annotations[f"{doc.uid}@@{extraction_ref}"], f, indent=2)

        doc_results = {
            "doc_id": doc.uid,
            "result_url": doc.get_annotations_url(extraction_ref)
        }

        self.results_url.append(doc_results)
        self.notifier(
            "PROGRESS",
            output={
                "dataset_url": self.dataset.get_absolute_url(),
                "results_url": [doc_results],
            },
        )

    def process_doc(self, doc: Document) -> bool:
        """
        Process a whole document, download it, process all images, save annotations
        """
        self.log(f"Downloading {doc.uid}...")

        doc.download()
        if not doc.has_images():
            self.log_error(f"No images were extracted from {doc.uid}")
            return False

        self.result_dir = doc.annotations_path
        os.makedirs(self.result_dir, exist_ok=True)

        # This way, same dataset can be extracted twice with same extraction model
        # is it what we want?
        extraction_ref = f"{self.extraction_model}+{self.experiment_id}"
        annotation_file = self.result_dir / f"{extraction_ref}.json"
        with open(annotation_file, "w"):
            pass

        extraction_id = f"{doc.uid}@@{extraction_ref}"

        self.print_and_log(f"DETECTING VISUAL ELEMENTS FOR {doc.uid} 🕵️")
        if self.process_doc_imgs(doc, extraction_id):
            self.store(doc, extraction_ref)

    def run_task(self) -> bool:
        """
        Run the extraction task
        """
        if not self.check_doc():
            self.print_and_log_warning("[task.extract_regions] No dataset to annotate")
            self.task_update(
                "ERROR",
                message=f"[API ERROR] Failed to download dataset for {self.dataset}",
                exception=Exception("No images where to extract regions"),
            )
            return False

        self.task_update("STARTED")
        self.log(f"Extraction task triggered with {self.model}!")

        try:
            self.initialize()
            for doc in self.jlogger.iterate(
                self.dataset.documents, "Processing documents"
            ):
                self.process_doc(doc)

            self.log(f"Task completed with status: SUCCESS")
            return True
        except Exception as e:
            self.task_update(
                "ERROR",
                message=[f"Error while extracting regions: {e}"] + self.error_list,
                exception=e,
            )
            return False
        finally:
            self.terminate()
