import traceback
import zipfile

import os
import torch

from pathlib import Path
from typing import Optional

from ..shared.dataset import Dataset
from ..shared.tasks import LoggedTask
from ..shared.utils.fileutils import download_file, get_model
from .const import (
    MODEL_CONFIG,
    MODEL_CHECKPOINT,
    VEC_RESULTS_PATH,
    DEMO_NAME,
    MODEL_PATH,
)

from .lib.src import build_model_main
from .lib.src.inference import (
    set_config,
    preprocess_img,
    generate_prediction,
    postprocess_preds,
    save_pred_as_svg,
)


def load_model(model_checkpoint_path=MODEL_CHECKPOINT, model_config_path=MODEL_CONFIG):
    # TODO allow for multiple models
    config = set_config(model_config_path)
    model, _, postprocessors = build_model_main(config)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
    return model, postprocessors


class ComputeVectorization(LoggedTask):
    def __init__(self, dataset: Dataset, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.model = model
        self.imgs = []
        self.results_url = []

    def check_dataset(self):
        # TODO add more checks
        if not self.dataset.documents:
            return False
        return True

    def store(self, doc):
        self.create_zip(doc.uid)
        doc_results = {
            "doc_id": doc.uid,
            "result_url": doc.get_results_url(DEMO_NAME)
        }

        self.results_url.append(doc_results)
        self.notifier(
            "PROGRESS",
            output={
                "dataset_url": self.dataset.get_absolute_url(),
                "results_url": [doc_results],
            },
        )

    def run_task(self):
        if not self.check_dataset():
            self.print_and_log_warning(f"[task.vectorization] No documents to download")
            raise ValueError(f"[task.vectorization] No documents to download")

        self.task_update("STARTED")
        try:
            model, postprocessors = load_model(get_model(self.model, MODEL_PATH))
            model.eval()

            for doc in self.jlogger.iterate(
                self.dataset.documents, "Processing documents"
            ):
                self.log(f"Vectorization task triggered for {doc.uid}!")
                try:
                    doc.download()
                    if not doc.has_images():
                        self.log_error(f"No images were extracted from {doc.uid}")
                        return False

                    output_dir = VEC_RESULTS_PATH / doc.uid
                    os.makedirs(output_dir, exist_ok=True)

                    for image in doc.list_images():
                        path = image.path
                        orig_img, tr_img = preprocess_img(path)
                        preds = generate_prediction(
                            orig_img, tr_img, model, postprocessors
                        )
                        preds = postprocess_preds(preds, orig_img.size)
                        save_pred_as_svg(
                            path,
                            img_name=os.path.splitext(os.path.basename(path))[0],
                            img_size=orig_img.size,
                            pred_dict=preds,
                            pred_dir=output_dir,
                        )
                    self.store(doc)
                except Exception as e:
                    self.notifier(
                        "ERROR", error=traceback.format_exc(), completed=False
                    )
                    self.log_error(f"Error when vectorizing {doc.uid}", e)

        except Exception as e:
            self.log_error(f"Error when computing vectorization", e)
            raise e

        return True

    def create_zip(self, doc_id):
        """
        Creates a zip file containing the vectorization results and saves it to disk
        Returns the path to the created zip file
        """
        output_dir = VEC_RESULTS_PATH / doc_id
        zip_path = output_dir / f"{doc_id}.zip"

        try:
            self.log(
                f"[task.vectorization] Zipping directory {output_dir}", color="blue"
            )

            try:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(output_dir):
                        for file in files:
                            # Skip the zip file itself if it exists
                            if file == f"{doc_id}.zip":
                                continue
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(output_dir)
                            zipf.write(file_path, arcname)

                return True

            except Exception as e:
                self.log_error(
                    f"Failed to create zip file for directory {output_dir}",
                    e,
                )
                raise e

        except Exception as e:
            self.log_error(
                f"Failed to zip directory {output_dir}", e
            )
            raise e
