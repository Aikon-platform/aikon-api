from .const import WATERMARKS_SOURCES_FOLDER
from pathlib import Path
from typing import Dict, List
import json
import torch


class WatermarkSource:
    def __init__(self, uid: str):
        self.uid = uid
        self.data_folder = WATERMARKS_SOURCES_FOLDER / uid

    @property
    def metadata_file(self) -> Path:
        return self.data_folder / "metadata.json"

    @property
    def metadata(self) -> Dict:
        if not hasattr(self, "_metadata"):
            with open(self.metadata_file, "r") as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def features_file(self) -> Path:
        return self.data_folder / "features.pth"

    @property
    def features(self):
        if not hasattr(self, "_features"):
            self._features = torch.load(
                self.features_file, map_location=torch.device("cpu"), weights_only=False
            )["features"]
        return self._features

    @staticmethod
    def list_available() -> list[dict[str, dict]]:
        return [
            {"uid": s.name, "metadata": WatermarkSource(s.name).metadata}
            for s in WATERMARKS_SOURCES_FOLDER.iterdir()
            if s.is_dir() and not (s / "deprecated").exists()
        ]
