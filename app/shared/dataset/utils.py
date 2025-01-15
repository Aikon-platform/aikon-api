from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

from ..utils.logging import console

if TYPE_CHECKING:
    from .document import Document


@dataclass
class Image:
    id: str
    src: str
    path: Path
    metadata: dict[str, str] = None
    document: "Document" = None

    def to_dict(self, relpath: Path) -> dict:
        return {
            "id": self.id,
            "src": self.src,
            "path": str(self.path.relative_to(relpath)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict, document: "Document", relpath: Path) -> "Image":
        return cls(
            id=data["id"],
            src=data["src"],
            path=relpath / data["path"],
            metadata=data.get("metadata", None),
            document=document,
        )


def pdf_to_img(pdf_path, img_path, dpi=500, max_size=3000):
    """
    Convert the PDF file to JPEG images
    """
    import subprocess

    file_prefix = pdf_path.stem
    try:
        command = f"pdftoppm -jpeg -r {dpi} -scale-to {max_size} {pdf_path} {img_path}/{file_prefix} -sep _ "
        subprocess.run(command, shell=True, check=True)

    except Exception as e:
        console(f"Error converting {pdf_path} to images: {e}", "red")
