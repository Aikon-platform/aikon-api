import typing
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, TypedDict
from typing_extensions import NotRequired

from ..utils.logging import console

if TYPE_CHECKING:
    from .document import Document


class ImageDict(TypedDict):
    uid: str
    src: str
    path: str
    metadata: dict[str, str]
    doc_uid: NotRequired[str]


@dataclass
class Image:
    id: str
    src: str
    path: Path
    metadata: dict[str, str] | None = None
    document: "Document" = None

    def to_dict(self, relpath: Path = None) -> dict:
        if relpath is None:
            relpath = self.document.path
        return {
            "id": self.id,
            "src": self.src,
            "path": str(self.path.relative_to(relpath)),
            "metadata": self.metadata,
        }

    # @property
    # def path(self) -> Path:
    #     """Returns the absolute path to the image file"""
    #     return Path(self.path)

    @classmethod
    def from_dict(
        cls, data: Dict, document: "Document", relpath: Path = None
    ) -> "Image":
        if relpath is None:
            relpath = document.path
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

@dataclass
class DocInRange:
    """
    A range of images from the same document, used to group images by document
    Mostly used in similarity to reference feature indices
    """
    document: Document
    range: range
    images: list[Image]

    def __eq__(self, value: "DocInRange") -> bool:
        return self.document.uid == value.document.uid

    def __hash__(self):
        return hash(self.document.uid)

    def slice(self, scale_by: int) -> slice:
        return slice(self.range.start * scale_by, self.range.stop * scale_by)

    def __str__(self):
        return (
            f"DocInFeatures({self.document.uid}, {self.range.start}-{self.range.stop})"
        )

    def __repr__(self):
        return str(self)

def group_by_documents(images: list[Image]) -> list[DocInRange]:
    """
    Identify groups of consecutive images from the same document
    """
    ranges = []
    p = 0
    for k, i in enumerate(images + [None]):
        if i is None or i.document != images[p].document:
            ranges.append(DocInRange(images[p].document, range(p, k), images[p:k]))
            p = k
    return ranges
