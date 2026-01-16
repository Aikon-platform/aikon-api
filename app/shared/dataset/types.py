from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, TYPE_CHECKING
from typing_extensions import NotRequired

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
    src: str | None
    path: Path
    metadata: dict[str, str] | None = None
    document: "Document" = None

    def remove_src(self):
        self.src = None
        return self

    def to_dict(self, relpath: Path = None) -> dict:
        if relpath is None:
            relpath = self.document.path
        return {
            "id": self.id,
            "src": self.src,
            "path": str(self.path.relative_to(relpath)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls, data: dict, document: "Document", relpath: Path = None
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


@dataclass
class DocInRange:
    """
    A range of images from the same document, used to group images by document
    Mostly used in similarity to reference feature indices
    """

    document: "Document"
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
