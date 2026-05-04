"""
Utilities for smart bounding box handling
"""

from typing import List, Tuple, Dict, Optional, TypeVar, Type, Iterable, Iterator
from collections.abc import Sequence


def base36_to_int(s: str) -> int:
    """
    Convert a base 36 string to an int.
    Raise ValueError if the input won't fit into an int.
    """
    if len(s) > 13:
        raise ValueError("Base36 input too large")
    return int(s, 36)


try:
    import numpy as np

    def int_to_base36(i: int) -> str:
        return np.base_repr(i, base=36).lower()

except ImportError:

    def int_to_base36(i: int) -> str:
        """
        Convert an integer to a base36 string.
        """
        char_set = "0123456789abcdefghijklmnopqrstuvwxyz"
        if i < 0:
            raise ValueError("Negative base36 conversion input.")
        if i < 36:
            return char_set[i]
        b36 = ""
        while i != 0:
            i, n = divmod(i, 36)
            b36 = char_set[n] + b36
        return b36


# Types for static analysis
TSegment = TypeVar("TSegment", bound="Segment")
TBbox = Tuple[float, float, float, float]


class Segment:
    """
    A bounding box segment (x, y, w, h) in relative (0-1) format,
    or None for full size
    """

    __slots__ = ("left", "top", "width", "height", "precision", "_fullpage")

    def __init__(
        self,
        left: Optional[float],
        top: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        precision: int = 2,
    ):

        self._fullpage = left is None

        if self._fullpage:
            self.left = 0.0
            self.top = 0.0
            self.width = 1.0
            self.height = 1.0
            self.precision = 1

            return

        assert (
            left is not None
            and top is not None
            and width is not None
            and height is not None
        )

        assert 0.0 <= left <= 1.0
        self.left = left

        assert 0.0 <= top <= 1.0
        self.top = top

        assert 0.0 <= width <= 1.0
        self.width = width

        assert 0.0 <= width <= 1.0
        self.height = height

        assert precision > 0
        self.precision = precision

    @classmethod
    def from_bbox(cls: Type[TSegment], array: TBbox, precision: int = 2) -> TSegment:
        """
        Create a segment from a bounding box tuple
        """
        return cls(*array, precision=precision)

    @classmethod
    def unserialize(cls: Type[TSegment], string: str) -> TSegment:
        """
        Create a segment from a serialized string
        """
        precision = len(string) // 4

        if precision == 0:
            return cls(None)

        maxi = 36**precision - 1
        values: List[float] = []
        for k in range(4):
            val = base36_to_int(string[k * precision : (k + 1) * precision])
            val = val / maxi
            values.append(val)
        return cls.from_bbox(
            (values[0], values[1], values[2], values[3]), precision=precision
        )

    @classmethod
    def auto_cast(cls: Type[TSegment], bbox) -> TSegment:
        """
        Cast to segment from various input types (str, tuple, Segment, None)
        """
        if isinstance(bbox, cls):
            return bbox

        if not bbox or bbox == "None":
            return cls(None)

        if isinstance(bbox, str):
            bbox = cls.unserialize(bbox)
        elif (
            isinstance(bbox, Sequence)
            or (hasattr(bbox, "__iter__") and hasattr(bbox, "__len__"))
        ) and len(bbox) == 4:
            bbox = cls.from_bbox((bbox[0], bbox[1], bbox[2], bbox[3]))
        else:
            raise ValueError("Unknown segment format")

        return bbox

    def serialize(self, precision: Optional[int] = None) -> Optional[str]:
        """
        Serialize the segment to a string, using the given precision in base36 encoding
        """
        if precision is None:
            precision = self.precision

        if self._fullpage:
            return None

        maxi = 36**precision - 1

        def encode(v):
            v = int_to_base36(round(v * maxi))
            return "0" * (precision - len(v)) + v

        return "".join(
            [
                encode(self.left),
                encode(self.top),
                encode(self.width),
                encode(self.height),
            ]
        )

    def __repr__(self) -> str:
        if self._fullpage:
            return "[full]"
        return ",".join("%.3f" % (x) for x in self)

    def __str__(self) -> str:
        s = self.serialize()
        if s is None:
            return "[full]"
        return s

    def __getitem__(self, i: int) -> float:
        return getattr(self, (["left", "top", "width", "height"])[i])

    def __len__(self) -> int:
        return 4

    def __eq__(self, other: object) -> bool:
        if not other:
            return self._fullpage
        return self.serialize() == self.auto_cast(other).serialize()

    def __bool__(self) -> bool:
        return not self._fullpage

    def __iter__(self) -> Iterator[float]:
        yield self.left
        yield self.top
        yield self.width
        yield self.height
        return

    @property
    def is_full(self) -> bool:
        return self._fullpage


class IIIFPageSegmentationItem:
    """
    A segmentation item of a IIIF document page
    """

    def __init__(self, page_uid: str, bbox, label: str):
        self.page_uid = page_uid
        self.bbox = Segment.auto_cast(bbox)
        self.label = label

    def as_dict(self) -> Dict:
        return {
            "page_uid": self.page_uid,
            "bbox": self.bbox.serialize(),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __eq__(self, other):
        return (
            self.page_uid == other.page_uid
            and self.label == other.label
            and self.bbox == other.bbox
        )

    def __str__(self):
        return f"Segmentation {self.bbox.serialize()} ({self.label}) of {self.page_uid}"


class IIIFDocumentSegmentation:
    """
    A wrapper for segmentation data of a IIIF document
    """

    def __init__(self, manifest_url, segments):
        self.manifest_url = manifest_url
        segments = [
            (
                s
                if isinstance(s, IIIFPageSegmentationItem)
                else IIIFPageSegmentationItem.from_dict(s)
            )
            for s in segments
        ]
        self.segments = segments

    def as_dict(self):
        return {
            "manifest_url": self.manifest_url,
            "segments": [s.as_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
