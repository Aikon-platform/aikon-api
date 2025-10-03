import os
import sys
from itertools import combinations_with_replacement
from enum import Enum
from PIL.Image import Transpose
import numpy as np

from ...shared.dataset import Document
from ...shared.dataset.utils import Image

class AllTranspose(Enum):
    NONE = -1
    HFLIP = Transpose.FLIP_LEFT_RIGHT.value
    VFLIP = Transpose.FLIP_TOP_BOTTOM.value
    ROT90 = Transpose.ROTATE_90.value
    ROT180 = Transpose.ROTATE_180.value
    ROT270 = Transpose.ROTATE_270.value
    # TRANSPOSE = Transpose.TRANSPOSE
    # TRANSVERSE = Transpose.TRANSVERSE


def doc_pairs(doc_ids: list):
    # NOT USED
    if isinstance(doc_ids, list) and len(doc_ids) > 0:
        return list(combinations_with_replacement(doc_ids, 2))
    raise ValueError("Input must be a non-empty list of ids.")


def best_matches(segswap_pairs, q_img, doc_pair):
    # NOT USED
    """
    segswap_pairs = [[score, img_doc1.jpg, img_doc2.jpg]
                     [score, img_doc1.jpg, img_doc2.jpg]
                     ...]
    q_img = "path/to/doc_hash/img_name.jpg"
    doc_pair = (doc1_hash, doc2_hash)
    """
    query_hash = os.path.dirname(q_img).split("/")[-1]
    query_doc = 1 if query_hash == doc_pair[0] else 2
    sim_doc = 2 if query_doc == 1 else 1
    sim_hash = doc_pair[1] if query_hash == doc_pair[0] else doc_pair[0]

    # Get pairs concerning the given query image q_img
    # img_pairs = segswap_pairs[segswap_pairs[:, query_doc] == q_img]
    img_pairs = segswap_pairs[segswap_pairs[:, query_doc] == os.path.basename(q_img)]

    # return sorted([(pair[0], f"{sim_hash}/{pair[sim_doc]}") for pair in img_pairs], key=lambda x: x[0], reverse=True)
    return [(float(pair[0]), f"{sim_hash}/{pair[sim_doc]}") for pair in img_pairs]


def handle_transpositions(
    sim_matrix: np.ndarray, n_trans_rows: int, n_trans_cols: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle multiple transpositions per image.
    """
    if n_trans_cols is None:
        n_trans_cols = n_trans_rows

    n_rows, n_cols = sim_matrix.shape
    n_im_rows = n_rows // n_trans_rows
    n_im_cols = n_cols // n_trans_cols
    assert n_rows % n_trans_rows == 0, "Features must be divisible by transpositions"
    assert n_cols % n_trans_cols == 0, "Features must be divisible by transpositions"

    # Reshape to get all transposition combinations
    sim_trans = (
        sim_matrix.reshape(n_im_rows, n_trans_rows, n_im_cols, n_trans_cols)
        .transpose(0, 2, 1, 3)
        .reshape(n_im_rows, n_im_cols, n_trans_rows * n_trans_cols)
    )

    # Find best transposition pairs
    best_trans = sim_trans.argmax(axis=2, keepdims=True)
    sim_matrix = np.take_along_axis(sim_trans, best_trans, axis=2).squeeze(axis=2)
    tr_i, tr_j = np.divmod(best_trans.squeeze(axis=2), n_trans_cols)

    return sim_matrix, tr_i, tr_j


@dataclass
class DocInFeatures:
    document: Document
    range: range
    images: list[Image]

    def __eq__(self, value: "DocInFeatures") -> bool:
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

def group_by_documents(images: list[Image]) -> list[DocInFeatures]:
    """
    Identify groups of consecutive images from the same document
    """
    ranges = []
    p = 0
    for k, i in enumerate(images + [None]):
        if i is None or i.document != images[p].document:
            ranges.append(DocInFeatures(images[p].document, range(p, k), images[p:k]))
            p = k
    return ranges
