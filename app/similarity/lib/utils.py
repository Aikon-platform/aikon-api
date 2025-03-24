import os
import sys
from itertools import combinations_with_replacement
from enum import Enum
from PIL.Image import Transpose


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
