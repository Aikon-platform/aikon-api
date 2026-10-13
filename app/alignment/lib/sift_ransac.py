"""
Align images onto a reference image with SIFT keypoint matching and RANSAC 
transform estimation
"""

from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from ...shared.dataset.types import Image

from .const import MAX_SIZE, RATIO_TEST, RANSAC_THRESHOLD
from .types import AlignedImage, Keypoint

# target_transform -> (minimal number of matches, RANSAC estimator returning (matrix, inlier mask))
ESTIMATORS = {
    "homographic": (
        4,
        lambda src, dst, t: cv2.findHomography(src, dst, cv2.RANSAC, t),
    ),
    "affine": (
        3,
        lambda src, dst, t: cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=t
        ),
    ),
    "scale+rotate": (
        2,
        lambda src, dst, t: cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=t
        ),
    ),
}


def extract_sift_features(
    path: Path, max_size: int = MAX_SIZE
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detect SIFT keypoints on a downscaled grayscale version of the image

    Returns:
        tuple of 
        - keypoint coordinates (N, 2) in original image pixels
        - descriptors (N, 128)
    """
    sift = cv2.SIFT_create()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")

    scale = min(1.0, max_size / max(img.shape))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    kps, desc = sift.detectAndCompute(img, None)

    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2) / scale
    return pts, desc


def match_to_reference(
    desc: np.ndarray | None,
    ref_desc: np.ndarray | None,
    ratio: float = RATIO_TEST,
) -> dict[int, int]:
    """
    Match image descriptors to reference descriptors with Lowe's ratio test,
    keeping at most one image keypoint per reference keypoint (the closest)

    Returns:
        {reference keypoint index: image keypoint index}
    """
    if desc is None or ref_desc is None or len(desc) < 2 or len(ref_desc) < 2:
        return {}

    best: dict[int, cv2.DMatch] = {}
    for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, ref_desc, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= ratio * n.distance:
            continue
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    return {ref_idx: m.queryIdx for ref_idx, m in best.items()}


def estimate_transform(
    src: np.ndarray, dst: np.ndarray, target_transform: str, threshold: float
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Estimate the transform mapping src points onto dst points with RANSAC

    Returns:
        3x3 matrix (or None if it could not be estimated), boolean inlier mask
    """
    min_matches, estimator = ESTIMATORS[target_transform]
    no_inliers = np.zeros(len(src), dtype=bool)
    if len(src) < min_matches:
        return None, no_inliers

    matrix, mask = estimator(src, dst, threshold)
    if matrix is None:
        return None, no_inliers
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return matrix, mask.ravel().astype(bool)


def to_keypoint(pt: np.ndarray, inlier: bool) -> Keypoint:
    return {
        "x": round(float(pt[0]), 2),
        "y": round(float(pt[1]), 2),
        "inlier": bool(inlier),
    }


def align_sift_ransac(
    images: list[Image],
    ref_idx: int,
    target_transform: str,
    log: Callable[[str], None] = print,
) -> list[AlignedImage]:
    """
    Align every image onto paths[ref_idx]

    The reference keypoints are those matched in at least one other image; the keypoints
    of every other image are listed in the same order (None where unmatched)
    """
    if target_transform not in ESTIMATORS:
        raise NotImplementedError(
            f"Transform '{target_transform}' is not supported by sift+ransac (available: {list(ESTIMATORS)})"
        )

    ref_img = images[ref_idx]

    # Reference descriptors
    ref_pts, ref_desc = extract_sift_features(ref_img.path)

    # TODO scale threshold?
    threshold = RANSAC_THRESHOLD

    log(f"Reference {ref_img.id}: {len(ref_pts)} keypoints")

    # image index -> (matrix, {ref kp index: (image point, inlier)})
    aligned: dict[
        int, tuple[np.ndarray | None, dict[int, tuple[np.ndarray, bool]]]
    ] = {}

    for i, img in enumerate(images):
        path = img.path
        if i == ref_idx:
            continue

        pts, desc = extract_sift_features(path)
        matches = match_to_reference(desc, ref_desc)

        ref_ids = sorted(matches)
        src = pts[[matches[r] for r in ref_ids]].reshape(-1, 2)
        matrix, inliers = estimate_transform(
            src, ref_pts[ref_ids].reshape(-1, 2), target_transform, threshold
        )
        aligned[i] = (matrix, {r: (p, inl) for r, p, inl in zip(ref_ids, src, inliers)})
        log(
            f"{img.id}: {len(matches)} matches, {int(inliers.sum())} inliers"
            + ("" if matrix is not None else " (no transform found)")
        )

    used = sorted({r for _, kps in aligned.values() for r in kps})
    ref_inliers = {
        r for _, kps in aligned.values() for r, (_, inl) in kps.items() if inl
    }

    results: list[AlignedImage] = []
    for i, img in enumerate(images):
        if i == ref_idx:
            results.append(
                {
                    "image_id": img.id,
                    "transform_matrix": np.eye(3).tolist(),
                    "keypoints": [
                        to_keypoint(ref_pts[r], r in ref_inliers) for r in used
                    ],
                    "is_reference": True,
                }
            )
            continue
        matrix, kps = aligned[i]
        results.append(
            {
                "image_id": img.id,
                "transform_matrix": matrix.tolist() if matrix is not None else None,
                "keypoints": [to_keypoint(*kps[r]) if r in kps else None for r in used],
            }
        )
    return results
