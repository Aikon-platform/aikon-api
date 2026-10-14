"""
Align images onto a reference image: keypoint extraction (SIFT, dense features of a
similarity model or SegSwap), matching to the reference keypoints and RANSAC transform estimation
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.transforms import functional as TF

from ...shared.dataset.types import Image
from ...shared.utils import clear_cuda, get_device
from ...similarity.lib import segswap
from ...similarity.lib.models import (
    IMAGENET_NORMALIZE,
    MODELS,
    get_model_path,
    get_transforms_for_model,
    load_model,
)

from .const import (
    CNN_SIZE_MULTIPLE,
    DEEP_MIN_SIZE,
    DEEP_RANSAC_THRESHOLD,
    DEEP_SCALES,
    DEEP_SIZE,
    DENSE_ARCHS,
    MAX_SIZE,
    RANSAC_THRESHOLD,
    RATIO_TEST,
    SEGSWAP_FEATURE,
    SEGSWAP_MASK_THRESHOLD,
    SEGSWAP_MODEL,
    SEGSWAP_SIZE,
    SIFT_FEATURE,
)
from .ransac import ransac
from .types import AlignedImage, Keypoint


class Features(NamedTuple):
    # keypoint coordinates (N, 2) in original image pixels
    pts: np.ndarray
    # descriptors (N, D)
    desc: np.ndarray | torch.Tensor | None
    # size of a feature pixel (SIFT) / cell at scale 1 (deep), in original image pixels
    unit: float

class Extractor:
    """
    Abstract interface for extracting keypoints from an image
    """
    requires_deduplication = False
    ransac_threshold = 1

    def extract(self, path: Path) -> Features:
        raise NotImplementedError

    def match(self, query: Features, ref: Features) -> np.ndarray:
        raise NotImplementedError

class SiftExtractor(Extractor):
    """
    SIFT keypoints on a downscaled grayscale version of the image, matched with Lowe's ratio test
    """

    # SIFT gives several keypoints at the same position (one per orientation)
    requires_deduplication = True
    ransac_threshold = RANSAC_THRESHOLD

    def __init__(self, max_size: int = MAX_SIZE):
        self.max_size = max_size
        self.sift = cv2.SIFT_create()

    def extract(self, path: Path) -> Features:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {path}")

        scale = min(1.0, self.max_size / max(img.shape))
        if scale < 1.0:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        kps, desc = self.sift.detectAndCompute(img, None)

        pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2) / scale
        return Features(pts, desc, 1 / scale)

    def match(
        self,
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


class DeepExtractor(Extractor):
    """
    Dense features of a similarity model: every cell of its feature map is a keypoint
    (at the centre of the cell), extracted at several scales; matched by mutual nearest
    neighbour on cosine similarity
    """

    requires_deduplication = False
    ransac_threshold = DEEP_RANSAC_THRESHOLD

    def __init__(self, feat_net: str, device: str):
        self.device = device
        self.model, self.feat_fn = load_model(None, feat_net, device, spatial=True)
        # ViT inputs must be a whole number of patches
        self.multiple = MODELS[feat_net].get("patch_size") or CNN_SIZE_MULTIPLE
        # We use our own transforms to override the resize part
        model_transforms = getattr(get_transforms_for_model(feat_net), "transforms", [])
        self.normalize = next(
            (t for t in model_transforms if isinstance(t, transforms.Normalize)),
            IMAGENET_NORMALIZE,
        )

    @torch.no_grad()
    def extract(self, path: Path) -> Features:
        img = PILImage.open(path).convert("RGB")
        w, h = img.size
        diag = (w**2 + h**2) ** 0.5

        pts, desc = [], []
        unit = 1.0
        for scale in DEEP_SCALES:
            k = DEEP_SIZE * scale / diag
            if min(w, h) * k < DEEP_MIN_SIZE:
                continue
            size = [max(self.multiple, round(v * k / self.multiple) * self.multiple) for v in (w, h)]
            x = self.normalize(TF.to_tensor(img.resize(size, PILImage.Resampling.BICUBIC)))
            fmap = self.feat_fn(self.model(x[None].to(self.device)))[0]  # (C, fh, fw)
            fh, fw = fmap.shape[1:]

            ys, xs = torch.meshgrid(
                torch.arange(fh) + 0.5, torch.arange(fw) + 0.5, indexing="ij"
            )
            pts.append(torch.stack([xs * w / fw, ys * h / fh], dim=-1).reshape(-1, 2))
            desc.append(F.normalize(fmap.flatten(1).T.float(), dim=1))
            unit = w / fw * scale

        if not pts:
            return Features(np.zeros((0, 2), dtype=np.float32), None, unit)
        return Features(torch.cat(pts).numpy(), torch.cat(desc), unit)

    def match(
        self, desc: torch.Tensor | None, ref_desc: torch.Tensor | None
    ) -> dict[int, int]:
        """
        Mutual nearest neighbours between image and reference descriptors

        Returns:
            {reference keypoint index: image keypoint index}
        """
        if desc is None or ref_desc is None:
            return {}

        sim = desc @ ref_desc.T
        fwd, bwd = sim.argmax(dim=1), sim.argmax(dim=0)
        idx = (bwd[fwd] == torch.arange(len(desc), device=desc.device)).nonzero()[:, 0]
        return dict(zip(fwd[idx].tolist(), idx.tolist()))


class SegSwapExtractor(Extractor):
    """
    SegSwap co-segmentation transformer: every cell of its ResNet feature map is a keypoint
    (at the centre of the cell); matching runs the transformer on the image pair, which
    predicts for each image cell its corresponding reference cell and whether it lies in
    the region shared by both images
    """

    requires_deduplication = False
    ransac_threshold = DEEP_RANSAC_THRESHOLD

    def __init__(self, device: str):
        self.device = device
        param = torch.load(
            get_model_path(SEGSWAP_MODEL), map_location=device, weights_only=False
        )
        self.backbone = segswap.load_backbone(param).to(device)
        self.encoder = segswap.load_encoder(param).to(device)

    @torch.no_grad()
    def extract(self, path: Path) -> Features:
        img = PILImage.open(path).convert("RGB")
        w, h = img.size
        x = IMAGENET_NORMALIZE(TF.to_tensor(segswap.resize(img, SEGSWAP_SIZE)))
        fmap = F.normalize(self.backbone(x[None].to(self.device)))  # (1, C, fh, fw)
        fh, fw = fmap.shape[2:]

        ys, xs = torch.meshgrid(
            torch.arange(fh) + 0.5, torch.arange(fw) + 0.5, indexing="ij"
        )
        pts = torch.stack([xs * w / fw, ys * h / fh], dim=-1).reshape(-1, 2)
        return Features(pts.numpy(), fmap, w / fw)

    @torch.no_grad()
    def match(self, desc: torch.Tensor, ref_desc: torch.Tensor) -> dict[int, int]:
        """
        Predicted correspondence of each image cell in the reference, kept where the
        cycle-consistent co-segmentation mask is confident; when several image cells
        point to the same reference cell, the one with the highest mask value is kept

        Returns:
            {reference keypoint index: image keypoint index}
        """
        out, ref_out = self.encoder(desc, ref_desc)
        out, ref_out = out.cpu(), ref_out.cpu()
        mask, _ = segswap.consistent_mask(
            out[:, 2:].numpy(), ref_out[:, 2:].numpy(), out, ref_out
        )
        mask = torch.from_numpy(mask).flatten()

        rh, rw = ref_desc.shape[2:]
        flow = out[0, :2].flatten(1)  # (2, N): x, y in [0, 1] of the reference
        ref_ids = (flow[1] * (rh - 1)).round().long() * rw + (
            flow[0] * (rw - 1)
        ).round().long()

        keep = (mask > SEGSWAP_MASK_THRESHOLD).nonzero()[:, 0]
        # ascending mask order: the best match of each reference cell is written last
        keep = keep[mask[keep].argsort()]
        return dict(zip(ref_ids[keep].tolist(), keep.tolist()))


Extractor = SiftExtractor | DeepExtractor | SegSwapExtractor


def make_extractor(ransac_feature: str, device: str) -> Extractor:
    if ransac_feature == SIFT_FEATURE:
        return SiftExtractor()
    if ransac_feature == SEGSWAP_FEATURE:
        return SegSwapExtractor(device)
    if MODELS.get(ransac_feature, {}).get("arch") in DENSE_ARCHS:
        return DeepExtractor(ransac_feature, device)
    raise ValueError(f"Feature '{ransac_feature}' cannot be used to compute keypoints")


def estimate_transform(
    src: np.ndarray,
    dst: np.ndarray,
    target_transform: str,
    threshold: float,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Estimate the transform mapping src points onto dst points with RANSAC

    Returns:
        3x3 matrix (or None if it could not be estimated), boolean inlier mask
    """
    to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    result = ransac(
        to_tensor(src), to_tensor(dst), target_transform, threshold, generator=generator
    )
    if result is None:
        return None, np.zeros(len(src), dtype=bool)
    return result.transform.double().cpu().numpy(), result.inliers.cpu().numpy()


def to_keypoint(pt: np.ndarray, inlier: bool) -> Keypoint:
    return {
        "x": round(float(pt[0]), 2),
        "y": round(float(pt[1]), 2),
        "inlier": bool(inlier),
    }


def find_duplicates(pts: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Find duplicate 2d points within eps distance

    Returns:
        for each point, the index of its representative among duplicates
    """
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    reverse_order = np.empty(len(pts), dtype=int)
    reverse_order[order] = range(len(pts))

    dist = np.linalg.norm(pts[order][1:] - pts[order][:-1], axis=1)

    steps = [True] + list(dist > eps)
    unique = np.where(steps)[0]
    rmap = np.cumsum(steps) - 1

    return order[unique[rmap][reverse_order]]


def align_ransac(
    images: list[Image],
    ref_idx: int,
    target_transform: str,
    ransac_feature: str = SIFT_FEATURE,
    log: Callable[[str], None] = print,
) -> list[AlignedImage]:
    """
    Align every image onto images[ref_idx]

    The reference keypoints are those matched in at least one other image; the keypoints
    of every other image are listed in the same order (None where unmatched)
    """
    device = get_device()
    extractor = make_extractor(ransac_feature, device)
    # fixed seed: reproducible results
    generator = torch.Generator(device=device).manual_seed(0)

    ref_img = images[ref_idx]
    ref = extractor.extract(ref_img.path)
    ref_canonical = (
        find_duplicates(ref.pts) if extractor.requires_deduplication else np.arange(len(ref.pts))
    )
    threshold = extractor.ransac_threshold * ref.unit

    log(f"Reference {ref_img.id}: {len(ref.pts)} keypoints")

    # image index -> (matrix, {ref kp index: (image point, inlier)})
    aligned: dict[
        int, tuple[np.ndarray | None, dict[int, tuple[np.ndarray, bool]]]
    ] = {}

    for i, img in enumerate(images):
        if i == ref_idx:
            continue

        feats = extractor.extract(img.path)
        matches = extractor.match(feats.desc, ref.desc)
        # There may be duplicate matched keypoints, RANSAC will deal with/filter it

        ref_ids = sorted(matches)
        src = feats.pts[[matches[r] for r in ref_ids]].reshape(-1, 2)
        matrix, inliers = estimate_transform(
            src,
            ref.pts[ref_ids].reshape(-1, 2),
            target_transform,
            threshold,
            device,
            generator,
        )
        # TODO deduplicate image keypoints too, and choose the best match as being part of inliers
        aligned[i] = (
            matrix,
            {
                ref_canonical[ref_id]: (p, inl)
                for ref_id, p, inl in zip(ref_ids, src, inliers)
            },
        )
        log(
            f"{img.id}: {len(matches)} matches, {int(inliers.sum())} inliers"
            + ("" if matrix is not None else " (no transform found)")
        )

    del extractor
    clear_cuda()

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
                        to_keypoint(ref.pts[r], r in ref_inliers) for r in used
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
