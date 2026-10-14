"""
Batched RANSAC estimation of 2D transforms between matched keypoints, in pure torch

Conventions:
- points are (N, 2) floating tensors of pixel coordinates, src[i] matched with dst[i]
- a transform is a 3x3 matrix H mapping src onto dst in homogeneous coordinates
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

# UTILS


def to_homogeneous(pts: Tensor) -> Tensor:
    """
    Convert points to homogeneous coordinates

    (..., 2) -> (..., 3), appending 1
    """
    return torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)


def _from_affine_rows(rows: Tensor) -> Tensor:
    """
    Convert affine transform rows to full (3x3) matrix

    (B, 2, 3) -> (B, 3, 3), appending [0, 0, 1]
    """
    last = torch.zeros_like(rows[:, :1])
    last[..., 2] = 1
    return torch.cat([rows, last], dim=1)


def _normalizing_transform(pts: Tensor) -> Tensor:
    """
    Center and scale so that mean distance is sqrt(2)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    mean = pts.mean(dim=1)
    dist = (pts - mean[:, None]).norm(dim=-1).mean(dim=1)
    scale = 2**0.5 / dist.clamp_min(torch.finfo(pts.dtype).eps)
    rows = torch.zeros(len(pts), 2, 3, dtype=pts.dtype, device=pts.device)
    rows[:, 0, 0] = rows[:, 1, 1] = scale
    rows[..., 2] = -scale[:, None] * mean
    return _from_affine_rows(rows)


def transfer_error(src: Tensor, dst: Tensor, h: Tensor) -> Tensor:
    """
    Distance between dst and the projection of src by each transform

    Args:
        src, dst: (N, 2)
        h: (B, 3, 3)

    Returns:
        (B, N) errors, inf/nan for points sent to infinity
    """
    proj = to_homogeneous(src) @ h.transpose(1, 2)  # (B, N, 3)
    return (dst - proj[..., :2] / proj[..., 2:]).norm(dim=-1)


def _count_inliers(
    src: Tensor, dst: Tensor, h: Tensor, threshold: float, min_det: float
) -> Tensor:
    """
    Number of inliers of each transform, 0 for invalid ones (non-finite or det(H) <= min_det)

    Batched for h: src and dst (N, 2), h (B, 3, 3), output (B,)
    """
    valid_h = torch.isfinite(h).all(dim=(1, 2)) & (torch.linalg.det(h) > min_det)
    return (transfer_error(src, dst, h) < threshold).sum(dim=1) * valid_h


def _centered(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    """
    Subtract the per-sample centroids, (B, n, 2) -> (B, n, 2)
    """
    return src - src.mean(dim=1, keepdim=True), dst - dst.mean(dim=1, keepdim=True)


def _with_translation(linear: Tensor, src: Tensor, dst: Tensor) -> Tensor:
    """
    Complete a linear part fitted on centered points with the least-squares translation
    (mapping the src centroid onto the dst centroid)

    Batched: linear (B, 2, 2), src and dst (B, n, 2), output (B, 3, 3)
    """
    shift = dst.mean(dim=1) - (linear @ src.mean(dim=1)[..., None])[..., 0]
    return _from_affine_rows(torch.cat([linear, shift[..., None]], dim=2))


# ESTIMATORS
# linear parts fitted on centered points are inf/nan for degenerate samples (rejected by ransac)


def estimate_translation(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares translation (1 point minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    eye = torch.eye(2, dtype=src.dtype, device=src.device).expand(len(src), 2, 2)
    return _with_translation(eye, src, dst)


def estimate_scale(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares uniform scaling + translation (2 points minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    src_c, dst_c = _centered(src, dst)
    scale = (src_c * dst_c).sum(dim=(1, 2)) / (src_c**2).sum(dim=(1, 2))  # (B,)
    eye = torch.eye(2, dtype=src.dtype, device=src.device)
    return _with_translation(scale[:, None, None] * eye, src, dst)


def estimate_scalexy(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares independent x / y scaling + translation (2 points minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    src_c, dst_c = _centered(src, dst)
    # per axis 1D linear regression
    scale = (src_c * dst_c).sum(dim=1) / (src_c**2).sum(dim=1)  # (B, 2)
    return _with_translation(torch.diag_embed(scale), src, dst)


def estimate_scale_rotate(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares similarity: uniform scaling + rotation + translation (2 points minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    src_c, dst_c = _centered(src, dst)
    (x, y), (x_, y_) = src_c.unbind(-1), dst_c.unbind(-1)
    norm = (x**2 + y**2).sum(dim=1)
    # linear part [[a, -b], [b, a]] = scale * rotation
    a = (x * x_ + y * y_).sum(dim=1) / norm
    b = (x * y_ - y * x_).sum(dim=1) / norm
    linear = torch.stack([torch.stack([a, -b], -1), torch.stack([b, a], -1)], dim=1)
    return _with_translation(linear, src, dst)


def estimate_scalexy_rotate(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares R @ diag(sx, sy) + translation: independent x / y scaling along the
    src axes, then rotation (3 points minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    src_c, dst_c = _centered(src, dst)
    # for a rotation u = (cos, sin), the optimal scales are sx = p.u / sxx and sy = q.u / syy,
    # leaving a residual of const - u^T m u: the best u is the top eigenvector of m
    # (u and -u give the same transform, with both scales negated)
    sxx, syy = (src_c**2).sum(dim=1).unbind(-1)
    p = (src_c[..., :1] * dst_c).sum(dim=1)  # (sum x x', sum x y')
    q = (src_c[..., 1:] * dst_c).sum(dim=1).flip(-1) * torch.tensor(
        [1.0, -1.0], dtype=src.dtype, device=src.device
    )  # (sum y y', -sum y x')
    m = (
        p[:, :, None] * p[:, None] / sxx[:, None, None]
        + q[:, :, None] * q[:, None] / syy[:, None, None]
    )
    # eigh may fail on non-finite input; the scales below stay inf/nan for degenerate samples
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    u = torch.linalg.eigh(m).eigenvectors[..., -1]  # (B, 2), eigenvalues ascending
    cos, sin = u.unbind(-1)
    rot = torch.stack(
        [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], dim=1
    )
    scale = torch.stack([(p * u).sum(-1) / sxx, (q * u).sum(-1) / syy], dim=-1)
    return _with_translation(rot @ torch.diag_embed(scale), src, dst)


def estimate_affine(src: Tensor, dst: Tensor) -> Tensor:
    """
    Least-squares affine transform (3 points minimum)

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    # normal equations linear @ cov_ss = cov_ds, with an explicit 2x2 inverse
    # (torch.linalg.lstsq / inv raise on singular samples on CUDA)
    src_c, dst_c = _centered(src, dst)
    cov_ss = src_c.transpose(1, 2) @ src_c  # (B, 2, 2)
    cov_ds = dst_c.transpose(1, 2) @ src_c
    a, b, d = cov_ss[:, 0, 0], cov_ss[:, 0, 1], cov_ss[:, 1, 1]
    adj = torch.stack([torch.stack([d, -b], -1), torch.stack([-b, a], -1)], dim=1)
    inv_ss = adj / (a * d - b * b)[:, None, None]
    return _with_translation(cov_ds @ inv_ss, src, dst)


def estimate_homography(src: Tensor, dst: Tensor) -> Tensor:
    """
    Homography by normalized DLT (4 points minimum), scaled so that H[2, 2] = 1

    Batched: input (B, n, 2), output (B, 3, 3)
    """
    # Hartley normalization: DLT on raw pixel coordinates is badly conditioned
    t_src, t_dst = _normalizing_transform(src), _normalizing_transform(dst)
    u, v = (to_homogeneous(src) @ t_src.transpose(1, 2)).unbind(-1)[:2]
    u_, v_ = (to_homogeneous(dst) @ t_dst.transpose(1, 2)).unbind(-1)[:2]
    zero, one = torch.zeros_like(u), torch.ones_like(u)

    # each match gives 2 linear constraints on the 9 entries of H
    a = torch.stack(
        [
            torch.stack([zero, zero, zero, -u, -v, -one, v_ * u, v_ * v, v_], dim=-1),
            torch.stack([u, v, one, zero, zero, zero, -u_ * u, -u_ * v, -u_], dim=-1),
        ],
        dim=2,
    ).flatten(1, 2)  # (B, 2n, 9)

    # H is the right singular vector of the smallest singular value
    h = torch.linalg.svd(a).Vh[:, -1].reshape(-1, 3, 3)
    h = torch.linalg.inv(t_dst) @ h @ t_src
    return h / h[:, 2:, 2:]


Estimator = Callable[[Tensor, Tensor], Tensor]


class Model(NamedTuple):
    min_points: int
    estimate: Estimator


MODELS: dict[str, Model] = {
    "translation": Model(1, estimate_translation),
    "scale": Model(2, estimate_scale),
    "scalexy": Model(2, estimate_scalexy),
    "scale+rotate": Model(2, estimate_scale_rotate),
    "scalexy+rotate": Model(3, estimate_scalexy_rotate),
    "affine": Model(3, estimate_affine),
    "homographic": Model(4, estimate_homography),
}


class RansacResult(NamedTuple):
    transform: Tensor  # (3, 3)
    inliers: Tensor  # (N,) bool


def ransac(
    src: Tensor,
    dst: Tensor,
    transform: str,
    threshold: float,
    n_iter: int = 1000,
    batch_size: int = 100,
    min_det: float = 1e-6,
    n_refit: int = 1,
    generator: torch.Generator | None = None,
) -> RansacResult | None:
    """
    Find the transform mapping the most src points within threshold of their dst match

    Every iteration fits the model on a random minimal sample of distinct matches;
    iterations are run batch_size at a time (memory is O(batch_size * N)).
    Candidates whose det(H) <= min_det (degenerate or mirroring) are discarded.
    The best candidate is then refitted on its inliers, up to n_refit times,
    as long as this does not lose inliers.

    Returns:
        best transform and its inlier mask, or None if no candidate has any inlier
    """
    min_points, estimate = MODELS[transform]
    n = len(src)
    if n < min_points:
        return None

    best_h, best_counts = [], []
    for start in range(0, n_iter, batch_size):
        b = min(batch_size, n_iter - start)
        # top-k of uniform noise = sampling without replacement
        samples = (
            torch.rand(b, n, device=src.device, generator=generator)
            .topk(min_points, dim=1)
            .indices
        )
        h = estimate(src[samples], dst[samples])
        counts = _count_inliers(src, dst, h, threshold, min_det)
        best = counts.argmax()
        best_h.append(h[best])
        best_counts.append(counts[best])

    # single device sync, after all batches
    best = torch.stack(best_counts).argmax()
    if best_counts[best] == 0:
        return None
    h, count = best_h[best], best_counts[best]

    for _ in range(n_refit):
        if count < min_points:
            break
        inliers = transfer_error(src, dst, h[None])[0] < threshold
        refit = estimate(src[inliers][None], dst[inliers][None])
        refit_count = _count_inliers(src, dst, refit, threshold, min_det)[0]
        if refit_count < count:
            break
        h, count = refit[0], refit_count

    return RansacResult(h, transfer_error(src, dst, h[None])[0] < threshold)
