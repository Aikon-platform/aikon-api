"""
Batched RANSAC estimation of 2D transforms between matched keypoints, in pure torch

Conventions:
- points are (N, 2) floating tensors of pixel coordinates, src[i] matched with dst[i]
- a transform is a 3x3 matrix H mapping src onto dst in homogeneous coordinates
"""

import math
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


def _ajugate_invert(h: Tensor) -> Tensor:
    """
    Inverse of 3x3 matrices by their adjugate, inf/nan for singular ones
    (torch.linalg.inv raises on singular matrices on CUDA)

    Batched: (B, 3, 3) -> (B, 3, 3)
    """
    r0, r1, r2 = h.unbind(1)
    # columns of the inverse: cross products of the rows, over the determinant
    adj = torch.stack(
        [
            torch.linalg.cross(r1, r2),
            torch.linalg.cross(r2, r0),
            torch.linalg.cross(r0, r1),
        ],
        dim=-1,
    )
    return adj / (r0 * adj[..., 0]).sum(-1)[:, None, None]


def symmetric_error(src: Tensor, dst: Tensor, h: Tensor) -> Tensor:
    """
    RMS of the forward (src -> dst) and backward (dst -> src) transfer errors, in dst pixels.

    Scaling factor between src & dst pixels is approximated by linear part of h 
    (= exact for affine transform)

    Args:
        src, dst: (N, 2)
        h: (B, 3, 3)

    Returns:
        (B, N) errors, inf/nan for points sent to infinity and singular transforms
    """
    fwd = transfer_error(src, dst, h)
    bwd = transfer_error(dst, src, _ajugate_invert(h))
    scale = (torch.linalg.det(h[:, :2, :2]) / h[:, 2, 2] ** 2).abs().sqrt()
    return ((fwd**2 + (bwd * scale[:, None]) ** 2) / 2).sqrt()


def _is_transform_valid(h: Tensor, min_det: float) -> Tensor:
    """
    Finite transforms with det(H) > min_det (neither degenerate nor mirroring)

    Batched: (B, 3, 3) -> (B,) bool
    """
    return torch.isfinite(h).all(dim=(1, 2)) & (torch.linalg.det(h) > min_det)


def _group_max_sum(values: Tensor, groups: Tensor, n_groups: int) -> Tensor:
    """
    Sum over groups of the largest value of each group

    Batched: values (B, N) >= 0, groups (N,) in [0, n_groups), output (B,)
    """
    out = values.new_zeros(len(values), n_groups)
    return out.scatter_reduce(1, groups.expand_as(values), values, "amax").sum(dim=1)


class FitScorer:
    """
    Counts the inliers of transforms
    
    Deduplicates src & dst points, by considering the smaller of the numbers 
    of src and dst points with an inlier
    """

    def __init__(
        self,
        src: Tensor,
        dst: Tensor,
        src_groups: Tensor,
        dst_groups: Tensor,
        min_det: float,
    ):
        self.src, self.dst, self.min_det = src, dst, min_det
        self.groups = [
            (g, int(g.max()) + 1 if len(g) else 0) for g in (src_groups, dst_groups)
        ]

    def errors(self, h: Tensor) -> Tensor:
        return symmetric_error(self.src, self.dst, h)

    def count(self, h: Tensor, threshold: float) -> Tensor:
        """
        (B, 3, 3) -> (B,), 0 for invalid transforms
        """
        inliers = (self.errors(h) < threshold).to(self.src.dtype)
        counts = [_group_max_sum(inliers, g, n) for g, n in self.groups]
        return torch.minimum(*counts) * _is_transform_valid(h, self.min_det)


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


def _needed_iterations(inlier_ratio: float, min_points: int, p_fail: float) -> float:
    """
    Number of random minimal samples to draw to get an all-inlier one with
    probability 1 - p_fail, i.e. log(p_fail) / log(1 - inlier_ratio ^ min_points)
    """
    p_good = inlier_ratio**min_points
    if p_good >= 1:
        return 0
    if p_good <= 0:
        return math.inf
    return math.log(p_fail) / math.log1p(-p_good)


def _refit_finer_transform(
    scorer: FitScorer,
    estimate: Estimator,
    h: Tensor,
    thresholds: list[float],
    min_points: int,
) -> Tensor:
    """
    Refit h on its inliers, with decreasing threshold, to tighten the transform estimate

    (3, 3) -> (3, 3)
    """
    for t in thresholds:
        inliers = scorer.errors(h[None])[0] < t
        if inliers.sum() < min_points:
            break
        refit = estimate(scorer.src[inliers][None], scorer.dst[inliers][None])
        if not _is_transform_valid(refit, scorer.min_det)[0]:
            break
        h = refit[0]
    return h


def ransac(
    src: Tensor,
    dst: Tensor,
    transform: str,
    threshold: float,
    *,
    src_groups: Tensor | None = None,
    dst_groups: Tensor | None = None,
    hypotheses: Tensor | None = None,
    loose_factor: float = 3.0,
    n_refit: int = 4,
    n_candidates: int = 8,
    max_iter: int = 10000,
    p_fail: float = 1e-3,
    batch_size: int = 256,
    min_inliers: int = 0,
    min_det: float = 1e-6,
    generator: torch.Generator | None = None,
) -> RansacResult | None:
    """
    Find the transform mapping src onto dst with the most inliers

    An inlier is a match whose symmetric_error is below threshold; if a point is shared
    by several matches, it counts once (cf _Scorer).

    1. Collect candidate transforms:
       - the given hypotheses, if any: typically one per match, each built from that
         single match (with SIFT, from the size and orientation of its two keypoints);
       - fits on random samples of min_points matches.
       All are scored at a loose threshold (threshold * loose_factor), and only the
       n_candidates best are kept.
    2. Refine each kept candidate: refit it on its inliers n_refit times, with a
       threshold shrinking from the loose one down to threshold.
    3. Return the best transform, refined or not.

    The hypotheses parameter is a way to provide candidate transforms obtained differently
    (e.g., one-match hypotheses). It helps: if a fraction w of the matches are right, a hypothesis
    built from a single match is right with probability w, while a random sample of
    min_points matches is right only with probability w ** min_points (e.g. 1e-4 for a
    homography, 4 matches, at w = 10%).

    Args:
        src, dst: (N, 2) matched points
        transform: model name, a key of MODELS
        threshold: inlier threshold, in dst pixels
        src_groups, dst_groups: (N,) id of the src / dst point of each match, to count
            a point shared by several matches once (default: all points distinct)
        hypotheses: (M, 3, 3) candidate transforms, e.g. one per match from the size and
            orientation of its two keypoints (see frame_hypotheses in align_ransac)
        loose_factor: threshold multiplier for scoring the unrefined candidates
        n_refit: number of refits of each candidate
        n_candidates: number of candidates kept for refinement
        max_iter: maximum number of random samples; sampling stops earlier once a right
            sample has been drawn with probability 1 - p_fail, given the best inlier
            ratio found so far
        p_fail: accepted probability of missing a right random sample
        batch_size: samples / hypotheses scored at once (memory is O(batch_size * N))
        min_inliers: below this number of inliers, no transform is returned
        min_det: transforms with det(H) <= min_det (degenerate or mirroring) are discarded
        generator: random generator, for reproducible results

    Returns:
        best transform and its inlier mask, or None if it has fewer than
        max(min_inliers, min_points) inliers
    """
    min_points, estimate = MODELS[transform]
    n = len(src)
    if n < max(min_points, min_inliers):
        return None

    every = torch.arange(n, device=src.device)
    scorer = FitScorer(
        src,
        dst,
        every if src_groups is None else src_groups,
        every if dst_groups is None else dst_groups,
        min_det,
    )
    loose = threshold * loose_factor

    # n_candidates best candidates, with their inlier count at the loose threshold, best first
    cand_h = src.new_empty(0, 3, 3)
    cand_s = src.new_empty(0)

    def add_candidates(h: Tensor):
        nonlocal cand_h, cand_s
        cand_h = torch.cat([cand_h, h])
        cand_s = torch.cat([cand_s, scorer.count(h, loose)])
        keep = cand_s.topk(min(n_candidates, len(cand_s))).indices
        cand_h, cand_s = cand_h[keep], cand_s[keep]

    if hypotheses is not None:
        for h in hypotheses.to(src).split(batch_size):
            add_candidates(h)

    done = 0
    while done < max_iter:
        ratio = (
            float(scorer.count(cand_h[:1], threshold)[0]) / n if len(cand_h) else 0.0
        )
        if done >= _needed_iterations(ratio, min_points, p_fail):
            break
        b = min(batch_size, max_iter - done)
        # top-k of uniform noise = sampling without replacement
        samples = (
            torch.rand(b, n, device=src.device, generator=generator)
            .topk(min_points, dim=1)
            .indices
        )
        add_candidates(estimate(src[samples], dst[samples]))
        done += b

    cand_h = cand_h[cand_s > 0]
    if not len(cand_h):
        return None

    thresholds = [
        loose * (threshold / loose) ** (i / max(n_refit - 1, 1)) for i in range(n_refit)
    ]
    refined = [
        _refit_finer_transform(scorer, estimate, h, thresholds, min_points) for h in cand_h
    ]

    # Chooser best transform
    final = torch.cat([cand_h, torch.stack(refined)])
    h = final[scorer.count(final, threshold).argmax()]

    if scorer.count(h[None], threshold)[0] < max(min_inliers, min_points):
        return None
    return RansacResult(h, scorer.errors(h[None])[0] < threshold)
