from typing import TypedDict


class Keypoint(TypedDict):
    x: float
    y: float
    inlier: bool


class AlignedImage(TypedDict):
    image_name: str
    # 3x3 matrix mapping image pixel coordinates to reference pixel coordinates
    transform_matrix: list[list[float]] | None
    # Same order as the reference keypoints, None where no match was found
    keypoints: list[Keypoint | None]
    is_reference: bool | None
