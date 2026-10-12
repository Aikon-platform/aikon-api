AVAILABLE_TRANSFORMS = [
    "scale",
    "scalexy",
    "scale+rotate",
    "scalexy+rotate",
    "affine",
    "homographic",
]
ALIGN_METHODS = ["sift+ransac"]

MAX_SIZE = 2000
# Lowe's ratio test threshold to keep a SIFT match
RATIO_TEST = 0.75
# RANSAC reprojection threshold, in pixels of the (downscaled) reference image
RANSAC_THRESHOLD = 3.0
