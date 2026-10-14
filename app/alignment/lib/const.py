AVAILABLE_TRANSFORMS = [
    "scale",
    "scalexy",
    "scale+rotate",
    "scalexy+rotate",
    "affine",
    "homographic",
]
ALIGN_METHODS = ["ransac"]
# Pseudo-feature for classical SIFT; any other ransac_feature is a similarity model name
SIFT_FEATURE = "SIFT"

# Similarity model architectures giving a feature map, usable as ransac_feature
DENSE_ARCHS = ("cnn", "vit_hook", "dinov2")

MAX_SIZE = 2000
# Lowe's ratio test threshold to keep a SIFT match
RATIO_TEST = 0.75
# RANSAC reprojection threshold, in pixels of the (downscaled to MAX_SIZE) reference image
RANSAC_THRESHOLD = 3.0

# Dense deep features: target image diagonal (pixels) at scale 1, and scales extracted
DEEP_SIZE = 640
DEEP_SCALES = (0.5, 1.0, 1.5, 2.0)
# Scales giving an image short side below this (pixels) are skipped
DEEP_MIN_SIZE = 112
# CNN input sides are rounded to a multiple of this (ViTs use their patch size)
CNN_SIZE_MULTIPLE = 32
# RANSAC reprojection threshold, in feature cells of the reference image at scale 1
DEEP_RANSAC_THRESHOLD = 1.5
