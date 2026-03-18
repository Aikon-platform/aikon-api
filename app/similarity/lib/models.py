import os
from collections import OrderedDict
from functools import partial
from typing import Tuple, Callable, Any

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

from .vit import VisionTransformer
from ..const import MODEL_PATH
from ...shared.utils import clear_cuda
from ...shared.utils.fileutils import download_file


# ── Feature extraction wrappers ──────────────────────────────────────────────


def _setup_cnn_features(model, node_name, spatial=False):
    model = create_feature_extractor(model, return_nodes={node_name: "feat"})
    if spatial:
        return model, lambda x: x["feat"]  # [B, C, H, W]
    return model, lambda x: x["feat"].mean(dim=[2, 3])  # [B, C]


def _setup_vit_hook(model, block_idx, spatial=False, patch_size=None):
    captured = {}
    model.blocks[block_idx].register_forward_hook(
        lambda m, i, o: captured.update(feat=o)
    )
    patch_tokens = lambda x: captured["feat"][:, 1:]  # [B, N, C]
    if spatial and patch_size:

        def feat_fn(x):
            tokens = patch_tokens(x)
            B, N, C = tokens.shape
            h = x.shape[2] // patch_size
            w = x.shape[3] // patch_size
            return tokens.view(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, h, w]

        return model, feat_fn
    if spatial:
        return model, lambda x: patch_tokens(x)  # [B, N, C]
    return model, lambda x: patch_tokens(x).mean(dim=1)  # [B, C]


class _DINOv2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._input_shape = None

    def forward(self, x):
        self._input_shape = x.shape
        return self.model.forward_features(x)


def _setup_dinov2_features(model, layer, spatial=False, patch_size=None):
    if layer == "last":
        wrapper = _DINOv2Wrapper(model)
        if spatial and patch_size:

            def feat_fn(x):
                tokens = x["x_norm_patchtokens"]  # [B, N, C]
                B, N, C = tokens.shape
                h = wrapper._input_shape[2] // patch_size
                w = wrapper._input_shape[3] // patch_size
                return tokens.view(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, h, w]

            return wrapper, feat_fn
        if spatial:
            return wrapper, lambda x: x["x_norm_patchtokens"]  # [B, N, C]
        return wrapper, lambda x: x["x_norm_clstoken"]  # [B, C]
    return _setup_vit_hook(
        model, block_idx=layer, spatial=spatial, patch_size=patch_size
    )


# ── Model loaders ────────────────────────────────────────────────────────────


def _load_moco_v2(weights_path, device):
    model = models.resnet50()
    state = torch.load(weights_path, map_location=device, weights_only=False)[
        "state_dict"
    ]
    model.load_state_dict(
        OrderedDict(
            (k[17:], v) for k, v in state.items() if k.startswith("module.encoder_q.")
        ),
        strict=False,
    )
    return model


def _load_resnet50(weights_path, device):
    return models.resnet50(
        torch.load(weights_path, weights_only=False, map_location=device)
    )


def _load_dino_deitsmall16(weights_path, device):
    model = VisionTransformer(patch_size=16, embed_dim=384, num_heads=6, qkv_bias=True)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    return model


def _load_dinov2_vitb14(_path, device):
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)


def _load_clip_vitb32(_path, device):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    return model.visual


def _load_resnet18_watermarks(weights_path, device, conv4=False):
    model = torch.load(str(weights_path), map_location=device, weights_only=False)
    if conv4:
        model.layer4 = nn.Identity()
    return model


# ── CNN layer nodes ──────────────────────────────────────────────────────────

CNN_LAYERS_RESNET50 = {
    "layer1": "layer1.2.bn3",
    "layer2": "layer2.3.bn3",
    "layer3": "layer3.5.bn3",
    "layer4": "layer4.2.bn3",
}

VIT_LAYERS = [2, 5, 8, 11]

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        IMAGENET_NORMALIZE,
    ]
)


# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "resnet50": {
        "loader": _load_resnet50,
        "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "transform": IMAGENET_TRANSFORM,
        "arch": "cnn",
        "default_layer": "layer3",
        "layers": CNN_LAYERS_RESNET50,
        "info": {
            "name": "ResNet 50",
            "desc": "A deep residual network trained for image classification.",
        },
    },
    "moco_v2_800ep_pretrain": {
        "loader": _load_moco_v2,
        "url": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
        "transform": IMAGENET_TRANSFORM,
        "arch": "cnn",
        "default_layer": "layer3",
        "layers": CNN_LAYERS_RESNET50,
        "info": {
            "name": "MoCo v2 800ep",
            "desc": "A contrastive learning model for image classification.",
        },
    },
    "dino_deitsmall16_pretrain": {
        "loader": _load_dino_deitsmall16,
        "url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        "transform": IMAGENET_TRANSFORM,
        "arch": "vit_hook",
        "default_layer": 8,
        "layers": VIT_LAYERS,
        "patch_size": 16,
        "info": {
            "name": "DINO DeiT-Small 16",
            "desc": "Data-efficient Image Transformer.",
        },
    },
    "dinov2_vitb14": {
        "loader": _load_dinov2_vitb14,
        "transform": IMAGENET_TRANSFORM,
        "arch": "dinov2",
        "default_layer": "last",
        "layers": VIT_LAYERS + ["last"],
        "patch_size": 14,
        "info": {
            "name": "DINOv2 ViT-Base 14",
            "desc": "Self-supervised Vision Transformer with strong transfer features.",
        },
    },
    "clip_vitb32": {
        "loader": _load_clip_vitb32,
        "transform": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        ),
        "arch": "fixed",
        "default_layer": "final",
        "layers": ["final"],
        "info": {
            "name": "CLIP ViT-B/32",
            "desc": "Vision-language model trained on image-text pairs.",
        },
    },
    **{
        k: {
            "loader": partial(_load_resnet18_watermarks, conv4=conv4),
            "url": {
                "repo_id": "seglinglin/Historical-Document-Backbone",
                "filename": "resnet18_watermarks.pth",
            },
            "transform": transforms.Compose(
                [
                    transforms.Resize((320, 320)),
                    transforms.Normalize(
                        mean=[0.75, 0.70, 0.65], std=[0.14, 0.15, 0.16]
                    ),
                ]
            ),
            "arch": "fixed",
            "default_layer": "final",
            "layers": ["final"],
            "info": {
                "name": name,
                "desc": desc,
            },
        }
        for k, conv4, name, desc in [
            (
                "resnet18_watermarks",
                False,
                "ResNet 18 for watermarks",
                "Deep residual network trained for watermarks comparison.",
            ),
            (
                "resnet18_watermarks_conv4",
                True,
                "ResNet 18 for watermarks (conv4)",
                "Deep residual network trained for watermarks comparison, more efficient for cross-domain.",
            ),
        ]
    },
    "hard_mining_neg5": {
        "loader": None,  # Loaded directly by segswap
        "url": "https://github.com/XiSHEN0220/SegSwap/raw/main/model/hard_mining_neg5.pth",
        "transform": None,
        "arch": "segswap",
        "default_layer": None,
        "layers": [],
        "info": {
            "name": "Hard Negative Mining",
            "desc": "A model trained with hard negative mining (not for feature extraction).",
        },
    },
}

AVAILABLE_MODELS = [k for k, v in MODELS.items() if v["loader"] is not None]

DEFAULT_MODEL_INFOS = {
    k: {"model": k, **v["info"]} for k, v in MODELS.items() if v.get("info")
}


# ── Weight management ────────────────────────────────────────────────────────


def download_model(model_name):
    os.makedirs(MODEL_PATH, exist_ok=True)

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    url = MODELS[model_name].get("url")
    if url is None:
        return

    if isinstance(url, str):
        download_file(url, MODEL_PATH / f"{model_name}.pth")
    else:
        from huggingface_hub import hf_hub_download

        hf_hub_download(local_dir=MODEL_PATH, **url)


def get_model_path(model_name):
    # Handle cropped model variants sharing the same weights file
    weights_name = model_name.replace("_conv4", "")

    if not os.path.exists(MODEL_PATH / f"{weights_name}.pth"):
        download_model(model_name)

    return f"{MODEL_PATH}/{weights_name}.pth"


# ── Public API ───────────────────────────────────────────────────────────────


def get_transforms_for_model(model_name):
    if model_name in MODELS:
        return MODELS[model_name]["transform"]
    return IMAGENET_TRANSFORM


def load_model(
    model_path: str,
    feat_net: str,
    device: str,
    spatial: bool = False,
) -> Tuple[nn.Module, Callable[Any, torch.Tensor]]:
    """
    Load a model and return (model, feat_fn).

    feat_fn output shapes:
      spatial=False → [B, C]        (global vector)
      spatial=True  → [B, C, H, W]  (feature map, for architectures that support it)

    Raises ValueError for spatial=True on architectures that don't support it (fixed, segswap).
    """
    if feat_net not in MODELS or MODELS[feat_net]["loader"] is None:
        raise ValueError(f"No loader registered for model: {feat_net}")

    cfg = MODELS[feat_net]

    if spatial and cfg["arch"] in ("fixed", "segswap"):
        raise ValueError(f"{feat_net} does not support spatial feature extraction")

    # Download weights if needed (for models requiring explicit files)
    if model_path is None and "url" in cfg:
        model_path = get_model_path(feat_net)

    print(f"Loading model {feat_net} from {model_path or 'torch.hub/package'}")
    clear_cuda()

    model = cfg["loader"](model_path, device)

    if isinstance(model, dict):
        raise ValueError(
            f"Invalid model state for {feat_net}: no loader could instantiate it."
        )

    model = model.eval().to(device)

    layer = cfg["default_layer"]
    patch_size = cfg.get("patch_size")

    match cfg["arch"]:
        case "cnn":
            return _setup_cnn_features(model, cfg["layers"][layer], spatial=spatial)
        case "vit_hook":
            return _setup_vit_hook(
                model, block_idx=layer, spatial=spatial, patch_size=patch_size
            )
        case "dinov2":
            return _setup_dinov2_features(
                model, layer, spatial=spatial, patch_size=patch_size
            )
        case "fixed":
            return model, lambda x: x

    raise ValueError(f"Unknown architecture '{cfg['arch']}' for {feat_net}")
