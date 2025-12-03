import os, torch
from typing import Tuple, Callable, Any
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models, transforms
from collections import OrderedDict
from functools import partial

from .vit import VisionTransformer
from ..const import MODEL_PATH
from ...shared.utils import clear_cuda
from ...shared.utils.fileutils import download_file

DEFAULT_MODEL_URLS = {
    "moco_v2_800ep_pretrain": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
    "dino_deitsmall16_pretrain": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    "dino_vitbase8_pretrain": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    "hard_mining_neg5": "https://github.com/XiSHEN0220/SegSwap/raw/main/model/hard_mining_neg5.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    **{k: {
        "repo_id": "seglinglin/Historical-Document-Backbone",
        "filename": "resnet18_watermarks.pth",
    } for k in ["resnet18_watermarks", "resnet18_watermarks_conv4"]},
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

DEFAULT_FEAT_LAYERS = {
    "resnet34": "layer3.5.bn2",
    "moco_v2_800ep_pretrain": "layer3.5.bn3",
}

DEFAULT_MODEL_INFOS = {
    "moco_v2_800ep_pretrain": {
        "name": "MoCo v2 800ep",
        "model": "moco_v2_800ep_pretrain",
        "desc": "A contrastive learning model for image classification.",
    },
    "dino_vitbase8_pretrain": {
        "name": "DINO ViT-Base 8",
        "model": "dino_vitbase8_pretrain",
        "desc": "A Vision Transformer feature extractor.",
    },
    "resnet34": {
        "name": "ResNet 34",
        "model": "resnet34",
        "desc": "A deep residual network trained for image classification.",
    },
    "resnet18_watermarks": {
        "name": "ResNet 18 for watermarks",
        "model": "resnet18_watermarks",
        "desc": "Deep residual network trained for watermarks comparison.",
    },
    "resnet18_watermarks_conv4": {
        "name": "ResNet 18 for watermarks, cropped to conv4",
        "model": "resnet18_watermarks_conv4",
        "desc": "Deep residual network trained for watermarks comparison, more efficient for cross-domain.",
    },
    "dino_deitsmall16_pretrain": {
        "name": "DINO DeiT-Small 16",
        "model": "dino_deitsmall16_pretrain",
        "desc": "Data-efficient Image Transformer.",
    },
    "hard_mining_neg5": {
        "name": "Hard Negative Mining",
        "model": "hard_mining_neg5",
        "desc": "A model trained with hard negative mining (not recommended for features extraction).",
    },
}


def _instantiate_moco_v2_800ep_pretrain(weights_path, device) -> torch.nn.Module:
    model = models.resnet50().to(device)
    pre_dict = torch.load(weights_path, weights_only=True)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in pre_dict.items():
        name = k[17:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def _instantiate_dino_vitbase8_pretrain(weights_path, device) -> torch.nn.Module:
    model = VisionTransformer(patch_size=8, embed_dim=768, num_heads=12, qkv_bias=True)
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=device)
    )
    return model


def _instantiate_resnet34(weights_path, device) -> torch.nn.Module:
    # try:
    #     weights = torch.load(weights_path, weights_only=True, map_location=device)
    # except RuntimeError:
    #     weights = torch.load(weights_path, weights_only=False, map_location=device)
    # # model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1, weights_only=True, map_location=device)
    # model = models.resnet34(weights=weights)
    model = models.resnet34(
        torch.load(weights_path, weights_only=False, map_location=device)
    )
    return model


def _instantiate_dino_deitsmall16_pretrain(weights_path, device) -> torch.nn.Module:
    model = VisionTransformer(patch_size=16, embed_dim=384, num_heads=6, qkv_bias=True)
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=device)
    )
    return model


def _instantiate_resnet18_watermarks(weights_path, device, conv4: bool=False) -> torch.nn.Module:
    path_str = str(weights_path)
    model = torch.load(path_str, map_location=device, weights_only=False)
    if conv4:
        model.layer4 = torch.nn.Identity()
    return model


DEFAULT_MODEL_LOADERS = {
    "moco_v2_800ep_pretrain": _instantiate_moco_v2_800ep_pretrain,
    "dino_vitbase8_pretrain": _instantiate_dino_vitbase8_pretrain,
    "dino_deitsmall16_pretrain": _instantiate_dino_deitsmall16_pretrain,
    "resnet34": _instantiate_resnet34,
    "resnet18_watermarks": partial(_instantiate_resnet18_watermarks, conv4=False),
    "resnet18_watermarks_conv4": partial(_instantiate_resnet18_watermarks, conv4=True)
}

DEFAULT_MODEL_TRANSFORMS = {
    **{k: transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.Normalize(mean=[0.75, 0.70, 0.65], std=[0.14, 0.15, 0.16]),
        ]
    ) for k in ["resnet18_watermarks", "resnet18_watermarks_conv4"]}
}

def download_model(model_name):
    os.makedirs(MODEL_PATH, exist_ok=True)

    if model_name not in DEFAULT_MODEL_URLS:
        raise ValueError("Unknown network for feature extraction.")

    url = DEFAULT_MODEL_URLS[model_name]

    if type(url) is str:
        download_file(url, MODEL_PATH / f"{model_name}.pth")
    else:
        from huggingface_hub import hf_hub_download

        hf_hub_download(local_dir=MODEL_PATH, **url)


def get_model_path(model_name):
    # TODO Hook to handle cropped models by removing "_conv4" suffix
    model_name = model_name.replace("_conv4", "")

    if not os.path.exists(MODEL_PATH / f"{model_name}.pth"):
        download_model(model_name)

    return f"{MODEL_PATH}/{model_name}.pth"


def get_transforms_for_model(model_name):
    # transforms.Lambda(lambda x: x.convert('RGB'))
    if model_name in DEFAULT_MODEL_TRANSFORMS:
        return DEFAULT_MODEL_TRANSFORMS[model_name]
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(
    model_path: str, feat_net: str, device: str
) -> Tuple[torch.nn.Module, Callable[Any, torch.Tensor]]:
    if model_path is None:
        model_path = get_model_path(feat_net)

    print(f"Loading model {feat_net} from {model_path}")
    clear_cuda()

    if feat_net in DEFAULT_MODEL_LOADERS:
        model = DEFAULT_MODEL_LOADERS[feat_net](model_path, device)
    else:
        model = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(model, dict):
        raise ValueError("Invalid network for feature extraction : no loader known.")

    model = model.eval().to(device)

    if feat_net in DEFAULT_FEAT_LAYERS:
        model = create_feature_extractor(
            model, return_nodes={DEFAULT_FEAT_LAYERS[feat_net]: "feat"}
        )
        feat_func = lambda x: x["feat"].flatten(start_dim=1)
    else:
        feat_func = lambda x: x
    return model, feat_func
