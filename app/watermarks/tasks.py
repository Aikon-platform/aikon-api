import dramatiq
from dramatiq.middleware import CurrentMessage
from PIL import ImageOps, Image
from torchvision import transforms
import torch
from typing import List, Dict, Optional
import json
import os

from ..config import TIME_LIMIT

from .const import (
    MODEL_PATHS,
    DEVICE,
    WATERMARKS_QUEUE,
    WATERMARKS_RESULTS_FOLDER,
)
from .sources import WatermarkSource
from .utils import box_to_xyxy

from ..shared.utils.logging import notifying, TLogger, LoggerHelper


FEATURE_TRANSFORMS = lambda sz: transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.75, 0.70, 0.65], std=[0.14, 0.15, 0.16]),
    ]
)

MODELS = {}


def auto_load_models(models: List[str] = None):
    global MODELS
    if models is None:
        models = MODEL_PATHS.keys()
    for model in models:
        if model not in MODELS:
            MODELS[model] = torch.load(
                MODEL_PATHS[model],
                map_location=torch.device(DEVICE),
                weights_only=False,
            ).eval()


@torch.no_grad()
def _detect_watermarks(model: torch.nn.Module, img: Image.Image) -> Dict:
    img = transforms.ToTensor()(img)
    h, w = img.shape[-2:]
    img = img.unsqueeze(0).to(next(model.parameters()).device)

    preds = model(img)

    return {
        "boxes": [
            [x0 / w, y0 / h, x1 / w, y1 / h]
            for x0, y0, x1, y1 in preds[0]["boxes"].cpu().numpy().tolist()
        ],
        "scores": preds[0]["scores"].cpu().numpy().tolist(),
    }


def detect_watermarks(model: torch.nn.Module, img: Image.Image) -> Dict:
    for size in [512, 800, 1400, 2000]:
        im0 = img.copy()
        im0.thumbnail((size, size))
        boxes = _detect_watermarks(model, im0)
        if len(boxes["boxes"]) > 0 and boxes["scores"][0] > 0.3:
            return boxes
    return boxes


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    images: List[Image.Image],
    resize_to: int = 352,
    transpositions: List[int | None] | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    imgs = []
    if transpositions is None:
        transpositions = [None]
    tf = FEATURE_TRANSFORMS(resize_to)

    for transp in transpositions:
        for image in images:
            img = image.transpose(transp) if transp else image
            img = tf(img).to(device)
            imgs.append(img)

    imgs = torch.stack(imgs)
    feats = model(imgs)

    return feats.reshape(len(transpositions), len(images), -1)


@torch.no_grad()
def get_closest_matches(
    queries: torch.Tensor, source: WatermarkSource, topk: int = 20, min_sim=0.3
):
    n_query_flips, n_queries = queries.shape[:2]
    n_compare_flips, n_compare, n_feats = source.features.shape
    queries = torch.nn.functional.normalize(queries, dim=-1)
    sim = torch.mm(
        queries.reshape((-1, n_feats)),
        source.features.to(queries.device, queries.dtype).reshape((-1, n_feats)).T,
    )
    sim = sim.reshape(n_query_flips, n_queries, n_compare_flips, n_compare)
    best_qsim, best_qflip = sim.max(dim=0)
    best_ssim, best_sflip = best_qsim.max(dim=1)
    tops = best_ssim.topk(topk, dim=1)
    return [
        [
            {
                "similarity": ssim.item(),
                "best_source_flip": best_sflip[i, j].item(),
                "best_query_flip": best_qflip[i, best_sflip[i, j], j].item(),
                "query_index": i,
                "source_index": j.item(),
            }
            for (j, ssim) in (zip(tops.indices[i], tops.values[i]))
            if ssim > min_sim
        ]
        for i in range(n_queries)
    ]


@torch.no_grad()
def _pipeline(
    image: Image.Image,
    detect: bool = True,
    compare_to: Optional[WatermarkSource] = None,
) -> Dict:
    to_load = []
    if detect:
        to_load.append("detection")
    if compare_to:
        to_load.append("features")
    auto_load_models(to_load)

    image = ImageOps.exif_transpose(image).convert("RGB")

    crops = [image]
    output = {}

    resize = compare_to.metadata.get("resize", 352) if compare_to else 352

    if detect:
        boxes = detect_watermarks(MODELS["detection"], image)
        output["detection"] = boxes
        crops = []
        if compare_to:
            for box, score in zip(boxes["boxes"], boxes["scores"]):
                if score < 0.5 and len(crops) > 0:
                    break
                x0, y0, x1, y1 = box_to_xyxy(box, image)
                crops.append(image.crop((x0, y0, x1, y1)).resize((resize, resize)))

    if compare_to and len(crops) > 0:
        feats = extract_features(
            MODELS["features"],
            crops,
            resize,
            [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        )
        output["matches"] = get_closest_matches(feats, compare_to)
        output["query_flips"] = [None, "rot90", "rot180", "rot270"]

    print(output)

    return output


@dramatiq.actor(
    time_limit=TIME_LIMIT,
    max_retries=0,
    store_results=True,
    queue_name=WATERMARKS_QUEUE,
)
@notifying
def pipeline(
    image_path: str,
    detect: bool = True,
    experiment_id: str = "",
    compare_to: Optional[str] = None,
    logger: TLogger = LoggerHelper,
    notifier=None,
    **kwargs,
):
    # TODO add notify_url to arguments?
    image = Image.open(image_path)
    compare_to = WatermarkSource(compare_to) if compare_to else None
    output = _pipeline(image, detect, compare_to)

    result_dir = WATERMARKS_RESULTS_FOLDER
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(
        result_dir / f"{CurrentMessage.get_current_message().message_id}.json", "w"
    ) as f:
        json.dump(output, f)
    os.unlink(image_path)

    return output
