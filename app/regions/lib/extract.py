import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import numpy as np
import cv2

from ultralytics.utils.plotting import Annotator, colors
from .bbox import Segment

from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.dataloaders import IMG_FORMATS
from .yolov5.utils.general import (
    check_file,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from .yolov5.utils.augmentations import letterbox
from .yolov5.utils.torch_utils import select_device, smart_inference_mode

from ...shared.utils import get_device
from ...shared.utils.fileutils import TPath
from ...shared.dataset import Image as DImage
from ...shared.utils.logging import console

FILE = Path(__file__).resolve()
LIB_ROOT = FILE.parents[0]  # lib root directory
if str(LIB_ROOT) not in sys.path:
    sys.path.append(str(LIB_ROOT))  # add LIB_ROOT to PATH
# LIB_ROOT = "api" / Path(os.path.relpath(LIB_ROOT, Path.cwd()))  # relative

# Constants
CONF_THRES = 0.25
IOU_THRES = 0.45
HIDE_LABEL = False
HIDE_CONF = False

# UTILS
def get_img_dim(source: TPath) -> Tuple[int, int]:
    """
    Get the dimensions of an image (width, height)
    """
    with Image.open(source) as img:
        return img.size[0], img.size[1]  # width, height


def setup_source(source: TPath) -> str:
    """
    Check if the source is a URL or a file

    If the source is a URL that points to an image file, download the image
    """
    source = str(source)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_file = Path(source).suffix[1:] in IMG_FORMATS
    if is_url and is_file:
        source = check_file(source)  # download
    return source


class ImageAnnotator:
    def __init__(self, image: DImage, img_w: int = None, img_h: int = None):
        path = image.path
        if img_w is None or img_h is None:
            img_w, img_h = get_img_dim(path)

        self.annotations = {
            "source": image.id,
            "source_info": image.to_dict(),
            "width": img_w,
            "height": img_h,
            "crops": [],
        }

    def add_region(self, x: int, y: int, w: int, h: int, conf: float):
        img_w = self.annotations["width"]
        img_h = self.annotations["height"]

        rel_x = x / img_w
        rel_y = y / img_h
        rel_w = w / img_w
        rel_h = h / img_h

        segment = Segment(rel_x, rel_y, rel_w, rel_h, precision=2)

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        self.annotations["crops"].append(
            {
                "bbox": segment.serialize(),  # compact string representation
                "crop_id": f'{self.annotations["source"]}-{segment.serialize()}',
                "source": self.annotations["source"],
                "confidence": round(conf, 4),
                "absolute": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": w,
                    "height": h,
                },
                "relative": {
                    "x1": round(rel_x, 4),
                    "y1": round(rel_y, 4),
                    "x2": round(rel_x + rel_w, 4),
                    "y2": round(rel_y + rel_h, 4),
                    "width": round(rel_w, 4),
                    "height": round(rel_h, 4),
                },
            }
        )


class BaseExtractor:
    """
    A base class for extracting regions from images
    """

    DEFAULT_IMG_SIZES = [640]  # used for multiscale inference

    def __init__(
        self,
        weights: TPath,
        device: str = None,
        input_sizes: list[int] = None,
        squarify: bool = False,
        margin: float = 0.0,
    ):
        self.weights = weights
        available_device = 0 if get_device() != "cpu" else "cpu"
        self.device = torch.device(device or available_device)
        self.input_sizes = (
            input_sizes if input_sizes is not None else self.DEFAULT_IMG_SIZES
        )
        self.model = self.get_model()
        self.squarify = squarify
        self.margin = margin

    def get_model(self):
        raise NotImplementedError()

    @smart_inference_mode()
    def extract_one(self, img: DImage, save_img: bool = False):
        raise NotImplementedError()

    @smart_inference_mode()
    def prepare_image(self, im):
        return transforms.ToTensor()(im).unsqueeze(0).to(self.device)

    @staticmethod
    def resize(img, size):
        resized_image = img.copy()
        resized_image.thumbnail((size, size))
        return resized_image

    @smart_inference_mode()
    def process_detections(
        self,
        detections: torch.Tensor,
        image_tensor: torch.Tensor,
        original_image: np.array,
        save_img: bool,
        source: TPath,
        writer: ImageAnnotator,
        class_names_examples: str = "abc",
    ) -> bool:
        annotator = (
            Annotator(original_image, line_width=2, example=str(class_names_examples))
            if save_img
            else None
        )

        if not len(detections):
            return False

        img_h, img_w = original_image.shape[:2]

        # Rescale boxes from img_size to im0 size
        detections[:, :4] = scale_boxes(
            image_tensor.shape[2:], detections[:, :4], original_image.shape
        ).round()

        # Write results
        for *xyxy, conf, cls in reversed(detections):
            # Extract coordinates
            x, y, w, h = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])

            if self.squarify:
                s = min(max(w, h), img_w, img_h)
                x -= (s - w) // 2
                y -= (s - h) // 2
                w = h = s

            if self.margin > 0 or self.squarify:
                x -= w * self.margin
                y -= h * self.margin
                w += 2 * self.margin * w
                h += 2 * self.margin * h

            w = min(w, img_w)
            h = min(h, img_h)
            x = min(max(0, x), img_w - w)
            y = min(max(0, y), img_h - h)

            x, y, w, h = int(x), int(y), int(w), int(h)
            writer.add_region(x, y, w, h, float(conf))

            if save_img:
                c = int(cls)
                label = (
                    None
                    if HIDE_LABEL
                    else (
                        class_names_examples[c]
                        if HIDE_CONF
                        else f"{class_names_examples[c]} {conf:.2f}"
                    )
                )
                annotator.box_label(xyxy, label, color=colors(c, True))

        if save_img:
            output_path = str(Path(source).parent / f"extracted_{Path(source).name}")
            cv2.imwrite(output_path, annotator.result())

        return True


class LineExtractor(BaseExtractor):
    """
    ------------------------------------------------------------------------
    Line Predictor
    Copyright (c) 2024 Raphaël Baena (Imagine team - LIGM)
    Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    Copied from LinePredictor (https://github.com/raphael-baena/LinePredictor)
    ------------------------------------------------------------------------
    """

    from .line_predictor.datasets import transforms

    config = LIB_ROOT / "line_predictor" / "config" / "DINO_4scale.py"
    T = transforms

    def get_model(self):
        from .line_predictor import build_model_main
        from .line_predictor.config.slconfig import SLConfig

        self.device = select_device(self.device)
        checkpoint = torch.load(self.weights, map_location="cpu")

        args = SLConfig.fromfile(self.config)
        args.device = self.device

        # TODO change nb queries
        for key, tensor in checkpoint["model"].items():
            if "tgt_embed.weight" in key:
                # adjust number of queries depending on the checkpoint
                args.num_queries = tensor.shape[0]
                break

        model, _, _ = build_model_main(args)
        model.load_state_dict(checkpoint["model"], strict=False)
        return model.eval()

    @staticmethod
    def renorm(
        img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ) -> torch.Tensor:
        """
        Function to re-normalize images: converts normalized tensors back to the original image scale
        Use for visualization
        """
        assert img.dim() in [3, 4], "Input tensor must have 3 or 4 dimensions"
        permutation = (1, 2, 0) if img.dim() == 3 else (0, 2, 3, 1)
        channels = img.size(0) if img.dim() == 3 else img.size(1)

        assert channels == 3, "Expected 3 channels in input tensor"
        img_perm = img.permute(*permutation)
        img_res = img_perm * torch.Tensor(std) + torch.Tensor(mean)

        img_renorm = img_res.permute(*permutation)
        return img_renorm.permute(1, 2, 0)

    @staticmethod
    def poly_to_bbox(poly):
        x0, y0, x1, y1 = poly[:, 0], poly[:, 1], poly[:, -4], poly[:, -1]
        # return torch.stack([x0, y0, x1, y1], dim=1)
        x_min, x_max = torch.min(x0, x1), torch.max(x0, x1)
        y_min, y_max = torch.min(y0, y1), torch.max(y0, y1)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def prepare_image(self, img: DImage):
        transform = self.T.Compose(
            [
                self.T.RandomResize([800], max_size=1333),
                self.T.ToTensor(),
                self.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(img, None)
        return image

    @staticmethod
    def scale(poly, w, h):
        return poly * torch.tensor([w, h]).repeat(10)

    def cleanup_detections(self, polygons, scores, curr_size):
        curr_w, curr_h = curr_size
        # scaled_polygons = self.scale(polygons, w_ratio, h_ratio)
        scaled_polygons = self.scale(polygons, curr_w, curr_h)

        bboxes = self.poly_to_bbox(scaled_polygons).to(self.device)
        scores = scores.to(self.device)

        # Perform Non-Maximum Suppression (NMS) TODO filter nms on polygons: https://github.com/WolodjaZ/PolyGoneNMS
        nms_filter = nms(bboxes, scores, iou_threshold=0.8).cpu()
        bboxes = bboxes[nms_filter]
        scores = scores[nms_filter].unsqueeze(1)
        labels = torch.zeros(len(scores), 1, device=self.device)

        return torch.cat(
            [bboxes, scores, labels],
            dim=1,
        )

    @smart_inference_mode()
    def extract_one(self, img: DImage, save_img: bool = False):
        source = setup_source(img.path)
        orig_img = Image.open(source).convert("RGB")
        orig_w, orig_h = orig_img.size
        writer = ImageAnnotator(img, img_w=orig_w, img_h=orig_h)

        for size in self.input_sizes:
            image = self.prepare_image(self.resize(orig_img, size))
            curr_h, curr_w = image.shape[1:]
            # h_ratio, w_ratio = float(curr_h) / float(orig_h), float(curr_w) / float(orig_w)

            output = self.model.to(self.device)(image[None].to(self.device))
            mask = output["pred_logits"].sigmoid().max(-1)[0] > 0.3
            polygons = output["pred_boxes"][mask].cpu().detach()
            # polygons = self.scale(output["pred_boxes"][mask].cpu().detach(), curr_w, curr_h)
            scores = output["pred_logits"][mask].sigmoid().max(-1)[0].cpu()

            preds = self.cleanup_detections(polygons, scores, (curr_w, curr_h))

            if self.process_detections(
                detections=preds,
                image_tensor=image.unsqueeze(0).to(self.device),
                original_image=np.array(orig_img),
                save_img=save_img,
                source=source,
                writer=writer,
            ):
                break
        return writer.annotations


class YOLOExtractor(BaseExtractor):
    def get_model(self):
        self.device = select_device(self.device)
        return DetectMultiBackend(self.weights, device=self.device, fp16=False)

    def prepare_image(self, im):
        return (torch.from_numpy(im).to(self.device).float() / 255.0).unsqueeze(
            0
        )  # no need to swap axes

    @smart_inference_mode()
    def extract_one(self, img: DImage, save_img: bool = False):
        img_path = img.path
        source = setup_source(img_path)

        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        im0 = cv2.imread(img_path)
        writer = ImageAnnotator(img, img_w=im0.shape[1], img_h=im0.shape[0])

        for s in self.input_sizes:
            imgsz = check_img_size([s, s], s=stride)
            self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *imgsz))

            im = letterbox(im0, imgsz, stride=stride, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)

            im = self.prepare_image(im)
            pred = self.model(im, augment=False)
            pred = non_max_suppression(
                pred, CONF_THRES, IOU_THRES, None, False, max_det=1000
            )

            if self.process_detections(
                detections=pred[0],
                image_tensor=im,
                original_image=im0,
                save_img=save_img,
                source=source,
                writer=writer,
                class_names_examples=names,
            ):
                break

        return writer.annotations


class FasterRCNNExtractor(BaseExtractor):
    DEFAULT_IMG_SIZES = [800, 1400, 2000]  # used for multiscale inference

    def get_model(self):
        model = torch.load(self.weights, map_location=self.device).eval()
        return model

    @staticmethod
    def cleanup_detections(boxes, scores, labels, img):
        # Remove low confidence detections
        mask = scores > 0.4
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        output = []
        crops = []
        # Remove overlapping boxes
        for k, box in enumerate(boxes):
            x0, y0, x1, y1 = [float(f) for f in box]
            # rescale to original size
            sx, sy = img.shape[-1], img.shape[-2]
            x0, y0, x1, y1 = x0 / sx, y0 / sy, x1 / sx, y1 / sy
            x0, y0, x1, y1 = np.clip([x0, y0, x1, y1], 0, 1)
            oarea = (x1 - x0) * (y1 - y0)
            if oarea < 0.01:
                continue
            # compute intersections with previous crops
            ignore = False
            for crop in crops:
                x0_, y0_, x1_, y1_ = crop["box"]
                intersect = (max(x0, x0_), max(y0, y0_), min(x1, x1_), min(y1, y1_))
                if intersect[2] < intersect[0] or intersect[3] < intersect[1]:
                    continue
                area = (intersect[2] - intersect[0]) * (intersect[3] - intersect[1])
                if area / oarea > 0.5:
                    ignore = True
                    print(
                        f"Ignoring box {k} overlapping box {crop['k']} by {area/oarea:0.2f}"
                    )
                    break
            if ignore:
                continue
            crops.append({"k": k, "box": (x0, y0, x1, y1)})
            output.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy, scores[k], labels[k]))

        return torch.tensor(output)

    @smart_inference_mode()
    def extract_one(self, img: DImage, save_img: bool = False):
        img_path = img.path
        source = setup_source(img_path)

        original_image = Image.open(source).convert("RGB")
        writer = ImageAnnotator(
            img, img_w=original_image.size[0], img_h=original_image.size[1]
        )

        for size in self.input_sizes:
            resized_image = self.prepare_image(self.resize(original_image, size))
            preds = self.model(resized_image)

            boxes = preds[0]["boxes"].cpu().numpy()
            scores = preds[0]["scores"].cpu().numpy()
            labels = preds[0]["labels"].cpu().numpy()

            preds = self.cleanup_detections(boxes, scores, labels, resized_image)

            if self.process_detections(
                detections=preds,
                image_tensor=resized_image,
                original_image=np.array(original_image),
                save_img=save_img,
                source=source,
                writer=writer,
            ):
                break

        return writer.annotations
