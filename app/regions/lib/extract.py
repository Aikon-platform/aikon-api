import os
import sys
from pathlib import Path
from typing import Tuple, List, Any

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

from ..const import MODEL_PATH
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

    def add_region(self, x: int, y: int, w: int, h: int, conf: float, class_info: Tuple[Any, Any] | None = None):
        """
        add a region to the Annotator. if the extraction algorithm also provides classification/labelling, the class and class name of the bounding box can be passed using the `class_info` tuple. in that case, an extra "class" field will be added

        :param class_info: a tuple of `(<class_id>, <class_label>)`
        """
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
        if class_info is not None and len(class_info):
            self.annotations["crops"][-1]["class"] = { "id": class_info[0], "label": class_info[1] }


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
        class_names: str | List = "abc",
        save_class: bool = False
    ) -> bool:
        """
        extract detections and write them

        :param detections: a tensor of shape [x, 6] where the first 4 cols are a bounding box in xyxy
            (top_left + bottom_right coordinates), 5th column is the scores, 6th column is the labels
        :param image_tensor
        :param original_image
        :param save_img
        :param source
        :param writer
        :param class_names
        :param save_class
        """
        annotator = (
            Annotator(original_image, line_width=2, example=str(class_names))
            if save_img
            else None
        )

        if not len(detections):
            return False

        img_h, img_w = original_image.shape[:2]

        # rescale boxes from tensor space to pixel space
        detections[:, :4] = scale_boxes(
            image_tensor.shape[2:], detections[:, :4], original_image.shape
        ).round()

        # NOTE the original, non-numpy version of those conversions can be found here: https://github.com/Aikon-platform/aikon-api/blob/80b7b6cc71c425778c693ccf0d0d66a8f188532e/app/regions/lib/extract.py

        detections = detections.cpu().numpy()  # move to cpu is necessary to perform numpy operations

        # convert xyxy to xywh
        xywh = np.column_stack([
            detections[:, 0],  # x
            detections[:, 1],  # y
            detections[:, 2] - detections[:, 0],  # w
            detections[:, 3] - detections[:, 1]  # h
        ])

        # squarify and add margins if necessary
        if self.squarify:
            square_dim = np.minimum(np.maximum(xywh[:, 2], xywh[:, 3]), min(img_w, img_h)) # min(max(w, h), min(img_w, img_h))
            xywh[:, 0] -= (square_dim - xywh[:, 2]) // 2 # x -= (square_dim - w) / 2
            xywh[:, 1] -= (square_dim - xywh[:, 3]) // 2 # y -= (square_dim - h) / 2
            xywh[:, 2:4] = square_dim[:, None] # w = h = square_dim

        has_margin = isinstance(self.margin, (int, float)) and self.margin > 0
        has_margins = isinstance(self.margin, list) and len(self.margin) == 2 and all(isinstance(_, (int, float)) for _ in self.margin)

        if has_margin or has_margins:
            # NOTE if squarify and self.margin = 0 it doesnt matter, so squarify should not be needed right?
            if has_margins:
                mx, my = self.margin
            else:
                mx, my = self.margin, self.margin

            xywh[:, 0] -= xywh[:, 2] * mx      # left
            xywh[:, 1] -= xywh[:, 3] * my      # top
            xywh[:, 2] += xywh[:, 2] * mx * 2  # right
            xywh[:, 3] += xywh[:, 3] * my * 2  # bottom


        xywh[:, 2] = np.minimum(xywh[:, 2], img_w)  # w cannot be > img_w
        xywh[:, 3] = np.minimum(xywh[:, 3], img_h)  # h cannot be > img_h
        xywh[:, 0] = np.clip(xywh[:, 0], 0, img_w - xywh[:, 2])  # x must be >= 0 + box can't exceed right limit
        xywh[:, 1] = np.clip(xywh[:, 1], 0, img_h - xywh[:, 3])  # y must be >= 0 + box can't exceed bottom limit

        detections = np.column_stack([xywh.astype(int), detections[:, -2:]])

        for x, y, w, h, conf, cls in reversed(detections):
            cls = int(cls)
            writer.add_region(
                x, y, w, h,
                float(conf),
                ( cls, class_names[cls] ) if save_class else None
                # if `save_class`, pass the class ID and class name to `add_region`
            )

            if save_img:
                xyxy = [x, y, x + w, y + h]
                label = (
                    None if HIDE_LABEL else
                    class_names[cls] if HIDE_CONF else
                    f"{class_names[cls]} {conf:.2f}"
                )
                annotator.box_label(xyxy, label, color=colors(cls, True))

        if save_img:
            output_path = str(Path(source).parent / f"extracted_{Path(source).name}")
            cv2.imwrite(output_path, annotator.result())

        return True


class OcrExtractor(BaseExtractor):
    """all shared data and methods between `LineExtractor` and `DtlrExtractor`"""

    T = None  # defined in sub-classes
    iou_threshold = None

    @property
    def transform(self):
        return self.T.Compose(
            [
                self.T.RandomResize([800], max_size=1333),
                self.T.ToTensor(),
                self.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def prepare_image(self, img: DImage):
        image, _ = self.transform(img, None)
        return image


class LineExtractor(OcrExtractor):
    """
    ------------------------------------------------------------------------
    Line Predictor
    Copyright (c) 2024 Raphaël Baena (Imagine team - LIGM)
    Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    Copied from LinePredictor (https://github.com/raphael-baena/LinePredictor)
    ------------------------------------------------------------------------
    """
    from .line_predictor.datasets import transforms

    T = transforms
    config = LIB_ROOT / "line_predictor" / "config" / "DINO_4scale.py"
    iou_threshold = 0.8

    def get_model(self):
        from .line_predictor import build_model_main
        from .line_predictor.config.slconfig import SLConfig

        self.device = select_device(self.device)
        checkpoint = torch.load(self.weights, map_location="cpu")

        args = SLConfig.fromfile(self.config)
        args.device = self.device
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

    #TODO fix bbox extraction on finetuned model
    @staticmethod
    def poly_to_bbox(poly):
        x0, y0, x1, y1 = poly[:, 0], poly[:, 1], poly[:, -4], poly[:, -1]
        x_min, x_max = torch.min(x0, x1), torch.max(x0, x1)
        y_min, y_max = torch.min(y0, y1), torch.max(y0, y1)
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    @staticmethod
    def scale(poly, w, h):
        return poly * torch.tensor([w, h]).repeat(10)

    def scale_and_bbox(self, polygons, curr_w, curr_h):
        scaled_polygons = self.scale(polygons, curr_w, curr_h)
        bboxes = self.poly_to_bbox(scaled_polygons).to(self.device)
        return bboxes

    # bboxes: tensor of shape [x, 4]
    def cleanup_detections(self, bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor|None=None) -> torch.Tensor:
        scores = scores.to(self.device)

        # Perform Non-Maximum Suppression (NMS)
        # TODO filter nms on polygons: https://github.com/WolodjaZ/PolyGoneNMS
        nms_filter = nms(bboxes, scores, iou_threshold=self.iou_threshold).cpu()
        bboxes = bboxes[nms_filter]
        scores = scores[nms_filter].unsqueeze(1)

        if labels is None:
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
            curr_h, curr_w = image.shape[1:] #list(image.shape[2:])
            # h_ratio, w_ratio = float(curr_h) / float(orig_h), float(curr_w) / float(orig_w)

            output = self.model.to(self.device)(image[None].to(self.device))
            #error here
            mask = output["pred_logits"].sigmoid().max(-1)[0] > 0.3
            polygons = output["pred_boxes"][mask].cpu().detach()
            # polygons = self.scale(output["pred_boxes"][mask].cpu().detach(), curr_w, curr_h)
            scores = output["pred_logits"][mask].sigmoid().max(-1)[0].cpu()

            bboxes = self.scale_and_bbox(polygons, curr_w, curr_h)
            preds = self.cleanup_detections(bboxes, scores, labels=None)

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


class DtlrExtractor(OcrExtractor):
    """
    ------------------------------------------------------------------------
    General Detection-based Text Line Recognition (DTLR)
    Copyright (c) 2024 Raphaël Baena (Imagine team - LIGM)
    Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    Copied from DTLR (https://github.com/raphael-baena/DTLR)

    @article{baena2024DTLR,
        title={General Detection-based Text Line Recognition},
        author={Raphael Baena and Syrine Kalleli and Mathieu Aubry},
        booktitle={NeurIPS},
        year={2024}},
        url={https://arxiv.org/abs/2409.17095},
    }
    ------------------------------------------------------------------------
    """
    from .dtlr.datasets import transforms

    config = LIB_ROOT / "dtlr" / "config" / "HWDB_full.py"
    labels = MODEL_PATH / "labels_icdar.pkl"
    T = transforms
    postprocessors = None  # defined in get_model
    charset = None         # defined in get_model
    iou_threshold = 0.2

    def get_model(self):
        import pickle
        from torch import nn
        from .dtlr import build_model_main
        from .dtlr.util.slconfig import SLConfig

        # 1 - define config
        self.device = select_device(self.device)
        args = SLConfig.fromfile(self.config)
        args.device = self.device
        args.CTC_training = False
        args.CTC_loss_coef = 0.25
        args.fix_size = False

        # 2 - load model, checpoint and charset
        model, _, postprocessors = build_model_main(args)
        checkpoint = torch.load(self.weights, map_location="cpu")

        with open(self.labels, mode="rb") as fh:
            labels_content = pickle.load(fh)
        charset = labels_content["charset"]["all_multi"]
        args.charset = charset
        charset_size = len(args.charset)

        # 3 - define class embeddigs
        #NOTE `new_class_embed` is defined twice: 1st as a single linear layer
        # (`nn.Linear`), then as an nn.ModuleList (6 stacked linear layers)
        features_dim = model.class_embed[0].weight.data.shape[1]
        new_class_embed = nn.Linear(features_dim, charset_size, )
        new_decoder_class_embed = nn.Linear(features_dim, charset_size, )
        new_enc_out_class_embed = nn.Linear(features_dim, charset_size, )

        # always true in our case => redefines `new_class_embed`
        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed
                for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)

        model.class_embed = new_class_embed.to(self.device)
        model.transformer.decoder.class_embed = new_decoder_class_embed.to(self.device)
        model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(self.device)
        model.label_enc = nn.Embedding(charset_size + 1, features_dim).to(self.device)

        # 4 - load state dict and fix mismatches. see: https://stackoverflow.com/a/76154523
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint["model"]
        fixed_state_dict={
            k:v
            if v.size()==model_state_dict[k].size()
            else model_state_dict[k]
            for k,v in zip(model_state_dict.keys(), checkpoint_state_dict.values())
        }

        model.load_state_dict(fixed_state_dict, strict=False)  # `strict=False` => skip layers with key mismatches in the checkpoint. see: https://stackoverflow.com/a/76154523
        model.to(self.device)

        self.postprocessors = postprocessors
        self.charset = charset
        return model.eval()


    #NOTE each image is a single line region extracted using `LineExtractor`
    @smart_inference_mode()
    def extract_one(self, img: DImage = None, save_img: bool = False):
        from .dtlr.util import box_ops

        #TODO move to `OcrMixin` ? (until `tensor_img = self.prepare_image(self.resize(orig_img, size))` included)
        source = setup_source(img.path)
        orig_img = Image.open(source).convert("RGB")
        orig_w, orig_h = orig_img.size
        writer = ImageAnnotator(img, img_w=orig_w, img_h=orig_h)

        for size in self.input_sizes:

            # 1 - resize image, convert resized image to tensor
            img_resize = self.resize(orig_img, size)
            resize_w, resize_h = img_resize.size
            tensor_img = self.prepare_image(img_resize)
            tensor_w, tensor_h = tensor_img.shape[2], tensor_img.shape[1]

            # 2 - inference
            #NOTE the model outputs bounding boxes in `xyxy` format, in a 0..1 range (0 = horizontal or vertical start of line)
            output = self.model.cuda()(tensor_img[None].cuda())

            # perform NMS
            self.postprocessors['bbox'].nms_iou_threshold = self.iou_threshold
            output = self.postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

            # 3 - extract relevant boxes
            # boxes = all character bounding boxes in `bbox`
            boxes = output['boxes']
            scores = output['scores']
            labels = output['labels']

            select_mask = scores > 0.1
            boxes = boxes[select_mask]
            scores = scores[select_mask]

            # 4 - extract labels
            # create a list of labels for characters in `bbox` and convert to utf-8
            to_utf8 = np.vectorize(lambda x: bytes(x, "utf-8").decode("unicode_escape"))
            labels = labels[select_mask]
            full_charset = to_utf8(np.array(self.charset))  # the entire character set in utf8
            labels_chars = full_charset[labels.cpu()]       # the characters in each bounding box, ordered

            # remove bounding boxes whose label is `" "` (aka, don't detect spaces) (and also remove their scores and labels)
            # there are errors in assigned labels (`o` detected as `e`...), but spaces are detected correctly (with some rare errors)
            idx_no_spaces = np.where(labels_chars!=" ")[0]
            boxes = boxes[idx_no_spaces]
            scores = scores[idx_no_spaces]
            labels = labels[idx_no_spaces]

            # 5 - convert bounding boxces from 0..1 space to tensor space. reshaping to pixel space is done in `process_detections`
            final_bboxes = boxes * torch.tensor([tensor_w, tensor_h, tensor_w, tensor_h]).cuda()

            # concat to fit the data model expected by `process_detections`
            preds = torch.cat([
                final_bboxes,
                scores.unsqueeze(1),
                labels.unsqueeze(1)
            ], dim=1)

            if self.process_detections(
                detections=preds,
                image_tensor=tensor_img.unsqueeze(0).to(self.device),
                original_image=np.array(orig_img),
                save_img=save_img,
                source=source,
                writer=writer,
                class_names=full_charset,
                save_class=True
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
                class_names=names,
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
