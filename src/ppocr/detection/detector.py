from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch

from ppocr import WEIGHTS_DIR
from ppocr.detection.db_fpn import RSEFPN
from ppocr.detection.db_head import DBHead
from ppocr.detection.db_postprocess import DBPostProcess
from ppocr.models.mobilenet_v3 import MobileNetV3Det
from ppocr.types import Box2D, Detection, Point2D
from ppocr.utils import load_module, maybe_download_github_asset


def _polygone_to_box2d(polygone: np.ndarray) -> Box2D:
    left = int(np.min(polygone[:, 0]))
    right = int(np.max(polygone[:, 0]))
    top = int(np.min(polygone[:, 1]))
    bottom = int(np.max(polygone[:, 1]))

    return Box2D(
        top_left=Point2D(x=left, y=top),
        bottom_right=Point2D(x=right, y=bottom),
    )


def _crop_polygone(image: np.ndarray, polygone: np.ndarray):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    img_crop_width = int(
        max(
            float(np.linalg.norm(polygone[0] - polygone[1])),
            float(np.linalg.norm(polygone[2] - polygone[3])),
        )
    )
    img_crop_height = int(
        max(
            float(np.linalg.norm(polygone[0] - polygone[3])),
            float(np.linalg.norm(polygone[1] - polygone[2])),
        )
    )
    pts_std = np.asarray(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(polygone, pts_std)
    crop = cv2.warpPerspective(
        image,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = crop.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        crop = np.rot90(crop)
    return crop


class PPOCRv3TextDetector(torch.nn.Module):
    algorithm = "DB"

    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3Det(
            model_name="large", scale=0.5, disable_se=True, in_channels=3
        )
        self.neck = RSEFPN(
            in_channels=self.backbone.out_channels, out_channels=96, shortcut=True
        )
        self.head = DBHead(in_channels=self.neck.out_channels, k=50)

    def forward(self, images: torch.Tensor) -> List[np.ndarray]:
        x = self.backbone(images)
        x = self.neck(x)
        x = self.head(x)
        return x


weights_files = {
    "PP-OCRv3": "ch_ptocr_v3_det_infer.pth",
}


class TextDetector:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.limit_side_len = 960

        if model_name not in weights_files:
            raise ValueError(f"Model {model_name} not found")

        weights_file_path = maybe_download_github_asset(
            file_name=weights_files[model_name], output_dir=WEIGHTS_DIR
        )
        self.model = PPOCRv3TextDetector()
        self.model = load_module(self.model, weights_file_path, self.device, eval=True)

        self.postprocess = DBPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            use_dilation=False,
        )

    def preprocess(self, image: np.ndarray):
        resize_height, resize_width = self._get_resize_dim(
            height=image.shape[0], width=image.shape[1]
        )

        image = cv2.resize(image, (resize_width, resize_height))
        image = image.astype(np.float32)
        image /= 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std

        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)

        return tensor

    @lru_cache()
    def _get_resize_dim(self, height: int, width: int):
        """
        Resize image to a size multiple of 32 which is required by the network
        """

        # limit the max side
        if max(height, width) > self.limit_side_len:
            if height > width:
                ratio = float(self.limit_side_len) / height
            else:
                ratio = float(self.limit_side_len) / width
        else:
            ratio = 1

        resize_height = int(height * ratio)
        resize_width = int(width * ratio)

        resize_height = max(int(round(resize_height / 32) * 32), 32)
        resize_width = max(int(round(resize_width / 32) * 32), 32)

        return resize_height, resize_width

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> list[Detection]:
        tensor = self.preprocess(image)
        preds = self.model(tensor)
        polygones = self.postprocess(
            preds.cpu().numpy(), input_height=image.shape[0], input_width=image.shape[1]
        )[0]

        detections = [
            Detection(
                box=_polygone_to_box2d(polygone),
                crop=_crop_polygone(image=image, polygone=polygone),
            )
            for polygone in polygones
        ]
        return detections
