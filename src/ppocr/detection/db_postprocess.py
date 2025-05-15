from typing import List

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class DBPostProcess:
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        thresh: float,
        box_thresh: float,
        max_candidates: int,
        unclip_ratio: float,
        use_dilation: bool,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.use_dilation = use_dilation
        self.dilation_kernel = np.ones((2, 2), dtype=np.uint8)

    @staticmethod
    def box_score(bitmap: np.ndarray, box_: np.ndarray) -> np.ndarray:
        """
        Compute the confidence score for a polygon : use bbox mean score as the mean score
        """
        h, w = bitmap.shape
        box = box_.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(
            bitmap[ymin : ymax + 1, xmin : xmax + 1].astype(np.float32), mask
        )[0]

    @staticmethod
    def get_mini_box(contour: np.ndarray):
        bbox = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bbox)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        points_arr = np.asarray(
            [points[index_1], points[index_2], points[index_3], points[index_4]]
        )
        min_size = min(bbox[1])
        return points_arr, min_size

    def unclip(self, box: np.ndarray) -> np.ndarray:
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        input_width: int,
        input_height: int,
    ):
        resize_height, resize_width = bitmap.shape

        # This is currently no CUDA implementation cv2.findContours.
        # One easier approach would be to reimplement the skimage find_contours, which uses
        # the marching squares algorithm
        # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
        # The rest could be implemented on CUDA (via numba or triton?)
        bitmap = bitmap.astype(np.uint8)
        if self.use_dilation:
            bitmap = cv2.dilate(bitmap, self.dilation_kernel)

        contours, _ = cv2.findContours(
            bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours[: self.max_candidates]

        boxes = []

        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            box_, sside = self.get_mini_box(contour)
            if sside < self.min_size:
                continue

            # Compute objectness
            score = self.box_score(pred, box_.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            box_ = self.unclip(box_).reshape(-1, 1, 2)

            # Remove too small boxes
            box_, sside = self.get_mini_box(box_)
            if sside < self.min_size + 2:
                continue

            # compute relative polygon to get rid of img shape
            box_[:, 0] = np.clip(
                np.round(box_[:, 0] / resize_width * input_width), 0, input_width
            )
            box_[:, 1] = np.clip(
                np.round(box_[:, 1] / resize_height * input_height), 0, input_height
            )
            boxes.append(box_)

        return np.asarray(boxes)

    def __call__(
        self,
        preds: np.ndarray,
        input_height: int,
        input_width: int,
    ) -> List[np.ndarray]:
        preds = np.squeeze(preds, axis=1)
        bit_maps = preds > self.thresh

        boxes = [
            self.bitmap_to_boxes(pred, bit_map, input_width, input_height)
            for pred, bit_map in zip(preds, bit_maps)
        ]

        return boxes
