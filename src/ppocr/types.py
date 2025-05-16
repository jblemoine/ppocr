from dataclasses import dataclass

import numpy as np


@dataclass
class Point2D:
    x: int
    y: int


@dataclass
class Box2D:
    top_left: Point2D
    bottom_right: Point2D


@dataclass
class Detection:
    box: Box2D
    crop: np.ndarray


@dataclass
class Recognition:
    text: str
    confidence: float


@dataclass
class OCRResult:
    box: Box2D
    text: str
    confidence: float
