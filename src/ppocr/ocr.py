import numpy as np
import torch

from ppocr.detection.detector import TextDetector
from ppocr.recognition.recognizer import TextRecognizer
from ppocr.types import OCRResult

__all__ = ["PPOCR"]


class PPOCR:
    def __init__(
        self,
        model_name: str = "PP-OCRv3",
        threshold: float = 0.50,
        device: str | None = None,
    ):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.threshold = threshold

        self.detector = TextDetector(model_name=model_name, device=device)
        self.recognizer = TextRecognizer(model_name=model_name, device=device)

    def predict(self, image: np.ndarray) -> list[OCRResult]:
        detections = self.detector.predict(image)
        results = self.recognizer.predict(
            crops=[detection.crop for detection in detections]
        )
        return results
