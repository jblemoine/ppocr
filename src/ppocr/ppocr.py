import numpy as np
import torch

from ppocr.detection.detector import TextDetector
from ppocr.recognition.recognizer import TextRecognizer
from ppocr.types import OCRResult

__all__ = ["PPOCR"]


class PPOCR:
    """
    PP-OCR model.

    Args:
        model_name: str
            The name of the model to use. Supported models:
            Supported models:
            - "PP-OCRv3"
        threshold: float
            The threshold to filter out low confidence results.
        device: str | None
            The device to run the model on. E.g. "cuda" or "cpu". If None, it will use "cuda" if available, otherwise "cpu".
    """

    def __init__(
        self,
        model_name: str = "PP-OCRv3",
        threshold: float = 0.50,
        device: str | None = None,
    ):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.threshold = threshold

        self.detector = TextDetector(model_name=model_name, device=self.device)
        self.recognizer = TextRecognizer(model_name=model_name, device=self.device)

    def predict_image(self, image: np.ndarray) -> list[OCRResult]:
        """
        Run OCR on a single image.

        Args:
            image: np.ndarray
                A RGB image with shape (H, W, 3).

        Returns:
            A list of OCRResult objects.
        """
        detections = self.detector.predict(image)
        crops = [detection.crop for detection in detections]
        recognitions = self.recognizer.predict(crops=crops)
        ocr_results = [
            OCRResult(
                box=detection.box,
                text=recognition.text,
                confidence=recognition.confidence,
            )
            for detection, recognition in zip(detections, recognitions)
            if recognition.confidence >= self.threshold
        ]
        return ocr_results
