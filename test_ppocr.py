import cv2
import numpy as np
import requests

from ppocr.ppocr import PPOCR
from src.ppocr.types import OCRResult


def test_ppocr():
    # Download a sample image with text
    url = (
        "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"
    )
    response = requests.get(url)
    response.raise_for_status()
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    ocr = PPOCR()
    results = ocr.predict(image)

    assert isinstance(results, list)
    assert all(isinstance(r, OCRResult) for r in results)
    # At least one result with non-empty text
    assert any(r.text.strip() for r in results)
