from pathlib import Path

import cv2

from ppocr.ppocr import PPOCR
from ppocr.types import OCRResult


def test_ppocr():
    img_path = Path(__file__).parent / "data" / "english.png"
    image = cv2.imread(str(img_path))
    ocr = PPOCR()
    results = ocr.predict_image(image)

    assert isinstance(results, list)
    assert all(isinstance(r, OCRResult) for r in results)
    # At least one result with non-empty text
    assert any(r.text.strip() for r in results)
