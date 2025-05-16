# PPOCR

PPOCR is a PyTorch implementation of the [PP-OCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/overview.html) model.

PP-OCR models, originally developed by Baidu as part of the PaddleOCR library and implemented in PaddlePaddle, are known for their speed and lightweight design. In my experience, they have outperformed other OCR solutions such as Tesseract.

This library provides a PyTorch-based version of the original PP-OCR model.

Currently, only the PP-OCRv3 model is supported.

## Installation

```bash
pip install ppocr
```

## Usage

```python
from ppocr.ocr import PPOCR
from PIL import Image
import numpy as np

ocr = PPOCR()

image = Image.open("path/to/image.jpg")
image = np.array(image)

results = ocr.predict(image)

for result in results:
    print(result)

# Output:
# [
#     OCRResult(
#         text=...,
#         confidence=...,
#         box=Box2D(top_left=Point2D(x, y), bottom_right=Point2D(x, y)),
#     ),
#     OCRResult(..),
#     ...
# ]
```

# Credits

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)

## TODO

- [] Add support for PP-OCRv4

