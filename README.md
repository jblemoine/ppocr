# PPOCR

PPOCR is a pytorch port of the [PP-OCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/overview.html) OCR model.

PPOCR is a lightweight and fast OCR library developed by Baidu and originally written in PaddlePaddle.

This library is a pytorch port of the original PPOCR model.

It only supports the PP-OCRv3 model for now.

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
#         box=...,
#         boxes=[Box2D(top_left=Point(x, y), bottom_right=Point(x, y)), ...],
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

