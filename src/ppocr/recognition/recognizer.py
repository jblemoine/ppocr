from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from ppocr import WEIGHTS_DIR
from ppocr.models.mobilenet_v1 import MobileNetV1Enhance
from ppocr.recognition.ctc_head import CTCHead
from ppocr.recognition.svtr import EncoderWithSVTR
from ppocr.types import Recognition
from ppocr.utils import load_module, maybe_download_github_asset


class CTCPostProcessor:
    def __init__(
        self,
        vocab_path: Path,
        use_space_char: bool = True,
    ) -> None:
        super(CTCPostProcessor, self).__init__()
        self.vocab = list(open(vocab_path, "r").read().split("\n"))
        if use_space_char:
            self.vocab.append(" ")
        self.ignore_vocab = "blank"
        self.ignore_token = 0
        self.vocab.insert(self.ignore_token, self.ignore_vocab)

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"

    def decode_seq(self, pred: np.ndarray) -> Tuple[str, float]:
        scores = pred.max(axis=-1)
        indexes = pred.argmax(axis=-1)

        last_character = ""
        word = []
        word_scores = []
        for index, score in zip(indexes, scores):
            character = self.vocab[index]

            if index != self.ignore_token and character != last_character:
                word.append(character)
                word_scores.append(score)

            last_character = character

        content = "".join(word)
        score = float(np.asarray(word_scores).mean()) if word_scores else 0
        return content, score

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        # Decode CTC
        return [self.decode_seq(pred) for pred in preds]


class PPOCRV3TextRecognizer(nn.Module):
    algorithm = "SVTR"

    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1Enhance(
            in_channels=3, scale=0.5, last_conv_stride=[1, 2], last_pool_type="avg"
        )

        self.neck = EncoderWithSVTR(
            in_channels=self.backbone.out_channels,
            dims=64,
            depth=2,
            hidden_dims=120,
            use_guide=True,
        )
        self.head = CTCHead(in_channels=self.neck.out_channels, out_channels=6625)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = self.neck(x)
        logits = self.head(x)
        preds = logits.softmax(dim=-1)
        feature = x.reshape(x.shape[0], -1)
        return preds, feature


weights_files = {
    "PP-OCRv3": "ch_ptocr_v3_rec_infer.pth",
}

vocab_file = {
    "PP-OCRv3": "ppocr_keys_v1.txt",
}


class TextRecognizer:
    def __init__(
        self,
        model_name: str = "PP-OCRv3",
        device: str | None = None,
    ):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.model_height = 32
        self.model_width = 320

        if model_name not in weights_files:
            raise ValueError(f"Model {model_name} not found")

        weights_file_path = maybe_download_github_asset(
            file_name=weights_files[model_name], output_dir=WEIGHTS_DIR
        )
        vocab_path = maybe_download_github_asset(
            file_name=vocab_file[model_name], output_dir=WEIGHTS_DIR
        )
        self.model = PPOCRV3TextRecognizer()
        self.model = load_module(self.model, weights_file_path, self.device, eval=True)
        self.postprocess = CTCPostProcessor(vocab_path=vocab_path, use_space_char=True)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, (self.model_width, self.model_height))
        image = image.astype(np.float32)
        image /= 255.0
        image -= 0.5
        image /= 0.5
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def _preprocess_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        return torch.stack([self._preprocess(crop) for crop in crops]).to(self.device)

    @torch.inference_mode()
    def predict(self, crops: list[np.ndarray]) -> list[Recognition]:
        tensor = self._preprocess_batch(crops)
        scores, feature = self.model(tensor)
        # feature are meant to be used later for traking across video frames
        results = self.postprocess(scores.cpu().numpy())

        return [
            Recognition(text=content, confidence=score) for (content, score) in results
        ]
