"""Hugging Face TrOCR recognizer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from .recognizer import Recognizer

logger = logging.getLogger(__name__)


def _chunks(seq: List[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class TrOCRRecognizer(Recognizer):
    """Line recognizer backed by Hugging Face TrOCR."""

    name = "trocr"

    def __init__(
        self,
        preset: str = "handwritten",
        model_id: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 4,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ):
        try:  # lazy import
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "TrOCR requires torch and transformers. Install with pip install torch transformers"
            ) from exc

        preset_map = {
            "printed": "microsoft/trocr-base-printed",
            "handwritten": "microsoft/trocr-base-handwritten",
        }
        if model_id is None:
            if preset not in preset_map:
                raise ValueError(f"Unknown preset {preset}")
            model_id = preset_map[preset]
        self.model_id = model_id

        self._torch = torch
        self._Image = Image
        self.device = self._select_device(device)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded TrOCR model %s on %s", model_id, self.device)

    def _select_device(self, device: str):
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda"):
            return device if self._torch.cuda.is_available() else "cpu"
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def recognize_lines(self, line_paths: List[Path]) -> List[str]:
        outputs: list[str] = []
        if not line_paths:
            return outputs

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": False,
        }

        with self._torch.inference_mode():
            for batch_paths in _chunks(line_paths, self.batch_size):
                images = [self._Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                pred_ids = self.model.generate(**inputs, **gen_kwargs)
                texts = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
                outputs.extend([t.strip() for t in texts])

        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
        return outputs
