"""Optional IAM best-practices recognizer (placeholder).

This module documents a hook for plugging in the popular IAM best-practices
model. Implementers can swap in their own checkpoint loading logic here.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .recognizer import Recognizer


class IAMBestPracticesRecognizer(Recognizer):
    """Placeholder for IAM best-practices model.

    To use, replace this placeholder with code that loads your IAM model, or
    point the CLI to another recognizer (e.g., trocr_handwritten).
    """

    name = "htr_best_practices_iam"

    def __init__(self, checkpoint_path: Path | None = None):
        raise RuntimeError(
            "IAM best-practices recognizer not wired in this repo. Use --recognizer trocr_handwritten "
            "or implement IAMBestPracticesRecognizer with your checkpoint."
        )

    def recognize_lines(self, line_paths: List[Path]) -> List[str]:
        raise RuntimeError(
            "IAM best-practices recognizer not implemented. Use a different recognizer."
        )
