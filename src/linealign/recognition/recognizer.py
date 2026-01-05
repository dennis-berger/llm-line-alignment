"""Recognition interfaces for OCR generation."""
from __future__ import annotations

from pathlib import Path
from typing import List


class Recognizer:
    """Abstract line recognizer."""

    name: str = "base"
    model_id: str | None = None

    def recognize_lines(self, line_paths: List[Path]) -> List[str]:
        raise NotImplementedError
