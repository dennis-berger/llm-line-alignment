"""Segmentation interfaces for OCR generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LineCrop:
    """Represents a single line crop on disk."""
    path: Path
    bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    line_index: Optional[int] = None
    confidence: Optional[float] = None


class Segmenter:
    """Abstract page-to-line segmenter."""

    name: str = "base"

    def segment_page(self, image_path: Path, cache_dir: Path) -> List[LineCrop]:
        raise NotImplementedError


class PassthroughSegmenter(Segmenter):
    """Returns the full page as a single line or uses pre-existing line crops.

    If `existing_lines_root` is provided and contains images under
    existing_lines_root/<page_stem>/*, those are used directly. Otherwise the
    full page is copied into the cache as a single line crop.
    """

    name = "none"

    def __init__(self, existing_lines_root: Optional[Path] = None):
        self.existing_lines_root = existing_lines_root

    def _find_existing(self, image_path: Path) -> Sequence[Path]:
        if not self.existing_lines_root:
            return []
        candidate_dir = self.existing_lines_root / image_path.stem
        if not candidate_dir.exists():
            return []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        files: list[Path] = []
        for ext in exts:
            files.extend(sorted(candidate_dir.glob(ext)))
        return files

    def segment_page(self, image_path: Path, cache_dir: Path) -> List[LineCrop]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self._find_existing(image_path))
        if existing:
            return [LineCrop(path=p, line_index=i) for i, p in enumerate(existing)]

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Pillow is required for passthrough segmentation when no pre-segmented lines are present."
            ) from exc

        img = Image.open(image_path)
        out_path = cache_dir / f"{image_path.stem}_line000.png"
        if not out_path.exists():
            img.save(out_path)
        return [LineCrop(path=out_path, line_index=0)]
