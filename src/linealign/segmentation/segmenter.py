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
    one of these layouts, those are used directly:

    - existing_lines_root/<sample_id>/<page_stem>/*
    - existing_lines_root/<sample_id>/<page_stem>_line*.png
    - existing_lines_root/<page_stem>/*

    Otherwise the full page is copied into the cache as a single line crop.
    """

    name = "none"

    def __init__(self, existing_lines_root: Optional[Path] = None):
        self.existing_lines_root = existing_lines_root

    def _discover_images(self, line_root: Path) -> list[Path]:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        files: list[Path] = []
        for ext in exts:
            files.extend(sorted(path for path in line_root.glob(ext) if not path.name.startswith(".")))
        return files

    def _sample_id_from_image_path(self, image_path: Path) -> str:
        if image_path.parent.name == "page":
            return image_path.parent.parent.name
        return image_path.parent.name

    def _prefixed_sample_files(self, sample_dir: Path, page_stem: str) -> list[Path]:
        matched = [
            path
            for path in self._discover_images(sample_dir)
            if path.stem == page_stem
            or path.stem.startswith(f"{page_stem}_")
            or path.stem.startswith(f"{page_stem}-")
        ]
        return matched

    def _find_existing(self, image_path: Path) -> Sequence[Path]:
        if not self.existing_lines_root:
            return []

        page_stem = image_path.stem
        sample_id = self._sample_id_from_image_path(image_path)

        nested_dir = self.existing_lines_root / sample_id / page_stem
        if nested_dir.exists():
            nested_files = self._discover_images(nested_dir)
            if nested_files:
                return nested_files

        sample_dir = self.existing_lines_root / sample_id
        if sample_dir.exists():
            matched_files = self._prefixed_sample_files(sample_dir, page_stem)
            if matched_files:
                return matched_files
            if sample_id == page_stem:
                all_sample_files = self._discover_images(sample_dir)
                if all_sample_files:
                    return all_sample_files

        legacy_dir = self.existing_lines_root / page_stem
        if legacy_dir.exists():
            return self._discover_images(legacy_dir)
        return []

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
