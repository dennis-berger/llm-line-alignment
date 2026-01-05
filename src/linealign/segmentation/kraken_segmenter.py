"""Kraken-based line segmentation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .segmenter import LineCrop, Segmenter

logger = logging.getLogger(__name__)


class KrakenSegmenter(Segmenter):
    """Segment pages into line crops using kraken."""

    name = "kraken"

    def __init__(self, pad: int = 2):
        self.pad = pad
        try:  # lazy import
            from kraken import binarization, pageseg
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Kraken is not installed. Install extra: pip install .[kraken] or pip install kraken"
            ) from exc
        self._binarization = binarization
        self._pageseg = pageseg
        self._Image = Image

    def _bbox_from_points(self, pts: Sequence[Tuple[int, int]]) -> Optional[tuple[int, int, int, int]]:
        if not pts:
            return None
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def _extract_lines(self, bounds) -> list:
        """Normalize kraken segmentation output to a list of line objects/dicts."""
        if not bounds:
            return []

        # Newer kraken returns a Segmentation object with a `.lines` attribute
        if hasattr(bounds, "lines"):
            return list(getattr(bounds, "lines"))

        # Older kraken may return a dict-like
        if isinstance(bounds, dict):
            if "lines" in bounds:
                return list(bounds.get("lines", []))
            if "boxes" in bounds:  # fallback: boxes only
                return [{"bbox": box} for box in bounds.get("boxes", [])]

        return []

    def _line_bbox(self, line_entry) -> Optional[tuple[int, int, int, int]]:
        if not line_entry:
            return None

        # Attribute-style access (kraken Line objects)
        for attr in ("boundary", "bbox", "poly", "polygon"):
            if hasattr(line_entry, attr):
                val = getattr(line_entry, attr)
                if val:
                    if attr == "bbox" and len(val) == 4:
                        x1, y1, x2, y2 = val
                        return (int(x1), int(y1), int(x2), int(y2))
                    return self._bbox_from_points(val)

        if hasattr(line_entry, "baseline") and getattr(line_entry, "baseline"):
            return self._bbox_from_points(getattr(line_entry, "baseline"))

        # Dict-style access
        if isinstance(line_entry, dict):
            for key in ("boundary", "bbox", "poly", "polygon"):
                if key in line_entry and line_entry[key]:
                    if key == "bbox" and len(line_entry[key]) == 4:
                        x1, y1, x2, y2 = line_entry[key]
                        return (int(x1), int(y1), int(x2), int(y2))
                    return self._bbox_from_points(line_entry[key])
            if "baseline" in line_entry and line_entry["baseline"]:
                return self._bbox_from_points(line_entry["baseline"])

        return None

    def segment_page(self, image_path: Path, cache_dir: Path) -> List[LineCrop]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        img = self._Image.open(image_path).convert("L")
        bin_img = self._binarization.nlbin(img)
        bounds = self._pageseg.segment(bin_img)
        lines_raw = self._extract_lines(bounds)
        if not lines_raw:
            logger.warning("kraken produced no lines for %s", image_path)
            return []

        crops: list[LineCrop] = []
        for idx, line in enumerate(lines_raw):
            bbox = self._line_bbox(line)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = x2 + self.pad
            y2 = y2 + self.pad
            crop = bin_img.crop((x1, y1, x2, y2))
            out_path = cache_dir / f"{image_path.stem}_line{idx:03d}.png"
            crop.save(out_path)

            prob = None
            if hasattr(line, "prob"):
                prob = getattr(line, "prob")
            elif isinstance(line, dict):
                prob = line.get("prob", None)

            crops.append(
                LineCrop(
                    path=out_path,
                    bbox=(x1, y1, x2, y2),
                    line_index=idx,
                    confidence=prob,
                )
            )

        crops.sort(key=lambda c: (c.bbox[1] if c.bbox else 0, c.bbox[0] if c.bbox else 0))
        return crops
