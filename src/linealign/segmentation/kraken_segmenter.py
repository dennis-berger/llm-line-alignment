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

    def __init__(
        self,
        pad: int = 2,
        merge_lines: bool = True,
        min_line_height_ratio: float = 0.4,
        vertical_overlap_threshold: float = 0.5,
        vertical_gap_ratio: float = 0.3,
    ):
        """Initialize the Kraken segmenter.

        Args:
            pad: Padding in pixels to add around each line crop.
            merge_lines: If True, merge over-segmented lines.
            min_line_height_ratio: Minimum height as ratio of median height.
                Lines shorter than this are filtered or merged.
            vertical_overlap_threshold: Merge lines with this much vertical overlap (0-1).
            vertical_gap_ratio: Merge lines if vertical gap is less than this ratio
                of the median line height.
        """
        self.pad = pad
        self.merge_lines = merge_lines
        self.min_line_height_ratio = min_line_height_ratio
        self.vertical_overlap_threshold = vertical_overlap_threshold
        self.vertical_gap_ratio = vertical_gap_ratio
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

    def _vertical_overlap(self, bbox1: tuple, bbox2: tuple) -> float:
        """Compute vertical overlap ratio between two bboxes."""
        _, y1_a, _, y2_a = bbox1
        _, y1_b, _, y2_b = bbox2
        overlap_start = max(y1_a, y1_b)
        overlap_end = min(y2_a, y2_b)
        if overlap_end <= overlap_start:
            return 0.0
        overlap = overlap_end - overlap_start
        height_a = y2_a - y1_a
        height_b = y2_b - y1_b
        min_height = min(height_a, height_b)
        return overlap / min_height if min_height > 0 else 0.0

    def _vertical_gap(self, bbox1: tuple, bbox2: tuple) -> int:
        """Compute vertical gap between two bboxes (negative if overlapping)."""
        _, y1_a, _, y2_a = bbox1
        _, y1_b, _, y2_b = bbox2
        # bbox1 is above bbox2
        if y2_a <= y1_b:
            return y1_b - y2_a
        # bbox2 is above bbox1
        if y2_b <= y1_a:
            return y1_a - y2_b
        # They overlap
        return -1

    def _merge_bboxes(self, bboxes: List[tuple]) -> tuple:
        """Merge multiple bboxes into one encompassing bbox."""
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    def _merge_overlapping_lines(self, bboxes: List[tuple]) -> List[tuple]:
        """Merge bboxes that have significant vertical overlap or are very close.
        
        Uses adaptive thresholds based on the actual distribution of gaps and heights.
        """
        if not bboxes:
            return []

        # Sort by vertical position (top to bottom)
        sorted_bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))

        # Compute heights and use a robust reference (75th percentile)
        heights = sorted([b[3] - b[1] for b in sorted_bboxes])
        if len(heights) >= 4:
            ref_height = heights[len(heights) * 3 // 4]
        else:
            ref_height = heights[len(heights) // 2] if heights else 50

        # Pre-filter: remove very tiny fragments (< 20% of reference height)
        pre_filter_threshold = int(ref_height * 0.2)
        filtered_bboxes = [b for b in sorted_bboxes if (b[3] - b[1]) > pre_filter_threshold]
        
        if not filtered_bboxes:
            return [max(sorted_bboxes, key=lambda b: b[3] - b[1])]

        # Compute actual gaps between consecutive filtered bboxes
        gaps = []
        for i in range(len(filtered_bboxes) - 1):
            gap = self._vertical_gap(filtered_bboxes[i], filtered_bboxes[i + 1])
            if gap >= 0:  # Only positive gaps (non-overlapping)
                gaps.append(gap)

        # Adaptive gap threshold: merge only if gap is much smaller than typical
        # If gaps vary a lot, use a smaller threshold (less aggressive merging)
        if gaps:
            sorted_gaps = sorted(gaps)
            median_gap = sorted_gaps[len(sorted_gaps) // 2]
            # Only merge if gap is less than 30% of median gap (i.e., unusually close)
            # But also cap at vertical_gap_ratio * ref_height for safety
            adaptive_max_gap = min(
                int(median_gap * 0.3),  # 30% of typical gap
                int(ref_height * self.vertical_gap_ratio)  # Original ratio-based threshold
            )
        else:
            # No gaps computed (all overlapping), use original approach
            adaptive_max_gap = int(ref_height * self.vertical_gap_ratio)

        min_height = int(ref_height * self.min_line_height_ratio)

        logger.debug(
            "Adaptive thresholds: ref_height=%d, median_gap=%s, adaptive_max_gap=%d",
            ref_height, median_gap if gaps else "N/A", adaptive_max_gap
        )

        merged = []
        current_group = [filtered_bboxes[0]]

        for bbox in filtered_bboxes[1:]:
            current_merged = self._merge_bboxes(current_group)

            # Check if this bbox should be merged with current group
            overlap = self._vertical_overlap(current_merged, bbox)
            gap = self._vertical_gap(current_merged, bbox)

            # Merge if:
            # 1. Significant vertical overlap (> threshold), OR
            # 2. Gap is very small compared to typical gaps
            should_merge = (
                overlap >= self.vertical_overlap_threshold
                or (0 <= gap <= adaptive_max_gap)
            )

            if should_merge:
                current_group.append(bbox)
            else:
                merged.append(self._merge_bboxes(current_group))
                current_group = [bbox]

        if current_group:
            merged.append(self._merge_bboxes(current_group))

        # Filter out lines that are too short (likely noise)
        filtered = [b for b in merged if (b[3] - b[1]) >= min_height]

        if not filtered and merged:
            filtered = [max(merged, key=lambda b: b[3] - b[1])]

        logger.debug(
            "Line merging: %d raw -> %d pre-filtered -> %d merged -> %d final",
            len(bboxes), len(filtered_bboxes), len(merged), len(filtered)
        )

        return filtered

    def segment_page(self, image_path: Path, cache_dir: Path) -> List[LineCrop]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        img = self._Image.open(image_path).convert("L")
        bin_img = self._binarization.nlbin(img)
        bounds = self._pageseg.segment(bin_img)
        lines_raw = self._extract_lines(bounds)
        if not lines_raw:
            logger.warning("kraken produced no lines for %s", image_path)
            return []

        # Extract all bboxes first
        raw_bboxes = []
        for line in lines_raw:
            bbox = self._line_bbox(line)
            if bbox:
                raw_bboxes.append(bbox)

        if not raw_bboxes:
            logger.warning("No valid bboxes extracted for %s", image_path)
            return []

        # Apply line merging if enabled
        if self.merge_lines:
            final_bboxes = self._merge_overlapping_lines(raw_bboxes)
            logger.info(
                "Segmentation for %s: %d raw lines -> %d merged lines",
                image_path.name, len(raw_bboxes), len(final_bboxes)
            )
        else:
            final_bboxes = raw_bboxes

        # Sort by vertical position
        final_bboxes = sorted(final_bboxes, key=lambda b: (b[1], b[0]))

        # Create line crops
        crops: list[LineCrop] = []
        for idx, bbox in enumerate(final_bboxes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = x2 + self.pad
            y2 = y2 + self.pad
            crop = bin_img.crop((x1, y1, x2, y2))
            out_path = cache_dir / f"{image_path.stem}_line{idx:03d}.png"
            crop.save(out_path)

            crops.append(
                LineCrop(
                    path=out_path,
                    bbox=(x1, y1, x2, y2),
                    line_index=idx,
                    confidence=None,
                )
            )

        return crops
