"""Pipeline to generate OCR/HTR outputs for a dataset sample."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from utils.common import write_text

from linealign.data.datasets import DatasetSpec
from linealign.segmentation.segmenter import LineCrop, Segmenter
from linealign.recognition.recognizer import Recognizer

logger = logging.getLogger(__name__)


def _page_cache_dir(cache_root: Path, sample_id: str, image_path: Path) -> Path:
    return cache_root / sample_id / image_path.stem


def _assemble_page_text(recognized_lines: List[str]) -> str:
    return "\n".join(line.strip() for line in recognized_lines if line is not None).strip()


def generate_ocr_for_id(
    dataset: DatasetSpec,
    sample_id: str,
    segmenter: Segmenter,
    recognizer: Recognizer,
    cache_root: Path,
    overwrite: bool = False,
    max_pages: Optional[int] = None,
    dry_run: bool = False,
    write_meta: bool = True,
) -> Dict[str, object]:
    images = dataset.image_paths(sample_id)
    if not images:
        raise FileNotFoundError(f"No images found for {sample_id} under {dataset.images_root}")
    if max_pages:
        images = images[:max_pages]

    ocr_path = dataset.ocr_output_path(sample_id)
    meta_path = dataset.meta_output_path(sample_id)

    if ocr_path.exists() and not overwrite:
        logger.info("Skip %s (exists). Use --overwrite to recompute.", sample_id)
        return {
            "id": sample_id,
            "skipped": True,
            "output_path": ocr_path,
        }

    if dry_run:
        logger.info("[dry-run] would process %s with %d page(s)", sample_id, len(images))
        return {
            "id": sample_id,
            "dry_run": True,
            "num_pages": len(images),
        }

    page_texts: list[str] = []
    total_lines = 0

    for page_idx, image_path in enumerate(images):
        cache_dir = _page_cache_dir(cache_root, sample_id, image_path)
        crops: List[LineCrop] = segmenter.segment_page(Path(image_path), cache_dir)
        if not crops:
            logger.warning("No lines found on page %s for %s", image_path, sample_id)
            continue
        line_paths = [c.path for c in crops]
        rec_lines = recognizer.recognize_lines(line_paths)
        total_lines += len(rec_lines)
        page_texts.append(_assemble_page_text(rec_lines))
        logger.info("%s page %d: %d lines", sample_id, page_idx + 1, len(rec_lines))

    combined = "\n\n".join(t for t in page_texts if t).strip()
    write_text(ocr_path, combined)

    meta = {
        "id": sample_id,
        "dataset": dataset.name,
        "segmenter": getattr(segmenter, "name", segmenter.__class__.__name__),
        "recognizer": getattr(recognizer, "name", recognizer.__class__.__name__),
        "recognizer_model": getattr(recognizer, "model_id", None),
        "num_pages": len(images),
        "num_lines": total_lines,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "ocr_path": str(ocr_path),
        "transcription_path": str(dataset.transcription_path(sample_id)),
    }
    if write_meta:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": sample_id,
        "output_path": ocr_path,
        "meta_path": meta_path,
        "num_pages": len(images),
        "num_lines": total_lines,
    }
