"""Pipeline to generate OCR/HTR outputs for a dataset sample."""
from __future__ import annotations

import json
import logging
import shutil
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


def _ensure_dataset_relative_crop_path(
    dataset: DatasetSpec,
    sample_id: str,
    crop_path: Path,
    page_stem: str,
    overwrite: bool = False,
) -> str:
    dataset_root = dataset.data_dir.resolve()
    resolved_crop = Path(crop_path).resolve()
    try:
        return str(resolved_crop.relative_to(dataset_root))
    except ValueError:
        portable_dir = dataset.line_images_root / sample_id
        portable_dir.mkdir(parents=True, exist_ok=True)
        portable_path = portable_dir / resolved_crop.name
        if overwrite or not portable_path.exists():
            shutil.copy2(resolved_crop, portable_path)
        return str(portable_path.resolve().relative_to(dataset_root))


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
    ocr_lines_path = dataset.ocr_lines_output_path(sample_id)
    meta_path = dataset.meta_output_path(sample_id)

    if ocr_path.exists() and ocr_lines_path.exists() and not overwrite:
        logger.info("Skip %s (exists). Use --overwrite to recompute.", sample_id)
        return {
            "id": sample_id,
            "skipped": True,
            "output_path": ocr_path,
            "ocr_lines_path": ocr_lines_path,
        }

    if dry_run:
        logger.info("[dry-run] would process %s with %d page(s)", sample_id, len(images))
        return {
            "id": sample_id,
            "dry_run": True,
            "num_pages": len(images),
            "ocr_lines_path": ocr_lines_path,
        }

    page_texts: list[str] = []
    total_lines = 0
    line_records: list[dict[str, object]] = []

    for page_idx, image_path in enumerate(images):
        cache_dir = _page_cache_dir(cache_root, sample_id, image_path)
        crops: List[LineCrop] = segmenter.segment_page(Path(image_path), cache_dir)
        if not crops:
            logger.warning("No lines found on page %s for %s", image_path, sample_id)
            continue
        line_paths = [c.path for c in crops]
        rec_lines = recognizer.recognize_lines(line_paths)
        if len(rec_lines) != len(crops):
            raise ValueError(
                f"Recognizer returned {len(rec_lines)} line(s) for {len(crops)} crop(s) on page {image_path}"
            )
        total_lines += len(rec_lines)
        page_texts.append(_assemble_page_text(rec_lines))
        for line_idx, (crop, text) in enumerate(zip(crops, rec_lines)):
            line_records.append(
                {
                    "page_index": page_idx,
                    "line_index": line_idx,
                    "text": text,
                    "crop_path": _ensure_dataset_relative_crop_path(
                        dataset,
                        sample_id,
                        crop.path,
                        image_path.stem,
                        overwrite=overwrite,
                    ),
                }
            )
        logger.info("%s page %d: %d lines", sample_id, page_idx + 1, len(rec_lines))

    combined = "\n\n".join(t for t in page_texts if t).strip()
    write_text(ocr_path, combined)
    ocr_lines_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_lines_path.write_text(
        json.dumps(
            {
                "id": sample_id,
                "dataset": dataset.name,
                "recognizer": getattr(recognizer, "name", recognizer.__class__.__name__),
                "num_pages": len(images),
                "num_lines": total_lines,
                "lines": line_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

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
        "ocr_lines_path": ocr_lines_path,
        "meta_path": meta_path,
        "num_pages": len(images),
        "num_lines": total_lines,
    }
