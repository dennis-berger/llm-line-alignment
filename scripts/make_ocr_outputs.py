#!/usr/bin/env python3
"""Unified OCR/HTR generation for all datasets."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure local src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Ensure top-level helpers (utils/) are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.data.datasets import get_dataset_spec
from linealign.pipelines.make_ocr import generate_ocr_for_id
from linealign.recognition.htr_best_practices_iam import IAMBestPracticesRecognizer
from linealign.recognition.trocr import TrOCRRecognizer
from linealign.segmentation.kraken_segmenter import KrakenSegmenter
from linealign.segmentation.segmenter import PassthroughSegmenter

logger = logging.getLogger(__name__)


# ---------- helpers ----------

def parse_ids_arg(ids_arg: Optional[str]) -> Optional[List[str]]:
    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [p.strip() for p in ids_arg.split(",") if p.strip()]


def build_segmenter(name: str, existing_lines_dir: Optional[str] = None):
    if name == "kraken":
        return KrakenSegmenter()
    if name == "none":
        return PassthroughSegmenter(existing_lines_root=Path(existing_lines_dir) if existing_lines_dir else None)
    raise ValueError(f"Unknown segmenter {name}")


def build_recognizer(
    name: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    model_id: Optional[str] = None,
):
    if name == "trocr_printed":
        return TrOCRRecognizer(
            preset="printed",
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    if name == "trocr_handwritten":
        return TrOCRRecognizer(
            preset="handwritten",
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    if name == "htr_best_practices_iam":
        return IAMBestPracticesRecognizer()
    if name == "none":
        raise RuntimeError("Recognizer 'none' is not supported. Choose a real recognizer.")
    raise ValueError(f"Unknown recognizer {name}")


def main():
    ap = argparse.ArgumentParser(description="Generate OCR/HTR outputs (ocr/<id>.txt) for supported datasets.")
    ap.add_argument("--dataset", choices=[
        "bullinger_handwritten",
        "bullinger_print",
        "easy_historical",
        "IAM_handwritten",
        "IAM_print",
        "children_handwritten",
    ], default="bullinger_handwritten")
    ap.add_argument("--data-dir", default=None, help="Root containing gt/, images/, transcription/. Defaults to datasets/<dataset>.")
    ap.add_argument("--ids", default=None, help="Comma-separated IDs or path to a file with one ID per line.")
    ap.add_argument("--segmenter", choices=["kraken", "none"], default=None)
    ap.add_argument("--recognizer", choices=["trocr_printed", "trocr_handwritten", "htr_best_practices_iam", "none"], default=None)
    ap.add_argument("--device", default="auto", help="Device for recognizer, e.g., cpu or cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--recognizer-model", default=None, help="Override model id/path for TrOCR recognizers")
    ap.add_argument("--cache-dir", default=None, help="Cache directory for line crops. Default: outputs/cache/<dataset>/lines")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit pages per ID (for quick tests)")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if ocr/<id>.txt exists")
    ap.add_argument("--dry-run", action="store_true", help="List actions without running")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    ap.add_argument("--existing-lines-dir", default=None, help="Use pre-segmented line images from this root when --segmenter none")
    ap.add_argument("--no-meta", action="store_true", help="Do not write ocr/<id>.meta.json")

    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    dataset = get_dataset_spec(args.dataset, Path(args.data_dir) if args.data_dir else None)

    segmenter_name = args.segmenter or dataset.default_segmenter
    recognizer_name = args.recognizer or dataset.default_recognizer

    ids_filter = parse_ids_arg(args.ids)
    ids = dataset.list_ids(ids_filter=ids_filter)
    if not ids:
        logger.error("No IDs found for dataset %s under %s", dataset.name, dataset.data_dir)
        sys.exit(1)

    cache_root = Path(args.cache_dir) if args.cache_dir else Path("outputs/cache") / dataset.name / "lines"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        if args.recognizer_model and not recognizer_name.startswith("trocr_"):
            raise ValueError("--recognizer-model is only supported for TrOCR recognizers.")
        segmenter = build_segmenter(segmenter_name, existing_lines_dir=args.existing_lines_dir)
        recognizer = build_recognizer(
            recognizer_name,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            model_id=args.recognizer_model,
        )
    except Exception as exc:
        logger.error("Failed to initialize backends: %s", exc)
        sys.exit(1)

    logger.info(
        "Starting OCR generation: dataset=%s ids=%d segmenter=%s recognizer=%s", dataset.name, len(ids), segmenter_name, recognizer_name
    )

    failures = 0
    for sample_id in ids:
        try:
            result = generate_ocr_for_id(
                dataset=dataset,
                sample_id=sample_id,
                segmenter=segmenter,
                recognizer=recognizer,
                cache_root=cache_root,
                overwrite=args.overwrite,
                max_pages=args.max_pages,
                dry_run=args.dry_run,
                write_meta=not args.no_meta,
            )
            logger.info("Done %s -> %s", sample_id, result.get("output_path"))
        except Exception as exc:
            failures += 1
            logger.error("Failed %s: %s", sample_id, exc, exc_info=True)

    if failures:
        logger.error("Completed with %d failures", failures)
        sys.exit(1)
    logger.info("Completed successfully")


if __name__ == "__main__":
    main()
