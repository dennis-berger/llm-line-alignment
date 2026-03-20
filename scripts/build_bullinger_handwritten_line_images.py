#!/usr/bin/env python3
"""Materialize Bullinger line images from PAGE XML once for reuse."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.nntp import extract_prepared_lines

logger = logging.getLogger(__name__)


def parse_ids_arg(ids_arg: Optional[str]) -> Optional[List[str]]:
    """Parse ``--ids`` as a comma-separated list or newline-delimited file."""

    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [value.strip() for value in ids_arg.split(",") if value.strip()]


def discover_ids(data_dir: Path, ids_filter: Optional[List[str]] = None) -> list[str]:
    """Discover available Bullinger sample IDs."""

    gt_dir = data_dir / "gt"
    ids = sorted(path.stem for path in gt_dir.glob("*.txt"))
    if ids_filter is None:
        return ids
    selected = set(ids_filter)
    return [sample_id for sample_id in ids if sample_id in selected]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Bullinger line_images/ from PAGE XML once for PyLaia/M4 reuse.")
    ap.add_argument(
        "--data-dir",
        default="datasets/bullinger_handwritten",
        help="Bullinger dataset root containing gt/, transcription/, and images/",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output line-image root. Defaults to <data-dir>/line_images",
    )
    ap.add_argument("--ids", default=None, help="Comma-separated IDs or a file with one ID per line.")
    ap.add_argument("--pad", type=int, default=8, help="Padding in pixels around each XML line crop.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cropped line images.")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "line_images"
    ids = discover_ids(data_dir, parse_ids_arg(args.ids))
    if not ids:
        raise SystemExit(f"No sample IDs found under {data_dir / 'gt'}")

    failures = 0
    total_lines = 0
    for sample_id in ids:
        try:
            prepared = extract_prepared_lines(
                data_dir,
                sample_id,
                out_dir,
                overwrite=args.overwrite,
                pad=args.pad,
                prepare_mode="pagexml",
            )
        except Exception as exc:
            failures += 1
            logger.error("Failed %s: %s", sample_id, exc, exc_info=True)
            continue

        total_lines += len(prepared)
        logger.info(
            "%s -> %s (%d line image(s))",
            sample_id,
            out_dir / sample_id,
            len(prepared),
        )

    if failures:
        raise SystemExit(f"Completed with {failures} failure(s)")
    logger.info("Completed successfully: %d sample(s), %d total line image(s)", len(ids), total_lines)


if __name__ == "__main__":
    main()
