#!/usr/bin/env python3
"""Generate cross-fitted PyLaia OCR artifacts for children_handwritten."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.data.children_pylaia_cv import CHILDREN_PYLAIA_FOLDS

logger = logging.getLogger(__name__)


def _fold_paths(assets_root: Path, fold_name: str) -> tuple[Path, Path, Path]:
    """Return the model, checkpoint, and syms paths for one fold."""

    fold_root = assets_root / fold_name
    return fold_root / "model", fold_root / "best.ckpt", fold_root / "syms.txt"


def _load_test_ids(manifest_dir: Path, fold_name: str) -> list[str]:
    """Load the held-out sample ids for one fold."""

    meta_path = manifest_dir / fold_name / "manifest_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(meta["test_ids"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Generate held-out OCR artifacts for children_handwritten.")
    parser.add_argument("--data-dir", default="datasets/children_handwritten", help="Canonical children dataset root.")
    parser.add_argument(
        "--manifest-dir",
        default="outputs/manifests/children_handwritten_pylaia_cv",
        help="Directory containing children PyLaia fold manifests.",
    )
    parser.add_argument(
        "--assets-root",
        default="outputs/pylaia/children_handwritten",
        help="Directory containing one subdirectory per fold with model, best.ckpt, and syms.txt.",
    )
    parser.add_argument("--cache-dir", default="outputs/cache/children_handwritten/lines", help="OCR line cache root.")
    parser.add_argument("--pylaia-work-dir", default="outputs/cache/children_handwritten/pylaia", help="Per-fold PyLaia batch root.")
    parser.add_argument("--batch-size", type=int, default=8, help="PyLaia batch size.")
    parser.add_argument("--fold", choices=("all", "fold_a", "fold_b", "fold_c"), default="all", help="Run all folds or one fold.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate OCR for the held-out fold ids.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    assets_root = Path(args.assets_root)
    cache_dir = Path(args.cache_dir)
    pylaia_work_dir = Path(args.pylaia_work_dir)
    selected_folds = list(CHILDREN_PYLAIA_FOLDS) if args.fold == "all" else [args.fold]

    for fold_name in selected_folds:
        model_path, checkpoint_path, syms_path = _fold_paths(assets_root, fold_name)
        ids = _load_test_ids(manifest_dir, fold_name)
        if not ids:
            logger.warning("Skipping %s because it has no held-out ids", fold_name)
            continue

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_ocr_outputs.py"),
            "--dataset",
            "children_handwritten",
            "--data-dir",
            str(data_dir),
            "--segmenter",
            "none",
            "--existing-lines-dir",
            str(data_dir / "line_images"),
            "--recognizer",
            "pylaia",
            "--cache-dir",
            str(cache_dir),
            "--batch-size",
            str(args.batch_size),
            "--pylaia-root",
            str(model_path.parent),
            "--pylaia-checkpoint",
            str(checkpoint_path),
            "--pylaia-syms",
            str(syms_path),
            "--pylaia-work-dir",
            str(pylaia_work_dir / fold_name),
            "--ids",
            ",".join(ids),
            "--log-level",
            args.log_level,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        logger.info("Running children OCR cross-fit fold %s on %d sample(s)", fold_name, len(ids))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
