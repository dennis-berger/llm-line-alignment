#!/usr/bin/env python3
"""Generate cross-fitted PyLaia OCR artifacts for washington_handwritten."""
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

logger = logging.getLogger(__name__)
WASHINGTON_CROSSFIT_FOLDS = ("train_a", "train_b")
DEFAULT_PYLAIA_ROOT = REPO_ROOT / "third_party" / "pylaia-iam"


def _fold_checkpoint_path(assets_root: Path, fold_name: str) -> Path:
    """Return the trained checkpoint path for one fold."""

    fold_root = assets_root / f"washington_handwritten_{fold_name}"
    return fold_root / "best.ckpt"


def _load_test_ids(manifest_dir: Path, fold_name: str) -> list[str]:
    """Load the held-out sample ids for one fold."""

    meta_path = manifest_dir / fold_name / "manifest_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(meta["test_ids"])


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Generate held-out OCR artifacts for washington_handwritten.")
    parser.add_argument("--data-dir", default="datasets/washington_handwritten", help="Canonical Washington dataset root.")
    parser.add_argument(
        "--manifest-dir",
        default="outputs/manifests/washington_handwritten_pylaia_cv",
        help="Directory containing Washington PyLaia fold manifests.",
    )
    parser.add_argument(
        "--assets-root",
        default="outputs/pylaia",
        help="Directory containing washington_handwritten_<fold> subdirectories with best.ckpt outputs.",
    )
    parser.add_argument("--cache-dir", default="outputs/cache/washington_handwritten/lines", help="OCR line cache root.")
    parser.add_argument(
        "--pylaia-work-dir",
        default="outputs/cache/washington_handwritten/pylaia",
        help="Per-fold PyLaia batch root.",
    )
    parser.add_argument(
        "--pylaia-root",
        default=str(DEFAULT_PYLAIA_ROOT),
        help="Base PyLaia root containing the shared model file and default syms.txt.",
    )
    parser.add_argument(
        "--pylaia-syms",
        default=None,
        help="Optional syms.txt override. Defaults to <pylaia-root>/syms.txt.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="PyLaia batch size.")
    parser.add_argument("--fold", choices=("all",) + WASHINGTON_CROSSFIT_FOLDS, default="all", help="Run all folds or one fold.")
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
    pylaia_root = Path(args.pylaia_root)
    pylaia_syms = Path(args.pylaia_syms) if args.pylaia_syms else pylaia_root / "syms.txt"
    selected_folds = list(WASHINGTON_CROSSFIT_FOLDS) if args.fold == "all" else [args.fold]

    for fold_name in selected_folds:
        checkpoint_path = _fold_checkpoint_path(assets_root, fold_name)
        ids = _load_test_ids(manifest_dir, fold_name)
        if not ids:
            logger.warning("Skipping %s because it has no held-out ids", fold_name)
            continue

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_ocr_outputs.py"),
            "--dataset",
            "washington_handwritten",
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
            str(pylaia_root),
            "--pylaia-checkpoint",
            str(checkpoint_path),
            "--pylaia-syms",
            str(pylaia_syms),
            "--pylaia-work-dir",
            str(pylaia_work_dir / fold_name),
            "--ids",
            ",".join(ids),
            "--log-level",
            args.log_level,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        logger.info("Running Washington OCR cross-fit fold %s on %d sample(s)", fold_name, len(ids))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
