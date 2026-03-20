#!/usr/bin/env python3
"""Build Washington PyLaia train/val/test manifests for 2-fold CV."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.data.washington_pylaia_cv import build_washington_pylaia_cv_manifests

DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "washington_handwritten"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "manifests" / "washington_handwritten_pylaia_cv"
DEFAULT_SYMS_PATH = REPO_ROOT / "third_party" / "pylaia-iam" / "syms.txt"

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build Washington PyLaia 2-fold CV manifests.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Canonical Washington dataset root.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for generated manifests.")
    parser.add_argument("--syms", default=str(DEFAULT_SYMS_PATH), help="PyLaia syms.txt used for label validation.")
    parser.add_argument(
        "--fixed-height",
        type=int,
        default=None,
        help="Optional fixed height for normalized training images written under the manifest output root.",
    )
    parser.add_argument(
        "--fold",
        choices=("all", "train_a", "train_b"),
        default="all",
        help="Generate both folds or only one specific fold.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio over training pages.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for the train/val page split.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    selected_folds = None if args.fold == "all" else [args.fold]
    manifest = build_washington_pylaia_cv_manifests(
        Path(args.data_dir),
        Path(args.out_dir),
        syms_path=Path(args.syms),
        fixed_height=args.fixed_height,
        val_ratio=args.val_ratio,
        seed=args.seed,
        selected_folds=selected_folds,
    )
    summary_path = Path(args.out_dir) / "manifest_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote Washington PyLaia manifest summary to %s", summary_path)
    for fold_name, fold_meta in manifest["folds"].items():
        logger.info(
            "Fold %s: train=%d lines val=%d lines test=%d lines",
            fold_name,
            fold_meta["counts"]["train"]["line_count"],
            fold_meta["counts"]["val"]["line_count"],
            fold_meta["counts"]["test"]["line_count"],
        )


if __name__ == "__main__":
    main()
