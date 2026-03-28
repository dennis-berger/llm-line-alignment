#!/usr/bin/env python3
"""Build children_handwritten PyLaia manifests and symbol table."""
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

from linealign.data.children_pylaia_cv import build_children_pylaia_cv_manifests, write_children_symbol_table

DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "children_handwritten"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "manifests" / "children_handwritten_pylaia_cv"
DEFAULT_SYMS_PATH = DEFAULT_OUT_DIR / "children.syms.txt"

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build children_handwritten PyLaia CV manifests.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Canonical children dataset root.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for generated manifests.")
    parser.add_argument("--syms-out", default=str(DEFAULT_SYMS_PATH), help="Where to write the generated children syms.txt.")
    parser.add_argument(
        "--fixed-height",
        type=int,
        default=None,
        help="Optional fixed height for normalized training images written under the manifest output root.",
    )
    parser.add_argument(
        "--fold",
        choices=("all", "fold_a", "fold_b", "fold_c"),
        default="all",
        help="Generate all folds or only one specific fold.",
    )
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

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    syms_path = Path(args.syms_out)
    syms_meta = write_children_symbol_table(data_dir, syms_path)
    selected_folds = None if args.fold == "all" else [args.fold]
    manifest = build_children_pylaia_cv_manifests(
        data_dir,
        out_dir,
        syms_path=syms_path,
        fixed_height=args.fixed_height,
        selected_folds=selected_folds,
    )
    summary = {
        "syms": syms_meta,
        "manifest": manifest,
    }
    summary_path = out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote children PyLaia manifest summary to %s", summary_path)
    logger.info("Wrote children PyLaia syms.txt to %s", syms_path)
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
