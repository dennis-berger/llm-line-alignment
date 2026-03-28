#!/usr/bin/env python3
"""Summarize the three children_handwritten NNTP CV evaluation CSVs."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.nntp.cv_summary import summarize_cv_eval_csvs, write_cv_summary_csv

DEFAULT_FOLD_A_CSV = REPO_ROOT / "children_handwritten_eval_nntp_fold_a.csv"
DEFAULT_FOLD_B_CSV = REPO_ROOT / "children_handwritten_eval_nntp_fold_b.csv"
DEFAULT_FOLD_C_CSV = REPO_ROOT / "children_handwritten_eval_nntp_fold_c.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "children_handwritten_eval_nntp_cv.csv"

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Summarize the three children NNTP CV fold evaluations.")
    parser.add_argument("--fold-a-csv", default=str(DEFAULT_FOLD_A_CSV), help="CSV for fold_a.")
    parser.add_argument("--fold-b-csv", default=str(DEFAULT_FOLD_B_CSV), help="CSV for fold_b.")
    parser.add_argument("--fold-c-csv", default=str(DEFAULT_FOLD_C_CSV), help="CSV for fold_c.")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Output CSV path.")
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

    named_csvs = [
        ("fold_a", Path(args.fold_a_csv)),
        ("fold_b", Path(args.fold_b_csv)),
        ("fold_c", Path(args.fold_c_csv)),
    ]
    header, rows = summarize_cv_eval_csvs(named_csvs)
    write_cv_summary_csv(Path(args.out_csv), header, rows)
    logger.info("Wrote children NNTP CV summary to %s", args.out_csv)


if __name__ == "__main__":
    main()
