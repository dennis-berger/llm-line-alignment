#!/usr/bin/env python3
"""Run cross-fitted NNTP evaluation for children_handwritten."""
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
from linealign.nntp.cv_summary import summarize_cv_eval_csvs, write_cv_summary_csv

logger = logging.getLogger(__name__)


def _fold_paths(assets_root: Path, fold_name: str) -> tuple[Path, Path, Path]:
    """Return the model, checkpoint, and syms paths for one fold."""

    fold_root = assets_root / fold_name
    return fold_root / "model", fold_root / "best.ckpt", fold_root / "syms.txt"


def _load_test_ids_path(manifest_dir: Path, fold_name: str) -> Path:
    """Return the held-out sample id file for one fold."""

    ids_path = manifest_dir / fold_name / "test_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing held-out ids file for {fold_name}: {ids_path}")
    return ids_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run cross-fitted children_handwritten NNTP evaluation.")
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
    parser.add_argument("--work-root", default="outputs/nntp/children_handwritten", help="Per-fold NNTP work root.")
    parser.add_argument("--pred-stem", default="children_handwritten_predictions_nntp", help="Prediction directory stem.")
    parser.add_argument("--eval-stem", default="children_handwritten_eval_nntp", help="Evaluation CSV stem.")
    parser.add_argument("--summary-csv", default="children_handwritten_eval_nntp_cv.csv", help="Merged CV summary CSV path.")
    parser.add_argument("--fold", choices=("all", "fold_a", "fold_b", "fold_c"), default="all", help="Run all folds or one fold.")
    parser.add_argument(
        "--stop-after",
        choices=("prepare", "netout", "convert", "align", "evaluate"),
        default="evaluate",
        help="Stop stage passed through to run_nntp_eval.py.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force regeneration of the external NNTP stages.")
    parser.add_argument("--pylaia-gpus", type=int, default=0, help="PyLaia netout GPU count.")
    parser.add_argument("--pylaia-auto-select-gpus", action="store_true", help="Enable PyLaia auto-select-gpus.")
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
    work_root = Path(args.work_root)
    selected_folds = list(CHILDREN_PYLAIA_FOLDS) if args.fold == "all" else [args.fold]

    named_csvs: list[tuple[str, Path]] = []
    for fold_name in selected_folds:
        model_path, checkpoint_path, syms_path = _fold_paths(assets_root, fold_name)
        ids_path = _load_test_ids_path(manifest_dir, fold_name)
        work_dir = work_root / fold_name
        pred_dir = f"{args.pred_stem}_{fold_name}"
        eval_csv = Path(f"{args.eval_stem}_{fold_name}.csv")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_nntp_eval.py"),
            "--data-dir",
            str(data_dir),
            "--work-dir",
            str(work_dir),
            "--prepare-mode",
            "presegmented",
            "--pylaia-root",
            str(model_path.parent),
            "--pylaia-checkpoint",
            str(checkpoint_path),
            "--pylaia-syms",
            str(syms_path),
            "--pred-dir",
            pred_dir,
            "--eval-csv",
            str(eval_csv),
            "--ids",
            str(ids_path),
            "--stop-after",
            args.stop_after,
            "--pylaia-gpus",
            str(args.pylaia_gpus),
            "--log-level",
            args.log_level,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.pylaia_auto_select_gpus:
            cmd.append("--pylaia-auto-select-gpus")

        logger.info("Running children NNTP fold %s", fold_name)
        subprocess.run(cmd, check=True)
        named_csvs.append((fold_name, eval_csv))

    if args.stop_after != "evaluate" or len(named_csvs) < 2:
        return

    header, rows = summarize_cv_eval_csvs(named_csvs)
    write_cv_summary_csv(Path(args.summary_csv), header, rows)
    logger.info("Wrote children NNTP CV summary to %s", args.summary_csv)


if __name__ == "__main__":
    main()
