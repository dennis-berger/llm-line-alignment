#!/usr/bin/env python3
"""Assemble thesis-ready handwritten evaluation artifacts into one dated bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


THESIS_PREDICTIONS_ROOT = Path(
    "/Users/dennisberger/Library/Mobile Documents/com~apple~CloudDocs/Dokumente/Uni/Master_Thesis/predictions"
)
EXPECTED_DATASET_COUNTS = {
    "bullinger_handwritten": 59,
    "children_handwritten": 63,
    "washington_handwritten": 20,
    "iam_handwritten_rwth_test_representative_20": 20,
}
DEFAULT_EXISTING_M5_SOURCES = {
    "bullinger_handwritten": THESIS_PREDICTIONS_ROOT / "bullinger_full_context100_2026-04-02",
    "washington_handwritten": THESIS_PREDICTIONS_ROOT / "washington_handwritten_context100_2026-04-02",
    "iam_handwritten_rwth_test_representative_20": THESIS_PREDICTIONS_ROOT / "iam_handwritten_rep20_context100_2026-04-02",
}


@dataclass(frozen=True)
class ArtifactSet:
    predictions: Path
    evaluation: Path
    checkpoints: Path | None
    traces: Path | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        required=True,
        help=(
            "Local path to the synced cluster run root produced by "
            "jobs/orchestrators/eval_thesis_handwritten_completion.sbatch."
        ),
    )
    ap.add_argument(
        "--dest-root",
        default=str(THESIS_PREDICTIONS_ROOT),
        help="Destination parent directory for the new thesis bundle.",
    )
    ap.add_argument(
        "--bundle-name",
        default=f"handwritten_thesis_bundle_{date.today().isoformat()}",
        help="Name of the new bundle directory inside --dest-root.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing bundle directory if it already exists.",
    )
    ap.add_argument(
        "--exclude-m5",
        action="append",
        default=[],
        help=(
            "Optional dataset:model exclusion for M5 artifacts that are not ready yet, "
            "for example 'bullinger_handwritten:mistral-large-2512'. Can be passed multiple times."
        ),
    )
    return ap.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.is_dir() else 0


def choose_best_dir(candidates: list[Path], pattern: str) -> Path:
    best_path: Path | None = None
    best_count = -1
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        direct_count = count_files(candidate, pattern)
        if direct_count > best_count:
            best_path = candidate
            best_count = direct_count
    if best_path is None or best_count <= 0:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find a directory matching {pattern}: {joined}")
    return best_path


def choose_first_file(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any of: {joined}")


def choose_existing_dir(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any directory in: {joined}")


def copy_any(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def discover_existing_m5_artifacts(source_root: Path) -> ArtifactSet:
    predictions = choose_best_dir(
        [source_root / "predictions", source_root / "predictions_m5_context100"],
        "*.txt",
    )
    evaluation = choose_first_file(
        [source_root / "evaluation.csv", source_root / "eval.csv", source_root / "eval_m5_context100.csv"]
    )
    checkpoints = choose_existing_dir(
        [source_root / "checkpoints", source_root / "checkpoints" / "m5_context100"],
    )
    traces = choose_best_dir(
        [source_root / "traces", source_root / "traces" / "m5_context100"],
        "*.json",
    )
    return ArtifactSet(predictions=predictions, evaluation=evaluation, checkpoints=checkpoints, traces=traces)


def discover_standard_artifacts(source_root: Path, require_traces: bool = False) -> ArtifactSet:
    predictions = source_root / "predictions"
    evaluation = source_root / "evaluation.csv"
    checkpoints = source_root / "checkpoints"
    traces = source_root / "traces" if require_traces else None

    if not predictions.is_dir():
        raise FileNotFoundError(f"Missing predictions directory: {predictions}")
    if not evaluation.is_file():
        raise FileNotFoundError(f"Missing evaluation CSV: {evaluation}")
    if not checkpoints.is_dir():
        raise FileNotFoundError(f"Missing checkpoints directory: {checkpoints}")
    if require_traces and not traces.is_dir():
        raise FileNotFoundError(f"Missing traces directory: {traces}")

    return ArtifactSet(predictions=predictions, evaluation=evaluation, checkpoints=checkpoints, traces=traces)


def verify_prediction_count(predictions_dir: Path, expected: int, label: str) -> int:
    count = count_files(predictions_dir, "*.txt")
    if count != expected:
        raise ValueError(f"{label}: expected {expected} predictions, found {count} in {predictions_dir}")
    return count


def verify_trace_count(traces_dir: Path, expected: int, label: str) -> int:
    count = count_files(traces_dir, "*.json")
    if count != expected:
        raise ValueError(f"{label}: expected {expected} traces, found {count} in {traces_dir}")
    return count


def copy_method_bundle(
    source: ArtifactSet,
    dest_root: Path,
    expected_predictions: int,
    label: str,
    require_traces: bool = False,
) -> dict[str, int]:
    predictions_count = verify_prediction_count(source.predictions, expected_predictions, label)
    trace_count = 0
    if require_traces:
        assert source.traces is not None
        trace_count = verify_trace_count(source.traces, expected_predictions, label)

    copy_any(source.predictions, dest_root / "predictions")
    copy_any(source.evaluation, dest_root / "evaluation.csv")
    if source.checkpoints is not None:
        copy_any(source.checkpoints, dest_root / "checkpoints")
    if require_traces and source.traces is not None:
        copy_any(source.traces, dest_root / "traces")

    return {
        "predictions": predictions_count,
        "traces": trace_count,
    }


def copy_children_nntp_bundle(source_root: Path, dest_root: Path) -> dict[str, int]:
    predictions_dir = source_root / "predictions"
    eval_dir = source_root / "eval"
    summary_csv = source_root / "children_handwritten_eval_nntp_cv.csv"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Missing children NNTP predictions dir: {predictions_dir}")
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing children NNTP eval dir: {eval_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing children NNTP summary CSV: {summary_csv}")

    total_predictions = sum(count_files(path, "*.txt") for path in predictions_dir.iterdir() if path.is_dir())
    if total_predictions != EXPECTED_DATASET_COUNTS["children_handwritten"]:
        raise ValueError(
            "children NNTP: expected 63 held-out predictions across folds, "
            f"found {total_predictions} in {predictions_dir}"
        )

    copy_any(predictions_dir, dest_root / "predictions")
    copy_any(eval_dir, dest_root / "eval")
    copy_any(summary_csv, dest_root / summary_csv.name)
    return {"predictions": total_predictions}


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    dest_root = Path(args.dest_root).expanduser().resolve()
    bundle_dir = dest_root / args.bundle_name
    ensure_clean_dir(bundle_dir, args.overwrite)

    excluded_m5 = set(args.exclude_m5)
    manifest: dict[str, object] = {
        "bundle_name": args.bundle_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "notes": [
            "Bullinger NNTP is excluded from this bundle and should be cited from the external paper baseline.",
            "Children M2/M3/M4/M5 entries in this bundle are intended to replace the stale pre-April-03 OCR-dependent results.",
            "Experimental m5_boundary_ocr_struct outputs are excluded from this bundle.",
        ],
        "excluded_m5": sorted(excluded_m5),
        "artifacts": [],
    }

    children_method_models = {
        "m2": ["gpt-5.4-2026-03-05", "mistral-large-2512", "qwen3-vl-8b-instruct"],
        "m3": ["gpt-5.4-2026-03-05", "mistral-large-2512", "qwen3-vl-8b-instruct"],
        "m4": ["gpt-5.4-2026-03-05", "mistral-large-2512"],
    }
    for method, model_suffixes in children_method_models.items():
        for model_suffix in model_suffixes:
            source_root = run_root / "children_handwritten" / method / model_suffix
            source = discover_standard_artifacts(source_root)
            dest = bundle_dir / "children_handwritten" / method / model_suffix
            counts = copy_method_bundle(
                source,
                dest,
                EXPECTED_DATASET_COUNTS["children_handwritten"],
                f"children {method} {model_suffix}",
            )
            manifest["artifacts"].append(
                {
                    "dataset": "children_handwritten",
                    "method": method,
                    "model": model_suffix,
                    "source_root": str(source_root),
                    "dest_root": str(dest),
                    "counts": counts,
                }
            )

    nntp_source_root = run_root / "children_handwritten" / "nntp"
    nntp_dest_root = bundle_dir / "children_handwritten" / "nntp"
    nntp_counts = copy_children_nntp_bundle(nntp_source_root, nntp_dest_root)
    manifest["artifacts"].append(
        {
            "dataset": "children_handwritten",
            "method": "nntp",
            "model": "cross_fit_cv",
            "source_root": str(nntp_source_root),
            "dest_root": str(nntp_dest_root),
            "counts": nntp_counts,
        }
    )

    for dataset in EXPECTED_DATASET_COUNTS:
        for model_suffix in ["gpt-5.4", "gemini-2.5-pro", "mistral-large-2512"]:
            exclusion_key = f"{dataset}:{model_suffix}"
            if exclusion_key in excluded_m5:
                continue
            if model_suffix == "gpt-5.4" and dataset in DEFAULT_EXISTING_M5_SOURCES:
                source_root = DEFAULT_EXISTING_M5_SOURCES[dataset]
                source = discover_existing_m5_artifacts(source_root)
            else:
                source_root = run_root / dataset / "m5_context100" / model_suffix
                source = discover_standard_artifacts(source_root, require_traces=True)

            dest = bundle_dir / dataset / "m5_context100" / model_suffix
            counts = copy_method_bundle(
                source,
                dest,
                EXPECTED_DATASET_COUNTS[dataset],
                f"{dataset} m5_context100 {model_suffix}",
                require_traces=True,
            )
            manifest["artifacts"].append(
                {
                    "dataset": dataset,
                    "method": "m5_context100",
                    "model": model_suffix,
                    "source_root": str(source_root),
                    "dest_root": str(dest),
                    "counts": counts,
                }
            )

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote thesis handwritten bundle to {bundle_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
