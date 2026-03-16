#!/usr/bin/env python3
"""Run the Bullinger handwritten NNTP baseline from local PAGE XML."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure local src is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.nntp import (
    concatenate_observations,
    convert_lattice_block,
    decode_alignment_segments,
    extract_prepared_lines,
    filter_transcription_text,
    load_symbol_table,
    parse_alignment_file,
    read_boundary_map,
    split_lattice_blocks,
    write_boundary_map,
    write_observation_file,
)
from utils.evaluation import evaluate_prediction
from utils.common import read_text, write_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CSV_HEADER = [
    "id",
    "len_gt",
    "len_pred",
    "wer",
    "cer",
    "wer_norm",
    "cer_norm",
    "line_acc",
    "line_acc_norm",
    "rev_line_acc",
    "rev_line_acc_norm",
    "exact_line_precision",
    "exact_line_recall",
    "exact_line_f1",
    "exact_line_precision_norm",
    "exact_line_recall_norm",
    "exact_line_f1_norm",
]
STOP_STAGES = ("prepare", "netout", "convert", "align", "evaluate")
DEFAULT_PYLAIA_ROOT = REPO_ROOT / "third_party" / "pylaia-dennis"
DEFAULT_PYLAIA_CHECKPOINT = DEFAULT_PYLAIA_ROOT / "epoch=170-lowest_va_cer.ckpt"
DEFAULT_PYLAIA_SYMS = DEFAULT_PYLAIA_ROOT / "syms.txt"
DEFAULT_NNTP_ROOT = (REPO_ROOT.parent / "nntp").resolve()


def parse_ids_arg(ids_arg: str | None) -> list[str] | None:
    """Parse `--ids` as a comma-separated list or a newline-delimited file."""

    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [value.strip() for value in ids_arg.split(",") if value.strip()]


def discover_sample_ids(data_dir: Path, ids_arg: str | None) -> list[str]:
    """Discover the sample IDs to process."""

    parsed_ids = parse_ids_arg(ids_arg)
    if parsed_ids is not None:
        return parsed_ids
    gt_dir = data_dir / "gt"
    return sorted(path.stem for path in gt_dir.glob("*.txt"))


def ensure_file(path: Path, label: str) -> Path:
    """Require a file to exist and return its resolved path."""

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def write_lines(path: Path, lines: list[str]) -> None:
    """Write one line per entry, ending with a trailing newline when non-empty."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Compute a stable SHA-256 digest for a small text file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage_metadata(path: Path) -> dict | None:
    """Load optional stage metadata JSON."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_netout_config(
    config_path: Path,
    experiment_dir: Path,
    model_path: Path,
    pylaia_gpus: int = 0,
    auto_select_gpus: bool = False,
) -> None:
    """Write a minimal PyLaia netout config for the generated line images."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    trainer_lines = [
        "trainer:",
        f"  auto_select_gpus: {'true' if auto_select_gpus else 'false'}",
        f"  gpus: {pylaia_gpus}",
    ]
    config_path.write_text(
        "\n".join(
            [
                "common:",
                f'  experiment_dirname: "{experiment_dir.resolve()}"',
                f'  model_filename: "{model_path.resolve()}"',
                "netout:",
                "  output_transform: softmax",
                *trainer_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_stage(args, data_dir: Path, symbol_table):
    """Extract line images and build NNTP label inputs."""

    line_image_root = Path(args.work_dir) / "line_images"
    prepare_dir = Path(args.work_dir) / "prepare"
    nntp_dir = Path(args.work_dir) / "nntp"

    selected_ids = discover_sample_ids(data_dir, args.ids)
    prepared_by_id = {}
    filtered_labels = {}
    stripped_report = {}

    for sample_id in selected_ids:
        gt_path = data_dir / "gt" / f"{sample_id}.txt"
        transcription_path = data_dir / "transcription" / f"{sample_id}.txt"
        if not gt_path.exists():
            logger.warning("Skipping %s because %s is missing", sample_id, gt_path)
            continue
        if not transcription_path.exists():
            logger.warning("Skipping %s because %s is missing", sample_id, transcription_path)
            continue

        try:
            prepared_lines = extract_prepared_lines(
                data_dir,
                sample_id,
                line_image_root,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            logger.warning("Skipping %s because PAGE XML preparation failed: %s", sample_id, exc)
            continue
        if not prepared_lines:
            logger.warning("Skipping %s because no content lines were extracted from PAGE XML", sample_id)
            continue

        filtered_label = filter_transcription_text(sample_id, read_text(transcription_path), symbol_table)
        if not filtered_label.tokens:
            logger.warning("Skipping %s because no supported characters remain after syms filtering", sample_id)
            continue

        prepared_by_id[sample_id] = prepared_lines
        filtered_labels[sample_id] = filtered_label
        stripped_report[sample_id] = {
            "original_length": len(filtered_label.original_text),
            "filtered_length": len(filtered_label.filtered_text),
            "stripped_counts": filtered_label.stripped_counts,
        }

    active_ids = list(prepared_by_id)
    if not active_ids:
        raise SystemExit("No samples are runnable for the NNTP pipeline.")

    pylaia_images_path = prepare_dir / "pylaia_images.txt"
    image_lines = [
        str(record.crop_path)
        for sample_id in active_ids
        for record in prepared_by_id[sample_id]
    ]
    write_lines(pylaia_images_path, image_lines)

    ids_path = nntp_dir / "faIds.txt"
    labels_path = nntp_dir / "faLabels.txt"
    write_lines(ids_path, active_ids)
    write_lines(labels_path, [" ".join(filtered_labels[sample_id].tokens) for sample_id in active_ids])

    stripped_report_path = prepare_dir / "stripped_chars.json"
    stripped_report_path.parent.mkdir(parents=True, exist_ok=True)
    stripped_report_path.write_text(json.dumps(stripped_report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Prepared %d sample(s) and %d cropped line images",
        len(active_ids),
        len(image_lines),
    )

    return {
        "ids": active_ids,
        "prepared_by_id": prepared_by_id,
        "filtered_labels": filtered_labels,
        "pylaia_images_path": pylaia_images_path,
        "fa_ids_path": ids_path,
        "fa_labels_path": labels_path,
        "stripped_report_path": stripped_report_path,
    }


def run_pylaia_netout(args, artifacts) -> Path:
    """Run `pylaia-htr-netout` to produce the raw lattice file."""

    pylaia_exe = shutil.which("pylaia-htr-netout")
    if pylaia_exe is None:
        raise RuntimeError("pylaia-htr-netout is not installed or not on PATH")

    pylaia_root = Path(args.pylaia_root)
    checkpoint_path = ensure_file(Path(args.pylaia_checkpoint), "PyLaia checkpoint")
    model_path = ensure_file(pylaia_root / "model", "PyLaia model file")

    netout_dir = Path(args.work_dir) / "netout"
    netout_dir.mkdir(parents=True, exist_ok=True)
    raw_lattice_path = (netout_dir / "lattice.txt").resolve()
    metadata_path = netout_dir / "netout_meta.json"
    metadata = load_stage_metadata(metadata_path)
    expected_metadata = {
        "ids": artifacts["ids"],
        "pylaia_images_sha256": file_sha256(artifacts["pylaia_images_path"]),
        "checkpoint": str(Path(args.pylaia_checkpoint).resolve()),
        "pylaia_gpus": args.pylaia_gpus,
        "pylaia_auto_select_gpus": args.pylaia_auto_select_gpus,
    }
    if raw_lattice_path.exists() and not args.overwrite and metadata == expected_metadata:
        logger.info("Reusing existing raw lattice: %s", raw_lattice_path)
        return raw_lattice_path

    config_path = netout_dir / "netout_generated.yaml"
    experiment_dir = Path(args.work_dir) / "pylaia_run"
    build_netout_config(
        config_path,
        experiment_dir,
        model_path,
        pylaia_gpus=args.pylaia_gpus,
        auto_select_gpus=args.pylaia_auto_select_gpus,
    )

    cmd = [
        pylaia_exe,
        str(Path(artifacts["pylaia_images_path"]).resolve()),
        "--config",
        str(config_path.resolve()),
        "--common.checkpoint",
        str(checkpoint_path),
        "--netout.lattice",
        str(raw_lattice_path),
    ]
    logger.info("Running PyLaia netout for %d sample(s)", len(artifacts["ids"]))
    subprocess.run(cmd, check=True)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    return raw_lattice_path


def convert_stage(args, artifacts, symbol_table, raw_lattice_path: Path):
    """Split the raw lattice, convert line observations, and concatenate per letter."""

    prepared_lines = [
        record
        for sample_id in artifacts["ids"]
        for record in artifacts["prepared_by_id"][sample_id]
    ]
    split_dir = Path(args.work_dir) / "split_lattices"
    line_obs_dir = Path(args.work_dir) / "observations_lines"
    letter_obs_dir = Path(args.work_dir) / "observations_letters"
    boundary_dir = Path(args.work_dir) / "boundaries"

    split_paths = split_lattice_blocks(raw_lattice_path, split_dir, prepared_lines)
    observation_paths = {}
    letter_observation_paths = {}
    boundary_paths = {}

    for sample_id in artifacts["ids"]:
        for record in artifacts["prepared_by_id"][sample_id]:
            split_path = split_paths[record.crop_path.resolve()]
            matrix = convert_lattice_block(split_path.read_text(encoding="utf-8").splitlines(True), symbol_table)
            observation_path = line_obs_dir / sample_id / f"{record.crop_path.stem}.txt"
            write_observation_file(matrix, observation_path)
            observation_paths[record.crop_path.resolve()] = observation_path

        combined_matrix, boundaries = concatenate_observations(
            artifacts["prepared_by_id"][sample_id],
            observation_paths,
        )
        letter_observation_path = letter_obs_dir / f"{sample_id}.txt"
        boundary_path = boundary_dir / f"{sample_id}.json"
        write_observation_file(combined_matrix, letter_observation_path)
        write_boundary_map(boundaries, boundary_path)
        letter_observation_paths[sample_id] = letter_observation_path
        boundary_paths[sample_id] = boundary_path

    logger.info("Converted lattices into %d letter-level observation file(s)", len(letter_observation_paths))
    return {
        "letter_observation_paths": letter_observation_paths,
        "boundary_paths": boundary_paths,
    }


def write_nntp_params(path: Path, observation_dir: Path, recognition_dir: Path) -> None:
    """Write the NNTP `faParams.txt` file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "obsLogBase logNat",
        f"dirObservation {observation_dir.resolve()}{Path('/').as_posix()}",
        f"dirRecognition {recognition_dir.resolve()}{Path('/').as_posix()}",
        "postfixObservation txt",
        "postfixRecognition rec",
        "windowSize 400",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nntp_align(args, artifacts, converted):
    """Compile NNTP and run forced alignment on the per-letter observation files."""

    nntp_root = Path(args.nntp_root)
    source_dir = nntp_root / "src" / "andreas"
    if not source_dir.exists():
        raise FileNotFoundError(f"NNTP source directory not found: {source_dir}")

    classes_dir = Path(args.work_dir) / "nntp" / "classes"
    recognition_dir = Path(args.work_dir) / "nntp" / "recognitions"
    params_path = Path(args.work_dir) / "nntp" / "faParams.txt"
    metadata_path = Path(args.work_dir) / "nntp" / "align_meta.json"
    write_nntp_params(params_path, Path(args.work_dir) / "observations_letters", recognition_dir)

    recognition_dir.mkdir(parents=True, exist_ok=True)
    rec_paths = {sample_id: recognition_dir / f"{sample_id}.rec" for sample_id in artifacts["ids"]}
    metadata = load_stage_metadata(metadata_path)
    expected_metadata = {
        "ids": artifacts["ids"],
        "fa_ids_sha256": file_sha256(artifacts["fa_ids_path"]),
        "fa_labels_sha256": file_sha256(artifacts["fa_labels_path"]),
    }
    if all(path.exists() for path in rec_paths.values()) and not args.overwrite and metadata == expected_metadata:
        logger.info("Reusing existing NNTP recognitions in %s", recognition_dir)
        return rec_paths

    javac = shutil.which("javac")
    java = shutil.which("java")
    if javac is None or java is None:
        raise RuntimeError("Java and javac must be installed to run NNTP")

    classes_dir.mkdir(parents=True, exist_ok=True)
    sources = [str(path) for path in sorted(source_dir.glob("*.java"))]
    subprocess.run([javac, "-d", str(classes_dir), *sources], check=True)
    cmd = [
        java,
        "-cp",
        str(classes_dir),
        "andreas.Main",
        "align",
        str(artifacts["fa_ids_path"]),
        str(params_path),
        str(artifacts["fa_labels_path"]),
    ]
    logger.info("Running NNTP forced alignment for %d sample(s)", len(artifacts["ids"]))
    subprocess.run(cmd, check=True, cwd=nntp_root)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    return rec_paths


def write_evaluation_csv(eval_csv: Path, rows: list[list], sums: dict[str, float], n: int) -> None:
    """Write the standard evaluation CSV plus its macro average row."""

    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    with eval_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
        if n > 0:
            writer.writerow([])
            writer.writerow(
                [
                    "macro_avg",
                    "",
                    "",
                    sums["wer"] / n,
                    sums["cer"] / n,
                    sums["wer_norm"] / n,
                    sums["cer_norm"] / n,
                    sums["line_acc"] / n,
                    sums["line_acc_norm"] / n,
                    sums["rev_line_acc"] / n,
                    sums["rev_line_acc_norm"] / n,
                    sums["exact_line_precision"] / n,
                    sums["exact_line_recall"] / n,
                    sums["exact_line_f1"] / n,
                    sums["exact_line_precision_norm"] / n,
                    sums["exact_line_recall_norm"] / n,
                    sums["exact_line_f1_norm"] / n,
                ]
            )


def evaluate_stage(args, data_dir: Path, artifacts, converted, rec_paths):
    """Decode NNTP alignments into predictions and evaluate them against GT."""

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n = 0
    sums = {
        "wer": 0.0,
        "cer": 0.0,
        "wer_norm": 0.0,
        "cer_norm": 0.0,
        "line_acc": 0.0,
        "line_acc_norm": 0.0,
        "rev_line_acc": 0.0,
        "rev_line_acc_norm": 0.0,
        "exact_line_precision": 0.0,
        "exact_line_recall": 0.0,
        "exact_line_f1": 0.0,
        "exact_line_precision_norm": 0.0,
        "exact_line_recall_norm": 0.0,
        "exact_line_f1_norm": 0.0,
    }

    for sample_id in artifacts["ids"]:
        segments = parse_alignment_file(rec_paths[sample_id])
        boundaries = read_boundary_map(converted["boundary_paths"][sample_id])
        pred = decode_alignment_segments(segments, boundaries)
        write_text(pred_dir / f"{sample_id}.txt", pred)

        gt = read_text(data_dir / "gt" / f"{sample_id}.txt")
        result = evaluate_prediction(gt, pred, sample_id)

        rows.append(
            [
                result["id"],
                result["len_gt"],
                result["len_pred"],
                result["wer"],
                result["cer"],
                result["wer_whitespace_normalized"],
                result["cer_whitespace_normalized"],
                result["line_accuracy"],
                result["line_accuracy_whitespace_normalized"],
                result["line_accuracy_reverse"],
                result["line_accuracy_whitespace_normalized_reverse"],
                result["exact_line_precision"],
                result["exact_line_recall"],
                result["exact_line_f1"],
                result["exact_line_precision_norm"],
                result["exact_line_recall_norm"],
                result["exact_line_f1_norm"],
            ]
        )

        sums["wer"] += result["wer"]
        sums["cer"] += result["cer"]
        sums["wer_norm"] += result["wer_whitespace_normalized"]
        sums["cer_norm"] += result["cer_whitespace_normalized"]
        sums["line_acc"] += result["line_accuracy"]
        sums["line_acc_norm"] += result["line_accuracy_whitespace_normalized"]
        sums["rev_line_acc"] += result["line_accuracy_reverse"]
        sums["rev_line_acc_norm"] += result["line_accuracy_whitespace_normalized_reverse"]
        sums["exact_line_precision"] += result["exact_line_precision"]
        sums["exact_line_recall"] += result["exact_line_recall"]
        sums["exact_line_f1"] += result["exact_line_f1"]
        sums["exact_line_precision_norm"] += result["exact_line_precision_norm"]
        sums["exact_line_recall_norm"] += result["exact_line_recall_norm"]
        sums["exact_line_f1_norm"] += result["exact_line_f1_norm"]
        n += 1

        logger.info(
            "[OK] %s: WER=%.3f CER=%.3f LineAcc=%.3f ExactLineF1=%.3f",
            sample_id,
            result["wer"],
            result["cer"],
            result["line_accuracy"],
            result["exact_line_f1"],
        )

    eval_csv = Path(args.eval_csv)
    write_evaluation_csv(eval_csv, rows, sums, n)
    logger.info("Wrote %s with %d sample(s)", eval_csv, n)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the NNTP runner."""

    parser = argparse.ArgumentParser(description="Run the Bullinger handwritten NNTP baseline.")
    parser.add_argument("--data-dir", default="datasets/bullinger_handwritten", help="Dataset root containing gt/, transcription/, and images/.")
    parser.add_argument("--work-dir", default="outputs/nntp/bullinger_handwritten", help="Disposable working directory for line images and NNTP intermediates.")
    parser.add_argument(
        "--pylaia-root",
        default=str(DEFAULT_PYLAIA_ROOT),
        help="Directory containing the vendored PyLaia model file and syms.txt.",
    )
    parser.add_argument(
        "--pylaia-checkpoint",
        default=str(DEFAULT_PYLAIA_CHECKPOINT),
        help="PyLaia checkpoint path passed to pylaia-htr-netout.",
    )
    parser.add_argument(
        "--pylaia-syms",
        default=str(DEFAULT_PYLAIA_SYMS),
        help="PyLaia syms.txt file used for label filtering and observation conversion.",
    )
    parser.add_argument("--nntp-root", default=str(DEFAULT_NNTP_ROOT), help="NNTP repository root.")
    parser.add_argument("--pylaia-gpus", type=int, default=0, help="Number of GPUs to request for PyLaia netout.")
    parser.add_argument(
        "--pylaia-auto-select-gpus",
        action="store_true",
        help="Let PyLaia/PyTorch Lightning auto-select the requested GPUs.",
    )
    parser.add_argument("--ids", default=None, help="Comma-separated sample IDs or a file with one ID per line.")
    parser.add_argument("--pred-dir", default="bullinger_handwritten_predictions_nntp", help="Output directory for decoded NNTP predictions.")
    parser.add_argument("--eval-csv", default="bullinger_handwritten_eval_nntp.csv", help="Evaluation CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute expensive external stages even when their outputs already exist.")
    parser.add_argument("--stop-after", choices=STOP_STAGES, default="evaluate", help="Stop after the named pipeline stage.")
    return parser


def main() -> None:
    """Entry point for the NNTP baseline runner."""

    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    symbol_table = load_symbol_table(ensure_file(Path(args.pylaia_syms), "PyLaia syms.txt"))
    artifacts = prepare_stage(args, data_dir, symbol_table)
    if args.stop_after == "prepare":
        return

    raw_lattice_path = run_pylaia_netout(args, artifacts)
    if args.stop_after == "netout":
        return

    converted = convert_stage(args, artifacts, symbol_table, raw_lattice_path)
    if args.stop_after == "convert":
        return

    rec_paths = run_nntp_align(args, artifacts, converted)
    if args.stop_after == "align":
        return

    evaluate_stage(args, data_dir, artifacts, converted, rec_paths)


if __name__ == "__main__":
    main()
