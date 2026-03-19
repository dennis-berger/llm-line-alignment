"""Build deterministic PyLaia CV manifests for the Washington dataset."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from linealign.nntp.symbols import filter_transcription_text, load_symbol_table
from utils.common import read_text

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")

FOLD_A_TRAIN_IDS = ("270", "271", "272", "273", "274", "275", "276", "279", "304", "305")
FOLD_B_TRAIN_IDS = ("277", "278", "300", "301", "302", "303", "306", "307", "308", "309")

WASHINGTON_PYLAIA_FOLDS = {
    "train_a": {
        "train_ids": FOLD_A_TRAIN_IDS,
        "test_ids": FOLD_B_TRAIN_IDS,
    },
    "train_b": {
        "train_ids": FOLD_B_TRAIN_IDS,
        "test_ids": FOLD_A_TRAIN_IDS,
    },
}


@dataclass(frozen=True)
class ManifestRow:
    """One Washington line paired with its GT text."""

    sample_id: str
    image_id: str
    image_path: str
    text: str
    tokenized_text: str


def _line_sort_key(path: Path) -> tuple[str, str]:
    """Sort line images stably by filename."""

    return path.stem, path.name


def _discover_line_images(line_root: Path) -> list[Path]:
    """Return all visible line images in reading order."""

    paths: list[Path] = []
    for pattern in IMAGE_GLOBS:
        paths.extend(path for path in line_root.glob(pattern) if not path.name.startswith("."))
    return sorted(paths, key=_line_sort_key)


def split_train_val_ids(sample_ids: Iterable[str], *, val_ratio: float = 0.1, seed: int = 42) -> tuple[list[str], list[str]]:
    """Split sample ids deterministically into disjoint train and validation sets."""

    ordered = sorted(sample_ids)
    if len(ordered) < 2:
        raise ValueError("Need at least two sample ids to create disjoint train/val splits")

    val_count = max(1, int(math.ceil(len(ordered) * val_ratio)))
    val_count = min(val_count, len(ordered) - 1)

    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    val_ids = sorted(shuffled[:val_count])
    train_ids = [sample_id for sample_id in ordered if sample_id not in set(val_ids)]
    return train_ids, val_ids


def build_manifest_rows(data_dir: Path, sample_ids: Iterable[str], syms_path: Path) -> list[ManifestRow]:
    """Pair Washington line images and GT lines in strict reading order."""

    symbol_table = load_symbol_table(syms_path)
    rows: list[ManifestRow] = []

    for sample_id in sample_ids:
        gt_path = data_dir / "gt" / f"{sample_id}.txt"
        line_root = data_dir / "line_images" / sample_id
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file for sample {sample_id}: {gt_path}")
        if not line_root.exists():
            raise FileNotFoundError(f"Missing line-image directory for sample {sample_id}: {line_root}")

        gt_lines = read_text(gt_path).splitlines()
        line_paths = _discover_line_images(line_root)
        if len(gt_lines) != len(line_paths):
            raise ValueError(
                f"Sample {sample_id} has {len(line_paths)} line image(s) but {len(gt_lines)} GT line(s)"
            )

        for image_path, text in zip(line_paths, gt_lines):
            filtered = filter_transcription_text(sample_id, text, symbol_table)
            if filtered.stripped_counts:
                raise ValueError(
                    f"Sample {sample_id} contains characters outside {syms_path}: {filtered.stripped_counts}"
                )
            if not filtered.tokens:
                raise ValueError(f"Sample {sample_id} line {image_path.name} is empty after filtering")

            image_id = str(Path("line_images") / sample_id / image_path.name)
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    image_id=image_id,
                    image_path=str(image_path.resolve()),
                    text=text,
                    tokenized_text=" ".join(filtered.tokens),
                )
            )

    return rows


def _write_tsv(path: Path, rows: list[ManifestRow]) -> None:
    """Write the absolute-path TSV requested for Washington training manifests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.image_path}\t{row.text}\n")


def _write_txt_table(path: Path, rows: list[ManifestRow]) -> None:
    """Write the PyLaia text table format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.image_id} {row.tokenized_text}\n")


def _write_sample_ids(path: Path, sample_ids: Iterable[str]) -> None:
    """Write one sample id per line."""

    ordered = list(sample_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(ordered)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def validate_manifest_dir(fold_dir: Path, data_dir: Path) -> dict:
    """Validate one generated fold directory."""

    required_files = (
        "train.tsv",
        "val.tsv",
        "train.txt",
        "val.txt",
        "test.txt",
        "train_ids.txt",
        "val_ids.txt",
        "test_ids.txt",
        "manifest_meta.json",
    )
    missing = [name for name in required_files if not (fold_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Fold directory {fold_dir} is missing required file(s): {', '.join(missing)}")

    meta = json.loads((fold_dir / "manifest_meta.json").read_text(encoding="utf-8"))
    train_ids = meta["train_ids"]
    val_ids = meta["val_ids"]
    test_ids = meta["test_ids"]
    if set(train_ids) & set(val_ids):
        raise ValueError(f"Train/val overlap in {fold_dir}")
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Train/test overlap in {fold_dir}")
    if set(val_ids) & set(test_ids):
        raise ValueError(f"Val/test overlap in {fold_dir}")

    for split_name in ("train", "val"):
        rows = (fold_dir / f"{split_name}.tsv").read_text(encoding="utf-8").splitlines()
        expected_count = meta["counts"][split_name]["line_count"]
        if len(rows) != expected_count:
            raise ValueError(f"{fold_dir / f'{split_name}.tsv'} has {len(rows)} row(s), expected {expected_count}")
        for row in rows:
            image_path, _text = row.split("\t", 1)
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Manifest image path does not exist: {image_path}")

    for split_name in ("train", "val", "test"):
        rows = (fold_dir / f"{split_name}.txt").read_text(encoding="utf-8").splitlines()
        expected_count = meta["counts"][split_name]["line_count"]
        if len(rows) != expected_count:
            raise ValueError(f"{fold_dir / f'{split_name}.txt'} has {len(rows)} row(s), expected {expected_count}")
        for row in rows:
            image_id, _tokens = row.split(" ", 1)
            if not (data_dir / image_id).exists():
                raise FileNotFoundError(f"PyLaia image id does not resolve under {data_dir}: {image_id}")

    return meta


def build_washington_pylaia_cv_manifests(
    data_dir: Path,
    out_dir: Path,
    *,
    syms_path: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    selected_folds: Iterable[str] | None = None,
    fold_specs: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> dict:
    """Build Washington CV manifests for PyLaia fine-tuning."""

    active_fold_specs = fold_specs or WASHINGTON_PYLAIA_FOLDS
    selected = list(selected_folds) if selected_folds is not None else list(active_fold_specs)
    unknown = [fold for fold in selected if fold not in active_fold_specs]
    if unknown:
        raise ValueError(f"Unknown Washington fold(s): {', '.join(sorted(unknown))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_dir": str(data_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "syms_path": str(syms_path.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "folds": {},
    }

    for fold_name in selected:
        fold_spec = active_fold_specs[fold_name]
        base_train_ids = list(fold_spec["train_ids"])
        base_test_ids = list(fold_spec["test_ids"])
        train_ids, val_ids = split_train_val_ids(base_train_ids, val_ratio=val_ratio, seed=seed)
        train_rows = build_manifest_rows(data_dir, train_ids, syms_path)
        val_rows = build_manifest_rows(data_dir, val_ids, syms_path)
        test_rows = build_manifest_rows(data_dir, base_test_ids, syms_path)

        fold_dir = out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(fold_dir / "train.tsv", train_rows)
        _write_tsv(fold_dir / "val.tsv", val_rows)
        _write_txt_table(fold_dir / "train.txt", train_rows)
        _write_txt_table(fold_dir / "val.txt", val_rows)
        _write_txt_table(fold_dir / "test.txt", test_rows)
        _write_sample_ids(fold_dir / "train_ids.txt", train_ids)
        _write_sample_ids(fold_dir / "val_ids.txt", val_ids)
        _write_sample_ids(fold_dir / "test_ids.txt", base_test_ids)

        fold_meta = {
            "fold": fold_name,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": base_test_ids,
            "counts": {
                "train": {
                    "sample_count": len(train_ids),
                    "line_count": len(train_rows),
                },
                "val": {
                    "sample_count": len(val_ids),
                    "line_count": len(val_rows),
                },
                "test": {
                    "sample_count": len(base_test_ids),
                    "line_count": len(test_rows),
                },
            },
        }
        (fold_dir / "manifest_meta.json").write_text(
            json.dumps(fold_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        validate_manifest_dir(fold_dir, data_dir)
        manifest["folds"][fold_name] = fold_meta

    return manifest


def manifest_rows_to_dicts(rows: Iterable[ManifestRow]) -> list[dict]:
    """Expose manifest rows as plain dictionaries for tests or reporting."""

    return [asdict(row) for row in rows]
