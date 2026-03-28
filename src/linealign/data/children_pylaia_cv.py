"""Build deterministic PyLaia CV manifests for children_handwritten."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from linealign.nntp.symbols import SPACE_TOKEN, filter_transcription_text, load_symbol_table
from utils.common import read_text

from .children_handwritten import canonical_doc_id

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")

CHILDREN_PYLAIA_FOLDS = {
    "fold_a": {
        "train_docs": ("1A_15", "2A_11", "2B_14", "3B_16"),
        "val_docs": ("1A_8", "2A_12"),
        "test_docs": ("3B_19", "1A_17", "1A_6"),
    },
    "fold_b": {
        "train_docs": ("1A_17", "2A_11", "3B_16", "3B_19"),
        "val_docs": ("1A_6", "2A_12"),
        "test_docs": ("2B_14", "1A_15", "1A_8"),
    },
    "fold_c": {
        "train_docs": ("1A_15", "1A_6", "2B_14", "3B_19"),
        "val_docs": ("1A_17", "1A_8"),
        "test_docs": ("2A_11", "3B_16", "2A_12"),
    },
}


@dataclass(frozen=True)
class ManifestRow:
    """One children line paired with its GT text."""

    sample_id: str
    doc_id: str
    image_id: str
    image_path: str
    text: str
    tokenized_text: str


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resize_to_fixed_height(src_path: Path, dst_path: Path, fixed_height: int) -> Path:
    """Resize one line image to a fixed height while preserving aspect ratio."""

    if fixed_height <= 0:
        raise ValueError(f"fixed_height must be positive, got {fixed_height}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        width, height = image.size
        if height <= 0:
            raise ValueError(f"Image has invalid height 0: {src_path}")
        if height == fixed_height:
            normalized = image.copy()
        else:
            new_width = max(1, int(round(width * (fixed_height / height))))
            normalized = image.resize((new_width, fixed_height), resample=Image.Resampling.LANCZOS)
        normalized.save(dst_path)
    return dst_path


def _discover_line_images(line_root: Path) -> list[Path]:
    """Return all visible line images in reading order."""

    paths: list[Path] = []
    for pattern in IMAGE_GLOBS:
        paths.extend(path for path in line_root.glob(pattern) if not path.name.startswith("."))
    return sorted(paths)


def _iter_normalized_gt_lines(data_dir: Path) -> list[str]:
    """Return all GT lines with normalized whitespace."""

    lines: list[str] = []
    for gt_path in sorted((data_dir / "gt").glob("*.txt")):
        for line in read_text(gt_path).splitlines():
            normalized = " ".join(line.split())
            if normalized:
                lines.append(normalized)
    return lines


def write_children_symbol_table(data_dir: Path, syms_path: Path) -> dict[str, object]:
    """Build a deterministic children ``syms.txt`` from GT text."""

    chars = set()
    for line in _iter_normalized_gt_lines(data_dir):
        chars.update(line)

    symbols = ["<ctc>", SPACE_TOKEN, *sorted(char for char in chars if char != " ")]
    payload = "\n".join(f"{symbol} {index}" for index, symbol in enumerate(symbols)) + "\n"
    _write_text(syms_path, payload)
    return {
        "syms_path": str(syms_path.resolve()),
        "symbol_count": len(symbols),
        "alphabet": symbols[2:],
    }


def _sample_ids_by_doc(data_dir: Path) -> dict[str, list[str]]:
    """Group dataset samples by canonical document id."""

    sample_ids = sorted(path.stem for path in (data_dir / "gt").glob("*.txt"))
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        grouped.setdefault(canonical_doc_id(sample_id), []).append(sample_id)
    return grouped


def _sample_ids_for_docs(grouped_ids: dict[str, list[str]], doc_ids: Iterable[str]) -> list[str]:
    """Expand ordered document ids into ordered sample ids."""

    sample_ids: list[str] = []
    for doc_id in doc_ids:
        sample_ids.extend(grouped_ids.get(doc_id, []))
    return sample_ids


def build_manifest_rows(
    data_dir: Path,
    sample_ids: Iterable[str],
    syms_path: Path,
    *,
    prepared_line_images_root: Path | None = None,
    fixed_height: int | None = None,
) -> list[ManifestRow]:
    """Pair line images and GT lines in strict reading order."""

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

        doc_id = canonical_doc_id(sample_id)
        for image_path, text in zip(line_paths, gt_lines):
            filtered = filter_transcription_text(sample_id, text, symbol_table)
            if filtered.stripped_counts:
                raise ValueError(
                    f"Sample {sample_id} contains characters outside {syms_path}: {filtered.stripped_counts}"
                )
            if not filtered.tokens:
                raise ValueError(f"Sample {sample_id} line {image_path.name} is empty after filtering")

            image_id = str(Path("line_images") / sample_id / image_path.name)
            if prepared_line_images_root is not None:
                target_path = prepared_line_images_root / image_id
                image_path = _resize_to_fixed_height(image_path, target_path, fixed_height or 0)
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    doc_id=doc_id,
                    image_id=image_id,
                    image_path=str(image_path.resolve()),
                    text=text,
                    tokenized_text=" ".join(SPACE_TOKEN if token == "sp" else token for token in filtered.tokens),
                )
            )

    return rows


def _write_tsv(path: Path, rows: list[ManifestRow]) -> None:
    """Write the absolute-path TSV format used for training manifests."""

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


def _write_ids(path: Path, ids: Iterable[str]) -> None:
    """Write ordered ids one per line."""

    ordered = list(ids)
    payload = "\n".join(ordered)
    if payload:
        payload += "\n"
    _write_text(path, payload)


def build_children_pylaia_cv_manifests(
    data_dir: Path,
    out_dir: Path,
    *,
    syms_path: Path,
    fixed_height: int | None = None,
    selected_folds: Iterable[str] | None = None,
    fold_specs: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, object]:
    """Build the deterministic children PyLaia CV manifests."""

    active_fold_specs = fold_specs or CHILDREN_PYLAIA_FOLDS
    selected = list(selected_folds) if selected_folds is not None else list(active_fold_specs)
    unknown = [fold for fold in selected if fold not in active_fold_specs]
    if unknown:
        raise ValueError(f"Unknown children fold(s): {', '.join(sorted(unknown))}")

    grouped_ids = _sample_ids_by_doc(data_dir)
    known_docs = set(grouped_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_dir": str(data_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "syms_path": str(syms_path.resolve()),
        "fixed_height": fixed_height,
        "folds": {},
    }
    prepared_line_images_root = out_dir / "prepared_line_images" if fixed_height is not None else None

    for fold_name in selected:
        fold_spec = active_fold_specs[fold_name]
        train_docs = list(fold_spec["train_docs"])
        val_docs = list(fold_spec["val_docs"])
        test_docs = list(fold_spec["test_docs"])

        overlap = (set(train_docs) & set(val_docs)) | (set(train_docs) & set(test_docs)) | (set(val_docs) & set(test_docs))
        if overlap:
            raise ValueError(f"Fold {fold_name} has overlapping document ids: {sorted(overlap)}")

        missing_docs = (set(train_docs) | set(val_docs) | set(test_docs)) - known_docs
        if missing_docs:
            raise ValueError(f"Fold {fold_name} references unknown document ids: {sorted(missing_docs)}")

        train_ids = _sample_ids_for_docs(grouped_ids, train_docs)
        val_ids = _sample_ids_for_docs(grouped_ids, val_docs)
        test_ids = _sample_ids_for_docs(grouped_ids, test_docs)

        train_rows = build_manifest_rows(
            data_dir,
            train_ids,
            syms_path,
            prepared_line_images_root=prepared_line_images_root,
            fixed_height=fixed_height,
        )
        val_rows = build_manifest_rows(
            data_dir,
            val_ids,
            syms_path,
            prepared_line_images_root=prepared_line_images_root,
            fixed_height=fixed_height,
        )
        test_rows = build_manifest_rows(
            data_dir,
            test_ids,
            syms_path,
            prepared_line_images_root=prepared_line_images_root,
            fixed_height=fixed_height,
        )

        fold_dir = out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(fold_dir / "train.tsv", train_rows)
        _write_tsv(fold_dir / "val.tsv", val_rows)
        _write_txt_table(fold_dir / "train.txt", train_rows)
        _write_txt_table(fold_dir / "val.txt", val_rows)
        _write_txt_table(fold_dir / "test.txt", test_rows)
        _write_ids(fold_dir / "train_docs.txt", train_docs)
        _write_ids(fold_dir / "val_docs.txt", val_docs)
        _write_ids(fold_dir / "test_docs.txt", test_docs)
        _write_ids(fold_dir / "train_ids.txt", train_ids)
        _write_ids(fold_dir / "val_ids.txt", val_ids)
        _write_ids(fold_dir / "test_ids.txt", test_ids)

        fold_meta = {
            "fold": fold_name,
            "fixed_height": fixed_height,
            "pylaia_img_dirs": [str(prepared_line_images_root.resolve())] if prepared_line_images_root else [str(data_dir.resolve())],
            "train_docs": train_docs,
            "val_docs": val_docs,
            "test_docs": test_docs,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "counts": {
                "train": {
                    "doc_count": len(train_docs),
                    "sample_count": len(train_ids),
                    "line_count": len(train_rows),
                },
                "val": {
                    "doc_count": len(val_docs),
                    "sample_count": len(val_ids),
                    "line_count": len(val_rows),
                },
                "test": {
                    "doc_count": len(test_docs),
                    "sample_count": len(test_ids),
                    "line_count": len(test_rows),
                },
            },
        }
        (fold_dir / "manifest_meta.json").write_text(json.dumps(fold_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest["folds"][fold_name] = {
            **fold_meta,
            "first_train_row": asdict(train_rows[0]) if train_rows else None,
            "first_val_row": asdict(val_rows[0]) if val_rows else None,
            "first_test_row": asdict(test_rows[0]) if test_rows else None,
        }

    return manifest
