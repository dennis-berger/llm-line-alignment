#!/usr/bin/env python3
"""Build an IAM dataset slice from the RWTH split definitions."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SPLIT_URL = "https://www.openslr.org/resources/56/splits.zip"
SPLIT_TO_FILENAME = {
    "train": "splits/train.uttlist",
    "validation": "splits/validation.uttlist",
    "test": "splits/test.uttlist",
}


@dataclass(frozen=True)
class IamLineRecord:
    """One official IAM line entry from ascii/lines.txt."""

    line_id: str
    form_id: str
    writer_id: str
    line_index: int
    status: str
    text: str
    line_image_path: Path


def parse_ids_arg(ids_arg: str | None) -> list[str] | None:
    """Parse --ids as a comma-separated list or a newline-delimited file."""

    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [value.strip() for value in ids_arg.split(",") if value.strip()]


def parse_line_index(line_id: str) -> int:
    """Extract the numeric line suffix from an IAM line id."""

    return int(line_id.rsplit("-", 1)[1])


def normalize_line_text(raw_text: str) -> str:
    """Convert IAM pipe-delimited tokens to plain text."""

    return raw_text.replace("|", " ").strip()


def load_rwth_split(split: str, split_url: str) -> list[str]:
    """Download and parse the RWTH form split list."""

    filename = SPLIT_TO_FILENAME[split]
    with urllib.request.urlopen(split_url, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        return [
            line.strip()
            for line in archive.read(filename).decode("utf-8").splitlines()
            if line.strip()
        ]


def load_line_records(iam_root: Path) -> dict[str, list[IamLineRecord]]:
    """Load and group official IAM line metadata by form id."""

    lines_path = iam_root / "ascii" / "lines.txt"
    if not lines_path.exists():
        raise FileNotFoundError(f"IAM lines.txt not found: {lines_path}")

    grouped: dict[str, list[IamLineRecord]] = {}
    for raw_line in lines_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        parts = raw_line.split(maxsplit=8)
        if len(parts) != 9:
            raise ValueError(f"Unexpected IAM line entry: {raw_line}")
        line_id, status = parts[0], parts[1]
        form_id = "-".join(line_id.split("-")[:2])
        writer_id = line_id.split("-")[0]
        line_index = parse_line_index(line_id)
        line_image_path = iam_root / "lines" / writer_id / form_id / f"{line_id}.png"
        record = IamLineRecord(
            line_id=line_id,
            form_id=form_id,
            writer_id=writer_id,
            line_index=line_index,
            status=status,
            text=normalize_line_text(parts[8]),
            line_image_path=line_image_path.resolve(),
        )
        grouped.setdefault(form_id, []).append(record)

    for records in grouped.values():
        records.sort(key=lambda record: record.line_index)
    return grouped


def resolve_form_image(iam_root: Path, form_id: str) -> Path:
    """Locate the IAM form image across the three official form folders."""

    for dirname in ("formsA-D", "formsE-H", "formsI-Z"):
        candidate = iam_root / dirname / f"{form_id}.png"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Form image not found for {form_id} under {iam_root}")


def materialize_path(source: Path, destination: Path, link_mode: str) -> None:
    """Create a symlink or copy at the requested destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "symlink":
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def select_form_ids(
    split_form_ids: list[str],
    ids_arg: str | None,
    max_forms: int | None,
) -> list[str]:
    """Apply optional user filters while keeping split order stable."""

    selected = split_form_ids
    requested_ids = parse_ids_arg(ids_arg)
    if requested_ids is not None:
        requested_set = set(requested_ids)
        unknown = [form_id for form_id in requested_ids if form_id not in set(split_form_ids)]
        if unknown:
            raise ValueError(f"Requested form(s) not present in the selected split: {', '.join(unknown)}")
        selected = [form_id for form_id in split_form_ids if form_id in requested_set]
    if max_forms is not None:
        selected = selected[:max_forms]
    return selected


def build_dataset(args: argparse.Namespace) -> None:
    """Materialize one RWTH split into the project dataset layout."""

    iam_root = Path(args.iam_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    split_form_ids = load_rwth_split(args.split, args.split_url)
    selected_form_ids = select_form_ids(split_form_ids, args.ids, args.max_forms)
    line_records_by_form = load_line_records(iam_root)

    status_counts = Counter()
    metadata_forms = []
    for form_id in selected_form_ids:
        if form_id not in line_records_by_form:
            raise FileNotFoundError(f"No line metadata found for RWTH form {form_id}")
        form_image_path = resolve_form_image(iam_root, form_id)
        form_records = line_records_by_form[form_id]

        gt_lines = [record.text for record in form_records]
        gt_path = out_dir / "gt" / f"{form_id}.txt"
        transcription_path = out_dir / "transcription" / f"{form_id}.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        transcription_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        transcription_path.write_text(" ".join(gt_lines) + "\n", encoding="utf-8")

        dataset_form_image = out_dir / "images" / form_id / f"{form_id}.png"
        materialize_path(form_image_path, dataset_form_image, args.link_mode)

        line_metadata = []
        line_status_counts = Counter()
        for record in form_records:
            if not record.line_image_path.exists():
                raise FileNotFoundError(f"Missing IAM line image for {record.line_id}: {record.line_image_path}")
            dataset_line_image = out_dir / "line_images" / form_id / f"{record.line_id}.png"
            materialize_path(record.line_image_path, dataset_line_image, args.link_mode)
            line_status_counts[record.status] += 1
            status_counts[record.status] += 1
            line_metadata.append(
                {
                    "line_id": record.line_id,
                    "line_index": record.line_index,
                    "status": record.status,
                    "text": record.text,
                    "source_line_image": str(record.line_image_path),
                    "dataset_line_image": str(dataset_line_image),
                }
            )

        metadata_forms.append(
            {
                "form_id": form_id,
                "split": args.split,
                "source_form_image": str(form_image_path),
                "dataset_form_image": str(dataset_form_image),
                "line_count": len(form_records),
                "status_counts": dict(line_status_counts),
                "lines": line_metadata,
            }
        )

    metadata = {
        "split": args.split,
        "split_url": args.split_url,
        "iam_root": str(iam_root),
        "out_dir": str(out_dir),
        "link_mode": args.link_mode,
        "form_count": len(metadata_forms),
        "line_count": sum(form["line_count"] for form in metadata_forms),
        "status_counts": dict(status_counts),
        "forms": metadata_forms,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Built IAM {args.split} dataset at {out_dir}")
    print(f"Forms: {metadata['form_count']}  Lines: {metadata['line_count']}")
    print(f"Statuses: {metadata['status_counts']}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description="Build an IAM dataset slice from the RWTH split files.")
    parser.add_argument("--iam-root", default="../iam/data", help="Root of the extracted IAM data directory.")
    parser.add_argument("--split", choices=tuple(SPLIT_TO_FILENAME), default="test", help="RWTH split to materialize.")
    parser.add_argument("--out-dir", default="datasets/IAM_handwritten_rwth_test", help="Output dataset directory.")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="copy",
        help="How to materialize images in the output dataset. Use copy for a portable dataset that works off-machine.",
    )
    parser.add_argument("--split-url", default=SPLIT_URL, help="URL of the RWTH split zip archive.")
    parser.add_argument("--ids", default=None, help="Comma-separated form IDs or a file with one form ID per line.")
    parser.add_argument("--max-forms", type=int, default=None, help="Optional limit for smoke tests or debug builds.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_arg_parser().parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
