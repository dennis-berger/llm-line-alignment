"""Materialize the children_handwritten dataset from the original alignment export."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_ID_ALIASES = {
    "3B-16_16-17": "3B_16_16-17",
}

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
LINE_INDEX_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")

EXAMPLE_SOURCE_DIR = Path("../children_hw_original/alignment_tests")
EXAMPLE_OUT_DIR = Path("datasets/children_handwritten")


@dataclass(frozen=True)
class ChildrenSample:
    """One aligned children sample."""

    sample_id: str
    doc_id: str
    source_sample_id: str
    gt_lines: tuple[str, ...]
    source_csv: str


def parse_ids_arg(ids_arg: str | None) -> list[str] | None:
    """Parse ``--ids`` as a comma-separated list or file."""

    if not ids_arg:
        return None
    path = Path(ids_arg)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [value.strip() for value in ids_arg.split(",") if value.strip()]


def canonical_doc_id(sample_id: str) -> str:
    """Return the canonical document id for one sample."""

    doc_id = sample_id.rsplit("_", 1)[0]
    return "3B_16" if doc_id == "3B-16" else doc_id


def resolve_source_sample_id(sample_id: str) -> str:
    """Map one canonical dataset id to the raw source id."""

    return SOURCE_ID_ALIASES.get(sample_id, sample_id)


def iter_aligned_lines(row: dict[str, str]) -> Iterable[str]:
    """Yield non-empty aligned line fragments in source CSV order."""

    # Preserve the header order from the alignment export. Sorting keys
    # lexicographically would place `_10` before `_2`.
    for key, value in row.items():
        if not key.startswith("_"):
            continue
        value = (value or "").strip()
        if value:
            yield value


def iter_children_samples(source_dir: Path) -> list[ChildrenSample]:
    """Parse all aligned CSV rows into normalized sample records."""

    csv_root = source_dir / "ground_truth" / "csv_aligned"
    if not csv_root.exists():
        raise FileNotFoundError(f"Aligned CSV directory not found: {csv_root}")

    samples: list[ChildrenSample] = []
    for csv_path in sorted(csv_root.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = row["ID"].strip()
                gt_lines = tuple(iter_aligned_lines(row))
                if not gt_lines:
                    continue
                samples.append(
                    ChildrenSample(
                        sample_id=sample_id,
                        doc_id=canonical_doc_id(sample_id),
                        source_sample_id=resolve_source_sample_id(sample_id),
                        gt_lines=gt_lines,
                        source_csv=str(csv_path.resolve()),
                    )
                )
    return samples


def _discover_line_images(line_root: Path) -> list[Path]:
    """Return visible source line crops in stable reading order."""

    line_paths: list[Path] = []
    for pattern in IMAGE_GLOBS:
        line_paths.extend(path for path in line_root.glob(pattern) if not path.name.startswith("."))

    def _sort_key(path: Path) -> tuple[str, int | str]:
        match = LINE_INDEX_RE.match(path.stem)
        if match:
            return match.group("prefix"), int(match.group("index"))
        return path.stem, path.name

    return sorted(line_paths, key=_sort_key)


def _materialize_path(source: Path, destination: Path, link_mode: str) -> None:
    """Copy or symlink one file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _reset_sample_dir(path: Path) -> None:
    """Remove one sample-owned output directory if it exists."""

    if path.exists() or path.is_symlink():
        shutil.rmtree(path)


def _write_text(path: Path, text: str) -> None:
    """Write text with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _materialize_sample(
    sample: ChildrenSample,
    *,
    source_dir: Path,
    out_dir: Path,
    link_mode: str,
    overwrite: bool,
) -> dict[str, object]:
    """Materialize one sample into the canonical dataset layout."""

    image_source = source_dir / "data" / "images" / f"{sample.source_sample_id}.png"
    if not image_source.exists():
        raise FileNotFoundError(f"Missing page image for {sample.sample_id}: {image_source}")

    line_source_root = source_dir / "output" / "disjoin" / sample.source_sample_id
    if not line_source_root.exists():
        raise FileNotFoundError(f"Missing source line-image directory for {sample.sample_id}: {line_source_root}")

    line_sources = _discover_line_images(line_source_root)
    if len(line_sources) != len(sample.gt_lines):
        raise ValueError(
            f"Sample {sample.sample_id} has {len(line_sources)} source line image(s) but {len(sample.gt_lines)} GT line(s)"
        )

    gt_path = out_dir / "gt" / f"{sample.sample_id}.txt"
    transcription_path = out_dir / "transcription" / f"{sample.sample_id}.txt"
    image_dir = out_dir / "images" / sample.sample_id
    line_dir = out_dir / "line_images" / sample.sample_id

    if overwrite:
        _reset_sample_dir(image_dir)
        _reset_sample_dir(line_dir)

    gt_payload = "\n".join(sample.gt_lines) + "\n"
    transcription_payload = " ".join(sample.gt_lines) + "\n"
    _write_text(gt_path, gt_payload)
    _write_text(transcription_path, transcription_payload)

    dataset_image = image_dir / f"{sample.sample_id}.png"
    _materialize_path(image_source, dataset_image, link_mode)

    line_dir.mkdir(parents=True, exist_ok=True)
    dataset_line_paths: list[str] = []
    for index, line_source in enumerate(line_sources):
        dataset_line = line_dir / f"{sample.sample_id}_line{index:03d}.png"
        _materialize_path(line_source, dataset_line, link_mode)
        dataset_line_paths.append(str(dataset_line))

    return {
        "sample_id": sample.sample_id,
        "doc_id": sample.doc_id,
        "source_sample_id": sample.source_sample_id,
        "source_csv": sample.source_csv,
        "source_image": str(image_source.resolve()),
        "dataset_image": str(dataset_image),
        "source_line_dir": str(line_source_root.resolve()),
        "dataset_line_dir": str(line_dir),
        "line_count": len(sample.gt_lines),
        "gt_lines": list(sample.gt_lines),
        "dataset_line_images": dataset_line_paths,
    }


def build_children_handwritten_dataset(
    source_dir: Path,
    out_dir: Path,
    *,
    ids_filter: Iterable[str] | None = None,
    link_mode: str = "copy",
    overwrite: bool = False,
) -> dict[str, object]:
    """Build the canonical children_handwritten dataset."""

    samples = iter_children_samples(source_dir)
    if ids_filter is not None:
        allowed = set(ids_filter)
        samples = [sample for sample in samples if sample.sample_id in allowed]

    if not samples:
        raise ValueError("No children_handwritten samples selected for materialization")

    metadata_samples = []
    for sample in samples:
        metadata_samples.append(
            _materialize_sample(
                sample,
                source_dir=source_dir,
                out_dir=out_dir,
                link_mode=link_mode,
                overwrite=overwrite,
            )
        )

    metadata = {
        "source_dir": str(source_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "link_mode": link_mode,
        "sample_count": len(metadata_samples),
        "line_count": sum(int(sample["line_count"]) for sample in metadata_samples),
        "source_id_aliases": dict(SOURCE_ID_ALIASES),
        "samples": metadata_samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the children dataset builder CLI."""

    parser = argparse.ArgumentParser(description="Build the canonical children_handwritten dataset.")
    parser.add_argument("--source-dir", default=str(EXAMPLE_SOURCE_DIR), help="Original alignment_tests root.")
    parser.add_argument("--out-dir", default=str(EXAMPLE_OUT_DIR), help="Output dataset directory.")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to materialize images and line images in the output dataset.",
    )
    parser.add_argument("--ids", default=None, help="Comma-separated sample ids or file with one id per line.")
    parser.add_argument("--overwrite", action="store_true", help="Replace per-sample image and line-image outputs.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_arg_parser().parse_args()
    metadata = build_children_handwritten_dataset(
        Path(args.source_dir),
        Path(args.out_dir),
        ids_filter=parse_ids_arg(args.ids),
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(
        f"Built children_handwritten dataset at {args.out_dir} "
        f"(samples={metadata['sample_count']} lines={metadata['line_count']})"
    )


if __name__ == "__main__":
    main()
