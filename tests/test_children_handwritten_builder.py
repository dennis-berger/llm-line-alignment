"""Tests for the children_handwritten dataset builder."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linealign.data.children_handwritten import build_children_handwritten_dataset


def _write_aligned_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda key: (not key.startswith("_"), key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_image(path: Path, size: tuple[int, int] = (80, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_builder_materializes_alias_and_numeric_line_order(tmp_path: Path) -> None:
    """Builder should resolve the raw alias and renumber source `_10` crops correctly."""

    source_dir = tmp_path / "children_original" / "alignment_tests"
    _write_aligned_csv(
        source_dir / "ground_truth" / "csv_aligned" / "children.csv",
        [
            {
                "ID": "3B-16_16-17",
                "Category": "test",
                "_0": "schön",
                "_1": "grün",
            },
            {
                "ID": "2A_11_16-17",
                "Category": "test",
                **{f"_{index}": f"line {index}" for index in range(11)},
            },
        ],
    )

    _save_image(source_dir / "data" / "images" / "3B_16_16-17.png")
    _save_image(source_dir / "data" / "images" / "2A_11_16-17.png")
    _save_image(source_dir / "output" / "disjoin" / "3B_16_16-17" / "3B_16_16-17_1.png")
    _save_image(source_dir / "output" / "disjoin" / "3B_16_16-17" / "3B_16_16-17_0.png")
    for index in range(11):
        _save_image(source_dir / "output" / "disjoin" / "2A_11_16-17" / f"2A_11_16-17_{index}.png")

    out_dir = tmp_path / "children_handwritten"
    metadata = build_children_handwritten_dataset(source_dir, out_dir, overwrite=True)

    assert metadata["sample_count"] == 2
    assert metadata["line_count"] == 13

    assert (out_dir / "images" / "3B-16_16-17" / "3B-16_16-17.png").exists()
    assert (out_dir / "gt" / "3B-16_16-17.txt").read_text(encoding="utf-8") == "schön\ngrün\n"
    assert (out_dir / "transcription" / "3B-16_16-17.txt").read_text(encoding="utf-8") == "schön grün\n"

    alias_lines = sorted((out_dir / "line_images" / "3B-16_16-17").glob("*.png"))
    assert [path.name for path in alias_lines] == [
        "3B-16_16-17_line000.png",
        "3B-16_16-17_line001.png",
    ]

    numeric_lines = sorted((out_dir / "line_images" / "2A_11_16-17").glob("*.png"))
    assert [path.name for path in numeric_lines] == [
        f"2A_11_16-17_line{index:03d}.png" for index in range(11)
    ]

    written_meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    alias_meta = next(sample for sample in written_meta["samples"] if sample["sample_id"] == "3B-16_16-17")
    assert alias_meta["doc_id"] == "3B_16"
    assert alias_meta["source_sample_id"] == "3B_16_16-17"
