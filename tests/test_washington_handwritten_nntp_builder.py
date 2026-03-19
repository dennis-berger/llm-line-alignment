"""Tests for the washington_handwritten NNTP dataset builder."""
from __future__ import annotations

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

from linealign.data.washington_handwritten_nntp import build_washington_handwritten_nntp_dataset
from linealign.segmentation.segmenter import LineCrop, Segmenter


class FakeSegmenter(Segmenter):
    """Simple segmenter for builder tests."""

    name = "fake"

    def segment_page(self, image_path: Path, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        boxes = [
            (50, 60, 120, 90),
            (10, 10, 120, 35),
            (15, 40, 130, 55),
        ]
        crops = []
        for idx, bbox in enumerate(boxes):
            out_path = cache_dir / f"unsorted_{idx}.png"
            Image.new("L", (bbox[2] - bbox[0], bbox[3] - bbox[1]), color=255).save(out_path)
            crops.append(LineCrop(path=out_path, bbox=bbox, line_index=idx))
        return crops


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_image(path: Path, size: tuple[int, int] = (200, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_builder_creates_workspace_with_metadata_and_previews(tmp_path: Path):
    """The builder should materialize sample inputs and ordered crop outputs."""

    source_dir = tmp_path / "washington_handwritten"
    sample_id = "270"
    _write_text(source_dir / "gt" / f"{sample_id}.txt", "line a\nline b\nline c\n")
    _write_text(source_dir / "transcription" / f"{sample_id}.txt", "line a line b line c\n")
    _write_text(source_dir / "ocr" / f"{sample_id}.txt", "ocr a\nocr b\nocr c\n")
    _save_image(source_dir / "images" / sample_id / f"{sample_id}.jpg")

    out_dir = tmp_path / "washington_handwritten_nntp"
    manifest = build_washington_handwritten_nntp_dataset(
        source_dir=source_dir,
        out_dir=out_dir,
        link_mode="copy",
        segmenter=FakeSegmenter(),
    )

    assert manifest["sample_count"] == 1
    assert manifest["gt_line_count"] == 3
    assert manifest["detected_line_count"] == 3

    line_dir = out_dir / "line_images" / sample_id
    assert [path.name for path in sorted(line_dir.glob("*.png"))] == [
        "270_line000.png",
        "270_line001.png",
        "270_line002.png",
    ]

    preview_path = out_dir / "previews" / sample_id / "270_overlay.png"
    assert preview_path.exists()

    metadata_path = out_dir / "metadata" / f"{sample_id}.json"
    sample_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert sample_metadata["gt_line_count"] == 3
    assert sample_metadata["detected_line_count"] == 3
    assert [line["bbox"] for line in sample_metadata["pages"][0]["lines"]] == [
        [10, 10, 120, 35],
        [15, 40, 130, 55],
        [50, 60, 120, 90],
    ]

    review_status = json.loads((out_dir / "review_status.json").read_text(encoding="utf-8"))
    assert review_status["allowed_statuses"] == [
        "ok",
        "needs_merge",
        "needs_split",
        "needs_reorder",
        "redo_segmentation",
    ]
    assert review_status["samples"][0]["status"] is None
