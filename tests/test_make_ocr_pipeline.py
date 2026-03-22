"""Regression tests for OCR generation pipeline edge cases."""
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

from linealign.data.datasets import get_dataset_spec
from linealign.pipelines.make_ocr import generate_ocr_for_id
from linealign.recognition.recognizer import Recognizer
from linealign.segmentation.segmenter import PassthroughSegmenter


class EchoRecognizer(Recognizer):
    """Recognizer stub returning line stems for deterministic tests."""

    name = "echo"
    model_id = "echo-test"

    def recognize_lines(self, line_paths: list[Path]) -> list[str]:
        return [path.stem for path in line_paths]


def save_image(path: Path, size: tuple[int, int] = (120, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_generate_ocr_for_id_uses_existing_line_images_without_page_images(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    (data_dir / "gt").mkdir(parents=True, exist_ok=True)
    (data_dir / "transcription").mkdir(parents=True, exist_ok=True)
    (data_dir / "gt" / "10069.txt").write_text("dummy\n", encoding="utf-8")
    (data_dir / "transcription" / "10069.txt").write_text("dummy\n", encoding="utf-8")

    save_image(data_dir / "line_images" / "10069" / "0001_line000.png")
    save_image(data_dir / "line_images" / "10069" / "0001_line001.png")
    save_image(data_dir / "line_images" / "10069" / "0002_line000.png")

    dataset = get_dataset_spec("bullinger_handwritten", data_dir)
    segmenter = PassthroughSegmenter(existing_lines_root=data_dir / "line_images")
    recognizer = EchoRecognizer()

    result = generate_ocr_for_id(
        dataset,
        "10069",
        segmenter,
        recognizer,
        cache_root=tmp_path / "cache",
        overwrite=True,
    )

    assert result["num_pages"] == 2
    assert result["num_lines"] == 3
    assert dataset.ocr_output_path("10069").read_text(encoding="utf-8") == "0001_line000\n0001_line001\n\n0002_line000"

    payload = json.loads(dataset.ocr_lines_output_path("10069").read_text(encoding="utf-8"))
    assert payload["num_pages"] == 2
    assert payload["num_lines"] == 3
    assert [line["page_index"] for line in payload["lines"]] == [0, 0, 1]
    assert [line["crop_path"] for line in payload["lines"]] == [
        "line_images/10069/0001_line000.png",
        "line_images/10069/0001_line001.png",
        "line_images/10069/0002_line000.png",
    ]


def test_generate_ocr_for_id_prefers_existing_line_images_over_empty_page_image(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    (data_dir / "gt").mkdir(parents=True, exist_ok=True)
    (data_dir / "transcription").mkdir(parents=True, exist_ok=True)
    (data_dir / "images" / "10333").mkdir(parents=True, exist_ok=True)
    (data_dir / "gt" / "10333.txt").write_text("dummy\n", encoding="utf-8")
    (data_dir / "transcription" / "10333.txt").write_text("dummy\n", encoding="utf-8")

    save_image(data_dir / "images" / "10333" / "0001.jpg")
    save_image(data_dir / "images" / "10333" / "0002.jpg")
    save_image(data_dir / "images" / "10333" / "0003.jpg")
    save_image(data_dir / "images" / "10333" / "0004.jpg")

    save_image(data_dir / "line_images" / "10333" / "0001_line000.png")
    save_image(data_dir / "line_images" / "10333" / "0002_line000.png")
    save_image(data_dir / "line_images" / "10333" / "0003_line000.png")

    dataset = get_dataset_spec("bullinger_handwritten", data_dir)
    segmenter = PassthroughSegmenter(existing_lines_root=data_dir / "line_images")
    recognizer = EchoRecognizer()

    result = generate_ocr_for_id(
        dataset,
        "10333",
        segmenter,
        recognizer,
        cache_root=tmp_path / "cache",
        overwrite=True,
    )

    assert result["num_pages"] == 3
    assert result["num_lines"] == 3

    payload = json.loads(dataset.ocr_lines_output_path("10333").read_text(encoding="utf-8"))
    assert payload["num_pages"] == 3
    assert payload["num_lines"] == 3
    assert [line["crop_path"] for line in payload["lines"]] == [
        "line_images/10333/0001_line000.png",
        "line_images/10333/0002_line000.png",
        "line_images/10333/0003_line000.png",
    ]
