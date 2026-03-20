"""Tests for Method 4 helpers and OCR-line artifact generation."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linealign.data.datasets import DatasetSpec
from linealign.pipelines.make_ocr import generate_ocr_for_id
from linealign.segmentation.segmenter import LineCrop, Segmenter
from linealign.recognition.recognizer import Recognizer
from run_eval_m4 import VLMMethod4Combiner
from src.linealign.vlm import VLMConfig
from utils.common import read_json, read_text
from utils.m4 import parse_m4_response, project_boundaries_to_transcription


def save_image(path: Path, size: tuple[int, int] = (100, 40)) -> None:
    """Create a placeholder image on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


class StubSegmenter(Segmenter):
    """Return a fixed set of line crops for one page."""

    name = "stub"

    def __init__(self, crops: list[Path]):
        self.crops = crops

    def segment_page(self, image_path: Path, cache_dir: Path) -> list[LineCrop]:
        return [LineCrop(path=crop, line_index=index) for index, crop in enumerate(self.crops)]


class StubRecognizer(Recognizer):
    """Return scripted OCR outputs matching the supplied crops."""

    name = "stub_recognizer"
    model_id = "stub-model"

    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    def recognize_lines(self, line_paths: list[Path]) -> list[str]:
        assert len(line_paths) == len(self.outputs)
        return list(self.outputs)


class ScriptedBackend:
    """Minimal backend stub for deterministic M4 tests."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.cleanup_calls = 0

    def generate(self, prompt: str, images=None) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No scripted backend responses left")
        return self.responses.pop(0)

    def cleanup(self):
        self.cleanup_calls += 1


def build_ocr_lines_payload(texts: list[str]) -> dict:
    """Build a minimal structured OCR-lines payload for tests."""

    return {
        "id": "sample",
        "dataset": "dummy",
        "recognizer": "pylaia_iam",
        "num_pages": 1,
        "num_lines": len(texts),
        "lines": [
            {
                "page_index": 0,
                "line_index": index,
                "text": text,
                "crop_path": f"line_images/sample/sample_line{index:03d}.png",
            }
            for index, text in enumerate(texts)
        ],
    }


def test_generate_ocr_for_id_writes_structured_line_artifact(tmp_path: Path):
    """OCR generation should also emit ordered, portable line hypotheses."""

    dataset_root = tmp_path / "dataset"
    sample_id = "0001"
    save_image(dataset_root / "images" / sample_id / "page_0001.png", size=(120, 60))
    (dataset_root / "transcription").mkdir(parents=True, exist_ok=True)
    (dataset_root / "transcription" / f"{sample_id}.txt").write_text("firstsecond", encoding="utf-8")

    external_crop_root = tmp_path / "external_crops"
    crop_a = external_crop_root / "page_0001_line000.png"
    crop_b = external_crop_root / "page_0001_line001.png"
    save_image(crop_a, size=(20, 10))
    save_image(crop_b, size=(30, 10))

    dataset = DatasetSpec(
        name="dummy_dataset",
        data_dir=dataset_root,
        default_segmenter="stub",
        default_recognizer="stub_recognizer",
    )

    result = generate_ocr_for_id(
        dataset=dataset,
        sample_id=sample_id,
        segmenter=StubSegmenter([crop_b, crop_a]),
        recognizer=StubRecognizer(["line-b", "line-a"]),
        cache_root=tmp_path / "cache",
        overwrite=True,
    )

    assert result["num_lines"] == 2
    assert read_text(dataset.ocr_output_path(sample_id)) == "line-b\nline-a"

    payload = read_json(dataset.ocr_lines_output_path(sample_id))
    assert payload["id"] == sample_id
    assert payload["dataset"] == "dummy_dataset"
    assert payload["recognizer"] == "stub_recognizer"
    assert payload["num_pages"] == 1
    assert payload["num_lines"] == 2
    assert [line["text"] for line in payload["lines"]] == ["line-b", "line-a"]
    assert [line["line_index"] for line in payload["lines"]] == [0, 1]
    assert all(not Path(line["crop_path"]).is_absolute() for line in payload["lines"])

    resolved_paths = [dataset_root / line["crop_path"] for line in payload["lines"]]
    assert all(path.exists() for path in resolved_paths)


def test_parse_m4_response_accepts_valid_json():
    """Valid JSON responses should parse into ordered line strings."""

    parsed = parse_m4_response('{"lines":["ab","cd"]}', expected_num_lines=2)

    assert parsed == ["ab", "cd"]


def test_m4_combiner_repairs_malformed_json():
    """A malformed first response should trigger a single repair retry."""

    backend = ScriptedBackend([
        "not json at all",
        '{"lines":["ab","cd"]}',
    ])
    combiner = VLMMethod4Combiner(VLMConfig(model_id="hf/dummy"), backend=backend)

    prediction = combiner.infer_line_breaks("abcd", build_ocr_lines_payload(["ab", "cd"]))

    assert prediction == "ab\ncd"
    assert len(backend.prompts) == 2
    assert "Your previous response was invalid." in backend.prompts[1]
    assert backend.cleanup_calls == 1


def test_m4_combiner_repairs_wrong_line_count():
    """A wrong line count should be retried once before succeeding."""

    backend = ScriptedBackend([
        '{"lines":["abcd"]}',
        '{"lines":["ab","cd"]}',
    ])
    combiner = VLMMethod4Combiner(VLMConfig(model_id="hf/dummy"), backend=backend)

    prediction = combiner.infer_line_breaks("abcd", build_ocr_lines_payload(["ab", "cd"]))

    assert prediction == "ab\ncd"
    assert len(backend.prompts) == 2
    assert "Expected exactly 2 lines but got 1" in backend.prompts[1]
    assert backend.cleanup_calls == 1


def test_project_boundaries_to_transcription_preserves_exact_text():
    """Boundary projection should recover exact transcription characters."""

    projected = project_boundaries_to_transcription(
        "helloworld",
        ["hello", "wurld"],
        expected_num_lines=2,
    )

    assert projected == ["hello", "world"]
    assert "".join(projected) == "helloworld"
    assert len(projected) == 2
