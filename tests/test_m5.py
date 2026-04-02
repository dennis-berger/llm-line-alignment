"""Tests for Method 5 helpers and line-image alignment behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_eval_m5 import VLMMethod5Combiner
from src.linealign.vlm import VLMConfig
from utils.common import select_few_shot_examples
from utils.m5 import build_stacked_line_image, resolve_line_images


def save_image(path: Path, size: tuple[int, int] = (100, 40), color: str = "white") -> None:
    """Create a placeholder image on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


class ScriptedVisionBackend:
    """Minimal backend stub for deterministic M5 tests."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.image_counts: list[int] = []
        self.cleanup_calls = 0

    def load_and_prepare_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def downscale_image(self, img: Image.Image, max_side: int = 1280) -> Image.Image:
        return img

    def generate(self, prompt: str, images=None) -> str:
        self.prompts.append(prompt)
        self.image_counts.append(len(images or []))
        if not self.responses:
            raise AssertionError("No scripted backend responses left")
        return self.responses.pop(0)

    def cleanup(self):
        self.cleanup_calls += 1


def build_ocr_lines_payload(entries: list[dict[str, object]]) -> dict:
    """Build a minimal structured OCR-lines payload for tests."""

    return {
        "id": "sample",
        "dataset": "dummy",
        "recognizer": "pylaia",
        "num_pages": 2,
        "num_lines": len(entries),
        "lines": entries,
    }


def test_resolve_line_images_preserves_payload_order(tmp_path: Path):
    """Line-image resolution should preserve the payload's stored reading order."""

    save_image(tmp_path / "line_images" / "sample" / "0001_line000.png")
    save_image(tmp_path / "line_images" / "sample" / "0001_line001.png")
    save_image(tmp_path / "line_images" / "sample" / "0002_line000.png")

    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "first",
                "crop_path": "line_images/sample/0001_line000.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": "second",
                "crop_path": "line_images/sample/0001_line001.png",
            },
            {
                "page_index": 1,
                "line_index": 0,
                "text": "third",
                "crop_path": "line_images/sample/0002_line000.png",
            },
        ]
    )

    resolved = resolve_line_images(payload, tmp_path)

    assert [item.page_index for item in resolved] == [0, 0, 1]
    assert [item.line_index for item in resolved] == [0, 1, 0]
    assert [item.crop_path.name for item in resolved] == [
        "0001_line000.png",
        "0001_line001.png",
        "0002_line000.png",
    ]


def test_build_stacked_line_image_inserts_black_separator(tmp_path: Path):
    """Stacked line-image mode should produce one composite image with separators."""

    first_path = tmp_path / "line_images" / "sample" / "line0.png"
    second_path = tmp_path / "line_images" / "sample" / "line1.png"
    save_image(first_path, size=(30, 10), color="white")
    save_image(second_path, size=(20, 12), color="white")

    stacked = build_stacked_line_image(
        resolve_line_images(
            build_ocr_lines_payload(
                [
                    {
                        "page_index": 0,
                        "line_index": 0,
                        "crop_path": "line_images/sample/line0.png",
                    },
                    {
                        "page_index": 0,
                        "line_index": 1,
                        "crop_path": "line_images/sample/line1.png",
                    },
                ]
            ),
            tmp_path,
        ),
        separator_height=4,
    )

    assert stacked.size == (30, 26)
    assert stacked.getpixel((0, 11)) == (0, 0, 0)


def test_select_few_shot_examples_loads_line_image_paths_for_m5(tmp_path: Path):
    """Method 5 few-shot loading should resolve line-image crops from ocr_lines."""

    data_dir = tmp_path / "dataset"
    (data_dir / "gt").mkdir(parents=True, exist_ok=True)
    (data_dir / "transcription").mkdir(parents=True, exist_ok=True)
    (data_dir / "ocr_lines").mkdir(parents=True, exist_ok=True)

    (data_dir / "gt" / "sample_a.txt").write_text("ab\ncd", encoding="utf-8")
    (data_dir / "gt" / "sample_b.txt").write_text("ef\ngh", encoding="utf-8")
    (data_dir / "transcription" / "sample_a.txt").write_text("abcd", encoding="utf-8")
    (data_dir / "transcription" / "sample_b.txt").write_text("efgh", encoding="utf-8")

    save_image(data_dir / "line_images" / "sample_a" / "line0.png")
    save_image(data_dir / "line_images" / "sample_a" / "line1.png")
    save_image(data_dir / "line_images" / "sample_b" / "line0.png")
    save_image(data_dir / "line_images" / "sample_b" / "line1.png")

    (data_dir / "ocr_lines" / "sample_a.json").write_text(
        (
            '{"lines":['
            '{"page_index":0,"line_index":0,"text":"ab","crop_path":"line_images/sample_a/line0.png"},'
            '{"page_index":0,"line_index":1,"text":"cd","crop_path":"line_images/sample_a/line1.png"}'
            ']}'
        ),
        encoding="utf-8",
    )
    (data_dir / "ocr_lines" / "sample_b.json").write_text(
        (
            '{"lines":['
            '{"page_index":0,"line_index":0,"text":"ef","crop_path":"line_images/sample_b/line0.png"},'
            '{"page_index":0,"line_index":1,"text":"gh","crop_path":"line_images/sample_b/line1.png"}'
            ']}'
        ),
        encoding="utf-8",
    )

    examples = select_few_shot_examples(
        data_dir=data_dir,
        n_shots=1,
        exclude_ids=["sample_b"],
        method="m5",
        seed=1,
    )

    assert len(examples) == 1
    assert [path.name for path in examples[0].line_image_paths] == ["line0.png", "line1.png"]


def test_m5_combiner_repairs_malformed_json_and_projects_exact_text(tmp_path: Path):
    """A malformed first response should trigger repair and preserve exact transcription."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png")
    save_image(tmp_path / "line_images" / "sample" / "line1.png")
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        "not json",
        '{"lines":["ab","xd"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert len(backend.prompts) == 2
    assert backend.image_counts == [2, 2]
    assert "Your previous response was invalid." in backend.prompts[1]
    assert backend.cleanup_calls == 1


def test_m5_boundary_prompt_variant_mentions_anchor_lines(tmp_path: Path):
    """Boundary-anchored M5 prompts should preserve short standalone lines explicitly."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png")
    save_image(tmp_path / "line_images" / "sample" / "line1.png")
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend(['{"lines":["ab","cd"]}'])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        prompt_variant="boundary_anchored_v1",
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert "Treat line image i as the anchor for output line i." in backend.prompts[0]
    assert "Preserve short, odd, or fragmentary standalone lines" in backend.prompts[0]


def test_m5_structural_repair_retries_when_short_hint_is_merged(tmp_path: Path):
    """Very short leading hints should trigger one extra repair attempt when merged."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png")
    save_image(tmp_path / "line_images" / "sample" / "line1.png")
    save_image(tmp_path / "line_images" / "sample" / "line2.png")
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "16.",
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": ".",
                "crop_path": "line_images/sample/line1.png",
            },
            {
                "page_index": 0,
                "line_index": 2,
                "text": "body",
                "crop_path": "line_images/sample/line2.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["116.","1 body text","tail"]}',
        '{"lines":["116.","1 ","body texttail"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("116.1 body texttail", payload)

    assert prediction == "116.\n1 \nbody texttail"
    assert len(backend.prompts) == 2
    assert combiner.last_trace["structural_repair_applied"] is True
    assert combiner.last_trace["attempts"][1]["kind"] == "structural_repair"


@pytest.mark.parametrize(
    ("line_image_mode", "expected_image_count"),
    [("separate", 2), ("stacked", 1)],
)
def test_m5_combiner_packages_images_by_mode(
    tmp_path: Path,
    line_image_mode: str,
    expected_image_count: int,
):
    """Separate and stacked modes should preserve output behavior while changing packaging."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png", size=(24, 10))
    save_image(tmp_path / "line_images" / "sample" / "line1.png", size=(20, 12))
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend(['{"lines":["ab","cd"]}'])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode=line_image_mode,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert backend.image_counts == [expected_image_count]


@pytest.mark.parametrize("line_image_mode", ["separate", "stacked"])
def test_m5_combiner_uses_ocr_text_fallback_for_both_packaging_modes(
    tmp_path: Path,
    line_image_mode: str,
):
    """Both packaging modes should share the same OCR-text fallback behavior."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png")
    save_image(tmp_path / "line_images" / "sample" / "line1.png")
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "ab",
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": "xd",
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        "still not json",
        "also not json",
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode=line_image_mode,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert len(backend.prompts) == 2
    assert backend.cleanup_calls == 1
    assert combiner.last_trace["resolution"]["mode"] == "fallback_projection"
    assert combiner.last_trace["resolution"]["fallback_source"] == "ocr_text"


@pytest.mark.parametrize("line_image_mode", ["separate", "stacked"])
def test_m5_combiner_uses_hidden_ocr_text_as_last_resort_fallback(
    tmp_path: Path,
    line_image_mode: str,
):
    """Non-OCR-text mode should still recover from repeated invalid responses."""

    save_image(tmp_path / "line_images" / "sample" / "line0.png")
    save_image(tmp_path / "line_images" / "sample" / "line1.png")
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "ab",
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": "xd",
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        "still not json",
        "also not json",
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode=line_image_mode,
        use_ocr_text=False,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert len(backend.prompts) == 2
    assert backend.cleanup_calls == 1
