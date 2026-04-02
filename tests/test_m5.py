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

from run_eval_m5 import (
    VLMMethod5Combiner,
    VLMMethod5JudgeEnsemble,
    parse_m5_candidate_judge_response,
)
from src.linealign.vlm import VLMConfig
from utils.common import select_few_shot_examples
from utils.m5 import build_numbered_strip_images, build_stacked_line_image, resolve_line_images


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


def test_build_numbered_strip_images_adds_left_gutter_labels(tmp_path: Path):
    """Numbered strip mode should add a left gutter and keep rows stacked in order."""

    first_path = tmp_path / "line_images" / "sample" / "line0.png"
    second_path = tmp_path / "line_images" / "sample" / "line1.png"
    save_image(first_path, size=(30, 10), color="white")
    save_image(second_path, size=(20, 12), color="white")

    strips = build_numbered_strip_images(
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
        lines_per_strip=4,
    )

    assert len(strips) == 1
    strip = strips[0]
    assert strip.size[0] > 30
    gutter_pixels = [
        strip.getpixel((x, y))
        for x in range(0, 24)
        for y in range(strip.size[1])
    ]
    assert any(pixel != (255, 255, 255) for pixel in gutter_pixels)


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


def test_m5_structural_repair_checks_short_trailing_lines(tmp_path: Path):
    """Short trailing lines should also trigger structural repair, not only headers."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png", "line4.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "aaa",
                "crop_path": "line_images/sample/line0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": "bbb",
                "crop_path": "line_images/sample/line1.png",
            },
            {
                "page_index": 0,
                "line_index": 2,
                "text": "ccc",
                "crop_path": "line_images/sample/line2.png",
            },
            {
                "page_index": 0,
                "line_index": 3,
                "text": "ff",
                "crop_path": "line_images/sample/line3.png",
            },
            {
                "page_index": 0,
                "line_index": 4,
                "text": "closing",
                "crop_path": "line_images/sample/line4.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["aaa","bbb","ccc","ffclosing",""]}',
        '{"lines":["aaa","bbb","ccc","ff","closing"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("aaabbbcccffclosing", payload)

    assert prediction == "aaa\nbbb\nccc\nff\nclosing"
    assert len(backend.prompts) == 2
    assert combiner.last_trace["structural_repair_applied"] is True
    assert combiner.last_trace["attempts"][1]["kind"] == "structural_repair"


def test_m5_combiner_reconciles_near_valid_model_lines_before_ocr_fallback(tmp_path: Path):
    """Wrong-count JSON should be salvaged from model lines before falling back to OCR text."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
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
                "text": "cd",
                "crop_path": "line_images/sample/line1.png",
            },
            {
                "page_index": 0,
                "line_index": 2,
                "text": "ef",
                "crop_path": "line_images/sample/line2.png",
            },
            {
                "page_index": 0,
                "line_index": 3,
                "text": "gh",
                "crop_path": "line_images/sample/line3.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["ab","cd","efgh"]}',
        '{"lines":["ab","cdef","gh"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcdefgh", payload)

    assert prediction == "ab\ncd\nef\ngh"
    assert len(backend.prompts) == 2
    assert backend.cleanup_calls == 1
    assert combiner.last_trace["resolution"]["mode"] == "fallback_projection"
    assert combiner.last_trace["resolution"]["fallback_source"] == "reconciled_model_lines"


def test_m5_combiner_prefers_repair_candidate_with_better_reference_alignment(tmp_path: Path):
    """Loose candidate selection should prefer the repair response when it fits OCR structure better."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
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
                "text": "cd",
                "crop_path": "line_images/sample/line1.png",
            },
            {
                "page_index": 0,
                "line_index": 2,
                "text": "ef",
                "crop_path": "line_images/sample/line2.png",
            },
            {
                "page_index": 0,
                "line_index": 3,
                "text": "gh",
                "crop_path": "line_images/sample/line3.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["abcx","d","e","f","gh"]}',
        '{"lines":["ab","c","d","e","f","gh"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcdefgh", payload)

    assert prediction == "ab\ncd\nef\ngh"
    assert combiner.last_trace["resolution"]["fallback_source"] == "reconciled_model_lines"
    assert combiner.last_trace["resolution"]["reconciled_from_response"] == "repair_response"


def test_m5_quality_repair_retries_exact_count_output_with_bad_reference_alignment(tmp_path: Path):
    """Exact-count outputs with poor OCR alignment should trigger one quality-focused retry."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
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
                "text": "cd",
                "crop_path": "line_images/sample/line1.png",
            },
            {
                "page_index": 0,
                "line_index": 2,
                "text": "ef",
                "crop_path": "line_images/sample/line2.png",
            },
            {
                "page_index": 0,
                "line_index": 3,
                "text": "gh",
                "crop_path": "line_images/sample/line3.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["abcd","","ef","gh"]}',
        '{"lines":["ab","cd","ef","gh"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcdefgh", payload)

    assert prediction == "ab\ncd\nef\ngh"
    assert combiner.last_trace["attempts"][1]["kind"] == "quality_repair"
    assert combiner.last_trace["resolution"]["mode"] == "reference_alignment_selection"
    assert combiner.last_trace["resolution"]["selected_source"] != "current_projection"


def test_m5_exact_count_selection_keeps_model_when_ocr_gain_is_only_marginal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """OCR projection should not overrule viable exact-count model candidates on a small edge."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": index,
                "text": hint,
                "crop_path": f"line_images/sample/line{index}.png",
            }
            for index, hint in enumerate(["o1", "o2", "o3", "o4"])
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["c1","c2","c3","c4"]}',
        '{"lines":["q1","q2","q3","q4"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    score_map = {
        ("c1", "c2", "c3", "c4"): (["cur-1", "cur-2", "cur-3", "cur-4"], 1.6),
        ("q1", "q2", "q3", "q4"): (["qual-1", "qual-2", "qual-3", "qual-4"], 1.4),
        ("o1", "o2", "o3", "o4"): (["ocr-1", "ocr-2", "ocr-3", "ocr-4"], 1.2),
    }

    monkeypatch.setattr(
        combiner,
        "_projected_reference_score",
        lambda transcription, candidate_lines, reference_lines, expected_num_lines: score_map[tuple(candidate_lines)],
    )

    prediction = combiner.infer_line_breaks("placeholder", payload)

    assert prediction == "qual-1\nqual-2\nqual-3\nqual-4"
    assert combiner.last_trace["resolution"]["selected_source"] == "quality_repair_projection"
    assert combiner.last_trace["reference_alignment"]["include_ocr_projection"] is False


def test_m5_exact_count_selection_still_uses_ocr_for_catastrophic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """OCR projection should remain available for clearly catastrophic exact-count outputs."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": index,
                "text": hint,
                "crop_path": f"line_images/sample/line{index}.png",
            }
            for index, hint in enumerate(["o1", "o2", "o3", "o4"])
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["c1","c2","c3","c4"]}',
        '{"lines":["q1","q2","q3","q4"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=backend,
    )

    score_map = {
        ("c1", "c2", "c3", "c4"): (["cur-1", "cur-2", "cur-3", "cur-4"], 3.2),
        ("q1", "q2", "q3", "q4"): (["qual-1", "qual-2", "qual-3", "qual-4"], 3.0),
        ("o1", "o2", "o3", "o4"): (["ocr-1", "ocr-2", "ocr-3", "ocr-4"], 1.6),
    }

    monkeypatch.setattr(
        combiner,
        "_projected_reference_score",
        lambda transcription, candidate_lines, reference_lines, expected_num_lines: score_map[tuple(candidate_lines)],
    )

    prediction = combiner.infer_line_breaks("placeholder", payload)

    assert prediction == "ocr-1\nocr-2\nocr-3\nocr-4"
    assert combiner.last_trace["resolution"]["selected_source"] == "ocr_projection"
    assert combiner.last_trace["reference_alignment"]["include_ocr_projection"] is True


def test_m5_short_prefix_hybrid_detects_dense_short_opening_hints(tmp_path: Path):
    """Dense short opening hints should expose an OCR-prefix hybrid candidate."""

    for name in ["line0.png", "line1.png", "line2.png", "line3.png", "line4.png", "line5.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": index,
                "text": hint,
                "crop_path": f"line_images/sample/line{index}.png",
            }
            for index, hint in enumerate(["aa", "bb", "cc", "dd", "ee", "ff"])
        ]
    )
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        backend=ScriptedVisionBackend([]),
    )

    hybrid = combiner._build_short_prefix_hybrid_hints(
        ["aabb", "cc", "dd", "ee", "ff", ""],
        resolve_line_images(payload, tmp_path),
    )

    assert hybrid is not None
    assert hybrid["prefix_len"] == 6
    assert hybrid["short_hint_count"] == 6
    assert hybrid["hybrid_lines"][:6] == ["aa", "bb", "cc", "dd", "ee", "ff"]


@pytest.mark.parametrize(
    ("line_image_mode", "expected_image_count"),
    [("separate", 2), ("stacked", 1), ("numbered_strips", 1)],
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


def test_m5_combiner_can_include_page_images_with_numbered_strips(tmp_path: Path):
    """Full page images should be prepended as global context when requested."""

    save_image(tmp_path / "images" / "sample" / "0001.jpg", size=(120, 80))
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
        line_image_mode="numbered_strips",
        include_page_images=True,
        prompt_variant="boundary_context_v2",
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert backend.image_counts == [2]
    assert "full page images" in backend.prompts[0]


def test_parse_m5_candidate_judge_response_normalizes_labels():
    """Judge parsing should accept a few natural label variants."""

    assert parse_m5_candidate_judge_response('{"winner":"A","reason":"ok"}') == {
        "winner": "A",
        "reason": "ok",
    }
    assert parse_m5_candidate_judge_response('{"winner":"candidate b"}') == {
        "winner": "B",
        "reason": "",
    }


def test_m5_judge_ensemble_selects_better_context_candidate(tmp_path: Path):
    """The ensemble should return the candidate chosen by the judge pass."""

    save_image(tmp_path / "images" / "sample" / "0001.jpg", size=(120, 80))
    save_image(tmp_path / "line_images" / "sample" / "line0.png", size=(24, 10))
    save_image(tmp_path / "line_images" / "sample" / "line1.png", size=(20, 12))
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
                "text": "cd",
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    primary_backend = ScriptedVisionBackend(['{"lines":["abc","d"]}'])
    secondary_backend = ScriptedVisionBackend(['{"lines":["ab","cd"]}'])
    judge_backend = ScriptedVisionBackend(['{"winner":"B","reason":"B matches the short first line."}'])
    primary = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode="separate",
        use_ocr_text=True,
        prompt_variant="boundary_anchored_v1",
        backend=primary_backend,
    )
    secondary = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode="numbered_strips",
        use_ocr_text=True,
        include_page_images=True,
        prompt_variant="boundary_context_v2",
        backend=secondary_backend,
    )
    ensemble = VLMMethod5JudgeEnsemble(
        primary,
        secondary,
        judge_cfg=VLMConfig(model_id="hf/dummy"),
        judge_backend=judge_backend,
    )

    prediction = ensemble.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert ensemble.last_trace["resolution"]["selected_winner"] == "B"
    assert judge_backend.image_counts == [2]
    assert "Candidate A" in judge_backend.prompts[0]


def test_m5_judge_ensemble_repairs_invalid_judge_response(tmp_path: Path):
    """An invalid judge response should trigger one repair prompt before selection."""

    save_image(tmp_path / "images" / "sample" / "0001.jpg", size=(120, 80))
    save_image(tmp_path / "line_images" / "sample" / "line0.png", size=(24, 10))
    save_image(tmp_path / "line_images" / "sample" / "line1.png", size=(20, 12))
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
                "text": "cd",
                "crop_path": "line_images/sample/line1.png",
            },
        ]
    )

    primary = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode="separate",
        use_ocr_text=True,
        prompt_variant="boundary_anchored_v1",
        backend=ScriptedVisionBackend(['{"lines":["abc","d"]}']),
    )
    secondary = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        line_image_mode="numbered_strips",
        use_ocr_text=True,
        include_page_images=True,
        prompt_variant="boundary_context_v2",
        backend=ScriptedVisionBackend(['{"lines":["ab","cd"]}']),
    )
    judge_backend = ScriptedVisionBackend([
        "not json",
        '{"winner":"B","reason":"repair works"}',
    ])
    ensemble = VLMMethod5JudgeEnsemble(
        primary,
        secondary,
        judge_cfg=VLMConfig(model_id="hf/dummy"),
        judge_backend=judge_backend,
    )

    prediction = ensemble.infer_line_breaks("abcd", payload)

    assert prediction == "ab\ncd"
    assert len(judge_backend.prompts) == 2
    assert "Your previous response was invalid." in judge_backend.prompts[1]


def test_m5_combiner_can_split_multi_page_samples_before_alignment(tmp_path: Path):
    """Multi-page samples can be aligned page-by-page using OCR-based coarse page splits."""

    for name in ["p0l0.png", "p0l1.png", "p1l0.png", "p1l1.png"]:
        save_image(tmp_path / "line_images" / "sample" / name)
    payload = build_ocr_lines_payload(
        [
            {
                "page_index": 0,
                "line_index": 0,
                "text": "ab",
                "crop_path": "line_images/sample/p0l0.png",
            },
            {
                "page_index": 0,
                "line_index": 1,
                "text": "cd",
                "crop_path": "line_images/sample/p0l1.png",
            },
            {
                "page_index": 1,
                "line_index": 0,
                "text": "ef",
                "crop_path": "line_images/sample/p1l0.png",
            },
            {
                "page_index": 1,
                "line_index": 1,
                "text": "gh",
                "crop_path": "line_images/sample/p1l1.png",
            },
        ]
    )

    backend = ScriptedVisionBackend([
        '{"lines":["ab","cd"]}',
        '{"lines":["ef","gh"]}',
    ])
    combiner = VLMMethod5Combiner(
        VLMConfig(model_id="hf/dummy"),
        dataset_root=tmp_path,
        use_ocr_text=True,
        split_by_page=True,
        backend=backend,
    )

    prediction = combiner.infer_line_breaks("abcdefgh", payload)

    assert prediction == "ab\ncd\nef\ngh"
    assert len(backend.prompts) == 2
    assert combiner.last_trace["mode"] == "page_split"
    assert len(combiner.last_trace["page_chunks"]) == 2


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
