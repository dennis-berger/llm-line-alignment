"""Helpers for Method 4 prompt formatting and response post-processing."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any


def extract_ocr_line_texts(ocr_lines_payload: Mapping[str, Any]) -> list[str]:
    """Return the ordered OCR line texts from an ``ocr_lines`` payload."""

    raw_lines = ocr_lines_payload.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("ocr_lines payload must contain a 'lines' array")

    texts: list[str] = []
    for index, item in enumerate(raw_lines):
        if not isinstance(item, Mapping):
            raise ValueError(f"ocr_lines entry {index} must be an object")
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError(f"ocr_lines entry {index} is missing a string 'text' field")
        texts.append(text)
    return texts


def render_ocr_line_hints(line_items: Sequence[Mapping[str, Any]]) -> str:
    """Render ordered OCR line hints for the M4 prompt."""

    formatted: list[str] = []
    for index, item in enumerate(line_items, start=1):
        page_index = item.get("page_index")
        line_index = item.get("line_index")
        text = item.get("text")
        if not isinstance(page_index, int) or not isinstance(line_index, int) or not isinstance(text, str):
            raise ValueError("Each OCR line hint must include integer page_index, integer line_index, and string text")
        hint_text = text if text else "[blank]"
        formatted.append(f"{index}. (page {page_index + 1}, line {line_index + 1}) {hint_text}")
    return "\n".join(formatted)


def _extract_json_candidate(response: str) -> str:
    stripped = response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start <= end:
        return stripped[start:end + 1]
    return stripped


def parse_m4_response(response: str, expected_num_lines: int) -> list[str]:
    """Parse and validate the strict M4 JSON response."""

    try:
        payload = json.loads(_extract_json_candidate(response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("Response JSON must include a 'lines' array")

    if len(raw_lines) != expected_num_lines:
        raise ValueError(f"Expected exactly {expected_num_lines} lines but got {len(raw_lines)}")

    parsed_lines: list[str] = []
    for index, line in enumerate(raw_lines):
        if not isinstance(line, str):
            raise ValueError(f"Response line {index} is not a string")
        parsed_lines.append(line)
    return parsed_lines


def _build_alignment_anchors(source: str, target: str) -> list[tuple[int, int]]:
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    anchors: list[tuple[int, int]] = [(0, 0)]
    for block in matcher.get_matching_blocks():
        anchors.append((block.a, block.b))
        anchors.append((block.a + block.size, block.b + block.size))

    deduped: list[tuple[int, int]] = []
    for anchor in sorted(anchors):
        if not deduped or anchor != deduped[-1]:
            deduped.append(anchor)
    return deduped


def _project_positions(source: str, target: str, positions: Sequence[int]) -> list[int]:
    if not positions:
        return []
    if not source:
        return [0 for _ in positions]

    anchors = _build_alignment_anchors(source, target)
    projected: list[int] = []
    anchor_index = 0

    for position in positions:
        bounded_position = max(0, min(position, len(source)))
        while (
            anchor_index + 1 < len(anchors)
            and anchors[anchor_index + 1][0] < bounded_position
        ):
            anchor_index += 1

        left_a, left_b = anchors[anchor_index]
        if anchor_index + 1 < len(anchors):
            right_a, right_b = anchors[anchor_index + 1]
        else:
            right_a, right_b = len(source), len(target)

        if right_a == left_a:
            mapped = left_b
        else:
            span_ratio = (bounded_position - left_a) / float(right_a - left_a)
            mapped = round(left_b + span_ratio * (right_b - left_b))

        projected.append(max(0, min(mapped, len(target))))

    monotonic: list[int] = []
    previous = 0
    for mapped in projected:
        clipped = max(previous, mapped)
        monotonic.append(clipped)
        previous = clipped
    return monotonic


def project_boundaries_to_transcription(
    transcription: str,
    hinted_lines: Sequence[str],
    expected_num_lines: int | None = None,
) -> list[str]:
    """Project hinted line boundaries back onto the exact transcription."""

    line_count = expected_num_lines if expected_num_lines is not None else len(hinted_lines)
    if line_count <= 0:
        return []

    if not hinted_lines:
        return [""] * (line_count - 1) + [transcription]

    if expected_num_lines is not None and len(hinted_lines) != expected_num_lines:
        raise ValueError(
            f"Expected {expected_num_lines} hinted lines but got {len(hinted_lines)}"
        )

    hinted_text = "".join(hinted_lines)
    if not hinted_text:
        return [""] * (line_count - 1) + [transcription]

    hint_boundaries: list[int] = []
    cursor = 0
    for line in hinted_lines[:-1]:
        cursor += len(line)
        hint_boundaries.append(cursor)

    projected_boundaries = _project_positions(hinted_text, transcription, hint_boundaries)

    exact_lines: list[str] = []
    start = 0
    for boundary in projected_boundaries:
        exact_lines.append(transcription[start:boundary])
        start = boundary
    exact_lines.append(transcription[start:])
    return exact_lines
