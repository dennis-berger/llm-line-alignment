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

    parsed_lines = parse_m4_response_loose(response)

    if len(parsed_lines) != expected_num_lines:
        raise ValueError(f"Expected exactly {expected_num_lines} lines but got {len(parsed_lines)}")

    return parsed_lines


def parse_m4_response_loose(response: str) -> list[str]:
    """Parse the JSON response without enforcing an exact line count."""

    try:
        payload = json.loads(_extract_json_candidate(response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object")

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError("Response JSON must include a 'lines' array")

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


def reconcile_lines_to_reference(
    source_lines: Sequence[str],
    reference_lines: Sequence[str],
    expected_num_lines: int | None = None,
) -> list[str]:
    """Re-slice ``source_lines`` so they follow the boundary count of ``reference_lines``."""

    line_count = expected_num_lines if expected_num_lines is not None else len(reference_lines)
    if line_count <= 0:
        return []

    if not source_lines:
        return [""] * line_count

    if expected_num_lines is not None and len(reference_lines) != expected_num_lines:
        raise ValueError(
            f"Expected {expected_num_lines} reference lines but got {len(reference_lines)}"
        )

    source_text = "".join(source_lines)
    if not source_text:
        return [""] * (line_count - 1) + [source_text]

    if not reference_lines:
        return [""] * (line_count - 1) + [source_text]

    reference_text = "".join(reference_lines)
    if not reference_text:
        return [""] * (line_count - 1) + [source_text]

    reference_boundaries: list[int] = []
    cursor = 0
    for line in reference_lines[:-1]:
        cursor += len(line)
        reference_boundaries.append(cursor)

    projected_boundaries = _project_positions(
        reference_text,
        source_text,
        reference_boundaries,
    )

    reconciled_lines: list[str] = []
    start = 0
    for boundary in projected_boundaries:
        reconciled_lines.append(source_text[start:boundary])
        start = boundary
    reconciled_lines.append(source_text[start:])
    return reconciled_lines


def _normalize_alignment_text(text: str) -> str:
    return " ".join(text.split())


def _line_group_alignment_cost(source_lines: Sequence[str], reference_line: str) -> float:
    source_text = _normalize_alignment_text("".join(source_lines))
    reference_text = _normalize_alignment_text(reference_line)
    if not source_text and not reference_text:
        return 0.0

    ratio = SequenceMatcher(None, source_text, reference_text, autojunk=False).ratio()
    length_scale = max(1, len(source_text), len(reference_text))
    length_penalty = abs(len(source_text) - len(reference_text)) / float(length_scale)
    merge_penalty = 0.05 * max(0, len(source_lines) - 1)
    return (1.0 - ratio) + 0.25 * length_penalty + merge_penalty


def score_lines_against_reference(
    source_lines: Sequence[str],
    reference_lines: Sequence[str],
    expected_num_lines: int | None = None,
) -> float:
    """Score how well ``source_lines`` align to ``reference_lines`` after reconciliation."""

    line_count = expected_num_lines if expected_num_lines is not None else len(reference_lines)
    if line_count <= 0:
        return 0.0

    if expected_num_lines is not None and len(reference_lines) != expected_num_lines:
        raise ValueError(
            f"Expected {expected_num_lines} reference lines but got {len(reference_lines)}"
        )

    if len(source_lines) > line_count:
        _, score = _merge_lines_to_reference_with_cost(
            source_lines,
            reference_lines,
            expected_num_lines=line_count,
        )
        return score

    if len(source_lines) < line_count:
        source_lines = reconcile_lines_to_reference(
            source_lines,
            reference_lines,
            expected_num_lines=line_count,
        )

    if len(source_lines) != line_count:
        raise ValueError(
            f"Expected {line_count} source lines after reconciliation but got {len(source_lines)}"
        )

    return sum(
        _line_group_alignment_cost([line], reference_line)
        for line, reference_line in zip(source_lines, reference_lines)
    )


def _merge_lines_to_reference_with_cost(
    source_lines: Sequence[str],
    reference_lines: Sequence[str],
    expected_num_lines: int | None = None,
) -> tuple[list[str], float]:
    """Merge contiguous source lines so they match the reference line count."""

    line_count = expected_num_lines if expected_num_lines is not None else len(reference_lines)
    if line_count <= 0:
        return []

    if expected_num_lines is not None and len(reference_lines) != expected_num_lines:
        raise ValueError(
            f"Expected {expected_num_lines} reference lines but got {len(reference_lines)}"
        )
    if len(source_lines) < line_count:
        raise ValueError(
            f"Cannot merge {len(source_lines)} source lines down to {line_count} lines"
        )

    source_count = len(source_lines)
    max_group_size = max(1, source_count - line_count + 1)
    inf = float("inf")
    dp: list[list[float]] = [[inf] * (line_count + 1) for _ in range(source_count + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (line_count + 1) for _ in range(source_count + 1)]
    dp[0][0] = 0.0

    for ref_index in range(1, line_count + 1):
        remaining_refs = line_count - ref_index
        for source_end in range(ref_index, source_count + 1):
            remaining_sources = source_count - source_end
            if remaining_sources < remaining_refs:
                continue

            start_min = max(ref_index - 1, source_end - max_group_size)
            start_max = source_end - 1
            best_cost = inf
            best_prev: tuple[int, int] | None = None

            for source_start in range(start_min, start_max + 1):
                previous_cost = dp[source_start][ref_index - 1]
                if previous_cost == inf:
                    continue

                group = source_lines[source_start:source_end]
                group_cost = _line_group_alignment_cost(group, reference_lines[ref_index - 1])
                total_cost = previous_cost + group_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev = (source_start, ref_index - 1)

            dp[source_end][ref_index] = best_cost
            back[source_end][ref_index] = best_prev

    if dp[source_count][line_count] == inf:
        raise ValueError(
            f"Could not merge {source_count} source lines down to {line_count} lines"
        )

    merged_reversed: list[str] = []
    source_index = source_count
    ref_index = line_count
    while ref_index > 0:
        previous = back[source_index][ref_index]
        if previous is None:
            raise ValueError("Backtracking failed while merging source lines")
        source_start, previous_ref_index = previous
        merged_reversed.append("".join(source_lines[source_start:source_index]))
        source_index = source_start
        ref_index = previous_ref_index

    return list(reversed(merged_reversed)), dp[source_count][line_count]


def merge_lines_to_reference(
    source_lines: Sequence[str],
    reference_lines: Sequence[str],
    expected_num_lines: int | None = None,
) -> list[str]:
    """Merge contiguous source lines so they match the reference line count."""

    merged_lines, _ = _merge_lines_to_reference_with_cost(
        source_lines,
        reference_lines,
        expected_num_lines=expected_num_lines,
    )
    return merged_lines
