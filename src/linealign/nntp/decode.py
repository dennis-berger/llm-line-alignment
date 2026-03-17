"""NNTP alignment parsing and prediction reconstruction."""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

from .models import AlignmentSegment, BoundaryRecord

EPSILON_LABELS = {"EPS", "eps", "<eps>", "<EPS>"}


def parse_alignment_file(path: Path) -> list[AlignmentSegment]:
    """Parse an NNTP `.rec` alignment file into character segments."""

    segments: list[AlignmentSegment] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == ".":
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        segments.append(
            AlignmentSegment(
                start=int(parts[0]),
                end=int(parts[1]),
                label=parts[2],
                score=float(parts[3]),
            )
        )
    return segments


def decode_alignment_segments(
    segments: list[AlignmentSegment],
    boundaries: list[BoundaryRecord],
) -> str:
    """Map NNTP character alignments back to visual lines."""

    if not segments or not boundaries:
        return ""

    sorted_boundaries = sorted(boundaries, key=lambda boundary: boundary.letter_line_index)
    end_positions = [boundary.end_timestep for boundary in sorted_boundaries]
    grouped: dict[int, list[str]] = {}

    for segment in segments:
        if segment.label in EPSILON_LABELS:
            continue
        midpoint = (segment.start + segment.end) / 2.0
        boundary_index = bisect_left(end_positions, midpoint)
        if boundary_index >= len(sorted_boundaries):
            boundary_index = len(sorted_boundaries) - 1
        while boundary_index > 0 and midpoint < sorted_boundaries[boundary_index].start_timestep:
            boundary_index -= 1
        line_index = sorted_boundaries[boundary_index].letter_line_index
        grouped.setdefault(line_index, []).append(" " if segment.label == "sp" else segment.label)

    lines: list[str] = []
    for line_index in sorted(grouped):
        line = "".join(grouped[line_index]).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
