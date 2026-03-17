"""NNTP alignment parsing and prediction reconstruction."""
from __future__ import annotations

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


def _sorted_content_segments(segments: list[AlignmentSegment]) -> list[AlignmentSegment]:
    """Return aligned segments sorted in reading order and without epsilon labels."""

    return sorted(
        (segment for segment in segments if segment.label not in EPSILON_LABELS),
        key=lambda segment: (segment.start, segment.end, segment.label),
    )


def _gap_centers(segments: list[AlignmentSegment]) -> list[float]:
    """Return the split-point centers between aligned segments."""

    if not segments:
        return []

    centers = [segments[0].start - 0.5]
    for index in range(1, len(segments)):
        prev_segment = segments[index - 1]
        segment = segments[index]
        centers.append((prev_segment.end + segment.start) / 2.0)
    centers.append(segments[-1].end + 0.5)
    return centers


def _optimal_split_indices(gap_centers: list[float], cuts: list[float]) -> list[int]:
    """Find monotone split indices whose gap centers best match the target cuts."""

    if not cuts:
        return []

    num_gaps = len(gap_centers)
    dp = [abs(center - cuts[0]) for center in gap_centers]
    backpointers: list[list[int]] = []

    for cut in cuts[1:]:
        prefix_costs = [0.0] * num_gaps
        prefix_indices = [0] * num_gaps
        best_cost = dp[0]
        best_index = 0
        prefix_costs[0] = best_cost
        prefix_indices[0] = best_index

        for index in range(1, num_gaps):
            if dp[index] <= best_cost:
                best_cost = dp[index]
                best_index = index
            prefix_costs[index] = best_cost
            prefix_indices[index] = best_index

        next_dp = [0.0] * num_gaps
        next_backpointer = [0] * num_gaps
        for index, center in enumerate(gap_centers):
            next_dp[index] = prefix_costs[index] + abs(center - cut)
            next_backpointer[index] = prefix_indices[index]

        dp = next_dp
        backpointers.append(next_backpointer)

    last_index = min(range(num_gaps), key=lambda index: (dp[index], index))
    split_indices = [last_index]
    for backpointer in reversed(backpointers):
        last_index = backpointer[last_index]
        split_indices.append(last_index)

    split_indices.reverse()
    return split_indices


def _partition_segments_by_boundaries(
    segments: list[AlignmentSegment],
    boundaries: list[BoundaryRecord],
) -> list[list[AlignmentSegment]]:
    """Split aligned segments into contiguous per-line groups."""

    if not boundaries:
        return []
    if not segments:
        return [[] for _ in boundaries]

    sorted_boundaries = sorted(boundaries, key=lambda boundary: boundary.letter_line_index)
    gap_centers = _gap_centers(segments)
    cuts = [
        (left.end_timestep + right.start_timestep) / 2.0
        for left, right in zip(sorted_boundaries, sorted_boundaries[1:])
    ]
    split_indices = [0, *_optimal_split_indices(gap_centers, cuts), len(segments)]

    groups: list[list[AlignmentSegment]] = []
    for start_index, end_index in zip(split_indices, split_indices[1:]):
        groups.append(segments[start_index:end_index])
    return groups


def _segments_to_line(segments: list[AlignmentSegment]) -> str:
    """Render one aligned line from its NNTP character segments."""

    return "".join(" " if segment.label == "sp" else segment.label for segment in segments).strip()


def decode_alignment_segments(
    segments: list[AlignmentSegment],
    boundaries: list[BoundaryRecord],
) -> str:
    """Map NNTP character alignments back to visual lines."""

    if not boundaries:
        return ""

    content_segments = _sorted_content_segments(segments)
    line_groups = _partition_segments_by_boundaries(content_segments, boundaries)
    lines = [_segments_to_line(group) for group in line_groups]

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)
