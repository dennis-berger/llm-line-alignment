"""PyLaia lattice conversion helpers for NNTP observations."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .models import BoundaryRecord, ObservationMatrix, PreparedLineRecord
from .symbols import CTC_TOKEN, SPACE_TOKEN, SymbolTable


def iter_lattice_blocks(raw_lattice_path: Path):
    """Yield `(header_path, block_lines)` pairs from a raw PyLaia lattice file."""

    current_header: str | None = None
    current_lines: list[str] = []

    with raw_lattice_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if len(stripped.split()) < 5:
                if current_header is not None and current_lines:
                    yield current_header, current_lines
                current_header = stripped
                current_lines = []
                continue
            current_lines.append(raw_line)

    if current_header is not None and current_lines:
        yield current_header, current_lines


def split_lattice_blocks(
    raw_lattice_path: Path,
    output_dir: Path,
    prepared_lines: list[PreparedLineRecord],
) -> dict[Path, Path]:
    """Split one raw lattice file into one lattice block per cropped line image."""

    output_dir.mkdir(parents=True, exist_ok=True)
    record_by_crop = {record.crop_path.resolve(): record for record in prepared_lines}
    split_paths: dict[Path, Path] = {}

    for header, block_lines in iter_lattice_blocks(raw_lattice_path):
        crop_path = Path(header).resolve()
        record = record_by_crop.get(crop_path)
        if record is None:
            raise KeyError(f"Raw lattice block header {header!r} does not match a prepared crop path")
        split_path = output_dir / record.sample_id / f"{record.crop_path.stem}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("".join(block_lines), encoding="utf-8")
        split_paths[crop_path] = split_path

    missing = [record.crop_path for record in prepared_lines if record.crop_path.resolve() not in split_paths]
    if missing:
        raise ValueError(f"Missing lattice blocks for {len(missing)} prepared line(s)")

    return split_paths


def _parse_block_values(block_lines: list[str]) -> tuple[list[int], dict[int, dict[int, float]]]:
    values_by_time: dict[int, dict[int, float]] = {}
    for line in block_lines:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            continue
        time = int(parts[0])
        label = int(parts[3])
        value = float(parts[4].split(",")[-1])
        values_by_time.setdefault(time, {})[label] = value
    if not values_by_time:
        raise ValueError("Lattice block does not contain any timestep values")
    times = sorted(values_by_time)
    return times, values_by_time


def _select_probability_transform(values_by_time: dict[int, dict[int, float]]):
    def negated(value: float) -> float:
        return max(0.0, -value)

    def exponentiated(value: float) -> float:
        return math.exp(value)

    candidates = [negated, exponentiated]
    scored: list[tuple[float, object]] = []
    for transform in candidates:
        error = 0.0
        for per_label in values_by_time.values():
            error += abs(1.0 - sum(transform(value) for value in per_label.values()))
        scored.append((error, transform))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def convert_lattice_block(block_lines: list[str], symbol_table: SymbolTable) -> ObservationMatrix:
    """Convert one raw PyLaia lattice block to NNTP's observation matrix format."""

    times, values_by_time = _parse_block_values(block_lines)
    transform = _select_probability_transform(values_by_time)

    raw_indices = sorted(symbol_table.raw_by_index)
    non_ctc_labels = [index + 1 for index in raw_indices if symbol_table.raw_by_index[index] != "<ctc>"]
    eps_label = raw_indices[0] + 1

    rows: list[list[float]] = []
    selected_columns: list[list[float]] = []
    for label in non_ctc_labels:
        selected_columns.append([transform(values_by_time[time].get(label, 0.0)) for time in times])
    selected_columns.append([transform(values_by_time[time].get(eps_label, 0.0)) for time in times])

    for timestep in range(len(times)):
        kept_total = sum(column[timestep] for column in selected_columns)
        if kept_total <= 0.0:
            raise ValueError(f"Observation timestep {timestep} does not retain any supported probability mass")
        for column in selected_columns:
            column[timestep] /= kept_total

    rows.extend(selected_columns)

    matrix = ObservationMatrix(symbols=symbol_table.observation_symbols, rows=rows)
    assert_probability_columns(matrix)
    return matrix


def decode_ctc_indices(indices: list[int], symbol_table: SymbolTable) -> str:
    """Greedily decode a CTC label sequence into plain text."""

    decoded_chars: list[str] = []
    previous_index: int | None = None

    for index in indices:
        if index == previous_index:
            continue
        previous_index = index
        symbol = symbol_table.raw_by_index.get(index)
        if symbol in (None, CTC_TOKEN):
            continue
        if symbol == SPACE_TOKEN:
            decoded_chars.append(" ")
        else:
            decoded_chars.append(symbol)

    return "".join(decoded_chars).strip()


def decode_lattice_block_greedy(block_lines: list[str], symbol_table: SymbolTable) -> str:
    """Greedily decode a raw PyLaia lattice block using CTC best path decoding."""

    times, values_by_time = _parse_block_values(block_lines)
    transform = _select_probability_transform(values_by_time)
    supported_labels = {index + 1: index for index in symbol_table.raw_by_index}
    best_path: list[int] = []

    for time in times:
        candidates = [
            (label, transform(value))
            for label, value in values_by_time[time].items()
            if label in supported_labels
        ]
        if not candidates:
            raise ValueError(f"Lattice timestep {time} does not contain any supported labels")
        best_label, _ = max(candidates, key=lambda item: item[1])
        best_path.append(supported_labels[best_label])

    return decode_ctc_indices(best_path, symbol_table)


def decode_lattice_file_greedy(raw_lattice_path: Path, symbol_table: SymbolTable) -> dict[Path, str]:
    """Greedily decode each lattice block in a raw PyLaia output file."""

    decoded: dict[Path, str] = {}
    for header, block_lines in iter_lattice_blocks(raw_lattice_path):
        decoded[Path(header).resolve()] = decode_lattice_block_greedy(block_lines, symbol_table)
    return decoded


def assert_probability_columns(matrix: ObservationMatrix, tolerance: float = 1e-3) -> None:
    """Validate that each timestep column still sums to approximately 1."""

    for timestep in range(matrix.num_timesteps):
        total = sum(row[timestep] for row in matrix.rows)
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Observation timestep {timestep} sums to {total:.6f}, expected approximately 1.0"
            )


def write_observation_file(matrix: ObservationMatrix, output_path: Path) -> None:
    """Write an NNTP observation matrix to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# " + " ".join(matrix.symbols)]
    for row in matrix.rows:
        lines.append(" ".join(f"{value:.8g}" for value in row))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_observation_file(path: Path) -> ObservationMatrix:
    """Read an NNTP observation matrix from disk."""

    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
        symbols = header.split()[1:]
        rows = [list(map(float, line.split())) for line in handle if line.strip()]
    return ObservationMatrix(symbols=symbols, rows=rows)


def concatenate_observations(
    prepared_lines: list[PreparedLineRecord],
    observation_paths: dict[Path, Path],
) -> tuple[ObservationMatrix, list[BoundaryRecord]]:
    """Concatenate line-level observation files into one letter-level observation."""

    if not prepared_lines:
        raise ValueError("No prepared lines available for concatenation")

    sorted_lines = sorted(prepared_lines, key=lambda record: record.letter_line_index)
    first_matrix = read_observation_file(observation_paths[sorted_lines[0].crop_path.resolve()])
    combined_rows = [[] for _ in first_matrix.rows]
    boundaries: list[BoundaryRecord] = []
    timestep_cursor = 0

    for record in sorted_lines:
        matrix = read_observation_file(observation_paths[record.crop_path.resolve()])
        if matrix.symbols != first_matrix.symbols:
            raise ValueError("Observation symbol headers do not match across line files")
        for row_index, row in enumerate(matrix.rows):
            combined_rows[row_index].extend(row)
        end_timestep = timestep_cursor + matrix.num_timesteps - 1
        boundaries.append(
            BoundaryRecord(
                sample_id=record.sample_id,
                page_index=record.page_index,
                page_stem=record.page_stem,
                page_line_index=record.page_line_index,
                letter_line_index=record.letter_line_index,
                crop_path=record.crop_path.resolve(),
                start_timestep=timestep_cursor,
                end_timestep=end_timestep,
            )
        )
        timestep_cursor = end_timestep + 1

    return ObservationMatrix(symbols=first_matrix.symbols, rows=combined_rows), boundaries


def write_boundary_map(boundaries: list[BoundaryRecord], output_path: Path) -> None:
    """Write line boundary metadata to JSON."""

    payload = [
        {
            "sample_id": boundary.sample_id,
            "page_index": boundary.page_index,
            "page_stem": boundary.page_stem,
            "page_line_index": boundary.page_line_index,
            "letter_line_index": boundary.letter_line_index,
            "crop_path": str(boundary.crop_path),
            "start_timestep": boundary.start_timestep,
            "end_timestep": boundary.end_timestep,
        }
        for boundary in boundaries
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_boundary_map(path: Path) -> list[BoundaryRecord]:
    """Load line boundary metadata from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        BoundaryRecord(
            sample_id=item["sample_id"],
            page_index=item["page_index"],
            page_stem=item["page_stem"],
            page_line_index=item["page_line_index"],
            letter_line_index=item["letter_line_index"],
            crop_path=Path(item["crop_path"]),
            start_timestep=item["start_timestep"],
            end_timestep=item["end_timestep"],
        )
        for item in payload
    ]
