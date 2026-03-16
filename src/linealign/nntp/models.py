"""Data models shared by the NNTP integration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageXmlLineRecord:
    """Represents a single text line parsed from PAGE XML."""

    sample_id: str
    page_index: int
    page_stem: str
    xml_path: Path
    image_path: Path
    region_id: str
    region_order: int
    textline_id: str
    line_order: int
    source_text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedLineRecord:
    """Represents a cropped line image prepared for PyLaia."""

    sample_id: str
    page_index: int
    page_stem: str
    page_line_index: int
    letter_line_index: int
    xml_path: Path
    image_path: Path
    crop_path: Path
    region_id: str
    region_order: int
    textline_id: str
    line_order: int
    source_text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class BoundaryRecord:
    """Maps one cropped line image to a timestep span in a letter observation file."""

    sample_id: str
    page_index: int
    page_stem: str
    page_line_index: int
    letter_line_index: int
    crop_path: Path
    start_timestep: int
    end_timestep: int


@dataclass(frozen=True)
class AlignmentSegment:
    """One aligned character segment emitted by NNTP."""

    start: int
    end: int
    label: str
    score: float


@dataclass
class ObservationMatrix:
    """NNTP observation matrix with the last row reserved for EPS."""

    symbols: list[str]
    rows: list[list[float]]

    @property
    def num_timesteps(self) -> int:
        if not self.rows:
            return 0
        return len(self.rows[0])
