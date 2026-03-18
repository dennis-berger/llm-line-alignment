"""Dispatcher for dataset-specific NNTP line preparation backends."""
from __future__ import annotations

from pathlib import Path

from .models import PreparedLineRecord
from .pagexml import extract_prepared_lines as extract_prepared_lines_from_pagexml
from .presegmented import extract_prepared_lines_from_presegmented

PREPARE_MODES = ("auto", "pagexml", "presegmented")


def detect_prepare_mode(data_dir: Path, sample_id: str) -> str:
    """Infer the best preparation backend for one sample."""

    line_dir = data_dir / "line_images" / sample_id
    if line_dir.exists():
        return "presegmented"
    return "pagexml"


def extract_prepared_lines(
    data_dir: Path,
    sample_id: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    pad: int = 8,
    prepare_mode: str = "auto",
) -> list[PreparedLineRecord]:
    """Prepare line images for one sample using the requested backend."""

    if prepare_mode not in PREPARE_MODES:
        raise ValueError(f"Unknown prepare_mode {prepare_mode!r}; expected one of {PREPARE_MODES}")

    resolved_mode = detect_prepare_mode(data_dir, sample_id) if prepare_mode == "auto" else prepare_mode
    if resolved_mode == "presegmented":
        return extract_prepared_lines_from_presegmented(
            data_dir,
            sample_id,
            output_dir,
            overwrite=overwrite,
        )
    return extract_prepared_lines_from_pagexml(
        data_dir,
        sample_id,
        output_dir,
        overwrite=overwrite,
        pad=pad,
    )
