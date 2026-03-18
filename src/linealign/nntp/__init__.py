"""Helpers for the NNTP baseline pipeline."""

from .decode import decode_alignment_segments, parse_alignment_file
from .lattice import (
    assert_probability_columns,
    concatenate_observations,
    convert_lattice_block,
    decode_ctc_indices,
    decode_lattice_block_greedy,
    decode_lattice_file_greedy,
    read_boundary_map,
    read_observation_file,
    split_lattice_blocks,
    write_boundary_map,
    write_observation_file,
)
from .models import AlignmentSegment, BoundaryRecord, ObservationMatrix, PageXmlLineRecord, PreparedLineRecord
from .pagexml import (
    bbox_from_points,
    is_placeholder_text,
    parse_pagexml,
    parse_points,
    parse_reading_order_index,
)
from .prepare import PREPARE_MODES, detect_prepare_mode, extract_prepared_lines
from .presegmented import extract_prepared_lines_from_presegmented
from .pylaia import (
    infer_pylaia_input_height,
    infer_pylaia_input_height_from_kwargs,
    load_pylaia_model_kwargs,
    resize_image_files,
    resize_prepared_line_images,
    write_pylaia_netout_config,
)
from .symbols import FilteredLabel, SymbolTable, filter_transcription_text, load_symbol_table

__all__ = [
    "AlignmentSegment",
    "BoundaryRecord",
    "FilteredLabel",
    "ObservationMatrix",
    "PREPARE_MODES",
    "PageXmlLineRecord",
    "PreparedLineRecord",
    "SymbolTable",
    "assert_probability_columns",
    "detect_prepare_mode",
    "bbox_from_points",
    "concatenate_observations",
    "convert_lattice_block",
    "decode_ctc_indices",
    "decode_alignment_segments",
    "decode_lattice_block_greedy",
    "decode_lattice_file_greedy",
    "extract_prepared_lines",
    "extract_prepared_lines_from_presegmented",
    "filter_transcription_text",
    "infer_pylaia_input_height",
    "infer_pylaia_input_height_from_kwargs",
    "is_placeholder_text",
    "load_pylaia_model_kwargs",
    "load_symbol_table",
    "parse_alignment_file",
    "parse_pagexml",
    "parse_points",
    "parse_reading_order_index",
    "read_boundary_map",
    "read_observation_file",
    "resize_image_files",
    "resize_prepared_line_images",
    "split_lattice_blocks",
    "write_boundary_map",
    "write_observation_file",
    "write_pylaia_netout_config",
]
