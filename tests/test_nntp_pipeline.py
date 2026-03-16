"""Unit tests for the NNTP Bullinger pipeline helpers."""
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

from linealign.nntp import (
    AlignmentSegment,
    PreparedLineRecord,
    concatenate_observations,
    convert_lattice_block,
    decode_alignment_segments,
    extract_prepared_lines,
    filter_transcription_text,
    load_symbol_table,
    write_observation_file,
)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_symbol_table(tmp_path: Path, symbols: list[str]):
    """Create a temporary syms.txt file and load it."""

    syms_path = tmp_path / "syms.txt"
    write_text(
        syms_path,
        "\n".join(f"{symbol} {index}" for index, symbol in enumerate(symbols)) + "\n",
    )
    return load_symbol_table(syms_path)


def test_extract_prepared_lines_uses_pagexml_order_and_filters_placeholders(tmp_path: Path):
    """PAGE XML lines should follow reading order and skip marker-only lines."""

    sample_root = tmp_path / "dataset" / "images" / "0001"
    sample_root.mkdir(parents=True, exist_ok=True)
    image_path = sample_root / "page_0001.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page_0001.png" imageWidth="100" imageHeight="100">
    <ReadingOrder>
      <OrderedGroup id="ro">
        <RegionRefIndexed index="0" regionRef="region_b"/>
        <RegionRefIndexed index="1" regionRef="region_a"/>
      </OrderedGroup>
    </ReadingOrder>
    <TextRegion id="region_a">
      <TextLine id="a1" custom="readingOrder {index:1;}">
        <Coords points="10,10 40,10 40,20 10,20"/>
        <TextEquiv><Unicode>second</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="a0" custom="readingOrder {index:0;}">
        <Coords points="10,30 40,30 40,40 10,40"/>
        <TextEquiv><Unicode>{MN}</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
    <TextRegion id="region_b">
      <TextLine id="b0">
        <Coords points="50,50 80,50 80,60 50,60"/>
        <TextEquiv><Unicode>first</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    write_text(sample_root / "page" / "page_0001.xml", xml_content)

    prepared = extract_prepared_lines(
        tmp_path / "dataset",
        "0001",
        tmp_path / "out" / "line_images",
        overwrite=True,
        pad=0,
    )

    assert [record.source_text for record in prepared] == ["first", "second"]
    assert [record.page_line_index for record in prepared] == [0, 1]
    assert [record.letter_line_index for record in prepared] == [0, 1]
    assert prepared[0].crop_path.name == "page_0001_line000.png"
    assert prepared[1].crop_path.name == "page_0001_line001.png"

    with Image.open(prepared[0].crop_path) as crop:
        assert crop.size == (30, 10)
    with Image.open(prepared[1].crop_path) as crop:
        assert crop.size == (30, 10)


def test_filter_transcription_text_strips_oov_chars(tmp_path: Path):
    """Unsupported characters should be removed and reported."""

    symbol_table = build_symbol_table(tmp_path, ["<ctc>", "<space>", "a", "b", "c"])
    filtered = filter_transcription_text("0001", "a à b— c", symbol_table)

    assert filtered.filtered_text == "a b c"
    assert filtered.tokens == ["a", "sp", "b", "sp", "c"]
    assert filtered.stripped_counts == {"à": 1, "—": 1}


def test_convert_lattice_block_and_concatenate_observations(tmp_path: Path):
    """Lattice conversion should preserve probability mass and concatenate cleanly."""

    symbol_table = build_symbol_table(tmp_path, ["<ctc>", "<space>", "a"])
    block_lines = [
        "0\t0\t0\t1\t-0.1\n",
        "0\t0\t0\t2\t-0.3\n",
        "0\t0\t0\t3\t-0.6\n",
        "1\t0\t0\t1\t-0.2\n",
        "1\t0\t0\t2\t-0.4\n",
        "1\t0\t0\t3\t-0.4\n",
    ]

    matrix = convert_lattice_block(block_lines, symbol_table)
    assert matrix.symbols == ["sp", "a"]
    assert matrix.rows[0] == pytest.approx([0.3, 0.4])
    assert matrix.rows[1] == pytest.approx([0.6, 0.4])
    assert matrix.rows[2] == pytest.approx([0.1, 0.2])

    obs_dir = tmp_path / "obs"
    first_obs = obs_dir / "0001" / "page_0001_line000.txt"
    second_obs = obs_dir / "0001" / "page_0001_line001.txt"
    write_observation_file(matrix, first_obs)
    write_observation_file(matrix, second_obs)

    prepared = [
        PreparedLineRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=0,
            letter_line_index=0,
            xml_path=tmp_path / "x1.xml",
            image_path=tmp_path / "x1.png",
            crop_path=(tmp_path / "line_0.png").resolve(),
            region_id="r1",
            region_order=0,
            textline_id="l1",
            line_order=0,
            source_text="line one",
            bbox=(0, 0, 1, 1),
        ),
        PreparedLineRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=1,
            letter_line_index=1,
            xml_path=tmp_path / "x1.xml",
            image_path=tmp_path / "x1.png",
            crop_path=(tmp_path / "line_1.png").resolve(),
            region_id="r1",
            region_order=0,
            textline_id="l2",
            line_order=1,
            source_text="line two",
            bbox=(0, 0, 1, 1),
        ),
    ]
    observation_paths = {
        prepared[0].crop_path.resolve(): first_obs,
        prepared[1].crop_path.resolve(): second_obs,
    }

    combined, boundaries = concatenate_observations(prepared, observation_paths)
    assert combined.rows[0] == pytest.approx([0.3, 0.4, 0.3, 0.4])
    assert combined.rows[2] == pytest.approx([0.1, 0.2, 0.1, 0.2])
    assert [(boundary.start_timestep, boundary.end_timestep) for boundary in boundaries] == [(0, 1), (2, 3)]


def test_convert_lattice_block_ignores_extra_labels_and_renormalizes(tmp_path: Path):
    """Unsupported lattice labels should be dropped and the kept rows renormalized."""

    symbol_table = build_symbol_table(tmp_path, ["<ctc>", "<space>", "a"])
    block_lines = [
        "0\t0\t0\t1\t-0.1\n",
        "0\t0\t0\t2\t-0.3\n",
        "0\t0\t0\t3\t-0.5\n",
        "0\t0\t0\t4\t-0.1\n",
    ]

    matrix = convert_lattice_block(block_lines, symbol_table)
    assert matrix.symbols == ["sp", "a"]
    assert matrix.rows[0] == pytest.approx([0.3 / 0.9])
    assert matrix.rows[1] == pytest.approx([0.5 / 0.9])
    assert matrix.rows[2] == pytest.approx([0.1 / 0.9])


def test_decode_alignment_segments_respects_line_boundaries():
    """NNTP alignments should decode back into newline-separated text."""

    segments = [
        AlignmentSegment(start=0, end=0, label="H", score=0.0),
        AlignmentSegment(start=1, end=1, label="sp", score=0.0),
        AlignmentSegment(start=2, end=2, label="B", score=0.0),
        AlignmentSegment(start=3, end=3, label="y", score=0.0),
        AlignmentSegment(start=4, end=4, label="e", score=0.0),
        AlignmentSegment(start=5, end=5, label="sp", score=0.0),
    ]
    from linealign.nntp.models import BoundaryRecord

    boundaries = [
        BoundaryRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=0,
            letter_line_index=0,
            crop_path=Path("/tmp/line0.png"),
            start_timestep=0,
            end_timestep=1,
        ),
        BoundaryRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=1,
            letter_line_index=1,
            crop_path=Path("/tmp/line1.png"),
            start_timestep=2,
            end_timestep=4,
        ),
        BoundaryRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=2,
            letter_line_index=2,
            crop_path=Path("/tmp/line2.png"),
            start_timestep=5,
            end_timestep=5,
        ),
    ]

    decoded = decode_alignment_segments(segments, boundaries)
    assert decoded == "H\nBye"
