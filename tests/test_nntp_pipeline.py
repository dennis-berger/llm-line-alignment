"""Unit tests for the NNTP Bullinger pipeline helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image
import torch

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
    decode_ctc_indices,
    decode_alignment_segments,
    decode_lattice_block_greedy,
    extract_prepared_lines,
    filter_transcription_text,
    infer_pylaia_input_height_from_kwargs,
    load_symbol_table,
    patch_pylaia_model_num_outputs,
    resize_prepared_line_images,
    write_pylaia_netout_config,
    write_observation_file,
)
from linealign.nntp.pagexml import load_pagexml_lines
from utils.common import find_images_for_id


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


def save_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Create a simple placeholder image on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_extract_prepared_lines_uses_pagexml_order_and_filters_placeholders(tmp_path: Path):
    """PAGE XML lines should follow reading order and skip marker-only lines."""

    sample_root = tmp_path / "dataset" / "images" / "0001"
    sample_root.mkdir(parents=True, exist_ok=True)
    image_path = sample_root / "page_0001.png"
    save_image(image_path)

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


def test_load_pagexml_lines_resolves_renamed_source_images(tmp_path: Path):
    """PAGE XML should bind to actual page files even when imageFilename was renamed."""

    sample_root = tmp_path / "images" / "10069"
    sample_root.mkdir(parents=True, exist_ok=True)
    save_image(sample_root / "0001.jpg", size=(100, 60))
    save_image(sample_root / "0002.jpg", size=(100, 60))

    xml_page_1 = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="0001_0001_p001.jpg" imageWidth="100" imageHeight="60">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="0,0 50,0 50,10 0,10"/>
        <TextEquiv><Unicode>first line</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    xml_page_2 = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="0002_0002_p002.jpg" imageWidth="100" imageHeight="60">
    <TextRegion id="r1">
      <TextLine id="l2">
        <Coords points="0,10 50,10 50,20 0,20"/>
        <TextEquiv><Unicode>second line</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    write_text(sample_root / "page" / "0001.xml", xml_page_1)
    write_text(sample_root / "page" / "0002.xml", xml_page_2)

    content_lines, image_paths = load_pagexml_lines(tmp_path / "images", "10069")

    assert [path.name for path in image_paths] == ["0001.jpg", "0002.jpg"]
    assert [record.image_path.name for record in content_lines] == ["0001.jpg", "0002.jpg"]
    assert [record.page_stem for record in content_lines] == ["0001", "0002"]
    assert [record.source_text for record in content_lines] == ["first line", "second line"]


def test_extract_prepared_lines_overwrite_prunes_stale_crops(tmp_path: Path):
    """Overwrite rebuilds should remove stale line-image leftovers for a sample."""

    sample_root = tmp_path / "dataset" / "images" / "0001"
    sample_root.mkdir(parents=True, exist_ok=True)
    save_image(sample_root / "0001.png", size=(100, 100))

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="0001.png" imageWidth="100" imageHeight="100">
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="0,0 50,0 50,10 0,10"/>
        <TextEquiv><Unicode>only line</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
    write_text(sample_root / "page" / "0001.xml", xml_content)

    stale_dir = tmp_path / "out" / "line_images" / "0001"
    stale_dir.mkdir(parents=True, exist_ok=True)
    save_image(stale_dir / "stale.png", size=(20, 20))

    prepared = extract_prepared_lines(
        tmp_path / "dataset",
        "0001",
        tmp_path / "out" / "line_images",
        overwrite=True,
        pad=0,
    )

    assert len(prepared) == 1
    assert sorted(path.name for path in stale_dir.glob("*.png")) == ["0001_line000.png"]


def test_extract_prepared_lines_uses_presegmented_images_when_available(tmp_path: Path):
    """Presegmented datasets should stage sorted line images without PAGE XML."""

    dataset_root = tmp_path / "dataset"
    write_text(dataset_root / "gt" / "iam-001.txt", "first line\nsecond line\n")
    save_image(dataset_root / "images" / "iam-001" / "iam-001.png", size=(80, 200))
    write_text(dataset_root / "images" / "iam-001" / "._iam-001.png", "sidecar")
    write_text(dataset_root / "line_images" / "iam-001" / "._iam-001-01.png", "sidecar")
    save_image(dataset_root / "line_images" / "iam-001" / "iam-001-01.png", size=(35, 10))
    save_image(dataset_root / "line_images" / "iam-001" / "iam-001-00.png", size=(25, 10))

    prepared = extract_prepared_lines(
        dataset_root,
        "iam-001",
        tmp_path / "out" / "line_images",
        overwrite=True,
    )

    assert [record.textline_id for record in prepared] == ["iam-001-00", "iam-001-01"]
    assert [record.source_text for record in prepared] == ["first line", "second line"]
    assert [record.crop_path.name for record in prepared] == ["iam-001-00.png", "iam-001-01.png"]
    assert prepared[0].image_path.name == "iam-001.png"
    with Image.open(prepared[0].crop_path) as crop:
        assert crop.size == (25, 10)
    with Image.open(prepared[1].crop_path) as crop:
        assert crop.size == (35, 10)


def test_find_images_for_id_ignores_hidden_sidecars(tmp_path: Path):
    """Hidden macOS sidecar files should not be treated as images."""

    images_root = tmp_path / "images"
    save_image(images_root / "0001" / "page_0001.png", size=(20, 20))
    write_text(images_root / "0001" / "._page_0001.png", "sidecar")

    image_paths = find_images_for_id(images_root, "0001")

    assert [path.name for path in image_paths] == ["page_0001.png"]


def test_extract_prepared_lines_rejects_presegmented_count_mismatches(tmp_path: Path):
    """Presegmented line images must match the GT line count."""

    dataset_root = tmp_path / "dataset"
    write_text(dataset_root / "gt" / "iam-001.txt", "only one line\n")
    save_image(dataset_root / "line_images" / "iam-001" / "iam-001-00.png", size=(20, 10))
    save_image(dataset_root / "line_images" / "iam-001" / "iam-001-01.png", size=(20, 10))

    with pytest.raises(ValueError, match="2 line image\\(s\\) but 1 GT line\\(s\\)"):
        extract_prepared_lines(
            dataset_root,
            "iam-001",
            tmp_path / "out" / "line_images",
            overwrite=True,
        )


def test_infer_pylaia_input_height_from_model_kwargs():
    """PyLaia input height should be inferred from sequencer size and pooling."""

    kwargs = {
        "image_sequencer": "none-16",
        "cnn_poolsize": [[2, 2], [2, 2], [0, 0], [2, 2]],
    }

    assert infer_pylaia_input_height_from_kwargs(kwargs) == 128


def test_resize_prepared_line_images_normalizes_height(tmp_path: Path):
    """Prepared line images should be resized in place to the PyLaia input height."""

    line_path = tmp_path / "line_0.png"
    save_image(line_path, size=(50, 25))
    prepared = [
        PreparedLineRecord(
            sample_id="0001",
            page_index=0,
            page_stem="page_0001",
            page_line_index=0,
            letter_line_index=0,
            xml_path=tmp_path / "x1.xml",
            image_path=tmp_path / "x1.png",
            crop_path=line_path.resolve(),
            region_id="r1",
            region_order=0,
            textline_id="l1",
            line_order=0,
            source_text="line one",
            bbox=(0, 0, 50, 25),
        )
    ]

    resize_prepared_line_images(prepared, 128)

    with Image.open(line_path) as image:
        assert image.size == (256, 128)


def test_patch_pylaia_model_num_outputs_rewrites_mismatched_metadata(tmp_path: Path):
    """Patched model metadata should match the checkpoint output dimension."""

    model_path = tmp_path / "model"
    checkpoint_path = tmp_path / "weights.ckpt"

    torch.save(
        {
            "kwargs": {
                "num_output_labels": 79,
                "image_sequencer": "none-16",
                "cnn_poolsize": [[2, 2], [2, 2], [0, 0], [2, 2]],
            }
        },
        model_path,
    )
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.zeros((98, 512)),
                "model.linear.bias": torch.zeros(98),
            }
        },
        checkpoint_path,
    )

    patched_path = patch_pylaia_model_num_outputs(
        model_path,
        checkpoint_path,
        output_dir=tmp_path / "patched",
    )

    assert patched_path != model_path.resolve()
    patched_model = torch.load(patched_path, map_location="cpu", weights_only=False)
    assert patched_model["kwargs"]["num_output_labels"] == 98


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


def test_decode_ctc_indices_preserves_repeats_split_by_blank(tmp_path: Path):
    """Greedy CTC decoding should keep repeated symbols when separated by blanks."""

    symbol_table = build_symbol_table(tmp_path, ["<ctc>", "<space>", "a", "b"])

    decoded = decode_ctc_indices([2, 2, 0, 2, 3, 3], symbol_table)

    assert decoded == "aab"


def test_decode_lattice_block_greedy_collapses_ctc_and_maps_spaces(tmp_path: Path):
    """Greedy lattice decoding should collapse repeats and decode spaces."""

    symbol_table = build_symbol_table(tmp_path, ["<ctc>", "<space>", "a", "b"])
    block_lines = [
        "0\t0\t0\t3\t-0.8\n",
        "0\t0\t0\t1\t-0.2\n",
        "1\t0\t0\t3\t-0.9\n",
        "1\t0\t0\t1\t-0.1\n",
        "2\t0\t0\t1\t-0.9\n",
        "2\t0\t0\t3\t-0.1\n",
        "3\t0\t0\t3\t-0.8\n",
        "3\t0\t0\t1\t-0.2\n",
        "4\t0\t0\t2\t-0.9\n",
        "4\t0\t0\t1\t-0.1\n",
        "5\t0\t0\t4\t-0.9\n",
        "5\t0\t0\t1\t-0.1\n",
    ]

    decoded = decode_lattice_block_greedy(block_lines, symbol_table)

    assert decoded == "aa b"


def test_write_pylaia_netout_config_writes_expected_trainer_values(tmp_path: Path):
    """The shared PyLaia config helper should encode GPU settings consistently."""

    config_path = tmp_path / "netout.yaml"
    experiment_dir = tmp_path / "run"
    model_path = tmp_path / "model"
    model_path.write_text("placeholder", encoding="utf-8")

    write_pylaia_netout_config(
        config_path,
        experiment_dir,
        model_path,
        pylaia_gpus=1,
        auto_select_gpus=True,
    )

    payload = config_path.read_text(encoding="utf-8")
    assert 'model_filename: "' in payload
    assert "auto_select_gpus: true" in payload
    assert "gpus: 1" in payload


def test_decode_alignment_segments_respects_line_boundaries():
    """NNTP alignments should decode back into newline-separated text."""

    segments = [
        AlignmentSegment(start=0, end=0, label="EPS", score=0.0),
        AlignmentSegment(start=0, end=0, label="eps", score=0.0),
        AlignmentSegment(start=0, end=0, label="H", score=0.0),
        AlignmentSegment(start=1, end=1, label="sp", score=0.0),
        AlignmentSegment(start=2, end=2, label="B", score=0.0),
        AlignmentSegment(start=3, end=3, label="y", score=0.0),
        AlignmentSegment(start=4, end=4, label="e", score=0.0),
        AlignmentSegment(start=5, end=5, label="sp", score=0.0),
        AlignmentSegment(start=5, end=5, label="<eps>", score=0.0),
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
