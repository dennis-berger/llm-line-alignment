"""Tests for Bullinger ICCV import helpers and CLI."""
from __future__ import annotations

import subprocess
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

from utils.common import filter_paths_by_stem, parse_ids_arg


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_image(path: Path, size: tuple[int, int] = (120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def build_pagexml(image_filename: str, lines: list[str]) -> str:
    line_xml = []
    for index, text in enumerate(lines):
        top = index * 10
        bottom = top + 8
        line_xml.append(
            f"""      <TextLine id="l{index}" custom="readingOrder {{index:{index};}}">
        <Coords points="0,{top} 80,{top} 80,{bottom} 0,{bottom}"/>
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>
      </TextLine>"""
        )
    lines_block = "\n".join(line_xml)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="{image_filename}" imageWidth="120" imageHeight="80">
    <TextRegion id="r1">
{lines_block}
    </TextRegion>
  </Page>
</PcGts>
"""


def test_parse_ids_arg_supports_csv_and_files(tmp_path: Path):
    ids_file = tmp_path / "ids.txt"
    write_text(ids_file, "10069\n10676\n")

    assert parse_ids_arg("10069,10676") == ["10069", "10676"]
    assert parse_ids_arg(str(ids_file)) == ["10069", "10676"]
    assert [path.stem for path in filter_paths_by_stem([tmp_path / "10069.txt", tmp_path / "99999.txt"], ["10069"])] == ["10069"]


def test_import_bullinger_iccv_testset_cli_builds_flat_dataset_and_subset_manifests(tmp_path: Path):
    source_dir = tmp_path / "iccv-testset"
    out_dir = tmp_path / "datasets" / "bullinger_handwritten"

    write_text(source_dir / "README.md", "source readme\n")

    subset1_sample = source_dir / "Subset1" / "10069"
    save_image(subset1_sample / "0001.jpg")
    save_image(subset1_sample / "0002.jpg")
    write_text(subset1_sample / "meta.xml", "<meta />\n")
    write_text(subset1_sample / "mets.xml", "<mets />\n")
    write_text(
        subset1_sample / "page" / "0001.xml",
        build_pagexml("0001_0001_p001.jpg", ["Pacem", "{MT}", "gratiam"]),
    )
    write_text(
        subset1_sample / "page" / "0002.xml",
        build_pagexml("0002_0002_p002.jpg", ["Vale", "{X}"]),
    )

    subset2_sample = source_dir / "Subset2" / "10676"
    save_image(subset2_sample / "0001.tif")
    write_text(
        subset2_sample / "page" / "0001.xml",
        build_pagexml("0005_0005_original_name.tif", ["foo", "{MN}", "bar"]),
    )

    write_text(out_dir / "line_images" / "stale" / "old.txt", "stale\n")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_bullinger_iccv_testset.py"),
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ],
        check=True,
        cwd=ROOT,
    )

    assert (out_dir / "README.source.md").exists()
    assert (out_dir / "images" / "10069" / "0001.jpg").exists()
    assert (out_dir / "images" / "10069" / "meta.xml").exists()
    assert (out_dir / "images" / "10069" / "page" / "0001.xml").exists()
    assert (out_dir / "images" / "10676" / "0001.tif").exists()
    assert not (out_dir / "line_images").exists()

    assert (out_dir / "gt" / "10069.txt").read_text(encoding="utf-8") == "Pacem\ngratiam\nVale"
    assert (out_dir / "transcription" / "10069.txt").read_text(encoding="utf-8") == "Pacem gratiam Vale"
    assert (out_dir / "gt" / "10676.txt").read_text(encoding="utf-8") == "foo\nbar"

    assert (out_dir / "subsets" / "subset1_ids.txt").read_text(encoding="utf-8") == "10069\n"
    assert (out_dir / "subsets" / "subset2_ids.txt").read_text(encoding="utf-8") == "10676\n"


@pytest.mark.parametrize(
    "script_name",
    ["run_eval_m1.py", "run_eval_m2.py", "run_eval_m3.py", "run_eval_m4.py"],
)
def test_eval_clis_expose_ids_option(script_name: str):
    result = subprocess.run(
        [sys.executable, str(ROOT / script_name), "--help"],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert "--ids" in result.stdout
