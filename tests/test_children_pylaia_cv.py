"""Tests for children PyLaia manifest and symbol-table helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linealign.data.children_pylaia_cv import build_children_pylaia_cv_manifests, write_children_symbol_table


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_image(path: Path, size: tuple[int, int] = (120, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _build_sample(dataset_root: Path, sample_id: str, lines: list[str]) -> None:
    _write_text(dataset_root / "gt" / f"{sample_id}.txt", "\n".join(lines) + "\n")
    for index, _line in enumerate(lines):
        _save_image(dataset_root / "line_images" / sample_id / f"{sample_id}_line{index:03d}.png")


def test_children_symbol_table_and_manifests_cover_all_fixed_folds(tmp_path: Path) -> None:
    """Children manifests should be doc-disjoint and preserve German umlauts."""

    dataset_root = tmp_path / "children_handwritten"
    sample_text = {
        "1A_15_0-1": ["alpha"],
        "1A_17_0": ["schön"],
        "1A_6_0": ["beta"],
        "1A_8_0": ["grün"],
        "2A_11_0-1": ["gamma"],
        "2A_12_0-1": ["mädchen"],
        "2B_14_0-1": ["delta"],
        "3B_16_0-1": ["epsilon"],
        "3B_19_0-1": ["zeta"],
    }
    for sample_id, lines in sample_text.items():
        _build_sample(dataset_root, sample_id, lines)

    syms_path = tmp_path / "children.syms.txt"
    syms_meta = write_children_symbol_table(dataset_root, syms_path)
    manifest = build_children_pylaia_cv_manifests(
        dataset_root,
        tmp_path / "out",
        syms_path=syms_path,
        fixed_height=32,
    )

    assert syms_meta["alphabet"].count("ä") == 1
    assert syms_meta["alphabet"].count("ö") == 1
    assert syms_meta["alphabet"].count("ü") == 1
    assert set(manifest["folds"]) == {"fold_a", "fold_b", "fold_c"}

    fold_a_meta = json.loads((tmp_path / "out" / "fold_a" / "manifest_meta.json").read_text(encoding="utf-8"))
    assert fold_a_meta["train_docs"] == ["1A_15", "2A_11", "2B_14", "3B_16"]
    assert fold_a_meta["val_docs"] == ["1A_8", "2A_12"]
    assert fold_a_meta["test_docs"] == ["3B_19", "1A_17", "1A_6"]
    assert set(fold_a_meta["train_docs"]).isdisjoint(fold_a_meta["val_docs"])
    assert set(fold_a_meta["train_docs"]).isdisjoint(fold_a_meta["test_docs"])
    assert set(fold_a_meta["val_docs"]).isdisjoint(fold_a_meta["test_docs"])

    all_test_ids = []
    for fold_name in ("fold_a", "fold_b", "fold_c"):
        meta = json.loads((tmp_path / "out" / fold_name / "manifest_meta.json").read_text(encoding="utf-8"))
        all_test_ids.extend(meta["test_ids"])
    assert sorted(all_test_ids) == sorted(sample_text)

    train_tsv_rows = (tmp_path / "out" / "fold_a" / "train.tsv").read_text(encoding="utf-8").splitlines()
    normalized_image = Path(train_tsv_rows[0].split("\t", 1)[0])
    with Image.open(normalized_image) as image:
        assert image.height == 32

    fold_b_test_txt = (tmp_path / "out" / "fold_b" / "test.txt").read_text(encoding="utf-8")
    assert "<space>" not in fold_b_test_txt
