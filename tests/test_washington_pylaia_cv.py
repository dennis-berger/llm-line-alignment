"""Tests for Washington PyLaia CV manifest and summary helpers."""
from __future__ import annotations

import csv
import json
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

from linealign.data.washington_pylaia_cv import build_washington_pylaia_cv_manifests
from linealign.nntp.cv_summary import summarize_cv_eval_csvs, write_cv_summary_csv


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_image(path: Path, size: tuple[int, int] = (100, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _build_sample(dataset_root: Path, sample_id: str, lines: list[str]) -> None:
    _write_text(dataset_root / "gt" / f"{sample_id}.txt", "\n".join(lines) + "\n")
    for index, _line in enumerate(lines):
        _save_image(dataset_root / "line_images" / sample_id / f"{sample_id}_line{index:03d}.png")


def test_build_washington_pylaia_cv_manifests_creates_disjoint_splits(tmp_path: Path) -> None:
    """Washington manifests should be disjoint at the page level and row-aligned."""

    dataset_root = tmp_path / "washington_handwritten"
    syms_path = tmp_path / "syms.txt"
    symbols = ["<ctc>", "<space>", *list("abcdefghijklmnopqrstuvwxyz")]
    _write_text(
        syms_path,
        "\n".join(f"{symbol} {index}" for index, symbol in enumerate(symbols)) + "\n",
    )

    for sample_id, lines in {
        "270": ["first line", "second line"],
        "271": ["third line", "fourth line"],
        "277": ["fifth line", "sixth line"],
        "278": ["seventh line", "eighth line"],
    }.items():
        _build_sample(dataset_root, sample_id, lines)

    manifest = build_washington_pylaia_cv_manifests(
        dataset_root,
        tmp_path / "out",
        syms_path=syms_path,
        val_ratio=0.5,
        seed=42,
        selected_folds=["train_a", "train_b"],
        fold_specs={
            "train_a": {"train_ids": ("270", "271"), "test_ids": ("277", "278")},
            "train_b": {"train_ids": ("277", "278"), "test_ids": ("270", "271")},
        },
    )

    assert set(manifest["folds"]) == {"train_a", "train_b"}

    train_a_meta = json.loads((tmp_path / "out" / "train_a" / "manifest_meta.json").read_text(encoding="utf-8"))
    assert sorted(train_a_meta["train_ids"] + train_a_meta["val_ids"]) == ["270", "271"]
    assert train_a_meta["test_ids"] == ["277", "278"]
    assert set(train_a_meta["train_ids"]).isdisjoint(train_a_meta["val_ids"])
    assert set(train_a_meta["train_ids"]).isdisjoint(train_a_meta["test_ids"])
    assert set(train_a_meta["val_ids"]).isdisjoint(train_a_meta["test_ids"])

    train_tsv_rows = (tmp_path / "out" / "train_a" / "train.tsv").read_text(encoding="utf-8").splitlines()
    val_tsv_rows = (tmp_path / "out" / "train_a" / "val.tsv").read_text(encoding="utf-8").splitlines()
    test_txt_rows = (tmp_path / "out" / "train_a" / "test.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_tsv_rows) == train_a_meta["counts"]["train"]["line_count"]
    assert len(val_tsv_rows) == train_a_meta["counts"]["val"]["line_count"]
    assert len(test_txt_rows) == train_a_meta["counts"]["test"]["line_count"]
    assert train_tsv_rows[0].startswith(str(dataset_root.resolve()))
    assert test_txt_rows[0].startswith("line_images/")


def test_build_washington_pylaia_cv_manifests_rejects_oov_text(tmp_path: Path) -> None:
    """OOV characters should fail manifest generation early."""

    dataset_root = tmp_path / "washington_handwritten"
    syms_path = tmp_path / "syms.txt"
    _write_text(syms_path, "<ctc> 0\n<space> 1\na 2\n")
    _build_sample(dataset_root, "270", ["a"])
    _build_sample(dataset_root, "271", ["à"])
    _build_sample(dataset_root, "277", ["a"])
    _build_sample(dataset_root, "278", ["a"])

    with pytest.raises(ValueError, match="outside"):
        build_washington_pylaia_cv_manifests(
            dataset_root,
            tmp_path / "out",
            syms_path=syms_path,
            val_ratio=0.5,
            seed=42,
            selected_folds=["train_a"],
            fold_specs={
                "train_a": {"train_ids": ("270", "271"), "test_ids": ("277", "278")},
            },
        )


def test_summarize_cv_eval_csvs_writes_macro_average(tmp_path: Path) -> None:
    """Two fold CSVs should produce two rows plus a macro-average row."""

    header = ["id", "wer", "cer", "line_acc"]
    fold_a_csv = tmp_path / "fold_a.csv"
    fold_b_csv = tmp_path / "fold_b.csv"

    for path, macro in (
        (fold_a_csv, {"wer": 0.1, "cer": 0.2, "line_acc": 0.3}),
        (fold_b_csv, {"wer": 0.5, "cer": 0.6, "line_acc": 0.7}),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerow({"id": "sample_1", "wer": 0.0, "cer": 0.0, "line_acc": 0.0})
            writer.writerow({"id": "macro_avg", **macro})

    out_csv = tmp_path / "summary.csv"
    out_header, rows = summarize_cv_eval_csvs(
        [("train_a_test_b", fold_a_csv), ("train_b_test_a", fold_b_csv)]
    )
    write_cv_summary_csv(out_csv, out_header, rows)

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))

    assert [row["id"] for row in written] == ["train_a_test_b", "train_b_test_a", "macro_avg"]
    assert float(written[-1]["wer"]) == pytest.approx(0.3)
    assert float(written[-1]["cer"]) == pytest.approx(0.4)
    assert float(written[-1]["line_acc"]) == pytest.approx(0.5)
