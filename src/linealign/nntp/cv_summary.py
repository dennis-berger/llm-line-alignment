"""Helpers for summarizing NNTP cross-validation evaluation CSVs."""
from __future__ import annotations

import csv
from pathlib import Path


def read_macro_avg_row(csv_path: Path) -> dict[str, float]:
    """Read one NNTP evaluation CSV and return its macro-average row."""

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("id") != "macro_avg":
                continue
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key == "id":
                    continue
                if value in (None, ""):
                    continue
                parsed[key] = float(value)
            return parsed
    raise ValueError(f"{csv_path} does not contain a macro_avg row")


def summarize_cv_eval_csvs(named_csvs: list[tuple[str, Path]]) -> tuple[list[str], list[dict[str, float | str]]]:
    """Return per-fold rows plus a macro-average row."""

    if not named_csvs:
        raise ValueError("Need at least one evaluation CSV to summarize")

    rows: list[dict[str, float | str]] = []
    keys: list[str] | None = None
    sums: dict[str, float] = {}

    for name, csv_path in named_csvs:
        metrics = read_macro_avg_row(csv_path)
        keys = keys or list(metrics)
        if list(metrics) != keys:
            raise ValueError(f"CSV {csv_path} has metric columns that do not match prior files")
        row: dict[str, float | str] = {"id": name}
        for key in keys:
            row[key] = metrics[key]
            sums[key] = sums.get(key, 0.0) + metrics[key]
        rows.append(row)

    macro_row: dict[str, float | str] = {"id": "macro_avg"}
    for key in keys or []:
        macro_row[key] = sums[key] / len(named_csvs)
    rows.append(macro_row)

    return ["id", *(keys or [])], rows


def write_cv_summary_csv(path: Path, header: list[str], rows: list[dict[str, float | str]]) -> None:
    """Write the Washington NNTP CV summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
