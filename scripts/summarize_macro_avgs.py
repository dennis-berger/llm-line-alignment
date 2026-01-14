#!/usr/bin/env python3
"""
Summarize evaluation CSVs by extracting the 'macro_avg' row from each file.

Input: multiple CSV files with a row whose first column 'id' == 'macro_avg'.
Output:
  - summary_long.csv : one row per (dataset, method) with all macro metrics
  - summary_wide.csv : one row per dataset, method-prefixed columns (m1_*, m2_*, m3_*, qwen_m1_*, ...)

It infers dataset + method from the filename using common patterns found in this project.
If inference fails, it still includes the file, but uses a conservative fallback.
python scripts/summarize_macro_avgs.py --in-dir "/Users/dennisberger/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master_Thesis/predictions//Users/dennisberger/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master_Thesis/predictions/2026_01_14_evalualation_all_methods" --glob "*.csv"

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# --------- filename parsing ---------

def infer_dataset_and_method(stem: str) -> Tuple[str, str]:
    """
    Infer (dataset, method_label) from a filename stem (no extension).

    Examples:
      bullinger_print_eval_m2          -> ("bullinger_print", "m2")
      bullinger_print_eval_qwen_m1     -> ("bullinger_print", "qwen_m1")
      evaluation_qwen_m1               -> ("bullinger_handwritten"? unknown) -> ("evaluation", "qwen_m1")
      evaluation_m1_iam_print          -> ("iam_print", "m1")
      easy_hist_eval_m3                -> ("easy_hist", "m3")
      iam_handwritten_eval_m2          -> ("iam_handwritten", "m2")
    """
    s = stem.lower()

    # Normalize some known prefixes
    # Pattern A: <dataset>_eval_<method>
    m = re.match(r"^(?P<dataset>.+?)_eval_(?P<method>.+?)$", s)
    if m:
        return m.group("dataset"), m.group("method")

    # Pattern B: evaluation_<method>  (dataset might be embedded or missing)
    m = re.match(r"^evaluation_(?P<rest>.+)$", s)
    if m:
        rest = m.group("rest")

        # evaluation_m1_iam_print -> dataset=iam_print, method=m1
        m2 = re.match(r"^(?P<method>m[123])_(?P<dataset>.+)$", rest)
        if m2:
            return m2.group("dataset"), m2.group("method")

        # evaluation_qwen_m1 -> dataset="evaluation", method="qwen_m1" (fallback)
        return "evaluation", rest

    # Fallback: no idea, treat whole stem as dataset
    return s, "unknown"


# --------- macro_avg extraction ---------

def read_macro_avg_row(csv_path: Path) -> Dict[str, float]:
    """
    Read CSV and return dict of macro_avg metrics as floats, plus len_gt/len_pred if present.
    """
    df = pd.read_csv(csv_path)

    if "id" not in df.columns:
        raise ValueError(f"{csv_path.name}: expected an 'id' column, got columns={list(df.columns)}")

    macro = df.loc[df["id"].astype(str) == "macro_avg"]
    if macro.empty:
        raise ValueError(f"{csv_path.name}: no row with id == 'macro_avg'")

    # Take first match (should be exactly one)
    row = macro.iloc[0].to_dict()

    # Convert numeric-like values (skip 'id')
    out: Dict[str, float] = {}
    for k, v in row.items():
        if k == "id":
            continue
        # Some fields might be empty/NaN
        try:
            out[k] = float(v) if pd.notna(v) and v != "" else float("nan")
        except Exception:
            # If a column is unexpectedly non-numeric, keep as NaN
            out[k] = float("nan")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract macro_avg rows from evaluation CSVs and summarize.")
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the evaluation CSV files.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Glob pattern to find CSVs in --in-dir (default: *.csv).",
    )
    ap.add_argument(
        "--out-long",
        type=Path,
        default=Path("summary_long.csv"),
        help="Output CSV (long format).",
    )
    ap.add_argument(
        "--out-wide",
        type=Path,
        default=Path("summary_wide.csv"),
        help="Output CSV (wide format).",
    )
    args = ap.parse_args()

    csv_paths = sorted(args.in_dir.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.in_dir} matching {args.glob}")

    rows = []
    for p in csv_paths:
        dataset, method = infer_dataset_and_method(p.stem)
        metrics = read_macro_avg_row(p)

        row = {
            "file": p.name,
            "dataset": dataset,
            "method": method,
            **metrics,
        }
        rows.append(row)

    summary_long = pd.DataFrame(rows)

    # Add n (number of samples) if you want it: here we compute it as (#rows excluding macro_avg)
    # This only works if your CSV has one row per sample + one macro_avg row.
    n_map = {}
    for p in csv_paths:
        df = pd.read_csv(p)
        if "id" in df.columns:
            n_map[p.name] = int((df["id"].astype(str) != "macro_avg").sum())
    summary_long["n"] = summary_long["file"].map(n_map)

    # Sort nicely
    summary_long = summary_long.sort_values(["dataset", "method", "file"]).reset_index(drop=True)

    # Wide version: one row per dataset, columns method_metric
    metric_cols = [c for c in summary_long.columns if c not in ("file", "dataset", "method")]
    wide = summary_long.pivot_table(
        index="dataset",
        columns="method",
        values=metric_cols,
        aggfunc="first",
    )

    # Flatten MultiIndex columns: (metric, method) -> f"{method}_{metric}"
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()

    # Save
    args.out_long.parent.mkdir(parents=True, exist_ok=True)
    args.out_wide.parent.mkdir(parents=True, exist_ok=True)

    summary_long.to_csv(args.out_long, index=False)
    wide.to_csv(args.out_wide, index=False)

    # Also print a quick preview
    print(f"Wrote {args.out_long} ({len(summary_long)} rows)")
    print(f"Wrote {args.out_wide} ({len(wide)} rows)")
    print("\nPreview (long):")
    preview_cols = ["dataset", "method", "n", "wer", "cer", "line_acc", "rev_line_acc", "exact_line_f1"]
    existing = [c for c in preview_cols if c in summary_long.columns]
    print(summary_long[existing].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
