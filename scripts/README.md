# Scripts Guide

## make_ocr_outputs.py

Generate OCR/HTR outputs for datasets.

### Basic Usage

```bash
# Generate OCR for a dataset
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten

# Process specific samples
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten --ids 10069,10676

# Use different recognizer
python scripts/make_ocr_outputs.py --dataset bullinger_print --recognizer trocr_printed
```

### Common Options

- `--dataset <name>` - Dataset name
- `--ids <list>` - Comma-separated IDs or path to file
- `--recognizer` - `trocr_handwritten`, `trocr_printed`, `pylaia`, `pylaia_iam` (legacy alias), `htr_best_practices_iam`
- `--segmenter` - `kraken` (default), `none` (passthrough)
- `--device` - `cuda`, `cpu`, `auto`
- `--batch-size` - Recognition batch size (default: 8)
- `--overwrite` - Regenerate existing outputs

### Output

Creates `ocr/<id>.txt`, `ocr_lines/<id>.json`, and optional `ocr/<id>.meta.json` in dataset directory.

For datasets that include `line_images/`, the script can reuse those presegmented crops with
`--segmenter none --existing-lines-dir ...`. For IAM handwritten datasets, it automatically switches to
`--segmenter none` and `--recognizer pylaia` unless you override those flags.

See [docs/ocr_pipeline.md](docs/ocr_pipeline.md) for technical details.

---

## import_bullinger_iccv_testset.py

Import the ICCV Bullinger export from `../iccv-testset` into the canonical flat
repo dataset layout while preserving paper subsets under `subsets/`.

### Basic Usage

```bash
python scripts/import_bullinger_iccv_testset.py \
  --source-dir ../iccv-testset \
  --out-dir datasets/bullinger_handwritten \
  --overwrite
```

### Output

Creates or replaces:

- **`images/<sample_id>/`** - page images plus PAGE XML and available sidecar XML
- **`gt/<sample_id>.txt`** - filtered line-broken PAGE text
- **`transcription/<sample_id>.txt`** - GT-collapsed transcription
- **`subsets/subset1_ids.txt`**, **`subsets/subset2_ids.txt`** - paper subset manifests
- **`subsets/manifest.json`** - subset/source metadata

---

## build_bullinger_handwritten_line_images.py

Materialize reusable Bullinger `line_images/` directly from the PAGE XML annotations.

This is useful when you want to:

- precompute deterministic line crops once
- run PyLaia OCR later as a separate preprocessing step
- keep Bullinger `M4` eval jobs focused on `run_eval_m4.py`

### Basic Usage

```bash
python scripts/build_bullinger_handwritten_line_images.py \
  --data-dir datasets/bullinger_handwritten \
  --out-dir datasets/bullinger_handwritten/line_images
```

### Output

Creates:

- **`line_images/<sample_id>/`** - one cropped image per XML line in reading order

---

## build_iam_rwth_dataset.py

Build an IAM dataset slice from the official RWTH split files used by the public PyLaia IAM checkpoint.

The default `--link-mode copy` makes the dataset portable to another machine or cluster.

### Basic Usage

```bash
# Build the full RWTH test split as a new dataset
python scripts/build_iam_rwth_dataset.py \
  --iam-root ../iam/data \
  --split test \
  --out-dir datasets/IAM_handwritten_rwth_test

# Smoke-test on a couple of forms
python scripts/build_iam_rwth_dataset.py \
  --iam-root ../iam/data \
  --split test \
  --max-forms 2 \
  --out-dir /tmp/iam_handwritten_rwth_test
```

### Output

Creates a dataset with:

- **`gt/`** - newline-separated ground truth per form
- **`transcription/`** - line-break-free text per form
- **`images/`** - form images
- **`line_images/`** - presegmented IAM line images for NNTP
- **`metadata.json`** - per-form and per-line provenance/status metadata

---

## build_children_handwritten_dataset.py

Build the canonical `children_handwritten` dataset directly from the original
`alignment_tests` export.

This script uses:

- `ground_truth/csv_aligned/` for GT lines
- `data/images/` for page images
- `output/disjoin/` for presegmented line crops

It also resolves the one raw/source naming alias for `3B-16_16-17` and rewrites
line-image filenames to zero-padded `<id>_lineNNN.png` names so downstream
PyLaia OCR and NNTP preparation preserve the correct line order.

### Basic Usage

```bash
python scripts/build_children_handwritten_dataset.py \
  --source-dir ../children_hw_original/alignment_tests \
  --out-dir datasets/children_handwritten \
  --overwrite
```

### Output

Creates or refreshes:

- **`gt/`** - newline-separated GT per sample
- **`transcription/`** - line-break-free transcription per sample
- **`images/`** - one page image per sample
- **`line_images/`** - canonical presegmented line crops
- **`metadata.json`** - source provenance and per-sample counts

---

## build_washington_handwritten_nntp_dataset.py

Build a scratch Washington NNTP workspace with raw Kraken line-image crops for manual review.

This script keeps the canonical dataset untouched and materializes a separate workspace that contains:

- copied or symlinked `gt/`, `transcription/`, `ocr/`, and `images/`
- auto-segmented `line_images/`
- per-sample metadata under `metadata/`
- overlay previews under `previews/`
- a review template in `review_status.json`

### Basic Usage

```bash
# Build a full scratch workspace from the canonical dataset
python scripts/build_washington_handwritten_nntp_dataset.py \
  --source-dir datasets/washington_handwritten \
  --out-dir /tmp/washington_handwritten_nntp

# Rebuild a small subset
python scripts/build_washington_handwritten_nntp_dataset.py \
  --source-dir datasets/washington_handwritten \
  --ids 270,271 \
  --out-dir /tmp/washington_handwritten_nntp \
  --overwrite
```

### Output

Creates a dataset with:

- **`gt/`**, **`transcription/`**, **`ocr/`**, **`images/`** - materialized source inputs
- **`line_images/`** - raw no-merge Kraken crops in reading order
- **`metadata/`** - one JSON file per page/sample with crop order and bounding boxes
- **`previews/`** - overlay images with crop boxes and line indices
- **`metadata.json`** - dataset-wide manifest
- **`review_status.json`** - template for manual verification statuses

---

## curate_washington_handwritten_nntp_dataset.py

Curate a Washington NNTP workspace so the final `line_images/` count matches the GT line count exactly.

This script preserves the raw workspace artifacts before rewriting the curated ones:

- raw crops move to `line_images_raw/`
- raw previews move to `previews_raw/`
- raw metadata is preserved in `metadata_raw/`, `metadata_raw.json`, and `review_status_raw.json`
- curated crops, previews, and manifests replace `line_images/`, `previews/`, `metadata/`, `metadata.json`, and `review_status.json`

### Basic Usage

```bash
python scripts/curate_washington_handwritten_nntp_dataset.py \
  --data-dir /tmp/washington_handwritten_nntp
```

### Output

Creates a curated dataset where:

- **`line_images/`** contains one crop per GT line
- **`line_images_raw/`** preserves the original no-merge Kraken output
- **`metadata.json`** reflects the curated line set
- **`review_status.json`** marks the curated Washington pages as ready for NNTP use

The committed canonical dataset already lives at `datasets/washington_handwritten/`. Use the builder/curator only when you want to reproduce or rework that curation in a scratch workspace.

---

## summarize_macro_avgs.py

Extract macro-average rows from evaluation CSVs.

---

## bundle_thesis_handwritten_predictions.py

Assemble the accepted handwritten thesis artifacts into one dated bundle under the thesis
predictions root.

The script is designed for the outputs of:

- `jobs/orchestrators/eval_thesis_handwritten_completion.sbatch`
- the existing valid April 2, 2026 `gpt-5.4` `M5 context100` bundles for Bullinger, Washington, and IAM representative 20

### Basic Usage

```bash
python scripts/bundle_thesis_handwritten_predictions.py \
  --run-root /path/to/synced/thesis_handwritten_completion_2026-04-09 \
  --bundle-name handwritten_thesis_bundle_2026-04-09
```

### Output

Creates a new bundle with:

- `children_handwritten/nntp/` - per-fold predictions, per-fold eval CSVs, and the merged CV summary CSV
- `children_handwritten/m2|m3|m4/<model>/` - rerun predictions, evaluation CSVs, and checkpoints
- `<dataset>/m5_context100/<model>/` - predictions, evaluation CSVs, checkpoints, and traces
- `manifest.json` - source paths, copied destinations, and count checks

### Basic Usage

```bash
# Summarize all CSVs in current directory
python scripts/summarize_macro_avgs.py

# Specify input directory
python scripts/summarize_macro_avgs.py --in-dir results/

# Custom glob pattern
python scripts/summarize_macro_avgs.py --glob "*gpt-5.2.csv"
```

### Output

- **`summary_long.csv`** - One row per (dataset, method)
- **`summary_wide.csv`** - One row per dataset, method-prefixed columns

---

## build_washington_pylaia_manifests.py

Build deterministic Washington 2-fold CV manifests for PyLaia fine-tuning.

This script reads the canonical `datasets/washington_handwritten/line_images/` and `gt/` pairs, validates them against `third_party/pylaia-iam/syms.txt`, and writes:

- user-facing `train.tsv` / `val.tsv` with absolute image paths and GT text
- PyLaia-native `train.txt` / `val.txt` / `test.txt`
- disjoint page ID lists for `train`, `val`, and `test`
- one `manifest_meta.json` per fold plus a top-level summary

### Basic Usage

```bash
# Build both CV folds
python scripts/build_washington_pylaia_manifests.py

# Rebuild only fold A
python scripts/build_washington_pylaia_manifests.py --fold train_a
```

### Output

Creates fold directories under `outputs/manifests/washington_handwritten_pylaia_cv/`:

- **`train_a/`**
- **`train_b/`**

Each fold directory contains:

- **`train.tsv`**, **`val.tsv`** - absolute-path TSV manifests
- **`train.txt`**, **`val.txt`**, **`test.txt`** - PyLaia text tables
- **`train_ids.txt`**, **`val_ids.txt`**, **`test_ids.txt`** - page/sample IDs
- **`manifest_meta.json`** - split counts and fold metadata

---

## build_children_pylaia_manifests.py

Build deterministic children-handwritten PyLaia manifests plus a dataset-specific
`syms.txt`.

The folds are document-level and fixed, so the held-out OCR and NNTP artifacts
can be generated with cross-fitted checkpoints rather than a single in-sample model.

### Basic Usage

```bash
python scripts/build_children_pylaia_manifests.py \
  --data-dir datasets/children_handwritten \
  --out-dir outputs/manifests/children_handwritten_pylaia_cv
```

### Output

Creates:

- **`outputs/manifests/children_handwritten_pylaia_cv/children.syms.txt`**
- **`outputs/manifests/children_handwritten_pylaia_cv/fold_a/`**
- **`outputs/manifests/children_handwritten_pylaia_cv/fold_b/`**
- **`outputs/manifests/children_handwritten_pylaia_cv/fold_c/`**
- **`manifest_summary.json`** - top-level fold and alphabet summary

---

## run_nntp_eval.py

Run the NNTP baseline using either local PAGE XML crops or presegmented line images.

### Basic Usage

```bash
# Bullinger smoke test from PAGE XML
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --ids 10069 \
  --stop-after prepare

# IAM smoke test from presegmented line images
python scripts/run_nntp_eval.py \
  --data-dir datasets/IAM_handwritten_rwth_test \
  --ids c04-110 \
  --stop-after prepare

# Full Bullinger NNTP pipeline
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --work-dir outputs/nntp/bullinger_handwritten \
  --nntp-root ../nntp

# Full IAM NNTP pipeline
python scripts/run_nntp_eval.py \
  --data-dir datasets/IAM_handwritten_rwth_test \
  --work-dir outputs/nntp/IAM_handwritten_rwth_test \
  --pred-dir iam_handwritten_rwth_test_predictions_nntp \
  --eval-csv iam_handwritten_rwth_test_eval_nntp.csv \
  --nntp-root ../nntp
```

### Output

- **Predictions:** dataset-specific `*_predictions_nntp/`
- **CSV metrics:** dataset-specific `*_eval_nntp.csv`
- **Intermediates:** dataset-specific `outputs/nntp/<dataset_name>/`

See [docs/nntp_pipeline.md](docs/nntp_pipeline.md) for stage details and caveats.

---

## run_children_crossfit_ocr.py

Generate held-out `ocr/` and `ocr_lines/` artifacts for `children_handwritten`
from the trained fold checkpoints.

### Basic Usage

```bash
python scripts/run_children_crossfit_ocr.py \
  --data-dir datasets/children_handwritten \
  --manifest-dir outputs/manifests/children_handwritten_pylaia_cv \
  --assets-root outputs/pylaia/children_handwritten
```

### Output

Writes canonical held-out artifacts directly under:

- **`datasets/children_handwritten/ocr/`**
- **`datasets/children_handwritten/ocr_lines/`**

---

## run_children_nntp_cv.py

Run the NNTP baseline fold-by-fold for `children_handwritten` and optionally
write a merged macro summary CSV.

### Basic Usage

```bash
python scripts/run_children_nntp_cv.py \
  --data-dir datasets/children_handwritten \
  --manifest-dir outputs/manifests/children_handwritten_pylaia_cv \
  --assets-root outputs/pylaia/children_handwritten
```

### Output

- **`children_handwritten_eval_nntp_fold_a.csv`**
- **`children_handwritten_eval_nntp_fold_b.csv`**
- **`children_handwritten_eval_nntp_fold_c.csv`**
- **`children_handwritten_eval_nntp_cv.csv`** - merged macro summary when all folds finish

---

## summarize_washington_nntp_cv.py

Summarize the two Washington NNTP fold evaluation CSVs into one 3-row CSV.

### Basic Usage

```bash
python scripts/summarize_washington_nntp_cv.py
```

### Output

- **`washington_handwritten_eval_nntp_cv_macro.csv`** - rows for `train_a_test_b`, `train_b_test_a`, and `macro_avg`

---

## summarize_children_nntp_cv.py

Summarize the three children NNTP fold evaluation CSVs into one macro CSV.

### Basic Usage

```bash
python scripts/summarize_children_nntp_cv.py
```

### Output

- **`children_handwritten_eval_nntp_cv.csv`** - rows for `fold_a`, `fold_b`, `fold_c`, and `macro_avg`

---

## Additional Utilities

Located in `scripts/` and `utils/`:

- **`convert_iam_dataset.py`** - Convert IAM database to project format
- **`build_iam_rwth_dataset.py`** - Build the RWTH IAM form split used by the NNTP baseline
- **`build_washington_pylaia_manifests.py`** - Build Washington PyLaia 2-fold CV manifests
- **`convert_washington_gt.py`** - Convert washington_handwritten ground truth
- **`copy_pages_to_images.py`** - Organize page images
- **`summarize_washington_nntp_cv.py`** - Combine the two Washington NNTP fold CSVs into one macro summary

---

## Related Documentation

- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - OCR generation details
- **[datasets/README.md](datasets/README.md)** - Dataset structure
- **[METRICS.md](METRICS.md)** - Metrics in CSV output
