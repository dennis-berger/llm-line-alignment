# Scripts Guide

## make_ocr_outputs.py

Generate OCR/HTR outputs for datasets.

### Basic Usage

```bash
# Generate OCR for a dataset
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten

# Process specific samples
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten --ids 0001,0002

# Use different recognizer
python scripts/make_ocr_outputs.py --dataset bullinger_print --recognizer trocr_printed
```

### Common Options

- `--dataset <name>` - Dataset name
- `--ids <list>` - Comma-separated IDs or path to file
- `--recognizer` - `trocr_handwritten`, `trocr_printed`, `pylaia_iam`, `htr_best_practices_iam`
- `--segmenter` - `kraken` (default), `none` (passthrough)
- `--device` - `cuda`, `cpu`, `auto`
- `--batch-size` - Recognition batch size (default: 8)
- `--overwrite` - Regenerate existing outputs

### Output

Creates `ocr/<id>.txt`, `ocr_lines/<id>.json`, and optional `ocr/<id>.meta.json` in dataset directory.

For datasets that include `line_images/`, the script can reuse those presegmented crops with
`--segmenter none --existing-lines-dir ...`. For IAM handwritten datasets, it automatically switches to
`--segmenter none` and `--recognizer pylaia_iam` unless you override those flags.

See [docs/ocr_pipeline.md](docs/ocr_pipeline.md) for technical details.

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

## run_nntp_eval.py

Run the NNTP baseline using either local PAGE XML crops or presegmented line images.

### Basic Usage

```bash
# Bullinger smoke test from PAGE XML
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --ids 10177 \
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

## summarize_washington_nntp_cv.py

Summarize the two Washington NNTP fold evaluation CSVs into one 3-row CSV.

### Basic Usage

```bash
python scripts/summarize_washington_nntp_cv.py
```

### Output

- **`washington_handwritten_eval_nntp_cv_macro.csv`** - rows for `train_a_test_b`, `train_b_test_a`, and `macro_avg`

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
