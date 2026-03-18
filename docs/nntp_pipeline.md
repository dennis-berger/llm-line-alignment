# NNTP Pipeline

This document describes the NNTP baseline in this repository for both PAGE XML datasets such as Bullinger and presegmented-line datasets such as IAM RWTH.

## Overview

The NNTP baseline runs in five stages:

1. `prepare`
2. `netout`
3. `convert`
4. `align`
5. `evaluate`

It supports two preparation backends:

- `pagexml` for datasets with PAGE XML geometry under `images/**/page/*.xml`
- `presegmented` for datasets that already provide `line_images/<id>/*.png`

`--prepare-mode auto` selects `presegmented` when `line_images/<id>/` exists and otherwise falls back to `pagexml`.

## External Dependencies

Two external tools are still required:

- `pylaia-htr-netout`
- Java (`java`, `javac`) for the sibling NNTP repo

Vendored PyLaia assets are included directly in this repo:

- PyLaia assets in `third_party/pylaia-dennis/`
  - `model`
  - `epoch=170-lowest_va_cer.ckpt`
  - `syms.txt`
- PyLaia assets in `third_party/pylaia-iam/`
  - `model`
  - `weights.ckpt`
  - `syms.txt`
- NNTP repo in `../nntp/`

## Basic Usage

### Bullinger from PAGE XML

```bash
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --ids 10177 \
  --stop-after prepare
```

### IAM RWTH test split from presegmented lines

Build the dataset once:

```bash
python scripts/build_iam_rwth_dataset.py \
  --iam-root ../iam/data \
  --split test \
  --out-dir datasets/IAM_handwritten_rwth_test
```

Smoke-test preparation:

```bash
python scripts/run_nntp_eval.py \
  --data-dir datasets/IAM_handwritten_rwth_test \
  --ids c04-110 \
  --stop-after prepare
```

Full runs:

```bash
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --work-dir outputs/nntp/bullinger_handwritten \
  --nntp-root ../nntp

python scripts/run_nntp_eval.py \
  --data-dir datasets/IAM_handwritten_rwth_test \
  --work-dir outputs/nntp/IAM_handwritten_rwth_test \
  --pred-dir iam_handwritten_rwth_test_predictions_nntp \
  --eval-csv iam_handwritten_rwth_test_eval_nntp.csv \
  --nntp-root ../nntp
```

## Main Outputs

Generated working files live under `outputs/nntp/<dataset_name>/`:

- `line_images/<id>/`
- `prepare/pylaia_images.txt`
- `prepare/stripped_chars.json`
- `netout/lattice.txt`
- `split_lattices/<id>/`
- `observations_lines/<id>/`
- `observations_letters/<id>.txt`
- `boundaries/<id>.json`
- `nntp/recognitions/<id>.rec`

Final comparison artifacts are dataset-specific:

- `<dataset>_predictions_nntp/<id>.txt`
- `<dataset>_eval_nntp.csv`

## Important Caveat

NNTP labels are filtered to the provided PyLaia `syms.txt` character set. Unsupported characters are stripped before alignment and are not reconstructed afterward.

This means the NNTP CSV can be compared directly to M1-M3, but it is not a perfectly apples-to-apples text comparison. The stripped-character report is written under the active work directory:

```text
outputs/nntp/<dataset_name>/prepare/stripped_chars.json
```

## Dataset-Specific Preparation Rules

For PAGE XML datasets, the pipeline:

- resolve the page image from the XML `Page imageFilename`
- read region order from `ReadingOrder/RegionRefIndexed`
- read line order from `readingOrder {index:...;}` when present
- crop each line from the `TextLine/Coords` polygon bbox with a small padding
- skip empty/structural marker lines such as `{MN}` and `{MT}[121.]`

For presegmented datasets, the pipeline:

- reads sorted line images from `line_images/<id>/`
- requires the line-image count to match `gt/<id>.txt`
- stages those line images into the NNTP work dir before netout
- uses the GT line order as the canonical source-text order for boundary reconstruction

## Notes

- The IAM RWTH dataset builder uses `../iam/data/ascii/lines.txt` as the transcription source and materializes form ids from the official OpenSLR SLR56 split lists.
- `--overwrite` forces regeneration of the expensive external stages.
- `--stop-after` is useful for debugging intermediate outputs.
- The evaluation CSV uses the same schema and macro-average row format as the existing `run_eval_m*.py` scripts.
- `jobs/eval/nntp/bullinger_handwritten.sbatch` is the cluster launcher matching the existing SLURM job layout.
- `jobs/eval/nntp/iam_handwritten.sbatch` runs the IAM RWTH test split with the public PyLaia IAM checkpoint.
