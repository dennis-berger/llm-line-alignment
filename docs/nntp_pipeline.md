# NNTP Pipeline

This document describes the Bullinger handwritten NNTP baseline added in this repository.

## Overview

The NNTP baseline runs in five stages:

1. `prepare`
2. `netout`
3. `convert`
4. `align`
5. `evaluate`

It uses the existing PAGE XML under `datasets/bullinger_handwritten/images/**/page/*.xml` to crop line images locally. It does not use Kraken and does not require the external `bullinger-htr` repository.

## External Dependencies

Two external tools are still required:

- `pylaia-htr-netout`
- Java (`java`, `javac`) for the sibling NNTP repo

Vendored PyLaia assets are included directly in this repo:

- PyLaia assets in `third_party/pylaia-dennis/`
  - `model`
  - `epoch=170-lowest_va_cer.ckpt`
  - `syms.txt`
- NNTP repo in `../nntp/`

## Basic Usage

Prepare-only smoke test:

```bash
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --ids 10177 \
  --stop-after prepare
```

Full run on Linux/cluster:

```bash
python scripts/run_nntp_eval.py \
  --data-dir datasets/bullinger_handwritten \
  --work-dir outputs/nntp/bullinger_handwritten \
  --nntp-root ../nntp
```

## Main Outputs

Generated working files live under `outputs/nntp/bullinger_handwritten/` by default:

- `line_images/<id>/`
- `prepare/pylaia_images.txt`
- `prepare/stripped_chars.json`
- `netout/lattice.txt`
- `split_lattices/<id>/`
- `observations_lines/<id>/`
- `observations_letters/<id>.txt`
- `boundaries/<id>.json`
- `nntp/recognitions/<id>.rec`

Final comparison artifacts:

- `bullinger_handwritten_predictions_nntp/<id>.txt`
- `bullinger_handwritten_eval_nntp.csv`

## Important Caveat

NNTP labels are filtered to the provided PyLaia `syms.txt` character set. Unsupported characters are stripped before alignment and are not reconstructed afterward.

This means the NNTP CSV can be compared directly to M1-M3, but it is not a perfectly apples-to-apples text comparison. The stripped-character report is written to:

```text
outputs/nntp/bullinger_handwritten/prepare/stripped_chars.json
```

## PAGE XML Rules

The pipeline uses PAGE XML like this:

- resolve the page image from the XML `Page imageFilename`
- read region order from `ReadingOrder/RegionRefIndexed`
- read line order from `readingOrder {index:...;}` when present
- crop each line from the `TextLine/Coords` polygon bbox with a small padding
- skip empty/structural marker lines such as `{MN}` and `{MT}[121.]`

## Notes

- `--overwrite` forces regeneration of the expensive external stages.
- `--stop-after` is useful for debugging intermediate outputs.
- The evaluation CSV uses the same schema and macro-average row format as the existing `run_eval_m*.py` scripts.
- `jobs/eval/nntp/bullinger_handwritten.sbatch` is the cluster launcher matching the existing SLURM job layout.
