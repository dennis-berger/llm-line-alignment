# OCR/HTR Generation Pipeline

This pipeline produces line-structured recognition artifacts used by the OCR/HTR
conditioned methods. The plain text output `ocr/<id>.txt` supports M2 and M3.
The structured output `ocr_lines/<id>.json` supports M4 and M5 by preserving the
ordered line list, page/line indices, and available crop paths.

## What it does
- Segments each page image into line crops (Kraken by default, caching crops).
- Recognizes each line with a selectable backend (TrOCR presets by default, PyLaia for IAM when presegmented line images are available).
- Reassembles page text with one line per visual line; multi-page letters join pages with a blank line.
- Writes OCR text to `<data-dir>/ocr/<id>.txt`.
- Writes structured line records to `<data-dir>/ocr_lines/<id>.json` when line
  crops or line-level metadata are available.
- Writes an optional `<id>.meta.json` sidecar with recognizer/segmenter metadata.

## Relationship To Thesis Methods

- **M2** uses `ocr/<id>.txt` as a noisy structural hint alongside page images.
- **M3** uses `ocr/<id>.txt` as the only structural hint.
- **M4** uses ordered `ocr_lines/<id>.json` text hypotheses and the expected line
  count.
- **M5** uses the `crop_path` entries in `ocr_lines/<id>.json` to load ordered
  line images, optionally using the OCR text fields as secondary hints.
- **NNTP** can reuse the same presegmented line-image assets, but it has its own
  pipeline described in [nntp_pipeline.md](nntp_pipeline.md).

## Dependencies
- Required: `pillow`, `torch`, `transformers` (already in requirements.txt).
- Optional for segmentation: `kraken` (`pip install kraken` or `pip install .[kraken]`).
- Optional for PyLaia recognition: `pylaia-htr-netout` available on `PATH`, plus vendored assets in `third_party/pylaia-bullinger/` or `third_party/pylaia-iam/`.
- Cluster note: CUDA is available on FAITH GPU jobs; CPU works on login nodes.

## CLI
```bash
python scripts/make_ocr_outputs.py \
  --dataset bullinger_handwritten \
  --data-dir datasets/bullinger_handwritten \
  --segmenter none \
  --existing-lines-dir datasets/bullinger_handwritten/line_images \
  --recognizer pylaia \
  --pylaia-root third_party/pylaia-bullinger \
  --pylaia-checkpoint third_party/pylaia-bullinger/epoch=170-lowest_va_cer.ckpt \
  --pylaia-syms third_party/pylaia-bullinger/syms.txt \
  --cache-dir outputs/cache/bullinger_handwritten/lines \
  --overwrite
```

Key flags:
- `--dataset`: one of bullinger_handwritten, bullinger_print, washington_handwritten, IAM_handwritten, IAM_print.
  - `children_handwritten` is also supported and can reuse canonical `line_images/` with `--segmenter none`.
- `--data-dir`: root containing gt/, images/, transcription/, ocr/ (defaults to datasets/<dataset>).
- `--ids`: comma list or file of IDs to process; defaults to all IDs in gt/ (fallback: transcription/).
- `--segmenter`: `kraken` (default) or `none` (uses pre-segmented lines or full page as one line).
- `--recognizer`: `trocr_printed`, `trocr_handwritten`, `pylaia`, `pylaia_iam` (legacy alias), `htr_best_practices_iam` (alias of the same PyLaia IAM checkpoint), `none` (not supported).
- `--device`: `auto`, `cpu`, or `cuda:0`.
- `--cache-dir`: where line crops are stored; defaults to `outputs/cache/<dataset>/lines`.
- `--pylaia-root`, `--pylaia-checkpoint`, `--pylaia-syms`: override the vendored PyLaia assets when using `pylaia`.
- `--pylaia-gpus`, `--pylaia-auto-select-gpus`, `--pylaia-fixed-height`: advanced PyLaia runtime overrides.
- `--max-pages`: limit pages per ID for smoke tests.
- `--overwrite`: recompute even if `ocr/<id>.txt` exists (default is skip-existing).
- `--dry-run`: list what would run without doing work.

## Dataset notes
- Paths are unified: `<data-dir>/ocr/<id>.txt` for every dataset.
- Structured line artifacts, when generated, live at
  `<data-dir>/ocr_lines/<id>.json`.
- IDs:
  - Bullinger (handwritten/print): `<id>` is a letter; may span multiple page images under `images/<id>/`.
  - washington_handwritten: treat `<id>` as a single page (e.g., `270`).
  - IAM (handwritten/print): `<id>` is a form/page ID.
- Bullinger handwritten reproducibility runs should use the precomputed PAGE-derived `line_images/` plus the Bullinger checkpoint in `third_party/pylaia-bullinger/`.
- IAM handwritten with `line_images/`: if `--segmenter` and `--recognizer` are not given, the script defaults to `--segmenter none` plus `--recognizer pylaia`.
- Children handwritten cross-fitted PyLaia runs should use `--segmenter none` with `datasets/children_handwritten/line_images/` so OCR and NNTP consume the same presegmented crops.
- Multi-page outputs: pages concatenate in order with a blank line between pages.

## Examples
- Bullinger handwritten with the Bullinger PyLaia checkpoint:
  ```bash
  python scripts/make_ocr_outputs.py \
    --dataset bullinger_handwritten \
    --data-dir datasets/bullinger_handwritten \
    --segmenter none \
    --existing-lines-dir datasets/bullinger_handwritten/line_images \
    --recognizer pylaia \
    --pylaia-root third_party/pylaia-bullinger \
    --pylaia-checkpoint third_party/pylaia-bullinger/epoch=170-lowest_va_cer.ckpt \
    --pylaia-syms third_party/pylaia-bullinger/syms.txt
  ```
- Bullinger print with printed recognizer:
  ```bash
  python scripts/make_ocr_outputs.py --dataset bullinger_print --recognizer trocr_printed
  ```
- Washington handwritten, limit to 2 IDs on CPU:
  ```bash
  python scripts/make_ocr_outputs.py --dataset washington_handwritten --device cpu --ids "270,271" --max-pages 1
  ```
- IAM handwritten RWTH subset with PyLaia and official line images:
  ```bash
  python scripts/make_ocr_outputs.py \
    --dataset IAM_handwritten \
    --data-dir datasets/IAM_handwritten_rwth_test_representative_20
  ```

- IAM handwritten forcing TrOCR on presegmented line images:
  ```bash
  python scripts/make_ocr_outputs.py \
    --dataset IAM_handwritten \
    --data-dir datasets/IAM_handwritten_rwth_test_representative_20 \
    --segmenter none \
    --existing-lines-dir datasets/IAM_handwritten_rwth_test_representative_20/line_images \
    --recognizer trocr_handwritten
  ```

- Children handwritten with held-out PyLaia fold assets:
  ```bash
  python scripts/make_ocr_outputs.py \
    --dataset children_handwritten \
    --data-dir datasets/children_handwritten \
    --segmenter none \
    --existing-lines-dir datasets/children_handwritten/line_images \
    --recognizer pylaia \
    --pylaia-root outputs/pylaia/children_handwritten/fold_a \
    --pylaia-checkpoint outputs/pylaia/children_handwritten/fold_a/best.ckpt \
    --pylaia-syms outputs/pylaia/children_handwritten/fold_a/syms.txt \
    --ids "$(paste -sd, outputs/manifests/children_handwritten_pylaia_cv/fold_a/test_ids.txt)"
  ```

## Tips
- Caching: line crops live under `outputs/cache/<dataset>/lines/<id>/`; reruns reuse them unless `--overwrite`.
- If Kraken is missing, install it or switch to `--segmenter none` with pre-segmented lines.
- For IAM handwritten fairness comparisons, prefer the RWTH datasets with `line_images/` so PyLaia and TrOCR both operate on the same official line crops.
- For FAITH Slurm jobs, request a GPU partition to use CUDA; on CPU runs keep `--batch-size` small.
- Reproducibility: models run in eval/inference_mode; no fixed seed enforced for speed.
