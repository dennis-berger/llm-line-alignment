# OCR/HTR Generation Pipeline

This pipeline produces `ocr/<id>.txt` for all datasets so Method 2 can run uniformly.

## What it does
- Segments each page image into line crops (Kraken by default, caching crops).
- Recognizes each line with a selectable backend (TrOCR presets by default).
- Reassembles page text with one line per visual line; multi-page letters join pages with a blank line.
- Writes outputs to `<data-dir>/ocr/<id>.txt` and an optional `<id>.meta.json` sidecar.

## Dependencies
- Required: `pillow`, `torch`, `transformers` (already in requirements.txt).
- Optional for segmentation: `kraken` (`pip install kraken` or `pip install .[kraken]`).
- Cluster note: CUDA is available on FAITH GPU jobs; CPU works on login nodes.

## CLI
```bash
python scripts/make_ocr_outputs.py \
  --dataset bullinger_handwritten \
  --data-dir datasets/bullinger_handwritten \
  --segmenter kraken \
  --recognizer trocr_handwritten \
  --device auto \
  --cache-dir outputs/cache/bullinger_handwritten/lines
```

Key flags:
- `--dataset`: one of bullinger_handwritten, bullinger_print, easy_historical, IAM_handwritten, IAM_print.
- `--data-dir`: root containing gt/, images/, transcription/, ocr/ (defaults to datasets/<dataset>).
- `--ids`: comma list or file of IDs to process; defaults to all IDs in gt/ (fallback: transcription/).
- `--segmenter`: `kraken` (default) or `none` (uses pre-segmented lines or full page as one line).
- `--recognizer`: `trocr_printed`, `trocr_handwritten` (default), `htr_best_practices_iam` (placeholder), `none` (not supported).
- `--device`: `auto`, `cpu`, or `cuda:0`.
- `--cache-dir`: where line crops are stored; defaults to `outputs/cache/<dataset>/lines`.
- `--max-pages`: limit pages per ID for smoke tests.
- `--overwrite`: recompute even if `ocr/<id>.txt` exists (default is skip-existing).
- `--dry-run`: list what would run without doing work.

## Dataset notes
- Paths are unified: `<data-dir>/ocr/<id>.txt` for every dataset.
- IDs:
  - Bullinger (handwritten/print): `<id>` is a letter; may span multiple page images under `images/<id>/`.
  - easy_historical: treat `<id>` as a single page (e.g., `270`).
  - IAM (handwritten/print): `<id>` is a form/page ID.
- Multi-page outputs: pages concatenate in order with a blank line between pages.

## Examples
- Bullinger handwritten (default everything):
  ```bash
  python scripts/make_ocr_outputs.py --dataset bullinger_handwritten
  ```
- Bullinger print with printed recognizer:
  ```bash
  python scripts/make_ocr_outputs.py --dataset bullinger_print --recognizer trocr_printed
  ```
- Easy historical, limit to 2 IDs on CPU:
  ```bash
  python scripts/make_ocr_outputs.py --dataset easy_historical --device cpu --ids "270,271" --max-pages 1
  ```
- IAM handwritten (using TrOCR handwritten):
  ```bash
  python scripts/make_ocr_outputs.py --dataset IAM_handwritten --recognizer trocr_handwritten
  ```

## Tips
- Caching: line crops live under `outputs/cache/<dataset>/lines/<id>/`; reruns reuse them unless `--overwrite`.
- If Kraken is missing, install it or switch to `--segmenter none` with pre-segmented lines.
- For FAITH Slurm jobs, request a GPU partition to use CUDA; on CPU runs keep `--batch-size` small.
- Reproducibility: models run in eval/inference_mode; no fixed seed enforced for speed.
