# bullinger-line-alignment-mwe

Line alignment experiments for historical OCR/HTR, with a focus on Bullinger letters. The goal is to insert **correct line breaks** into clean transcriptions by leveraging page images and/or noisy OCR line structure, then evaluate alignment quality with line‑level metrics.

**At A Glance**
1. Methods M1–M3 align line breaks using different inputs (image, transcription, OCR).
2. Unified OCR/HTR generation writes `ocr/<id>.txt` per dataset.
3. Evaluation reports WER/CER plus line‑level metrics, including exact‑line precision/recall/F1.

**Methods**
1. `M1` (image + correct transcription): insert line breaks using visual layout only.
2. `M2` (image + transcription + OCR): use OCR line structure as a guide, verify with image.
3. `M3` (transcription + OCR): align line breaks without images.

**Repository Layout**
1. `src/linealign/`: core pipeline code (segmentation, recognition, OCR generation).
2. `scripts/`: CLI entry points (e.g., OCR generation).
3. `run_eval_m1.py`, `run_eval_m2.py`, `run_eval_m3.py`: evaluation runners for each method.
4. `datasets/`: dataset roots and standardized subfolders.
5. `utils/`: shared helpers and prompts.
6. `tests/`: unit tests for metrics.
7. `docs/`: pipeline documentation.

**Datasets**
Datasets live under `datasets/` with a consistent structure:
1. `gt/`: ground‑truth text with line breaks.
2. `transcription/`: correct text without line breaks.
3. `images/`: page images per letter or page.
4. `ocr/`: OCR/HTR output with line breaks (generated).

See `DATASET_ORGANIZATION.md` for details and migration notes.

**Setup**
Python version is pinned in `.python-version` (currently `3.11.9`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional dependency for segmentation:
```bash
pip install kraken
```

**Quickstart**
Generate OCR/HTR outputs (required for M2/M3):
```bash
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten
```

Run evaluations:
```bash
python run_eval_m1.py
python run_eval_m2.py
python run_eval_m3.py
```

Run on another dataset:
```bash
python run_eval_m1.py --data-dir datasets/bullinger_print
```

**OCR/HTR Generation Pipeline**
Full details are in `docs/ocr_pipeline.md`. The CLI writes `ocr/<id>.txt` and optional `ocr/<id>.meta.json` per dataset sample.

**Evaluation Metrics**
Evaluation includes:
1. `WER` / `CER` (raw and whitespace‑normalized).
2. `line_accuracy` (forward and reverse; raw and normalized).
3. `exact_line_precision`, `exact_line_recall`, `exact_line_f1` (raw and normalized).

See `EXACT_LINE_METRICS_IMPLEMENTATION.md` for definitions and tests.

**Outputs**
Each eval script writes:
1. Per‑sample predictions to `predictions_m*/<id>.txt`.
2. A CSV summary (e.g., `evaluation_qwen_m1.csv`).

Generated OCR lives under each dataset’s `ocr/` directory.

**Reproducibility Tips**
1. Record the model ID and GPU type for any run (Qwen/TrOCR/kraken versions matter).
2. Use `--shots-seed` for stable few‑shot selection.
3. Keep a copy of evaluation CSVs and `ocr/*.meta.json` as run artifacts.

**Testing**
Run metric tests:
```bash
pytest tests/test_exact_line_metrics.py -v
```

**Troubleshooting**
1. If kraken is missing, install it or run `--segmenter none` with pre‑segmented lines.
2. If CUDA is unavailable, use `--device cpu` for OCR generation.
3. For quick smoke tests, use `--max-pages 1` or a small `--ids` list.
