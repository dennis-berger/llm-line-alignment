# llm-line-alignment

**Master's thesis project evaluating line alignment methods for document transcription.** The goal is to insert correct line breaks into clean diplomatic transcriptions by leveraging page images and/or noisy OCR/HTR line structure.

## Overview

Five methods (M1-M5) align line breaks using different input combinations: page image only, page image + OCR hints, text-only OCR alignment, structured OCR-line alignment, and line-image alignment. Each method is evaluated against multiple datasets (historical handwritten, modern handwritten, and printed) with character-level and line-level metrics.

**Key features:**
- Vision-language models for layout-aware alignment (M1, M2)
- Text-only alignment using HTR line structure hints (M3)
- Structured OCR-line alignment with exact line-count control (M4)
- Vision alignment with ordered line-image crops, optionally plus OCR text hints (M5)
- Comprehensive metrics: WER/CER, line accuracy, exact line matching (P/R/F1)
- Support for multi-page documents and various writing styles

## Quick Start

**1. Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Refresh Bullinger from the ICCV export when needed:**
```bash
python scripts/import_bullinger_iccv_testset.py \
  --source-dir ../iccv-testset \
  --out-dir datasets/bullinger_handwritten \
  --overwrite
```

**3. Rebuild Bullinger line images and PyLaia OCR outputs (required for M2, M3, M4, and M5):**
```bash
python scripts/build_bullinger_handwritten_line_images.py \
  --data-dir datasets/bullinger_handwritten \
  --out-dir datasets/bullinger_handwritten/line_images \
  --overwrite

python scripts/make_ocr_outputs.py \
  --dataset bullinger_handwritten \
  --data-dir datasets/bullinger_handwritten \
  --segmenter none \
  --existing-lines-dir datasets/bullinger_handwritten/line_images \
  --recognizer pylaia \
  --pylaia-root third_party/pylaia-bullinger \
  --pylaia-checkpoint third_party/pylaia-bullinger/epoch=170-lowest_va_cer.ckpt \
  --pylaia-syms third_party/pylaia-bullinger/syms.txt \
  --overwrite
```

**4. Run an evaluation:**
```bash
python run_eval_m1.py --data-dir datasets/bullinger_handwritten  # Image-only alignment
```

**5. Try other methods or datasets:**
```bash
python run_eval_m2.py --data-dir datasets/bullinger_handwritten  # Image + OCR hints
python run_eval_m3.py --data-dir datasets/bullinger_handwritten  # Text-only alignment
python run_eval_m4.py --data-dir datasets/bullinger_handwritten  # Structured OCR-line alignment
python run_eval_m5.py --data-dir datasets/bullinger_handwritten  # Line-image alignment
python run_eval_m1.py --data-dir datasets/bullinger_print
```

**6. Build children_handwritten with cross-fitted PyLaia artifacts:**
```bash
python scripts/build_children_handwritten_dataset.py \
  --source-dir ../children_hw_original/alignment_tests \
  --out-dir datasets/children_handwritten \
  --overwrite

python scripts/build_children_pylaia_manifests.py \
  --data-dir datasets/children_handwritten \
  --out-dir outputs/manifests/children_handwritten_pylaia_cv

# Train one fold at a time on the cluster
sbatch jobs/training/train_children_handwritten_pylaia_cv.sbatch

# After the three fold checkpoints exist, generate held-out OCR hints
python scripts/run_children_crossfit_ocr.py
```

## Documentation

- **[METHODS.md](METHODS.md)** - Detailed explanation of M1, M2, and M3 approaches
- **[METRICS.md](METRICS.md)** - All evaluation metrics with formulas and interpretation
- **[datasets/README.md](datasets/README.md)** - Dataset structure and characteristics
- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - OCR/HTR generation pipeline details
- **[docs/nntp_pipeline.md](docs/nntp_pipeline.md)** - NNTP baseline pipeline from local PAGE XML
- **[jobs/README.md](jobs/README.md)** - Running evaluations on HPC clusters
- **[scripts/README.md](scripts/README.md)** - Utility scripts for data processing
- **[tests/README.md](tests/README.md)** - Testing guide

## Project Structure

```
├── datasets/           # Datasets
├── src/linealign/      # Core pipeline (segmentation, recognition)
├── scripts/            # Data processing utilities
├── run_eval_m*.py      # Evaluation scripts for each method
├── utils/              # Shared helpers and prompts
├── tests/              # Unit tests
├── jobs/               # HPC cluster batch scripts
└── docs/               # Technical documentation
```

## Outputs

Evaluation scripts produce:
- **Predictions:** `predictions_m1/`, `predictions_m2/`, `predictions_m3/`
- **CSV metrics:** `evaluation_qwen_m1.csv` (per-sample and macro-average)
- **Generated OCR:** `datasets/*/ocr/` (cached for reproducibility)

## Testing

```bash
pytest tests/ -v
```

## Requirements

- Python 3.11.9 (see `.python-version`)
- PyTorch with CUDA support (recommended)
- Optional: `kraken` for line segmentation

## License & Citation

This is research code for a Master's thesis on line alignment in document transcription. If you use this code, please cite the associated thesis (details TBD).
