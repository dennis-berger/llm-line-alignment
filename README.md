# bullinger-line-alignment-mwe

**Master's thesis project evaluating line alignment methods for document transcription.** The goal is to insert correct line breaks into clean diplomatic transcriptions by leveraging page images and/or noisy OCR/HTR line structure.

## Overview

Three methods (M1, M2, M3) align line breaks using different input combinations—image only, image + OCR hints, or text-only alignment. Each method is evaluated against multiple datasets (historical handwritten, modern handwritten, and printed) with character-level and line-level metrics.

**Key features:**
- Vision-language models for layout-aware alignment (M1, M2)
- Text-only alignment using HTR line structure hints (M3)
- Comprehensive metrics: WER/CER, line accuracy, exact line matching (P/R/F1)
- Support for multi-page documents and various writing styles

## Quick Start

**1. Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Generate OCR outputs (required for M2 and M3):**
```bash
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten
```

**3. Run an evaluation:**
```bash
python run_eval_m1.py  # Image-only alignment
```

**4. Try other methods or datasets:**
```bash
python run_eval_m2.py  # Image + OCR hints
python run_eval_m3.py  # Text-only alignment
python run_eval_m1.py --data-dir datasets/bullinger_print
```

## Documentation

- **[METHODS.md](METHODS.md)** - Detailed explanation of M1, M2, and M3 approaches
- **[METRICS.md](METRICS.md)** - All evaluation metrics with formulas and interpretation
- **[datasets/README.md](datasets/README.md)** - Dataset structure and characteristics
- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - OCR/HTR generation pipeline details
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
