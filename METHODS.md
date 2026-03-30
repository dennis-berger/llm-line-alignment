# Line Alignment Methods

Five methods (M1-M5) for inserting line breaks into clean transcriptions. The newer methods add structured OCR-line hints and ordered line-image crops on top of the original page-image and text-only baselines.

## Method 1: Image + Transcription

**Inputs:** Page images + correct transcription (no line breaks)

**Task:** Insert line breaks by analyzing visual layout

**Usage:**
```bash
python run_eval_m1.py
```

## Method 2: Image + Transcription + HTR

**Inputs:** Page images + correct transcription + HTR output (with line breaks)

**Task:** Use HTR line structure as primary guide, verify with image

**Usage:**
```bash
python run_eval_m2.py
```

## Method 3: Transcription + HTR (Text-Only)

**Inputs:** Correct transcription + HTR output (no images)

**Task:** Align texts and transfer line breaks from HTR to transcription

**Usage:**
```bash
python run_eval_m3.py
```

## Method 4: Transcription + Structured OCR Lines

**Inputs:** Correct transcription + ordered OCR/HTR line hypotheses from `ocr_lines/<id>.json`

**Task:** Align the transcription to the ordered OCR line list and return exactly one output line per OCR line

```bash
python run_eval_m4.py
```

## Method 5: Transcription + Line Images

**Inputs:** Correct transcription + ordered line images from `ocr_lines/<id>.json` crop paths

**Task:** Align the transcription to the ordered line-image crops and return exactly one output line per supplied line image

```bash
python run_eval_m5.py
```

## Few-Shot Learning

All methods support 0-shot or N-shot evaluation:

```bash
# 0-shot (no examples)
python run_eval_m1.py --shots 0

# 1-shot (one example)
python run_eval_m1.py --shots 1 --shots-seed 42

# 3-shot
python run_eval_m2.py --shots 3 --shots-seed 42
```

## Multi-Page Handling

For multi-page documents:
- **M1/M2:** Load all page images, concatenate
- **M3:** HTR already contains all pages

## Design Constraints

All methods must:
1. Never change characters (only insert `\n`)
2. Preserve text order
3. Handle edge cases (empty lines, Unicode, etc.)

## Related Documentation

- **[METRICS.md](METRICS.md)** - Evaluation metrics
- **[datasets/README.md](datasets/README.md)** - Dataset structure
- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - HTR generation
