# Line Alignment Methods

Three methods (M1, M2, M3) for inserting line breaks into clean transcriptions. All use vision-language models (Qwen2-VL-7B-Instruct).

## Method 1: Image + Transcription

**Inputs:** Page images + correct transcription (no line breaks)

**Task:** Insert line breaks by analyzing visual layout

**Strengths:**
- Direct visual evidence
- No dependency on OCR

**Weaknesses:**
- VLM may hallucinate or skip text
- Requires quality page images

**Usage:**
```bash
python run_eval_m1.py
```

## Method 2: Image + Transcription + HTR

**Inputs:** Page images + correct transcription + HTR output (with line breaks)

**Task:** Use HTR line structure as primary guide, verify with image

**Strengths:**
- HTR provides explicit line positions
- Image verification prevents errors
- Most constrained (typically best performance)

**Weaknesses:**
- Requires pre-generated HTR
- HTR errors can propagate

**Usage:**
```bash
python run_eval_m2.py
```

## Method 3: Transcription + HTR (Text-Only)

**Inputs:** Correct transcription + HTR output (no images)

**Task:** Align texts and transfer line breaks from HTR to transcription

**Strengths:**
- No images needed (faster, less memory)
- Pure text alignment

**Weaknesses:**
- No visual verification
- Fully dependent on HTR quality

**Usage:**
```bash
python run_eval_m3.py
```

## Comparison

| Feature | M1 | M2 | M3 |
|---------|----|----|-----|
| **Images** | ✅ | ✅ | ❌ |
| **HTR** | ❌ | ✅ | ✅ |
| **Complexity** | Low | Medium | Low |
| **Speed** | Slower | Slower | Faster |
| **Typical Performance** | Medium | High | Variable |

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

## Configuration

**Default model:** `Qwen/Qwen2-VL-7B-Instruct`

**Generation:**
- max_new_tokens = 2048
- temperature = 0.0 (deterministic)
- do_sample = False

**Hardware:**
- M1/M2: ~16GB GPU
- M3: ~12GB GPU
- CPU mode available (slow)

## Related Documentation

- **[METRICS.md](METRICS.md)** - Evaluation metrics
- **[datasets/README.md](datasets/README.md)** - Dataset structure
- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - HTR generation
