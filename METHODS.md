# Line Alignment Methods

This repository evaluates methods for inserting line breaks into a clean
transcription. The desired output is the same character sequence as the input
transcription, with newline characters inserted so that the text matches the
document layout.

The methods differ in the evidence available to the model and in how strongly
the output contract is enforced. `M1`-`M3` rely on prompt instructions and score
the raw model response. `M4` and `M5` require structured JSON, repair invalid
responses, and project accepted line boundaries back onto the exact
transcription before scoring.

For a thesis-facing map of methods, artifacts, and caveats, see
[docs/thesis_reader_guide.md](docs/thesis_reader_guide.md).

## Shared Inputs

All datasets use the same core file vocabulary:

- `gt/<id>.txt`: line-broken ground truth for evaluation only.
- `transcription/<id>.txt`: clean transcription with line breaks removed.
- `images/<id>/`: page images for image-conditioned methods.
- `ocr/<id>.txt`: generated OCR/HTR text with noisy line breaks.
- `ocr_lines/<id>.json`: ordered line-level OCR/HTR hints, usually with crop
  paths and page/line indices.
- `line_images/<id>/`: reusable line crops used by OCR, M5, and NNTP workflows.

## Method 1: Image + Transcription

**Inputs:** Page images + correct transcription (no line breaks)

**Task:** Insert line breaks by analyzing visual layout

**Implementation note:** Multi-page samples are processed page by page. Because
the clean transcription is stored at sample level, `run_eval_m1.py` splits it
into page-sized chunks using a word-boundary character-length heuristic before
prompting each page.

**Usage:**
```bash
python run_eval_m1.py
```

## Method 2: Image + Transcription + HTR

**Inputs:** Page images + correct transcription + HTR output (with line breaks)

**Task:** Use HTR line structure as primary guide, verify with image

**Implementation note:** The image remains a layout signal, while the OCR/HTR
text provides a noisy line-break scaffold. As in M1, multi-page text is chunked
heuristically across page images.

**Usage:**
```bash
python run_eval_m2.py
```

## Method 3: Transcription + HTR (Text-Only)

**Inputs:** Correct transcription + HTR output (no images)

**Task:** Align texts and transfer line breaks from HTR to transcription

**Implementation note:** M3 is the pure text-only LLM condition. It tests how
much line structure can be recovered from OCR/HTR line breaks without visual
evidence.

**Usage:**
```bash
python run_eval_m3.py
```

## Method 4: Transcription + Structured OCR Lines

**Inputs:** Correct transcription + ordered OCR/HTR line hypotheses from `ocr_lines/<id>.json`

**Task:** Align the transcription to the ordered OCR line list and return exactly one output line per OCR line

**Control path:**
- The prompt asks for strict JSON: `{"lines": ["...", "..."]}`.
- The expected line count is the number of entries in `ocr_lines/<id>.json`.
- If the first response cannot be parsed or has the wrong line count, the script
  sends a repair prompt.
- If valid model lines still do not concatenate to the exact transcription, their
  boundaries are projected back onto the clean transcription before evaluation.
- If both model responses are invalid, a deterministic fallback projects the OCR
  hint boundaries onto the clean transcription.

**Useful flags:**
- `--ocr-lines-dir`: choose the structured hint directory.
- `--prompt-variant`: switch between the baseline and boundary-anchored prompt.
- `--trace-dir`: write per-sample prompt, response, repair, and resolution JSON.

```bash
python run_eval_m4.py
```

## Method 5: Transcription + Line Images

**Inputs:** Correct transcription + ordered line images from `ocr_lines/<id>.json` crop paths

**Task:** Align the transcription to the ordered line-image crops and return exactly one output line per supplied line image

**Control path:**
- The prompt asks for strict JSON with exactly one string per supplied line image.
- Line images can be sent separately, as one stacked image, or as numbered strip
  composites.
- Optional OCR text hints and full page images can be added as secondary context.
- Invalid or suspicious outputs can trigger repair prompts.
- Accepted boundaries are projected back to the exact transcription before
  scoring when needed.
- `--judge-candidates` can generate multiple M5 views and ask the model to select
  the better candidate; the trace records this decision path.

**Useful flags:**
- `--line-image-mode`: `separate`, `stacked`, or `numbered_strips`.
- `--use-ocr-text`: include OCR text hints in addition to the line images.
- `--include-page-images`: add full page images as global layout context.
- `--split-by-page`: split very large multi-page samples into page-wise calls.
- `--trace-dir`: write per-sample prompt, response, repair, projection, and
  candidate-judging metadata.

```bash
python run_eval_m5.py
```

## NNTP Baseline

**Inputs:** Correct transcription plus PAGE XML geometry or presegmented line
images, with PyLaia network outputs.

**Task:** Use a deterministic forced-alignment baseline rather than a prompted
LLM/VLM.

**Implementation note:** The NNTP pipeline is documented separately because it
has external Java and PyLaia dependencies, produces intermediate CTC lattice
artifacts, and can filter characters outside the active PyLaia symbol inventory.
See [docs/nntp_pipeline.md](docs/nntp_pipeline.md).

## Few-Shot Learning

All methods support 0-shot or N-shot evaluation:

```bash
# 0-shot (no examples)
python run_eval_m1.py --data-dir datasets/bullinger_handwritten --n-shots 0

# 1-shot (one example)
python run_eval_m1.py --data-dir datasets/bullinger_handwritten --n-shots 1 --shots-seed 42

# 3-shot
python run_eval_m2.py --data-dir datasets/bullinger_handwritten --n-shots 3 --shots-seed 42
```

## Multi-Page Handling

For multi-page documents:
- **M1/M2:** Load all page images and split the sample-level transcription into
  page chunks by a heuristic before concatenating page-level predictions.
- **M3/M4:** Use sample-level text inputs in a single text-only call.
- **M5:** Can run sample-level, or page-wise when `--split-by-page` is enabled.

## Design Constraints

All methods must:
1. Never change characters (only insert `\n`)
2. Preserve text order
3. Handle edge cases (empty lines, Unicode, etc.)

The first constraint is an intended task contract for every method. It is
enforced mostly by prompting in M1-M3 and more strongly by parsing/projection in
M4-M5.

## Related Documentation

- **[docs/thesis_reader_guide.md](docs/thesis_reader_guide.md)** - Reader-facing method-to-artifact map
- **[METRICS.md](METRICS.md)** - Evaluation metrics
- **[datasets/README.md](datasets/README.md)** - Dataset structure
- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - HTR generation
