# Thesis Reader Guide

This repository is the implementation companion to the thesis. It is meant to
answer questions such as "which script produced this method?", "which files are
inputs rather than generated artifacts?", and "which implementation caveats
matter when reading the result tables?"

## Method Map

| Thesis label | Main script | Primary evidence | Output control | Reader note |
| --- | --- | --- | --- | --- |
| M1 | `run_eval_m1.py` | page image(s) + clean transcription | prompt only | For multi-page samples, the clean transcription is split into page-sized chunks by a character-length heuristic before each page call. |
| M2 | `run_eval_m2.py` | page image(s) + clean transcription + OCR/HTR text | prompt only | OCR/HTR is a structural hint. The clean transcription remains the intended character source. |
| M3 | `run_eval_m3.py` | clean transcription + OCR/HTR text | prompt only | This is the text-only LLM alignment condition; no image evidence is sent to the model. |
| M4 | `run_eval_m4.py` | clean transcription + ordered `ocr_lines/<id>.json` line texts | JSON parse, repair, projection | The model is asked for exactly one line per OCR/HTR hint. Final scored text is projected back to the exact clean transcription if needed. |
| M5 | `run_eval_m5.py` | clean transcription + ordered line-image crops, optionally OCR text and page context | JSON parse, repair, projection, optional candidate judging | The model is given the line count through the supplied crop sequence. Reported `context100` runs use a richer M5 configuration than the minimal default CLI. |
| NNTP | `scripts/run_nntp_eval.py` | clean transcription + PAGE XML or presegmented line images + PyLaia netout | deterministic forced alignment | NNTP is evaluated with the same metric family, but unsupported characters can be stripped according to the active PyLaia `syms.txt`. |

All five LLM/VLM methods target the same task: preserve the supplied
transcription character sequence and insert newline characters. The degree of
enforcement differs. M1-M3 rely on prompt instructions and raw model output,
whereas M4-M5 add structured parsing, repair prompts, and deterministic boundary
projection before scoring.

## Artifact Vocabulary

- `gt/<id>.txt`: line-broken ground truth used only as the evaluation reference.
- `transcription/<id>.txt`: the clean text input with line breaks removed.
- `images/<id>/`: page images used by image-conditioned methods.
- `ocr/<id>.txt`: generated OCR/HTR text with line breaks, used by M2 and M3.
- `ocr_lines/<id>.json`: structured ordered OCR/HTR line hints, used by M4 and
  M5. Entries may include page/line indices, crop paths, and OCR text.
- `line_images/<id>/`: presegmented or curated line crops used by OCR generation,
  M5, and NNTP workflows.
- `predictions_*`, `evaluation_*.csv`, `checkpoints/`, and `outputs/cache/`:
  generated outputs. They are useful for audit and debugging, but they are not
  the source definitions of the methods.

## Reproducibility Notes

- Evaluation scripts write prediction text files plus CSV metric summaries. The
  last CSV row is `macro_avg`, computed as the arithmetic mean over processed
  sample IDs.
- API-backed runs can stop because of quotas. Checkpoints under `checkpoints/`
  let the same command resume without recomputing already processed samples.
- `--trace-dir` on M4 and M5 stores prompt, response, repair, projection, and
  resolution metadata per sample when enabled. Retained thesis artifacts have
  much stronger trace coverage for M5 than for M4.
- The thesis often reports a combined normalized directional line-accuracy view,
  computed outside the core CSV as `(line_acc_norm + rev_line_acc_norm) / 2`.
- Model IDs use provider prefixes such as `hf/`, `openai/`, `gemini/`, and
  `mistral/`. API providers can change hosted model behavior over time, so
  retained predictions and dated run roots are the audit source for submitted
  numbers.
- M4 and M5 know the expected output line count from `ocr_lines/<id>.json` or
  the associated crop list. This makes them stronger control conditions than
  M1-M3, and it should be kept visible when comparing methods.

## Where To Look First

- Method definitions and constraints: [METHODS.md](../METHODS.md)
- Metric formulas and interpretation: [METRICS.md](../METRICS.md)
- Dataset schema and artifact roles: [datasets/README.md](../datasets/README.md)
- OCR/HTR preprocessing: [docs/ocr_pipeline.md](ocr_pipeline.md)
- NNTP baseline: [docs/nntp_pipeline.md](nntp_pipeline.md)
- Cluster execution and job families: [jobs/README.md](../jobs/README.md)
- Utility scripts and packaging helpers: [scripts/README.md](../scripts/README.md)

## Caveats For Thesis Interpretation

- Do not read high CER/WER values as only line-break errors. They can also
  indicate that a prompt-only model copied, normalized, omitted, or inserted
  characters despite the newline-only instruction.
- Exact-line precision/recall/F1 is order-insensitive. It helps diagnose whether
  correct lines exist somewhere in the output, but it can hide severe reordering.
- Forward and reverse line accuracy are order-sensitive diagnostics. A gap
  between them often points to drift that starts near one end of a document.
- Printed datasets are useful supporting context, but the final thesis argument
  centers on the handwritten datasets and the available denominator-matched
  comparison regimes.
