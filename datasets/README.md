# Dataset Organization

All datasets follow a standard structure under `datasets/`:

## Standard Structure

```
datasets/<dataset_name>/
├── gt/                   # Ground truth with line breaks
│   ├── 0001.txt
│   └── ...
├── transcription/        # Correct text without line breaks
│   ├── 0001.txt
│   └── ...
├── images/               # Page images
│   ├── 0001/
│   │   ├── page_1.jpg
│   │   └── ...
│   └── ...
├── ocr/                  # HTR/OCR output (generated)
│   ├── 0001.txt
│   └── ...
├── line_images/          # Optional presegmented line images for NNTP/OCR
│   ├── 0001/
│   │   ├── 0001_line000.png
│   │   └── ...
│   └── ...
├── metadata/             # Optional per-sample NNTP provenance
│   ├── 0001.json
│   └── ...
├── previews/             # Optional overlay previews for line review
│   ├── 0001/
│   │   ├── 0001_overlay.png
│   │   └── ...
│   └── ...
├── metadata.json         # Optional dataset-level NNTP manifest
└── review_status.json    # Optional review status manifest
```

## Directory Descriptions

- **`gt/`** - Line-broken text matching original layout (evaluation reference)
- **`transcription/`** - Correct text without line breaks (input to methods)
- **`images/`** - Page images (required for M1/M2, not used in M3)
- **`ocr/`** - Generated HTR output with line breaks (required for M2/M3)
- **`ocr_lines/`** - Structured ordered OCR/HTR line hints with optional crop paths (required for M4/M5)
- **`line_images/`** - Optional curated line crops for presegmented OCR or NNTP
- **`metadata/`, `metadata.json`** - Optional NNTP provenance and bounding-box manifests
- **`previews/`, `review_status.json`** - Optional visual review artifacts for curated line images

## Method Compatibility

| Artifact | M1 | M2 | M3 | M4 | M5 | NNTP |
| --- | --- | --- | --- | --- | --- | --- |
| `gt/` | eval | eval | eval | eval | eval | eval |
| `transcription/` | input | input | input | input | input | input |
| `images/` | input | input | - | - | optional context | PAGE/crop source |
| `ocr/` | - | input | input | - | optional provenance | - |
| `ocr_lines/` | - | - | - | input | input | optional source |
| `line_images/` | - | OCR source | OCR source | OCR source | input | input |

`gt/` should not be used as model input for M1-M5. It is the held-out reference
for scoring. Some preprocessing steps derive line crops or structured hints from
dataset-specific ground-truth geometry; when reading thesis comparisons, keep
that evidence regime separate from the prompt target itself.

## Available Datasets

- **`bullinger_handwritten/`** - Historical handwritten letters (16th century)
- **`bullinger_print/`** - Historical printed texts
- **`washington_handwritten/`** - Easier historical handwriting
- **`IAM_handwritten/`** - Modern English handwriting (IAM dataset)
- **`IAM_print/`** - Modern printed text (IAM dataset)
- **`children_handwritten/`** - Children's handwriting

`washington_handwritten/` also ships curated `line_images/` plus NNTP review/provenance metadata.
Some thesis artifacts use auxiliary slices such as
`IAM_handwritten_rwth_test_representative_20`; treat those as fixed evaluation
snapshots rather than interchangeable aliases for the full IAM handwritten root.

## Generating OCR

For datasets without pre-generated OCR:

```bash
python scripts/make_ocr_outputs.py --dataset <dataset_name>
```

See [../scripts/README.md](../scripts/README.md) for details.

## Adding a New Dataset

1. Create directory structure: `mkdir -p datasets/my_dataset/{gt,transcription,images,ocr}`
2. Add ground truth files to `gt/`
3. Add transcriptions to `transcription/`
4. Add images to `images/<id>/`
5. Generate OCR: `python scripts/make_ocr_outputs.py --dataset my_dataset`

For M4/M5 or NNTP experiments, also provide either reusable `line_images/` or a
pipeline that can generate `ocr_lines/<id>.json` with stable reading order.

## File Naming

- Document IDs: Zero-padded numbers (`0001`, `0002`) or descriptive names
- Must be consistent across all subdirectories
- Images: `page_1.jpg`, `page_2.jpg`, etc. for multi-page documents
- UTF-8 encoding required for text files
- Optional NNTP assets follow the same sample IDs under `line_images/`, `metadata/`, and `previews/`

## Related Documentation

- **[../docs/ocr_pipeline.md](../docs/ocr_pipeline.md)** - OCR generation details
- **[../METHODS.md](../METHODS.md)** - How datasets are used
- **[../scripts/README.md](../scripts/README.md)** - Data processing scripts
