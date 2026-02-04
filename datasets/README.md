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
└── ocr/                  # HTR/OCR output (generated)
    ├── 0001.txt
    └── ...
```

## Directory Descriptions

- **`gt/`** - Line-broken text matching original layout (evaluation reference)
- **`transcription/`** - Correct text without line breaks (input to methods)
- **`images/`** - Page images (required for M1/M2, not used in M3)
- **`ocr/`** - Generated HTR output with line breaks (required for M2/M3)

## Available Datasets

- **`bullinger_handwritten/`** - Historical handwritten letters (16th century)
- **`bullinger_print/`** - Historical printed texts
- **`easy_historical/`** - Easier historical handwriting
- **`IAM_handwritten/`** - Modern English handwriting (IAM dataset)
- **`IAM_print/`** - Modern printed text (IAM dataset)
- **`children_handwritten/`** - Children's handwriting

## Generating OCR

For datasets without pre-generated OCR:

```bash
python scripts/make_ocr_outputs.py --dataset <dataset_name>
```

See [scripts/README.md](scripts/README.md) for details.

## Adding a New Dataset

1. Create directory structure: `mkdir -p datasets/my_dataset/{gt,transcription,images,ocr}`
2. Add ground truth files to `gt/`
3. Add transcriptions to `transcription/`
4. Add images to `images/<id>/`
5. Generate OCR: `python scripts/make_ocr_outputs.py --dataset my_dataset`

## File Naming

- Document IDs: Zero-padded numbers (`0001`, `0002`) or descriptive names
- Must be consistent across all subdirectories
- Images: `page_1.jpg`, `page_2.jpg`, etc. for multi-page documents
- UTF-8 encoding required for text files

## Related Documentation

- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - OCR generation details
- **[METHODS.md](METHODS.md)** - How datasets are used
- **[scripts/README.md](scripts/README.md)** - Data processing scripts
