# Scripts Guide

## make_ocr_outputs.py

Generate OCR/HTR outputs for datasets.

### Basic Usage

```bash
# Generate OCR for a dataset
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten

# Process specific samples
python scripts/make_ocr_outputs.py --dataset bullinger_handwritten --ids 0001,0002

# Use different recognizer
python scripts/make_ocr_outputs.py --dataset bullinger_print --recognizer trocr_printed
```

### Common Options

- `--dataset <name>` - Dataset name
- `--ids <list>` - Comma-separated IDs or path to file
- `--recognizer` - `trocr_handwritten`, `trocr_printed`, `iam_best_practices`
- `--segmenter` - `kraken` (default), `none` (passthrough)
- `--device` - `cuda`, `cpu`, `auto`
- `--batch-size` - Recognition batch size (default: 8)
- `--overwrite` - Regenerate existing outputs

### Output

Creates `ocr/<id>.txt` and optional `ocr/<id>.meta.json` in dataset directory.

See [docs/ocr_pipeline.md](docs/ocr_pipeline.md) for technical details.

---

## summarize_macro_avgs.py

Extract macro-average rows from evaluation CSVs.

### Basic Usage

```bash
# Summarize all CSVs in current directory
python scripts/summarize_macro_avgs.py

# Specify input directory
python scripts/summarize_macro_avgs.py --in-dir results/

# Custom glob pattern
python scripts/summarize_macro_avgs.py --glob "*gpt-5.2.csv"
```

### Output

- **`summary_long.csv`** - One row per (dataset, method)
- **`summary_wide.csv`** - One row per dataset, method-prefixed columns

---

## Additional Utilities

Located in `utils/` directory:

- **`convert_iam_dataset.py`** - Convert IAM database to project format
- **`convert_easy_hist_gt.py`** - Convert easy_historical ground truth
- **`copy_pages_to_images.py`** - Organize page images

---

## Related Documentation

- **[docs/ocr_pipeline.md](docs/ocr_pipeline.md)** - OCR generation details
- **[datasets/README.md](datasets/README.md)** - Dataset structure
- **[METRICS.md](METRICS.md)** - Metrics in CSV output
