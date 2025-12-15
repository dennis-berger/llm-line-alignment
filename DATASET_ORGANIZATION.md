# Dataset Organization

## Overview

All datasets have been consolidated into a single `datasets/` directory with descriptive names.

## Directory Structure

```
datasets/
├── bullinger_handwritten/    # Main validation dataset (handwritten Bullinger letters)
│   ├── gt/                   # Ground-truth line-broken text
│   ├── images/               # Page images per letter
│   ├── transcription/        # Correct diplomatic transcriptions (no line breaks)
│   └── ocr/                  # HTR outputs (with noisy line breaks)
│
├── bullinger_print/          # Printed Bullinger letters dataset
│   ├── gt/                   # Ground-truth line-broken text
│   ├── images/               # Page images per letter
│   ├── transcription/        # Correct diplomatic transcriptions (no line breaks)
│   └── pdf/                  # Original PDF files
│
└── easy_historical/          # Easier historical dataset
    ├── gt/                   # Ground-truth line-broken text
    ├── images/               # Page images
    └── transcription/        # Correct diplomatic transcriptions (no line breaks)
```

## Migration from Old Names

| Old Name                   | New Name                          | Description                          |
|----------------------------|-----------------------------------|--------------------------------------|
| `data_val/`                | `datasets/bullinger_handwritten/` | Main validation set (handwritten)    |
| `bullinger_print_dataset/` | `datasets/bullinger_print/`       | Printed Bullinger letters            |
| `easy_hist_dataset/`       | `datasets/easy_historical/`       | Easier historical handwriting        |

## Updated References

All references to the old dataset paths have been updated in:

### Job Scripts (`jobs/`)
- ✅ `eval_gpu_m1.sbatch` → `datasets/bullinger_handwritten`
- ✅ `eval_gpu_m2.sbatch` → `datasets/bullinger_handwritten`
- ✅ `eval_gpu_m3.sbatch` → `datasets/bullinger_handwritten`
- ✅ `eval_gpu_qwen.sbatch` → `datasets/bullinger_handwritten`
- ✅ `eval_gpu_qwen_m1_print.sbatch` → `datasets/bullinger_print`
- ✅ `eval_gpu_qwen_m1_easy-historical.sbatch` → `datasets/easy_historical`

### Python Scripts
- ✅ `run_eval.py` → default `--data-dir datasets/bullinger_handwritten`
- ✅ `run_eval_m1.py` → default `--data-dir datasets/bullinger_handwritten`
- ✅ `run_eval_m2.py` → default `--data-dir datasets/bullinger_handwritten`
- ✅ `run_eval_m3.py` → default `--data-dir datasets/bullinger_handwritten`
- ✅ `run_eval_qwen.py` → default `--data-dir datasets/bullinger_handwritten`

### Utility Scripts (`utils/`)
- ✅ `convert_easy_hist_gt.py` → usage examples updated
- ✅ `copy_pages_to_images.py` → usage examples updated

### Documentation
- ✅ `IMPROVEMENTS.md` → example commands updated

## Benefits

1. **Better Organization**: All datasets in one place
2. **Clear Naming**: Self-documenting names that describe the content
3. **Consistency**: Uniform naming convention across all datasets
4. **Scalability**: Easy to add new datasets to the `datasets/` folder

## Usage Examples

```bash
# Run Method 1 on handwritten Bullinger letters (default)
python run_eval_m1.py

# Run Method 1 on printed Bullinger letters
python run_eval_m1.py --data-dir datasets/bullinger_print

# Run Method 1 on easier historical dataset
python run_eval_m1.py --data-dir datasets/easy_historical

# Submit job for handwritten validation set
sbatch jobs/eval_gpu_m1.sbatch

# Submit job for printed dataset
sbatch jobs/eval_gpu_qwen_m1_print.sbatch

# Submit job for easy historical dataset
sbatch jobs/eval_gpu_qwen_m1_easy-historical.sbatch
```

## Dataset Characteristics

### Bullinger Handwritten (`datasets/bullinger_handwritten`)
- **Content**: Handwritten Bullinger correspondence
- **Purpose**: Main validation/test set for line alignment methods
- **Features**: Complex historical handwriting, multi-page letters
- **Has OCR**: Yes (noisy HTR outputs for Method 2 & 3)

### Bullinger Print (`datasets/bullinger_print`)
- **Content**: Printed editions of Bullinger texts
- **Purpose**: Testing on cleaner, printed historical text
- **Features**: Better quality, consistent typography
- **Has OCR**: Partial

### Easy Historical (`datasets/easy_historical`)
- **Content**: Easier historical handwriting samples
- **Purpose**: Testing on less challenging handwriting
- **Features**: Clearer handwriting, simpler layout
- **Has OCR**: No
