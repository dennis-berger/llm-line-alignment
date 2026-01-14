# SLURM Job Files for Line Alignment Evaluation

This directory contains SLURM batch jobs for running line alignment evaluations on the cluster with various methods, datasets, and few-shot configurations.

## Directory Structure

```
jobs/
├── README.md                     # This file
├── orchestrators/                # Master submission scripts
│   ├── eval_all_0shot.sbatch    # Submit all zero-shot jobs (15 jobs)
│   └── eval_all_1shot.sbatch    # Submit all 1-shot jobs (13 jobs)
├── eval/                         # Evaluation jobs organized by method
│   ├── m1/                       # Method 1: Image + Transcription
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── bullinger_print_0shot.sbatch
│   │   ├── easy_historical_0shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   ├── iam_print_0shot.sbatch
│   │   └── iam_print_1shot.sbatch
│   ├── m2/                       # Method 2: Image + Transcription + HTR
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── bullinger_print_0shot.sbatch
│   │   ├── bullinger_print_1shot.sbatch
│   │   ├── easy_historical_0shot.sbatch
│   │   ├── easy_historical_1shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   ├── iam_print_0shot.sbatch
│   │   └── iam_print_1shot.sbatch
│   └── m3/                       # Method 3: Transcription + HTR (no images)
│       ├── bullinger_handwritten_0shot.sbatch
│       ├── bullinger_handwritten_1shot.sbatch
│       ├── bullinger_print_0shot.sbatch
│       ├── bullinger_print_1shot.sbatch
│       ├── easy_historical_0shot.sbatch
│       ├── easy_historical_1shot.sbatch
│       ├── iam_handwritten_0shot.sbatch
│       ├── iam_handwritten_1shot.sbatch
│       ├── iam_print_0shot.sbatch
│       └── iam_print_1shot.sbatch
└── preprocessing/                # OCR/HTR generation jobs
    ├── make_ocr_easy_historical.sbatch
    ├── make_ocr_iam_handwritten.sbatch
    └── make_ocr_iam_print.sbatch
```

## Quick Start

### Submit All Jobs (Recommended)

**Zero-shot evaluation (all 15 jobs):**
```bash
sbatch jobs/orchestrators/eval_all_0shot.sbatch
```

**1-shot evaluation (all 13 jobs):**
```bash
sbatch jobs/orchestrators/eval_all_1shot.sbatch
```

### Submit Individual Jobs

**Single method/dataset combination:**
```bash
sbatch jobs/eval/m1/bullinger_handwritten_0shot.sbatch
sbatch jobs/eval/m2/iam_handwritten_1shot.sbatch
sbatch jobs/eval/m3/easy_historical_0shot.sbatch
```

**All jobs for one method:**
```bash
for job in jobs/eval/m2/*_0shot.sbatch; do sbatch "$job"; done
```

**All jobs for one dataset:**
```bash
sbatch jobs/eval/m1/iam_print_1shot.sbatch
sbatch jobs/eval/m2/iam_print_1shot.sbatch
sbatch jobs/eval/m3/iam_print_1shot.sbatch
```

## Methods Overview

### Method 1 (M1): Image + Transcription
- **Input:** Page image(s) + correct transcription (no line breaks)
- **Task:** Insert line breaks based on visual layout
- **Datasets:** 5 (all datasets)

### Method 2 (M2): Image + Transcription + HTR
- **Input:** Page image(s) + correct transcription + HTR output
- **Task:** Use HTR line breaks as hints while preserving correct text
- **Datasets:** 5 (all datasets)

### Method 3 (M3): Transcription + HTR (text-only)
- **Input:** Correct transcription + HTR output (no images)
- **Task:** Transfer line breaks from HTR to transcription
- **Datasets:** 5 (all datasets)

## Few-Shot Configuration

### Zero-Shot (0-shot)
- No examples provided to the model
- Default behavior for all `*_0shot.sbatch` files

### 1-Shot
- 1 example randomly selected from the same dataset
- Fixed seed (42) for reproducibility
- Automatically excludes test samples from example pool

### Future: 2-shot, 3-shot, etc.
To create additional few-shot configurations:
1. Copy `*_1shot.sbatch` files
2. Change filename to `*_2shot.sbatch`
3. Update `--n-shots 2` parameter
4. Create corresponding orchestrator in `orchestrators/`

## Monitoring Jobs

**Check job status:**
```bash
squeue -u $USER
```

**Watch job output in real-time:**
```bash
tail -f logs/bullinger_m1_1shot_<job_id>.out
tail -f logs/bullinger_m1_1shot_<job_id>.err
```

**Cancel jobs:**
```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel all 1-shot jobs
scancel -u $USER -n "*_1shot"

# Cancel all jobs for a method
scancel -u $USER -n "bullinger_*"
```

## Output Files

Results are automatically organized by shot count:

### Zero-Shot Outputs
- Predictions: `predictions_m1_0shot/`, `predictions_m2_0shot/`, `predictions_m3_0shot/`
- CSV results: `evaluation_qwen_m1_0shot.csv`, etc.

### 1-Shot Outputs
- Predictions: `predictions_m1_1shot/`, `predictions_m2_1shot/`, `predictions_m3_1shot/`
- CSV results: `evaluation_qwen_m1_1shot.csv`, etc.

### Dataset-Specific Naming
Some jobs use dataset-specific output naming:
- `bullinger_print_predictions_m1/`
- `iam_handwritten_eval_m2.csv`

## Resource Allocation

All evaluation jobs use:
- **GPU:** 1 GPU (any available)
- **Memory:** 64GB RAM
- **CPUs:** 6 cores
- **Time:** 2 hours
- **Partition:** GPU

Orchestrator jobs use minimal resources (1GB RAM, 10 minutes).

## Key Parameters

All jobs use:
- `--hf-model` - Qwen3-VL-8B-Instruct (local or HuggingFace)
- `--hf-device cuda` - Use GPU acceleration
- `--max-new-tokens 800` - Maximum output length

Few-shot specific:
- `--n-shots N` - Number of examples (0, 1, 2, ...)
- `--shots-seed 42` - Random seed for reproducibility
- `--shots-dataset-scope same` - Use examples from same dataset (default)

## Troubleshooting

**Job fails immediately:**
- Check logs in `logs/<job_name>_<job_id>.err`
- Verify conda environment is activated
- Check GPU availability with `nvidia-smi`

**Out of memory:**
- Reduce `--max-new-tokens`
- Check if 4-bit quantization is working (bitsandbytes)
- Some nodes have less VRAM (32GB vs 48GB)

**Model not found:**
- Ensure Qwen3-VL-8B-Instruct is downloaded to `.hf/Qwen3-VL-8B-Instruct/`
- Or rely on automatic download from HuggingFace (slower first run)

**Few-shot examples not loading:**
- Verify dataset has gt/, transcription/, and ocr/ directories
- Check that dataset has enough samples (need at least 2: 1 test + 1 example)

## Adding New Datasets

To add a new dataset:
1. Create job files following the naming pattern: `{dataset}_{nshot}.sbatch`
2. Place in appropriate method directory: `eval/m1/`, `eval/m2/`, or `eval/m3/`
3. Update orchestrator scripts to include new dataset
4. Ensure dataset directory structure matches expectations:
   ```
   datasets/{dataset}/
   ├── gt/              # Ground truth with line breaks
   ├── images/          # Page images (for M1/M2)
   ├── transcription/   # Correct text without line breaks
   └── ocr/             # HTR output (for M2/M3)
   ```
