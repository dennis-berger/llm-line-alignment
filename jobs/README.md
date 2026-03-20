# SLURM Job Files for Line Alignment Evaluation

This directory contains SLURM batch jobs for running line alignment evaluations on the cluster with various methods, datasets, and few-shot configurations.

## Directory Structure

```
jobs/
├── README.md                     # This file
├── orchestrators/                # Master submission scripts
│   ├── eval_all_0shot.sbatch    # Submit all zero-shot jobs (18 jobs)
│   ├── eval_all_1shot.sbatch    # Submit all 1-shot jobs (18 jobs)
│   └── eval_all_2shot.sbatch    # Submit all 2-shot jobs (18 jobs)
├── eval/                         # Evaluation jobs organized by method
│   ├── m1/                       # Method 1: Image + Transcription
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── bullinger_print_0shot.sbatch
│   │   ├── washington_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   ├── iam_print_0shot.sbatch
│   │   ├── iam_print_1shot.sbatch
│   │   └── ... 2-shot variants
│   ├── m2/                       # Method 2: Image + Transcription + HTR
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── bullinger_print_0shot.sbatch
│   │   ├── bullinger_print_1shot.sbatch
│   │   ├── washington_handwritten_0shot.sbatch
│   │   ├── washington_handwritten_1shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   ├── iam_print_0shot.sbatch
│   │   ├── iam_print_1shot.sbatch
│   │   └── ... 2-shot variants
│   ├── m3/                       # Method 3: Transcription + HTR (no images)
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── bullinger_print_0shot.sbatch
│   │   ├── bullinger_print_1shot.sbatch
│   │   ├── washington_handwritten_0shot.sbatch
│   │   ├── washington_handwritten_1shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   ├── iam_print_0shot.sbatch
│   │   ├── iam_print_1shot.sbatch
│   │   └── ... 2-shot variants
│   ├── m4/                       # Method 4: PyLaia-guided LLM alignment
│   │   ├── common_m4_eval.sbatch
│   │   ├── bullinger_handwritten_0shot.sbatch
│   │   ├── bullinger_handwritten_1shot.sbatch
│   │   ├── washington_handwritten_0shot.sbatch
│   │   ├── washington_handwritten_1shot.sbatch
│   │   ├── iam_handwritten_0shot.sbatch
│   │   ├── iam_handwritten_1shot.sbatch
│   │   └── ... 2-shot variants
│   └── nntp/                     # NNTP baselines
│       ├── bullinger_handwritten.sbatch
│       ├── iam_handwritten.sbatch
│       └── washington_handwritten.sbatch
├── training/                     # Training / fine-tuning jobs
│   ├── build_cvl_faust_manifest.sbatch
│   ├── train_trocr_cvl_faust.sbatch
│   └── train_washington_handwritten_pylaia_cv.sbatch
└── preprocessing/                # OCR/HTR generation jobs
    ├── make_ocr_children_handwritten.sbatch
    ├── make_ocr_washington_handwritten.sbatch
    ├── make_ocr_iam_handwritten.sbatch
    └── make_ocr_iam_print.sbatch
```

## Quick Start

### Submit All Jobs (Recommended)

**Zero-shot evaluation (all 18 jobs):**
```bash
sbatch jobs/orchestrators/eval_all_0shot.sbatch
```

**1-shot evaluation (all 18 jobs):**
```bash
sbatch jobs/orchestrators/eval_all_1shot.sbatch
```

### Submit Individual Jobs

**Single method/dataset combination:**
```bash
sbatch jobs/eval/m1/bullinger_handwritten_0shot.sbatch
sbatch jobs/eval/m2/iam_handwritten_1shot.sbatch
sbatch jobs/eval/m3/washington_handwritten_0shot.sbatch
sbatch jobs/eval/m4/washington_handwritten_0shot.sbatch
sbatch jobs/eval/nntp/bullinger_handwritten.sbatch
sbatch jobs/eval/nntp/iam_handwritten.sbatch
sbatch jobs/eval/nntp/washington_handwritten.sbatch
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
- **Datasets:** 6 (all datasets)

### Method 2 (M2): Image + Transcription + HTR
- **Input:** Page image(s) + correct transcription + HTR output
- **Task:** Use HTR line breaks as hints while preserving correct text
- **Datasets:** 6 (all datasets)

### Method 3 (M3): Transcription + HTR (text-only)
- **Input:** Correct transcription + HTR output (no images)
- **Task:** Transfer line breaks from HTR to transcription
- **Datasets:** 6 (all datasets)

### Method 4 (M4): PyLaia-Guided LLM Alignment
- **Input:** Correct transcription + structured PyLaia line hypotheses (`ocr_lines/<id>.json`)
- **Task:** Use PyLaia line order as structural hints while preserving exact transcription characters
- **Datasets:** Current cluster wrappers target `bullinger_handwritten`, `IAM_handwritten`, and `washington_handwritten`
- **Jobs:** `jobs/eval/m4/*_{0,1,2}shot.sbatch`; current defaults match the other eval jobs and start with `Qwen3-VL-8B-Instruct`
- **Note:** Each M4 job first refreshes or creates `ocr_lines/<id>.json` with PyLaia, then runs `run_eval_m4.py`

### NNTP Baseline: PyLaia + NNTP
- **Input:** Correct transcription plus either PAGE XML line geometry or presegmented line images
- **Task:** Prepare line images, generate CTC lattices with PyLaia, and align with NNTP
- **Datasets:** 3 scheduled jobs today (`bullinger_handwritten`, `IAM_handwritten_rwth_test`, `washington_handwritten`)

Washington uses curated presegmented `line_images/` plus the explicit IAM PyLaia assets from `third_party/pylaia-iam`.

Before running the IAM NNTP job, build the RWTH test dataset once:

```bash
python scripts/build_iam_rwth_dataset.py \
  --iam-root ../iam/data \
  --split test \
  --out-dir datasets/IAM_handwritten_rwth_test
```

Washington zero-adaptation NNTP baseline:

```bash
sbatch jobs/eval/nntp/washington_handwritten.sbatch
```

Washington PyLaia 2-fold fine-tuning plus opposite-fold NNTP evaluation:

```bash
TRAIN_FOLD=train_a sbatch jobs/training/train_washington_handwritten_pylaia_cv.sbatch
TRAIN_FOLD=train_b sbatch jobs/training/train_washington_handwritten_pylaia_cv.sbatch
```

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

Results are saved with dataset-specific naming for easy comparison:

**Naming Pattern:**
- CSV: `{dataset}_eval_{method}_{nshot}.csv`
- Predictions: `{dataset}_predictions_{method}_{nshot}/`

**Examples:**

### Zero-Shot (0-shot)
- `bullinger_handwritten_eval_m1_0shot.csv` + `bullinger_handwritten_predictions_m1_0shot/`
- `iam_print_eval_m2_0shot.csv` + `iam_print_predictions_m2_0shot/`
- `washington_handwritten_eval_m3_0shot.csv` + `washington_handwritten_predictions_m3_0shot/`

### 1-Shot
- `bullinger_handwritten_eval_m1_1shot.csv` + `bullinger_handwritten_predictions_m1_1shot/`
- `iam_handwritten_eval_m2_1shot.csv` + `iam_handwritten_predictions_m2_1shot/`
- `bullinger_print_eval_m3_1shot.csv` + `bullinger_print_predictions_m3_1shot/`

NNTP baseline:
- `bullinger_handwritten_eval_nntp.csv` + `bullinger_handwritten_predictions_nntp/`
- `iam_handwritten_rwth_test_eval_nntp.csv` + `iam_handwritten_rwth_test_predictions_nntp/`

This makes it easy to compare:
- **Across shots**: `iam_print_eval_m2_0shot.csv` vs `iam_print_eval_m2_1shot.csv`
- **Across methods**: `bullinger_handwritten_eval_m1_1shot.csv` vs `bullinger_handwritten_eval_m2_1shot.csv`
- **Across datasets**: `iam_print_eval_m2_1shot.csv` vs `washington_handwritten_eval_m2_1shot.csv`

## Resource Allocation

All evaluation jobs use:
- **GPU:** 1 GPU (any available)
- **Memory:** 64GB RAM
- **CPUs:** 6 cores
- **Time:** 2 hours
- **Partition:** GPU

The NNTP baseline jobs use the same GPU/CPU/memory profile, but with a 4 hour wall clock limit to cover PyLaia netout plus Java alignment.

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

## API Rate Limits & Auto-Resume (Gemini)

Gemini API has strict rate limits on the free tier:
- **Per-minute:** 25 requests/minute
- **Per-day:** 250 requests/day (resets at midnight PT)

### How It Works

1. **Throttling:** Requests are automatically spaced (2.4s apart) to stay under the per-minute limit
2. **Retry:** Per-minute 429 errors trigger exponential backoff retries
3. **Checkpointing:** Progress is saved after each sample to `checkpoints/` directory
4. **Daily Quota:** When daily limit is hit, job exits with code 75
5. **Auto-Resubmit:** Jobs automatically resubmit themselves with `--begin=now+24hours`

### Monitoring Resumed Jobs

When a job hits daily quota and resubmits:

```bash
# Check pending jobs (shows BeginTime for scheduled jobs)
squeue -u $USER

# See when jobs are scheduled to run
squeue -u $USER --start

# View checkpoint files (shows progress saved)
ls -la checkpoints/
```

### Manual Resume

If a job failed without auto-resubmit, you can manually resume:

```bash
# Simply resubmit the same job - it will resume from checkpoint
sbatch jobs/eval/m1/bullinger_handwritten_0shot.sbatch
```

### Checkpoint Files

Checkpoints are stored in `checkpoints/` with naming pattern:
```
checkpoint_{method}_{dataset}_{provider}_{model}_{nshot}shot.json
```

Checkpoints are automatically deleted when evaluation completes successfully.

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
