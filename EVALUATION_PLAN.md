# Comprehensive Evaluation Plan

## Overview

This document provides a step-by-step plan to evaluate all three methods (M1, M2, M3) on all available datasets using the FAITH HPC cluster.

## Dataset & Method Compatibility Matrix

| Dataset                     | Method 1 | Method 2 | Method 3 | Notes                                    |
|-----------------------------|----------|----------|----------|------------------------------------------|
| `bullinger_handwritten`     | ✅       | ✅       | ✅       | Has ocr/ folder - supports all methods   |
| `bullinger_print`           | ✅       | ❌       | ❌       | No ocr/ folder - M1 only                 |
| `easy_historical`           | ✅       | ❌       | ❌       | No ocr/ folder - M1 only                 |

**Method Requirements:**
- **Method 1**: Requires `gt/`, `images/`, `transcription/`
- **Method 2**: Requires `gt/`, `images/`, `transcription/`, **`ocr/`**
- **Method 3**: Requires `gt/`, `transcription/`, **`ocr/`** (no images needed)

## Available Job Scripts

### Existing Jobs (5 total)
1. ✅ `eval_gpu_m1.sbatch` - bullinger_handwritten + Method 1
2. ✅ `eval_gpu_m2.sbatch` - bullinger_handwritten + Method 2
3. ✅ `eval_gpu_m3.sbatch` - bullinger_handwritten + Method 3
4. ✅ `eval_gpu_qwen_m1_print.sbatch` - bullinger_print + Method 1
5. ✅ `eval_gpu_qwen_m1_easy-historical.sbatch` - easy_historical + Method 1

### Jobs That Cannot Be Created
- ❌ bullinger_print + Method 2/3 (missing ocr/ folder)
- ❌ easy_historical + Method 2/3 (missing ocr/ folder)

## Evaluation Outputs

Each job will produce:
- **Predictions directory**: Contains predicted line-broken text files
- **CSV file**: Evaluation metrics (WER, CER, line accuracy)

### Output Files Structure

```
cluster:~/projects/bullinger-line-alignment-mwe/
├── predictions_m1/                      # M1 on bullinger_handwritten
│   ├── 0001.txt
│   ├── 0002.txt
│   └── ...
├── predictions_m2/                      # M2 on bullinger_handwritten
│   └── ...
├── predictions_m3/                      # M3 on bullinger_handwritten
│   └── ...
├── bullinger_print_predictions_m1/     # M1 on bullinger_print
│   └── ...
├── easy_hist_predictions_m1/           # M1 on easy_historical
│   └── ...
├── evaluation_qwen_m1.csv              # M1 metrics (handwritten)
├── evaluation_qwen_m2.csv              # M2 metrics (handwritten)
├── evaluation_qwen_m3.csv              # M3 metrics (handwritten)
├── bullinger_print_eval_qwen_m1.csv   # M1 metrics (print)
└── easy_hist_eval_qwen_m1.csv         # M1 metrics (easy_historical)
```

## Step-by-Step Execution Plan

### Phase 1: Preparation (Local Machine)

#### 1.1 Verify Local Repository is Updated
```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# Check that datasets are organized
ls -la datasets/

# Verify job scripts exist
ls -la jobs/*.sbatch
```

#### 1.2 Push Latest Changes to Git (if using version control)
```bash
git add .
git commit -m "Update dataset organization and job scripts"
git push origin main
```

### Phase 2: Cluster Setup

#### 2.1 Login to FAITH Cluster
```bash
ssh bergerd@diufrd200.unifr.ch
```

#### 2.2 Navigate to Project Directory
```bash
cd ~/projects/bullinger-line-alignment-mwe
```

#### 2.3 Pull Latest Changes (if using git)
```bash
git pull origin main
```

#### 2.4 Verify Dataset Structure on Cluster
```bash
ls -la datasets/
ls -la datasets/bullinger_handwritten/
ls -la datasets/bullinger_print/
ls -la datasets/easy_historical/
```

#### 2.5 Create Logs Directory
```bash
mkdir -p logs
```

#### 2.6 Verify/Setup Conda Environment
```bash
# Activate environment
conda activate bullinger-mwe

# If environment doesn't exist, create it
# conda create -n bullinger-mwe python=3.10 -y
# conda activate bullinger-mwe
# pip install -r requirements.txt

# Verify packages
pip list | grep -E "torch|transformers|pillow"
```

### Phase 3: Job Submission

#### 3.1 Submit All Jobs
Submit jobs in order of priority (handwritten dataset first, as it has all three methods):

```bash
# Bullinger Handwritten (all 3 methods)
sbatch jobs/eval_gpu_m1.sbatch
sbatch jobs/eval_gpu_m2.sbatch
sbatch jobs/eval_gpu_m3.sbatch

# Bullinger Print (Method 1 only)
sbatch jobs/eval_gpu_qwen_m1_print.sbatch

# Easy Historical (Method 1 only)
sbatch jobs/eval_gpu_qwen_m1_easy-historical.sbatch
```

**Note:** You can submit all jobs at once. The SLURM scheduler will queue them appropriately.

#### 3.2 Monitor Job Queue
```bash
# Check your jobs
squeue -u $USER

# Watch queue continuously (Ctrl+C to exit)
watch -n 5 'squeue -u $USER'

# Check detailed job info
scontrol show job <JOB_ID>
```

#### 3.3 Monitor Job Logs (Real-time)
```bash
# Follow the output of a specific job (replace with actual job ID)
tail -f logs/bullinger_qwen_m1_<JOB_ID>.out

# Check for errors
tail -f logs/bullinger_qwen_m1_<JOB_ID>.err

# View all recent logs
ls -lt logs/ | head -20
```

### Phase 4: Results Collection

#### 4.1 Verify Job Completion
Wait for all jobs to complete. Check completion status:

```bash
# On cluster
cd ~/projects/bullinger-line-alignment-mwe

# Check if output files exist
ls -la *.csv
ls -la predictions_m1/
ls -la predictions_m2/
ls -la predictions_m3/
ls -la bullinger_print_predictions_m1/
ls -la easy_hist_predictions_m1/

# Quick check of job logs for success/errors
grep -i "done\|error\|failed" logs/*.out
```

#### 4.2 Copy Results to Local Machine

**Option A: Copy Everything (Recommended)**
```bash
# From your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# Create results directory
mkdir -p results_cluster_run_$(date +%Y%m%d)
cd results_cluster_run_$(date +%Y%m%d)

# Copy all CSV files
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*.csv .

# Copy all prediction directories
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m1 .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m2 .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m3 .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/bullinger_print_predictions_m1 .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/easy_hist_predictions_m1 .

# Copy logs for debugging (optional)
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/logs .
```

**Option B: Copy Only CSV Files (Quick)**
```bash
# From your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# Copy just the evaluation CSVs
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/evaluation_qwen_m*.csv .
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/bullinger_print_eval_qwen_m1.csv .
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/easy_hist_eval_qwen_m1.csv .
```

**Option C: Use rsync (More Efficient for Large Files)**
```bash
# From your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# Sync results
rsync -avz --progress bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*.csv ./results/
rsync -avz --progress bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m* ./results/
rsync -avz --progress bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/bullinger_print_predictions_m1 ./results/
rsync -avz --progress bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/easy_hist_predictions_m1 ./results/
```

### Phase 5: Results Analysis

#### 5.1 Examine CSV Files Locally
```bash
# On your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# View evaluation results
cat evaluation_qwen_m1.csv
cat evaluation_qwen_m2.csv
cat evaluation_qwen_m3.csv
cat bullinger_print_eval_qwen_m1.csv
cat easy_hist_eval_qwen_m1.csv

# Or open in spreadsheet software (Excel, Numbers, LibreOffice)
open evaluation_qwen_m1.csv
```

#### 5.2 Compare Predictions (Optional)
```bash
# Compare a specific prediction with ground truth
diff datasets/bullinger_handwritten/gt/0001.txt predictions_m1/0001.txt
diff predictions_m1/0001.txt predictions_m2/0001.txt
diff predictions_m2/0001.txt predictions_m3/0001.txt
```

## Estimated Time & Resources

### Job Duration Estimates
- **Method 1** (with images): ~1-2 hours per dataset (depends on number of pages)
- **Method 2** (images + OCR): ~1-2 hours
- **Method 3** (text only): ~30-60 minutes (faster, no image processing)

### Total Cluster Time
- 5 jobs × ~1.5 hours average = **~7.5 hours total wall time**
- With parallel execution on multiple GPUs: Could complete in **~2 hours** if all run simultaneously

### Storage Requirements
- Predictions: ~10-50 MB per dataset
- Logs: ~5-20 MB per job
- Total: **~100-300 MB**

## Troubleshooting

### Common Issues

#### Job Fails with Out of Memory
```bash
# Edit the job script to increase memory
#SBATCH --mem=64G  # Try 96G or 128G if available
```

#### Conda Environment Not Found
```bash
# On cluster, recreate environment
conda create -n bullinger-mwe python=3.10 -y
conda activate bullinger-mwe
pip install -r requirements.txt
```

#### Model Download Issues
```bash
# Pre-download model on login node
python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', trust_remote_code=True); \
print('Model cached')"
```

#### Job Stuck in Queue
```bash
# Check available resources
sinfo

# Check partition status
squeue --partition=GPU

# Try different partition or node if available
# Edit job script: #SBATCH --partition=gpu-long
```

### Getting Help

#### Check Job Details
```bash
# Detailed job info
scontrol show job <JOB_ID>

# Job history
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,DerivedExitCode
```

#### View Full Logs
```bash
# On cluster
cat logs/bullinger_qwen_m1_<JOB_ID>.out
cat logs/bullinger_qwen_m1_<JOB_ID>.err
```

## Results Summary Template

Once you have all results, create a summary table:

| Dataset                  | Method | WER (avg) | CER (avg) | Line Acc | Notes |
|--------------------------|--------|-----------|-----------|----------|-------|
| bullinger_handwritten    | M1     | ?         | ?         | ?        |       |
| bullinger_handwritten    | M2     | ?         | ?         | ?        |       |
| bullinger_handwritten    | M3     | ?         | ?         | ?        |       |
| bullinger_print          | M1     | ?         | ?         | ?        |       |
| easy_historical          | M1     | ?         | ?         | ?        |       |

Fill in the `?` values from the CSV files after copying results locally.

## Quick Reference Commands

### Cluster Login
```bash
ssh bergerd@diufrd200.unifr.ch
```

### Submit All Jobs (copy-paste)
```bash
cd ~/projects/bullinger-line-alignment-mwe
sbatch jobs/eval_gpu_m1.sbatch
sbatch jobs/eval_gpu_m2.sbatch
sbatch jobs/eval_gpu_m3.sbatch
sbatch jobs/eval_gpu_qwen_m1_print.sbatch
sbatch jobs/eval_gpu_qwen_m1_easy-historical.sbatch
```

### Check Job Status
```bash
squeue -u $USER
```

### Copy All Results (single command)
```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe && \
mkdir -p results_$(date +%Y%m%d) && cd results_$(date +%Y%m%d) && \
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*.csv . && \
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m* . && \
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/bullinger_print_predictions_m1 . && \
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/easy_hist_predictions_m1 .
```

## Next Steps After Evaluation

1. **Analyze Results**: Compare metrics across methods and datasets
2. **Error Analysis**: Examine failed predictions to understand model limitations
3. **Visualization**: Create charts/graphs of performance metrics
4. **Report Writing**: Document findings in your thesis
5. **Iteration**: Based on results, decide if further experiments are needed
