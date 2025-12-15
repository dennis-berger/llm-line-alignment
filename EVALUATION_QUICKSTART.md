# Evaluation Quick Start Guide

## TL;DR - Run Everything Now

### 1. Login to Cluster
```bash
ssh bergerd@diufrd200.unifr.ch
```

### 2. Go to Project & Submit All Jobs
```bash
cd ~/projects/bullinger-line-alignment-mwe
mkdir -p logs

# Submit all 5 jobs
sbatch jobs/eval_gpu_m1.sbatch
sbatch jobs/eval_gpu_m2.sbatch
sbatch jobs/eval_gpu_m3.sbatch
sbatch jobs/eval_gpu_qwen_m1_print.sbatch
sbatch jobs/eval_gpu_qwen_m1_easy-historical.sbatch
```

### 3. Monitor Progress
```bash
# Check queue
squeue -u $USER

# Watch newest log
ls -t logs/*.out | head -1 | xargs tail -f
```

### 4. Copy Results to Local (when done)
```bash
# On your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe
mkdir -p results_$(date +%Y%m%d)
cd results_$(date +%Y%m%d)

# Copy everything
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*.csv .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m* .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*_predictions_m1 .
```

### 5. View Results
```bash
# Open CSV files in your spreadsheet app
open *.csv
```

---

## What Gets Run

| Job # | Dataset               | Method | Output Directory              | CSV File                        | Time   |
|-------|-----------------------|--------|-------------------------------|---------------------------------|--------|
| 1     | bullinger_handwritten | M1     | predictions_m1/               | evaluation_qwen_m1.csv          | ~1.5h  |
| 2     | bullinger_handwritten | M2     | predictions_m2/               | evaluation_qwen_m2.csv          | ~1.5h  |
| 3     | bullinger_handwritten | M3     | predictions_m3/               | evaluation_qwen_m3.csv          | ~1h    |
| 4     | bullinger_print       | M1     | bullinger_print_predictions_m1/ | bullinger_print_eval_qwen_m1.csv | ~1h    |
| 5     | easy_historical       | M1     | easy_hist_predictions_m1/     | easy_hist_eval_qwen_m1.csv      | ~1h    |

**Total:** 5 jobs, ~6-7 hours if run in parallel

---

## Why Only These Jobs?

- **Method 1** needs: `gt/`, `images/`, `transcription/` ✅ All datasets have these
- **Method 2** needs: `gt/`, `images/`, `transcription/`, **`ocr/`** ❌ Only bullinger_handwritten has ocr/
- **Method 3** needs: `gt/`, `transcription/`, **`ocr/`** ❌ Only bullinger_handwritten has ocr/

---

## Expected Files After Completion

```
cluster:~/projects/bullinger-line-alignment-mwe/
├── evaluation_qwen_m1.csv                    # ← Main results file
├── evaluation_qwen_m2.csv                    # ← Main results file
├── evaluation_qwen_m3.csv                    # ← Main results file
├── bullinger_print_eval_qwen_m1.csv          # ← Print results
├── easy_hist_eval_qwen_m1.csv                # ← Easy results
├── predictions_m1/                           # 10 text files
├── predictions_m2/                           # 10 text files
├── predictions_m3/                           # 10 text files
├── bullinger_print_predictions_m1/           # 10 text files
├── easy_hist_predictions_m1/                 # 20 text files
└── logs/                                     # Job logs
    ├── bullinger_qwen_m1_12345.out
    ├── bullinger_qwen_m2_12346.out
    └── ...
```

---

## Troubleshooting

### Jobs stuck in queue?
```bash
sinfo  # Check available nodes
```

### Job failed?
```bash
tail -100 logs/bullinger_qwen_m1_*.err  # Check error log
```

### Out of memory?
Edit job file and change:
```bash
#SBATCH --mem=64G  →  #SBATCH --mem=96G
```

---

## See Full Documentation

For detailed step-by-step instructions, see: `EVALUATION_PLAN.md`
