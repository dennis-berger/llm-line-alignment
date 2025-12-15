# FAITH Cluster Jobs

These Slurm job scripts run the evaluation experiments on the **FAITH HPC cluster** at Uni Fribourg.

## Available Job Scripts

### Bullinger Handwritten Dataset (All Methods)
- `eval_gpu_m1.sbatch` - Method 1: Images + Transcription
- `eval_gpu_m2.sbatch` - Method 2: Images + Transcription + HTR/OCR
- `eval_gpu_m3.sbatch` - Method 3: Transcription + HTR/OCR (no images)

### Bullinger Print Dataset (Method 1 Only)
- `eval_gpu_qwen_m1_print.sbatch` - Method 1: Images + Transcription

### Easy Historical Dataset (Method 1 Only)
- `eval_gpu_qwen_m1_easy-historical.sbatch` - Method 1: Images + Transcription

### Legacy Scripts
- `eval_gpu_qwen.sbatch` - Original Qwen evaluation script (use M1/M2/M3 scripts instead)

**Note:** Method 2 and 3 require an `ocr/` folder with HTR outputs. Currently, only `bullinger_handwritten` has this data.

## Quick Start

See `../EVALUATION_QUICKSTART.md` for the fastest way to run all evaluations.

## Usage

### Logging into the Faith Cluster

Before submitting jobs, connect to the cluster login node:
```bash
ssh bergerd@diufrd200.unifr.ch
```

### Submit All Jobs at Once
```bash
cd ~/projects/bullinger-line-alignment-mwe
mkdir -p logs

sbatch jobs/eval_gpu_m1.sbatch
sbatch jobs/eval_gpu_m2.sbatch
sbatch jobs/eval_gpu_m3.sbatch
sbatch jobs/eval_gpu_qwen_m1_print.sbatch
sbatch jobs/eval_gpu_qwen_m1_easy-historical.sbatch
```

### Monitor Jobs
```bash
# Check queue status
squeue -u $USER

# Watch queue continuously (Ctrl+C to exit)
watch -n 5 'squeue -u $USER'

# Follow a specific job's output
tail -f logs/bullinger_qwen_m1_*.out
```

### Copy Results to Local Machine

After jobs complete, copy results from the cluster:

```bash
# On your local machine
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Uni/Master_Thesis/bullinger-line-alignment-mwe

# Create results directory
mkdir -p results_$(date +%Y%m%d)
cd results_$(date +%Y%m%d)

# Copy all evaluation CSV files
scp bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/*.csv .

# Copy all prediction directories
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/predictions_m* .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/bullinger_print_predictions_m1 .
scp -r bergerd@diufrd200.unifr.ch:~/projects/bullinger-line-alignment-mwe/easy_hist_predictions_m1 .
```

## Documentation

For detailed documentation of the Faith HPC Cluster visit: https://diuf-doc.unifr.ch/books/faith-hpc-cluster

For a comprehensive evaluation plan with all steps, see: `../EVALUATION_PLAN.md`
