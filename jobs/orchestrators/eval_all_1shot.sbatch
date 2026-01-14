#!/bin/bash
#SBATCH --job-name=bullinger_all_1shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ==============================================================================
# Central Orchestration Job: Submit All 1-Shot Dataset-Method Combinations
# ==============================================================================
# This job submits all 13 evaluation jobs with 1-shot few-shot prompting.
# The cluster scheduler will handle GPU resource allocation automatically.
#
# Usage:
#   sbatch jobs/eval_gpu_all_1shot.sbatch
#
# To cancel all submitted jobs:
#   scancel -u $USER -n *_1shot
# ==============================================================================

set -euo pipefail

echo "=== Submitting All 1-Shot Evaluation Jobs ==="
date

# SLURM sets SLURM_SUBMIT_DIR to where the job was submitted from
# Navigate to project root (assuming submission from project root)
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $PWD"

# Use absolute paths for job submissions
JOBS_DIR="$PWD/jobs"

# Track submitted job IDs
declare -a job_ids

# ------------------------------------------------------------------------------
# Method 1: Image + Transcription (3 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 1 (1-shot) jobs..."
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m1/bullinger_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m1/iam_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m1/iam_print_1shot.sbatch"))

# ------------------------------------------------------------------------------
# Method 2: Image + Transcription + HTR (5 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 2 (1-shot) jobs..."
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m2/bullinger_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m2/bullinger_print_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m2/easy_historical_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m2/iam_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m2/iam_print_1shot.sbatch"))

# ------------------------------------------------------------------------------
# Method 3: Transcription + HTR, no images (5 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 3 (1-shot) jobs..."
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m3/bullinger_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m3/bullinger_print_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m3/easy_historical_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m3/iam_handwritten_1shot.sbatch"))
job_ids+=($(sbatch --parsable "$JOBS_DIR/eval/m3/iam_print_1shot.sbatch"))

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=== Submission Complete ==="
echo "Total 1-shot jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all 1-shot jobs with:"
echo "  scancel ${job_ids[*]}"
echo "  # or: scancel -u \$USER -n *_1shot"
echo ""
date
echo "Done."
