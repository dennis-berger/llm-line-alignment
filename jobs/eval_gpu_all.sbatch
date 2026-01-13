#!/bin/bash
#SBATCH --job-name=bullinger_all_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ==============================================================================
# Central Orchestration Job: Submit All Dataset-Method Combinations
# ==============================================================================
# This job submits all 15 evaluation jobs (5 datasets × 3 methods) to the cluster.
# The cluster scheduler will handle GPU resource allocation automatically.
#
# Usage:
#   sbatch jobs/eval_gpu_all.sbatch
#
# To cancel all submitted jobs:
#   scancel -u $USER -n bullinger_*
#   scancel -u $USER -n easy_historical_*
#   scancel -u $USER -n iam_*
# ==============================================================================

set -euo pipefail

echo "=== Submitting All Evaluation Jobs ==="
date

# Ensure we're in the project root (parent of jobs/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Working directory: $PWD"

# Track submitted job IDs
declare -a job_ids

# ------------------------------------------------------------------------------
# Dataset: bullinger_handwritten (Base dataset)
# ------------------------------------------------------------------------------
echo "Submitting bullinger_handwritten jobs..."
job_ids+=($(sbatch --parsable jobs/eval_gpu_m1.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m2.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m3.sbatch))

# ------------------------------------------------------------------------------
# Dataset: bullinger_print
# ------------------------------------------------------------------------------
echo "Submitting bullinger_print jobs..."
job_ids+=($(sbatch --parsable jobs/eval_gpu_qwen_m1_print.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m2_bullinger_print.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m3_bullinger_print.sbatch))

# ------------------------------------------------------------------------------
# Dataset: easy_historical
# ------------------------------------------------------------------------------
echo "Submitting easy_historical jobs..."
job_ids+=($(sbatch --parsable jobs/eval_gpu_qwen_m1_easy-historical.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m2_easy_historical.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m3_easy_historical.sbatch))

# ------------------------------------------------------------------------------
# Dataset: IAM_handwritten
# ------------------------------------------------------------------------------
echo "Submitting IAM_handwritten jobs..."
job_ids+=($(sbatch --parsable jobs/eval_gpu_m1_iam_handwritten.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m2_iam_handwritten.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m3_iam_handwritten.sbatch))

# ------------------------------------------------------------------------------
# Dataset: IAM_print
# ------------------------------------------------------------------------------
echo "Submitting IAM_print jobs..."
job_ids+=($(sbatch --parsable jobs/eval_gpu_m1_iam_print.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m2_iam_print.sbatch))
job_ids+=($(sbatch --parsable jobs/eval_gpu_m3_iam_print.sbatch))

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=== Submission Complete ==="
echo "Total jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all jobs with:"
echo "  scancel ${job_ids[*]}"
echo "  # or: scancel -u \$USER"
echo ""
date
echo "Done."
