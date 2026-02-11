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

# SLURM sets SLURM_SUBMIT_DIR to where the job was submitted from
# Navigate to project root (assuming submission from project root)
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $PWD"

# Use absolute paths for job submissions
JOBS_DIR="$PWD/jobs"

# ==============================================================================
# MODEL CONFIGURATION - Change this to use a different model for all jobs
# ==============================================================================
# For HuggingFace models (local or remote):
#   MODEL_NAME="hf/Qwen/Qwen3-VL-8B-Instruct"
#   MODEL_LOCAL="./.hf/Qwen3-VL-8B-Instruct"
#
# For OpenAI API models (requires OPENAI_API_KEY env var):
#   MODEL_NAME="openai/gpt-5.2-vision"
#   MODEL_LOCAL=""  # Not used for API models
#
MODEL_NAME="openai/gpt-5.2-vision"
MODEL_LOCAL=""
# Extract short model suffix for output naming (e.g., "qwen3-vl-32b-instruct")
MODEL_SUFFIX=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
export SBATCH_EXPORT="MODEL_NAME=$MODEL_NAME,MODEL_LOCAL=$MODEL_LOCAL,MODEL_SUFFIX=$MODEL_SUFFIX"
echo "Using model: $MODEL_NAME (local: $MODEL_LOCAL, suffix: $MODEL_SUFFIX)"

# Track submitted job IDs
declare -a job_ids

# ------------------------------------------------------------------------------
# Dataset: bullinger_handwritten (Base dataset)
# ------------------------------------------------------------------------------
echo "Submitting bullinger_handwritten jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/bullinger_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/bullinger_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/bullinger_handwritten_0shot.sbatch"))

# ------------------------------------------------------------------------------
# Dataset: bullinger_print
# ------------------------------------------------------------------------------
echo "Submitting bullinger_print jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/bullinger_print_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/bullinger_print_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/bullinger_print_0shot.sbatch"))

# ------------------------------------------------------------------------------
# Dataset: easy_historical
# ------------------------------------------------------------------------------
echo "Submitting easy_historical jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/easy_historical_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/easy_historical_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/easy_historical_0shot.sbatch"))

# ------------------------------------------------------------------------------
# Dataset: children_handwritten
# ------------------------------------------------------------------------------
echo "Submitting children_handwritten jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/children_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/children_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/children_handwritten_0shot.sbatch"))

# ------------------------------------------------------------------------------
# Dataset: IAM_handwritten
# ------------------------------------------------------------------------------
echo "Submitting IAM_handwritten jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/iam_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/iam_handwritten_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/iam_handwritten_0shot.sbatch"))

# ------------------------------------------------------------------------------
# Dataset: IAM_print
# ------------------------------------------------------------------------------
echo "Submitting IAM_print jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/iam_print_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/iam_print_0shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/iam_print_0shot.sbatch"))

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
