#!/bin/bash
#SBATCH --job-name=bullinger_all_2shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# ==============================================================================
# Central Orchestration Job: Submit All 2-Shot Dataset-Method Combinations
# ==============================================================================
# This job submits all 18 evaluation jobs with 2-shot few-shot prompting.
# The cluster scheduler will handle GPU resource allocation automatically.
#
# Usage:
#   sbatch jobs/orchestrators/eval_all_2shot.sbatch
#
# To cancel all submitted jobs:
#   scancel -u $USER -n *_2shot
# ==============================================================================

set -euo pipefail

echo "=== Submitting All 2-Shot Evaluation Jobs ==="
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
#   MODEL_NAME="openai/gpt-5.2"
#   MODEL_LOCAL=""  # Not used for API models
#
MODEL_NAME="openai/gpt-5.2"
MODEL_LOCAL=""
# Extract short model suffix for output naming (e.g., "gpt-5.2")
MODEL_SUFFIX=$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
export SBATCH_EXPORT="MODEL_NAME=$MODEL_NAME,MODEL_LOCAL=$MODEL_LOCAL,MODEL_SUFFIX=$MODEL_SUFFIX"
echo "Using model: $MODEL_NAME (suffix: $MODEL_SUFFIX)"

# Track submitted job IDs
declare -a job_ids

# ------------------------------------------------------------------------------
# Method 1: Image + Transcription (6 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 1 (2-shot) jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/bullinger_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/bullinger_print_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/easy_historical_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/children_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/iam_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m1/iam_print_2shot.sbatch"))

# ------------------------------------------------------------------------------
# Method 2: Image + Transcription + HTR (6 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 2 (2-shot) jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/bullinger_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/bullinger_print_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/easy_historical_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/children_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/iam_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m2/iam_print_2shot.sbatch"))

# ------------------------------------------------------------------------------
# Method 3: Transcription + HTR, no images (6 datasets)
# ------------------------------------------------------------------------------
echo "Submitting Method 3 (2-shot) jobs..."
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/bullinger_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/bullinger_print_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/easy_historical_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/children_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/iam_handwritten_2shot.sbatch"))
job_ids+=($(sbatch --parsable --export="$SBATCH_EXPORT" "$JOBS_DIR/eval/m3/iam_print_2shot.sbatch"))

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=== Submission Complete ==="
echo "Total 2-shot jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all 2-shot jobs with:"
echo "  scancel ${job_ids[*]}"
echo "  # or: scancel -u \$USER -n *_2shot"
echo ""
date
echo "Done."
