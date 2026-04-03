#!/bin/bash
#SBATCH --job-name=iam_rep20_m5_ctx100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export REPO_ROOT
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m5/iam_handwritten_rep20_context100_0shot.sbatch}"
export DATASET_KEY="${DATASET_KEY:-IAM_handwritten}"
export DATA_DIR="${DATA_DIR:-datasets/IAM_handwritten_rwth_test_representative_20}"
export DATASET_TAG="${DATASET_TAG:-iam_handwritten_rwth_test_representative_20}"
export N_SHOTS="${N_SHOTS:-0}"
export SHOTS_SEED="${SHOTS_SEED:-42}"
export SEGMENTER="${SEGMENTER:-none}"
export EXISTING_LINES_DIR="${EXISTING_LINES_DIR:-$DATA_DIR/line_images}"
export LINE_IMAGE_MODE="${LINE_IMAGE_MODE:-numbered_strips}"
export USE_OCR_TEXT="${USE_OCR_TEXT:-1}"
export INCLUDE_PAGE_IMAGES="${INCLUDE_PAGE_IMAGES:-1}"
export STRIP_LINES_PER_IMAGE="${STRIP_LINES_PER_IMAGE:-12}"
export SPLIT_BY_PAGE="${SPLIT_BY_PAGE:-1}"
export SPLIT_MIN_LINES="${SPLIT_MIN_LINES:-100}"
export PROMPT_VARIANT="${PROMPT_VARIANT:-boundary_context_v2}"
export SKIP_OCR="${SKIP_OCR:-1}"

exec bash "$REPO_ROOT/jobs/eval/m5/common_m5_eval.sbatch"
