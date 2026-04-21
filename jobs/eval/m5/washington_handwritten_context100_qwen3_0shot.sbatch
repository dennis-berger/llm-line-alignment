#!/bin/bash
#SBATCH --job-name=wash_hw_m5_q32_ctx100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd205
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export REPO_ROOT
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m5/washington_handwritten_context100_qwen3_0shot.sbatch}"
export DATASET_KEY="${DATASET_KEY:-washington_handwritten}"
export DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
export DATASET_TAG="${DATASET_TAG:-washington_handwritten}"
export MODEL_NAME="${MODEL_NAME:-hf/Qwen/Qwen3-VL-32B-Instruct}"
export MODEL_SUFFIX="${MODEL_SUFFIX:-qwen3-vl-32b-instruct}"
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
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3-vl-32b-m5-washington}"
export EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-transformers>=4.57.0 accelerate bitsandbytes}"

exec bash "$REPO_ROOT/jobs/eval/m5/common_m5_eval.sbatch"
