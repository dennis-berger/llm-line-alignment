#!/bin/bash
#SBATCH --job-name=children_handwritten_m4_0shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export REPO_ROOT
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m4/children_handwritten_0shot.sbatch}"
export DATASET_KEY="${DATASET_KEY:-children_handwritten}"
export DATA_DIR="${DATA_DIR:-datasets/children_handwritten}"
export DATASET_TAG="${DATASET_TAG:-children_handwritten}"
export N_SHOTS="${N_SHOTS:-0}"
export SHOTS_SEED="${SHOTS_SEED:-42}"
export SEGMENTER="${SEGMENTER:-none}"
export EXISTING_LINES_DIR="${EXISTING_LINES_DIR:-datasets/children_handwritten/line_images}"
export SKIP_OCR="${SKIP_OCR:-1}"

exec bash "$REPO_ROOT/jobs/eval/m4/common_m4_eval.sbatch"
