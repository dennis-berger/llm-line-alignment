#!/bin/bash
#SBATCH --job-name=ocr_bullinger_pylaia
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export REPO_ROOT
export RESUBMIT_SCRIPT="$REPO_ROOT/jobs/preprocessing/make_ocr_bullinger_handwritten_pylaia.sbatch"
export DATASET_KEY="bullinger_handwritten"
export DATA_DIR="datasets/bullinger_handwritten"
export DATASET_TAG="bullinger_handwritten"
export SEGMENTER="none"
export EXISTING_LINES_DIR="datasets/bullinger_handwritten/line_images"
export PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-bullinger}"
export PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/epoch=170-lowest_va_cer.ckpt}"
export PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
export SKIP_EVAL=1

exec bash "$REPO_ROOT/jobs/eval/m4/common_m4_eval.sbatch"
