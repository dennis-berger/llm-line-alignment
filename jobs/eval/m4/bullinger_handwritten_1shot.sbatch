#!/bin/bash
#SBATCH --job-name=bullinger_handwritten_m4_1shot
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
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m4/bullinger_handwritten_1shot.sbatch}"
export DATASET_KEY="${DATASET_KEY:-bullinger_handwritten}"
export DATA_DIR="${DATA_DIR:-datasets/bullinger_handwritten}"
export DATASET_TAG="${DATASET_TAG:-bullinger_handwritten}"
export N_SHOTS="${N_SHOTS:-1}"
export SHOTS_SEED="${SHOTS_SEED:-42}"
export SEGMENTER="${SEGMENTER:-none}"
export PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-bullinger}"
export PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/epoch=170-lowest_va_cer.ckpt}"
export PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
export EXISTING_LINES_DIR="${EXISTING_LINES_DIR:-datasets/bullinger_handwritten/line_images}"
export SKIP_OCR="${SKIP_OCR:-1}"

exec bash "$REPO_ROOT/jobs/eval/m4/common_m4_eval.sbatch"
