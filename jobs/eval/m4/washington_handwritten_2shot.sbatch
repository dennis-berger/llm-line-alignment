#!/bin/bash
#SBATCH --job-name=washington_handwritten_m4_2shot
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
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m4/washington_handwritten_2shot.sbatch}"
export DATASET_KEY="${DATASET_KEY:-washington_handwritten}"
export DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
export DATASET_TAG="${DATASET_TAG:-washington_handwritten}"
export N_SHOTS="${N_SHOTS:-2}"
export SHOTS_SEED="${SHOTS_SEED:-42}"
export SEGMENTER="${SEGMENTER:-none}"
export PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-iam}"
export PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/weights.ckpt}"
export PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
export EXISTING_LINES_DIR="${EXISTING_LINES_DIR:-datasets/washington_handwritten/line_images}"

exec bash "$REPO_ROOT/jobs/eval/m4/common_m4_eval.sbatch"
