#!/bin/bash
#SBATCH --job-name=washington_nntp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
export WORK_DIR="${WORK_DIR:-outputs/nntp/washington_handwritten_iam_zero_adapt}"
export PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-iam}"
export PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/weights.ckpt}"
export PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
export NNTP_ROOT="${NNTP_ROOT:-../nntp}"
export PREPARE_MODE="${PREPARE_MODE:-presegmented}"
export PRED_DIR="${PRED_DIR:-washington_handwritten_predictions_nntp_iam_zero_adapt}"
export EVAL_CSV="${EVAL_CSV:-washington_handwritten_eval_nntp_iam_zero_adapt.csv}"

exec bash "$REPO_ROOT/jobs/eval/nntp/iam_handwritten.sbatch"
