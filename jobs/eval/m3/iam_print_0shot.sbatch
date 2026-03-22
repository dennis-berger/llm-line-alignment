#!/bin/bash
#SBATCH --job-name=iam_print_m3_0shot
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
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m3/iam_print_0shot.sbatch}"
export DATA_DIR="${DATA_DIR:-datasets/IAM_print}"
export DATASET_TAG="${DATASET_TAG:-iam_print}"
export N_SHOTS="${N_SHOTS:-0}"
export SHOTS_SEED="${SHOTS_SEED:-42}"

exec bash "$REPO_ROOT/jobs/eval/m3/common_m3_eval.sbatch"
