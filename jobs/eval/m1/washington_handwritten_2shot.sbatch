#!/bin/bash
#SBATCH --job-name=washington_handwritten_m1_2shot
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
export RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT:-$REPO_ROOT/jobs/eval/m1/washington_handwritten_2shot.sbatch}"
export DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
export DATASET_TAG="${DATASET_TAG:-washington_handwritten}"
export N_SHOTS="${N_SHOTS:-2}"
export SHOTS_SEED="${SHOTS_SEED:-42}"

exec bash "$REPO_ROOT/jobs/eval/m1/common_m1_eval.sbatch"
