#!/bin/bash
#SBATCH --job-name=handwritten_nntp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting handwritten NNTP baselines ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
TARGET_NNTP="${TARGET_NNTP:-washington,children}"
CONDA_ENV_NAME_BASE="${CONDA_ENV_NAME_BASE:-${CONDA_ENV_NAME:-bullinger-mwe}}"
CONDA_PKGS_DIRS_BASE="${CONDA_PKGS_DIRS_BASE:-${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs-clean}}"

make_export_list() {
  local export_list="ALL"
  local part
  for part in "$@"; do
    if [ -n "$part" ]; then
      export_list="${export_list},${part}"
    fi
  done
  printf '%s\n' "$export_list"
}

afterok_dependency() {
  local ids=()
  local id
  for id in "$@"; do
    if [ -n "$id" ]; then
      ids+=("$id")
    fi
  done
  if [ "${#ids[@]}" -eq 0 ]; then
    return 0
  fi

  local dep="afterok:${ids[0]}"
  local idx
  for ((idx = 1; idx < ${#ids[@]}; idx++)); do
    dep="${dep}:${ids[idx]}"
  done
  printf '%s\n' "$dep"
}

submit_job() {
  local dependency="$1"
  local export_vars="$2"
  local script="$3"
  local exclude_nodes="${SBATCH_EXCLUDE:-}"

  local args=(sbatch --parsable --export="$export_vars")
  if [ -n "$dependency" ]; then
    args+=(--dependency="$dependency")
  fi
  if [ -n "$exclude_nodes" ]; then
    args+=(--exclude="$exclude_nodes")
  fi
  args+=("$script")

  "${args[@]}"
}

target_nntp_enabled() {
  local target="$1"
  case ",${TARGET_NNTP}," in
    *,"${target}",*) return 0 ;;
    *) return 1 ;;
  esac
}

WASHINGTON_TRAIN_A_JOB_ID=""
WASHINGTON_TRAIN_B_JOB_ID=""
if target_nntp_enabled "washington"; then
  echo "Submitting Washington 2-fold PyLaia CV + NNTP jobs..."
  WASHINGTON_TRAIN_A_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "TRAIN_FOLD=train_a" "RUN_EVAL=1" "RUN_SUMMARY=1" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-washington-train-a" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-washington-train-a" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-washington-train-a-pylaia")" \
      "$JOBS_DIR/training/train_washington_handwritten_pylaia_cv.sbatch"
  )"
  WASHINGTON_TRAIN_B_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "TRAIN_FOLD=train_b" "RUN_EVAL=1" "RUN_SUMMARY=1" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-washington-train-b" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-washington-train-b" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-washington-train-b-pylaia")" \
      "$JOBS_DIR/training/train_washington_handwritten_pylaia_cv.sbatch"
  )"
fi

CHILDREN_FOLD_A_JOB_ID=""
CHILDREN_FOLD_B_JOB_ID=""
CHILDREN_FOLD_C_JOB_ID=""
CHILDREN_EVAL_JOB_ID=""
if target_nntp_enabled "children"; then
  echo "Submitting children 3-fold PyLaia CV training jobs..."
  CHILDREN_FOLD_A_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "TRAIN_FOLD=fold_a" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-children-fold-a" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-children-fold-a" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-children-fold-a-pylaia")" \
      "$JOBS_DIR/training/train_children_handwritten_pylaia_cv.sbatch"
  )"
  CHILDREN_FOLD_B_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "TRAIN_FOLD=fold_b" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-children-fold-b" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-children-fold-b" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-children-fold-b-pylaia")" \
      "$JOBS_DIR/training/train_children_handwritten_pylaia_cv.sbatch"
  )"
  CHILDREN_FOLD_C_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "TRAIN_FOLD=fold_c" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-children-fold-c" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-children-fold-c" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-children-fold-c-pylaia")" \
      "$JOBS_DIR/training/train_children_handwritten_pylaia_cv.sbatch"
  )"

  CHILDREN_EVAL_DEPENDENCY="$(
    afterok_dependency \
      "$CHILDREN_FOLD_A_JOB_ID" \
      "$CHILDREN_FOLD_B_JOB_ID" \
      "$CHILDREN_FOLD_C_JOB_ID"
  )"
  CHILDREN_EVAL_JOB_ID="$(
    submit_job \
      "$CHILDREN_EVAL_DEPENDENCY" \
      "$(make_export_list "REPO_ROOT=$REPO_ROOT" "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-children-eval" "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-children-eval" "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-children-eval-pylaia")" \
      "$JOBS_DIR/eval/nntp/children_handwritten.sbatch"
  )"
fi

echo ""
echo "=== Submission Complete ==="
echo "Washington fold jobs: ${WASHINGTON_TRAIN_A_JOB_ID} ${WASHINGTON_TRAIN_B_JOB_ID}"
echo "Children fold jobs: ${CHILDREN_FOLD_A_JOB_ID} ${CHILDREN_FOLD_B_JOB_ID} ${CHILDREN_FOLD_C_JOB_ID}"
echo "Children NNTP eval job: ${CHILDREN_EVAL_JOB_ID}"
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
