#!/bin/bash
#SBATCH --job-name=handwritten_m5_qwen3_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting handwritten M5 context100 Qwen3-VL-32B full sweep ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_ROOT="${RUN_ROOT:-cluster_runs/handwritten_m5_context100_qwen3_32b_${RUN_DATE}}"
MODEL_NAME="${MODEL_NAME:-hf/Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SUFFIX="${MODEL_SUFFIX:-qwen3-vl-32b-instruct}"
QWEN_NODELIST="${QWEN_NODELIST:-diufrd205}"
QWEN_GRES="${QWEN_GRES:-gpu:rtx6000:1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3-vl-32b-m5}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-transformers>=4.57.0 accelerate bitsandbytes}"

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

submit_job() {
  local export_vars="$1"
  local script="$2"
  shift 2

  local args=(sbatch --parsable --export="$export_vars")
  args+=("$@" "$script")
  "${args[@]}"
}

dataset_job_script() {
  local dataset="$1"
  case "$dataset" in
    bullinger_handwritten)
      printf '%s\n' "$REPO_ROOT/jobs/eval/m5/bullinger_handwritten_context100_qwen3_0shot.sbatch"
      ;;
    children_handwritten)
      printf '%s\n' "$REPO_ROOT/jobs/eval/m5/children_handwritten_context100_qwen3_0shot.sbatch"
      ;;
    washington_handwritten)
      printf '%s\n' "$REPO_ROOT/jobs/eval/m5/washington_handwritten_context100_qwen3_0shot.sbatch"
      ;;
    iam_handwritten_rwth_test_representative_20)
      printf '%s\n' "$REPO_ROOT/jobs/eval/m5/iam_handwritten_rep20_context100_qwen3_0shot.sbatch"
      ;;
    *)
      echo "ERROR: Unsupported dataset '$dataset'." >&2
      return 1
      ;;
  esac
}

for dataset in \
  bullinger_handwritten \
  children_handwritten \
  washington_handwritten \
  iam_handwritten_rwth_test_representative_20
do
  dataset_script="$(dataset_job_script "$dataset")"
  artifact_root="$RUN_ROOT/$dataset/m5_context100/$MODEL_SUFFIX"
  job_id="$(
    submit_job \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "MODEL_NAME=$MODEL_NAME" \
        "MODEL_SUFFIX=$MODEL_SUFFIX" \
        "OUT_DIR=$artifact_root/predictions" \
        "EVAL_CSV=$artifact_root/evaluation.csv" \
        "CHECKPOINT_DIR=$artifact_root/checkpoints" \
        "TRACE_DIR=$artifact_root/traces" \
        "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
        "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
      )" \
      "$dataset_script" \
      --nodelist="$QWEN_NODELIST" \
      --gres="$QWEN_GRES"
  )"
  echo "Qwen3-VL-32B M5 context100 ${dataset}: ${job_id}"
done

echo "Run root: $RUN_ROOT"
echo "Done."
