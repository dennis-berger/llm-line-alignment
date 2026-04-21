#!/bin/bash
#SBATCH --job-name=handwritten_m5_qwen3_submit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting handwritten M5 context100 Qwen3-VL-32B sweep ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_ROOT="${RUN_ROOT:-cluster_runs/handwritten_m5_context100_qwen3_32b_${RUN_DATE}}"
MODEL_NAME="${MODEL_NAME:-hf/Qwen/Qwen3-VL-32B-Instruct}"
MODEL_SUFFIX="${MODEL_SUFFIX:-qwen3-vl-32b-instruct}"
QWEN_NODELIST="${QWEN_NODELIST:-diufrd205}"
QWEN_SMOKE_GRES="${QWEN_SMOKE_GRES:-gpu:rtx6000:1}"
QWEN_SMOKE_GRES_FALLBACK="${QWEN_SMOKE_GRES_FALLBACK:-gpu:rtx6000:2}"
SMOKE_IDS="${SMOKE_IDS:-10069,10095,10100}"
SMOKE_EXPECTED_COUNT="${SMOKE_EXPECTED_COUNT:-3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3-vl-32b-m5}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-transformers>=4.57.0 accelerate bitsandbytes}"
FULL_ORCHESTRATOR="$REPO_ROOT/jobs/orchestrators/eval_handwritten_m5_context100_qwen3_32b_full.sbatch"
CONTROLLER_SCRIPT="$REPO_ROOT/jobs/orchestrators/eval_handwritten_m5_context100_qwen3_32b_smoke_controller.sbatch"
SMOKE_SCRIPT="$REPO_ROOT/jobs/eval/m5/bullinger_handwritten_context100_qwen3_0shot.sbatch"
SMOKE_ARTIFACT_ROOT="${RUN_ROOT}/bullinger_handwritten/m5_context100/${MODEL_SUFFIX}_smoke_g1"

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
  local dependency="$1"
  local export_vars="$2"
  local script="$3"
  shift 3

  local args=(sbatch --parsable --export="$export_vars")
  if [ -n "$dependency" ]; then
    args+=(--dependency="$dependency")
  fi
  args+=("$@" "$script")
  "${args[@]}"
}

SMOKE_JOB_ID="$(
  submit_job \
    "" \
    "$(make_export_list \
      "REPO_ROOT=$REPO_ROOT" \
      "MODEL_NAME=$MODEL_NAME" \
      "MODEL_SUFFIX=$MODEL_SUFFIX" \
      "IDS=$SMOKE_IDS" \
      "OUT_DIR=$SMOKE_ARTIFACT_ROOT/predictions" \
      "EVAL_CSV=$SMOKE_ARTIFACT_ROOT/evaluation.csv" \
      "CHECKPOINT_DIR=$SMOKE_ARTIFACT_ROOT/checkpoints" \
      "TRACE_DIR=$SMOKE_ARTIFACT_ROOT/traces" \
      "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
      "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
    )" \
    "$SMOKE_SCRIPT" \
    --nodelist="$QWEN_NODELIST" \
    --gres="$QWEN_SMOKE_GRES"
)"
echo "Qwen smoke (1 GPU): ${SMOKE_JOB_ID}"

CONTROLLER_JOB_ID="$(
  submit_job \
    "afterany:${SMOKE_JOB_ID}" \
    "$(make_export_list \
      "REPO_ROOT=$REPO_ROOT" \
      "RUN_ROOT=$RUN_ROOT" \
      "MODEL_NAME=$MODEL_NAME" \
      "MODEL_SUFFIX=$MODEL_SUFFIX" \
      "QWEN_NODELIST=$QWEN_NODELIST" \
      "SMOKE_IDS=$SMOKE_IDS" \
      "SMOKE_EXPECTED_COUNT=$SMOKE_EXPECTED_COUNT" \
      "SMOKE_STAGE=g1" \
      "SMOKE_ARTIFACT_ROOT=$SMOKE_ARTIFACT_ROOT" \
      "FULL_QWEN_GRES=$QWEN_SMOKE_GRES" \
      "QWEN_SMOKE_GRES_FALLBACK=$QWEN_SMOKE_GRES_FALLBACK" \
      "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
      "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
      "FULL_ORCHESTRATOR=$FULL_ORCHESTRATOR" \
      "SMOKE_SCRIPT=$SMOKE_SCRIPT" \
      "CONTROLLER_SCRIPT=$CONTROLLER_SCRIPT" \
    )" \
    "$CONTROLLER_SCRIPT"
)"
echo "Qwen smoke controller: ${CONTROLLER_JOB_ID}"

echo "Run root: $RUN_ROOT"
echo "Done."
