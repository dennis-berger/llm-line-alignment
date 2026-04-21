#!/bin/bash
#SBATCH --job-name=handwritten_m5_qwen3_ctl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

RUN_ROOT="${RUN_ROOT:?Missing RUN_ROOT}"
MODEL_NAME="${MODEL_NAME:?Missing MODEL_NAME}"
MODEL_SUFFIX="${MODEL_SUFFIX:?Missing MODEL_SUFFIX}"
QWEN_NODELIST="${QWEN_NODELIST:?Missing QWEN_NODELIST}"
SMOKE_IDS="${SMOKE_IDS:?Missing SMOKE_IDS}"
SMOKE_EXPECTED_COUNT="${SMOKE_EXPECTED_COUNT:?Missing SMOKE_EXPECTED_COUNT}"
SMOKE_STAGE="${SMOKE_STAGE:?Missing SMOKE_STAGE}"
SMOKE_ARTIFACT_ROOT="${SMOKE_ARTIFACT_ROOT:?Missing SMOKE_ARTIFACT_ROOT}"
FULL_QWEN_GRES="${FULL_QWEN_GRES:?Missing FULL_QWEN_GRES}"
QWEN_SMOKE_GRES_FALLBACK="${QWEN_SMOKE_GRES_FALLBACK:-gpu:rtx6000:2}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:?Missing CONDA_ENV_NAME}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
FULL_ORCHESTRATOR="${FULL_ORCHESTRATOR:?Missing FULL_ORCHESTRATOR}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:?Missing SMOKE_SCRIPT}"
CONTROLLER_SCRIPT="${CONTROLLER_SCRIPT:?Missing CONTROLLER_SCRIPT}"

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

prediction_count=0
trace_count=0
if [ -d "$SMOKE_ARTIFACT_ROOT/predictions" ]; then
  prediction_count="$(find "$SMOKE_ARTIFACT_ROOT/predictions" -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')"
fi
if [ -d "$SMOKE_ARTIFACT_ROOT/traces" ]; then
  trace_count="$(find "$SMOKE_ARTIFACT_ROOT/traces" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
fi

if [ -f "$SMOKE_ARTIFACT_ROOT/evaluation.csv" ] && [ "$prediction_count" = "$SMOKE_EXPECTED_COUNT" ] && [ "$trace_count" = "$SMOKE_EXPECTED_COUNT" ]; then
  FULL_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "RUN_ROOT=$RUN_ROOT" \
        "MODEL_NAME=$MODEL_NAME" \
        "MODEL_SUFFIX=$MODEL_SUFFIX" \
        "QWEN_NODELIST=$QWEN_NODELIST" \
        "QWEN_GRES=$FULL_QWEN_GRES" \
        "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
        "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
      )" \
      "$FULL_ORCHESTRATOR"
  )"
  echo "Smoke succeeded (${SMOKE_STAGE}); submitted full Qwen sweep: ${FULL_JOB_ID}"
  exit 0
fi

if [ "$SMOKE_STAGE" = "g1" ]; then
  RETRY_ARTIFACT_ROOT="${RUN_ROOT}/bullinger_handwritten/m5_context100/${MODEL_SUFFIX}_smoke_g2"
  RETRY_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "MODEL_NAME=$MODEL_NAME" \
        "MODEL_SUFFIX=$MODEL_SUFFIX" \
        "IDS=$SMOKE_IDS" \
        "OUT_DIR=$RETRY_ARTIFACT_ROOT/predictions" \
        "EVAL_CSV=$RETRY_ARTIFACT_ROOT/evaluation.csv" \
        "CHECKPOINT_DIR=$RETRY_ARTIFACT_ROOT/checkpoints" \
        "TRACE_DIR=$RETRY_ARTIFACT_ROOT/traces" \
        "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
        "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
      )" \
      "$SMOKE_SCRIPT" \
      --nodelist="$QWEN_NODELIST" \
      --gres="$QWEN_SMOKE_GRES_FALLBACK"
  )"
  echo "Smoke failed on 1 GPU; submitted 2-GPU retry: ${RETRY_JOB_ID}"

  RETRY_CONTROLLER_JOB_ID="$(
    submit_job \
      "afterany:${RETRY_JOB_ID}" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "RUN_ROOT=$RUN_ROOT" \
        "MODEL_NAME=$MODEL_NAME" \
        "MODEL_SUFFIX=$MODEL_SUFFIX" \
        "QWEN_NODELIST=$QWEN_NODELIST" \
        "SMOKE_IDS=$SMOKE_IDS" \
        "SMOKE_EXPECTED_COUNT=$SMOKE_EXPECTED_COUNT" \
        "SMOKE_STAGE=g2" \
        "SMOKE_ARTIFACT_ROOT=$RETRY_ARTIFACT_ROOT" \
        "FULL_QWEN_GRES=$QWEN_SMOKE_GRES_FALLBACK" \
        "QWEN_SMOKE_GRES_FALLBACK=$QWEN_SMOKE_GRES_FALLBACK" \
        "CONDA_ENV_NAME=$CONDA_ENV_NAME" \
        "EXTRA_PIP_PACKAGES=$EXTRA_PIP_PACKAGES" \
        "FULL_ORCHESTRATOR=$FULL_ORCHESTRATOR" \
        "SMOKE_SCRIPT=$SMOKE_SCRIPT" \
        "CONTROLLER_SCRIPT=$CONTROLLER_SCRIPT" \
      )" \
      "$CONTROLLER_SCRIPT"
  )"
  echo "Submitted 2-GPU smoke controller: ${RETRY_CONTROLLER_JOB_ID}"
  exit 0
fi

echo "Smoke failed after 2-GPU fallback. Predictions=${prediction_count}, traces=${trace_count}"
exit 1
