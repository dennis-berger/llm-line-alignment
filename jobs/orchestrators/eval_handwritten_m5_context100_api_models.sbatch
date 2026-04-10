#!/bin/bash
#SBATCH --job-name=handwritten_m5_ctx100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting handwritten M5 context100 API sweep ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_ROOT="${RUN_ROOT:-cluster_runs/thesis_handwritten_completion_${RUN_DATE}}"

OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-openai/gpt-5.4}"
OPENAI_MODEL_FALLBACK="${OPENAI_MODEL_FALLBACK:-openai/gpt-5.4-2026-03-05}"
GEMINI_MODEL_NAME="${GEMINI_MODEL_NAME:-gemini/gemini-2.5-pro}"
MISTRAL_MODEL_NAME="${MISTRAL_MODEL_NAME:-mistral/mistral-large-2512}"
TARGET_MODELS="${TARGET_MODELS:-openai,gemini,mistral}"
OPENAI_DATASETS="${OPENAI_DATASETS:-children_handwritten}"
GEMINI_DATASETS="${GEMINI_DATASETS:-bullinger_handwritten,children_handwritten,washington_handwritten,iam_handwritten_rwth_test_representative_20}"
MISTRAL_DATASETS="${MISTRAL_DATASETS:-bullinger_handwritten,children_handwritten,washington_handwritten,iam_handwritten_rwth_test_representative_20}"

CONDA_ENV_NAME_BASE="${CONDA_ENV_NAME_BASE:-${CONDA_ENV_NAME:-bullinger-mwe}}"
CONDA_PKGS_DIRS_BASE="${CONDA_PKGS_DIRS_BASE:-${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs-clean}}"

to_model_suffix() {
  basename "$1" | tr '[:upper:]' '[:lower:]'
}

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

resolve_openai_model() {
  local preferred="$1"
  local fallback="$2"

  if [ -z "${OPENAI_API_KEY:-}" ] || ! command -v curl >/dev/null 2>&1; then
    printf '%s\n' "$preferred"
    return 0
  fi

  local preferred_model="${preferred#openai/}"
  local status
  status="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      "https://api.openai.com/v1/models/${preferred_model}" \
      -H "Authorization: Bearer ${OPENAI_API_KEY}" \
      || printf '000'
  )"

  if [ "$status" = "200" ]; then
    printf '%s\n' "$preferred"
    return 0
  fi
  if [ "$status" = "404" ]; then
    echo "OpenAI model ${preferred_model} is unavailable; falling back to ${fallback#openai/}." >&2
    printf '%s\n' "$fallback"
    return 0
  fi

  echo "OpenAI model probe returned HTTP ${status}; keeping ${preferred_model}." >&2
  printf '%s\n' "$preferred"
}

target_model_enabled() {
  local model_label="$1"
  case ",${TARGET_MODELS}," in
    *,"${model_label}",*) return 0 ;;
    *) return 1 ;;
  esac
}

model_has_credentials() {
  local model_label="$1"
  case "$model_label" in
    openai)
      [ -n "${OPENAI_API_KEY:-}" ]
      return
      ;;
    gemini)
      [ -n "${GOOGLE_API_KEY:-}" ]
      return
      ;;
    mistral)
      [ -n "${MISTRAL_API_KEY:-}" ]
      return
      ;;
    *)
      return 1
      ;;
  esac
}

dataset_in_csv() {
  local dataset="$1"
  local csv_list="$2"
  case ",${csv_list}," in
    *,"${dataset}",*) return 0 ;;
    *) return 1 ;;
  esac
}

dataset_job_script() {
  local dataset="$1"
  case "$dataset" in
    bullinger_handwritten)
      printf '%s\n' "$JOBS_DIR/eval/m5/bullinger_handwritten_context100_0shot.sbatch"
      ;;
    children_handwritten)
      printf '%s\n' "$JOBS_DIR/eval/m5/children_handwritten_context100_0shot.sbatch"
      ;;
    washington_handwritten)
      printf '%s\n' "$JOBS_DIR/eval/m5/washington_handwritten_context100_0shot.sbatch"
      ;;
    iam_handwritten_rwth_test_representative_20)
      printf '%s\n' "$JOBS_DIR/eval/m5/iam_handwritten_rep20_context100_0shot.sbatch"
      ;;
    *)
      echo "ERROR: Unsupported dataset '$dataset'." >&2
      return 1
      ;;
  esac
}

m5_artifact_root() {
  local dataset="$1"
  local model_suffix="$2"
  printf '%s\n' "$RUN_ROOT/$dataset/m5_context100/$model_suffix"
}

submit_m5_dataset_job() {
  local dataset="$1"
  local model_name="$2"
  local model_local="$3"
  local chain_label="$4"

  local model_suffix
  local dataset_script
  local env_name
  local pkgs_dir
  local pylaia_env
  local artifact_root
  local export_vars=()
  local job_id=""

  model_suffix="$(to_model_suffix "$model_name")"
  dataset_script="$(dataset_job_script "$dataset")"
  env_name="${CONDA_ENV_NAME_BASE}-${chain_label}-${dataset}-m5"
  pkgs_dir="${CONDA_PKGS_DIRS_BASE}-${chain_label}-${dataset}-m5"
  pylaia_env="${CONDA_ENV_NAME_BASE}-${chain_label}-${dataset}-m5-pylaia"
  artifact_root="$(m5_artifact_root "$dataset" "$model_suffix")"

  export_vars=(
    "REPO_ROOT=$REPO_ROOT"
    "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_handwritten_m5_context100_api_models.sbatch"
    "MODEL_NAME=$model_name"
    "MODEL_LOCAL=$model_local"
    "MODEL_SUFFIX=$model_suffix"
    "N_SHOTS=0"
    "SHOTS_SEED=42"
    "LINE_IMAGE_MODE=numbered_strips"
    "USE_OCR_TEXT=1"
    "INCLUDE_PAGE_IMAGES=1"
    "SPLIT_BY_PAGE=1"
    "SPLIT_MIN_LINES=100"
    "PROMPT_VARIANT=boundary_context_v2"
    "SKIP_OCR=1"
    "OUT_DIR=$artifact_root/predictions"
    "EVAL_CSV=$artifact_root/evaluation.csv"
    "CHECKPOINT_DIR=$artifact_root/checkpoints"
    "TRACE_DIR=$artifact_root/traces"
    "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE"
    "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE"
    "CONDA_ENV_NAME=$env_name"
    "CONDA_PKGS_DIRS=$pkgs_dir"
    "PYLAIA_CONDA_ENV=$pylaia_env"
  )

  job_id="$(
    submit_job \
      "" \
      "$(make_export_list "${export_vars[@]}")" \
      "$dataset_script"
  )"
  echo "${chain_label} M5 context100 ${dataset} (${model_name}): ${job_id}"
}

submit_model_group() {
  local chain_label="$1"
  local model_name="$2"
  local model_local="$3"
  local dataset_csv="$4"

  local dataset
  for dataset in \
    bullinger_handwritten \
    children_handwritten \
    washington_handwritten \
    iam_handwritten_rwth_test_representative_20
  do
    if dataset_in_csv "$dataset" "$dataset_csv"; then
      submit_m5_dataset_job "$dataset" "$model_name" "$model_local" "$chain_label"
    fi
  done
}

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "WARNING: GOOGLE_API_KEY is not set in the submission environment."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set in the submission environment."
fi
if [ -z "${MISTRAL_API_KEY:-}" ]; then
  echo "WARNING: MISTRAL_API_KEY is not set in the submission environment."
fi

OPENAI_CHAIN_MODEL="$(resolve_openai_model "$OPENAI_MODEL_NAME" "$OPENAI_MODEL_FALLBACK")"

if target_model_enabled "openai" && model_has_credentials "openai"; then
  submit_model_group "openai" "$OPENAI_CHAIN_MODEL" "" "$OPENAI_DATASETS"
elif target_model_enabled "openai"; then
  echo "Skipping openai M5 context100 jobs because OPENAI_API_KEY is missing."
fi
if target_model_enabled "gemini" && model_has_credentials "gemini"; then
  submit_model_group "gemini" "$GEMINI_MODEL_NAME" "" "$GEMINI_DATASETS"
elif target_model_enabled "gemini"; then
  echo "Skipping gemini M5 context100 jobs because GOOGLE_API_KEY is missing."
fi
if target_model_enabled "mistral" && model_has_credentials "mistral"; then
  submit_model_group "mistral" "$MISTRAL_MODEL_NAME" "" "$MISTRAL_DATASETS"
elif target_model_enabled "mistral"; then
  echo "Skipping mistral M5 context100 jobs because MISTRAL_API_KEY is missing."
fi

echo ""
echo "Run root: $RUN_ROOT"
echo "=== Submission Complete ==="
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
