#!/bin/bash
#SBATCH --job-name=handwritten_matrix_0shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting handwritten 0-shot matrix ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
IAM_REP20_DATA_DIR="datasets/IAM_handwritten_rwth_test_representative_20"
IAM_REP20_TAG="iam_handwritten_rwth_test_representative_20"
IAM_REP20_LINES_DIR="$IAM_REP20_DATA_DIR/line_images"

HF_MODEL_NAME="${HF_MODEL_NAME:-hf/Qwen/Qwen3-VL-8B-Instruct}"
HF_MODEL_LOCAL="${HF_MODEL_LOCAL:-./.hf/Qwen3-VL-8B-Instruct}"
GEMINI_MODEL_NAME="${GEMINI_MODEL_NAME:-gemini/gemini-3.1-pro-preview}"
OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-openai/gpt-5.4-2026-03-05}"
OPENAI_MODEL_FALLBACK="${OPENAI_MODEL_FALLBACK:-openai/gpt-5.4}"
MISTRAL_MODEL_NAME="${MISTRAL_MODEL_NAME:-mistral/mistral-large-2512}"
TARGET_CHAINS="${TARGET_CHAINS:-hf,gemini,openai,mistral}"
SKIP_SHARED_PREP="${SKIP_SHARED_PREP:-0}"

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

  local args=(sbatch --parsable --export="$export_vars")
  if [ -n "$dependency" ]; then
    args+=(--dependency="$dependency")
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
    echo "OpenAI snapshot ${preferred_model} is unavailable; falling back to ${fallback#openai/}." >&2
    printf '%s\n' "$fallback"
    return 0
  fi

  echo "OpenAI snapshot probe returned HTTP ${status}; keeping ${preferred_model}." >&2
  printf '%s\n' "$preferred"
}

submit_chain_step() {
  local tail_job_id="$1"
  local export_vars="$2"
  local script="$3"
  shift 3

  local dependency
  dependency="$(afterok_dependency "$tail_job_id" "$@")"
  submit_job "$dependency" "$export_vars" "$script"
}

submit_handwritten_chain() {
  local chain_label="$1"
  local model_name="$2"
  local model_local="$3"

  local model_suffix
  model_suffix="$(to_model_suffix "$model_name")"

  local base_exports=(
    "REPO_ROOT=$REPO_ROOT"
    "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_handwritten_0shot_matrix.sbatch"
    "MODEL_NAME=$model_name"
    "MODEL_LOCAL=$model_local"
    "MODEL_SUFFIX=$model_suffix"
    "N_SHOTS=0"
    "SHOTS_SEED=42"
    "TARGET_CHAINS=$chain_label"
    "SKIP_SHARED_PREP=1"
  )

  local tail_job_id=""
  local job_id=""
  local export_vars=""
  declare -a chain_job_ids=()

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m1/bullinger_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m1/children_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}" "DATA_DIR=$IAM_REP20_DATA_DIR" "DATASET_TAG=$IAM_REP20_TAG")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m1/iam_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m1/washington_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m2/bullinger_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m2/children_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}" "DATA_DIR=$IAM_REP20_DATA_DIR" "DATASET_TAG=$IAM_REP20_TAG")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m2/iam_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m2/washington_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m3/bullinger_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m3/children_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}" "DATA_DIR=$IAM_REP20_DATA_DIR" "DATASET_TAG=$IAM_REP20_TAG")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m3/iam_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m3/washington_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m4/bullinger_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m4/children_handwritten_0shot.sbatch")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(
    make_export_list \
      "${base_exports[@]}" \
      "DATASET_KEY=IAM_handwritten" \
      "DATA_DIR=$IAM_REP20_DATA_DIR" \
      "DATASET_TAG=$IAM_REP20_TAG" \
      "SEGMENTER=none" \
      "EXISTING_LINES_DIR=$IAM_REP20_LINES_DIR" \
      "SKIP_OCR=1"
  )"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m4/iam_handwritten_0shot.sbatch" "$IAM_REP20_PREP_JOB_ID")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars="$(make_export_list "${base_exports[@]}" "SKIP_OCR=1")"
  job_id="$(submit_chain_step "$tail_job_id" "$export_vars" "$JOBS_DIR/eval/m4/washington_handwritten_0shot.sbatch" "$WASHINGTON_PREP_JOB_ID")"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  echo "${chain_label} chain (${model_name}) job IDs: ${chain_job_ids[*]}"
}

target_chain_enabled() {
  local chain_label="$1"
  case ",${TARGET_CHAINS}," in
    *,"${chain_label}",*) return 0 ;;
    *) return 1 ;;
  esac
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

WASHINGTON_PREP_JOB_ID=""
IAM_REP20_PREP_JOB_ID=""
if [ "$SKIP_SHARED_PREP" != "1" ]; then
  echo "Submitting shared M4 OCR prep jobs..."
  WASHINGTON_PREP_EXPORTS="$(make_export_list "REPO_ROOT=$REPO_ROOT" "SKIP_EVAL=1")"
  WASHINGTON_PREP_JOB_ID="$(
    submit_job \
      "" \
      "$WASHINGTON_PREP_EXPORTS" \
      "$JOBS_DIR/eval/m4/washington_handwritten_0shot.sbatch"
  )"
  echo "Washington M4 prep job ID: ${WASHINGTON_PREP_JOB_ID}"

  IAM_REP20_PREP_EXPORTS="$(
    make_export_list \
      "REPO_ROOT=$REPO_ROOT" \
      "DATASET_KEY=IAM_handwritten" \
      "DATA_DIR=$IAM_REP20_DATA_DIR" \
      "DATASET_TAG=$IAM_REP20_TAG" \
      "SEGMENTER=none" \
      "EXISTING_LINES_DIR=$IAM_REP20_LINES_DIR" \
      "SKIP_EVAL=1"
  )"
  IAM_REP20_PREP_JOB_ID="$(
    submit_job \
      "" \
      "$IAM_REP20_PREP_EXPORTS" \
      "$JOBS_DIR/eval/m4/iam_handwritten_0shot.sbatch"
  )"
  echo "IAM rep20 M4 prep job ID: ${IAM_REP20_PREP_JOB_ID}"
else
  echo "Skipping shared M4 OCR prep jobs because SKIP_SHARED_PREP=1."
fi

OPENAI_CHAIN_MODEL="$(resolve_openai_model "$OPENAI_MODEL_NAME" "$OPENAI_MODEL_FALLBACK")"

if target_chain_enabled "hf"; then
  submit_handwritten_chain "hf" "$HF_MODEL_NAME" "$HF_MODEL_LOCAL"
fi
if target_chain_enabled "gemini"; then
  submit_handwritten_chain "gemini" "$GEMINI_MODEL_NAME" ""
fi
if target_chain_enabled "openai"; then
  submit_handwritten_chain "openai" "$OPENAI_CHAIN_MODEL" ""
fi
if target_chain_enabled "mistral"; then
  submit_handwritten_chain "mistral" "$MISTRAL_MODEL_NAME" ""
fi

echo ""
echo "=== Submission Complete ==="
echo "Shared prep jobs: ${WASHINGTON_PREP_JOB_ID} ${IAM_REP20_PREP_JOB_ID}"
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
