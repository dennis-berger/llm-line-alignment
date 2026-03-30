#!/bin/bash
#SBATCH --job-name=washington_ocr_dependent_0shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting Washington OCR-dependent 0-shot rerun ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
DATASET_TAG="${DATASET_TAG:-washington_handwritten}"

HF_MODEL_NAME="${HF_MODEL_NAME:-hf/Qwen/Qwen3-VL-8B-Instruct}"
HF_MODEL_LOCAL="${HF_MODEL_LOCAL:-./.hf/Qwen3-VL-8B-Instruct}"
GEMINI_MODEL_NAME="${GEMINI_MODEL_NAME:-gemini/gemini-3.1-pro-preview}"
OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-openai/gpt-5.4-2026-03-05}"
OPENAI_MODEL_FALLBACK="${OPENAI_MODEL_FALLBACK:-openai/gpt-5.4}"
MISTRAL_MODEL_NAME="${MISTRAL_MODEL_NAME:-mistral/mistral-large-2512}"
TARGET_CHAINS="${TARGET_CHAINS:-hf,gemini,openai,mistral}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_PROMPT_COMPARE="${RUN_PROMPT_COMPARE:-1}"
PROMPT_COMPARE_VARIANT="${PROMPT_COMPARE_VARIANT:-boundary_anchored_v1}"
RESULT_SUFFIX="${RESULT_SUFFIX:-}"
IDS="${IDS:-}"
SKIP_PREFLIGHT_CLEANUP="${SKIP_PREFLIGHT_CLEANUP:-0}"

CONDA_ENV_NAME_BASE="${CONDA_ENV_NAME_BASE:-${CONDA_ENV_NAME:-bullinger-mwe}}"
CONDA_PKGS_DIRS_BASE="${CONDA_PKGS_DIRS_BASE:-${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs-clean}}"

to_model_suffix() {
  basename "$1" | tr '[:upper:]' '[:lower:]'
}

to_checkpoint_safe() {
  printf '%s\n' "$1" | tr '/:' '__'
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
    echo "OpenAI snapshot ${preferred_model} is unavailable; falling back to ${fallback#openai/}." >&2
    printf '%s\n' "$fallback"
    return 0
  fi

  echo "OpenAI snapshot probe returned HTTP ${status}; keeping ${preferred_model}." >&2
  printf '%s\n' "$preferred"
}

target_chain_enabled() {
  local chain_label="$1"
  case ",${TARGET_CHAINS}," in
    *,"${chain_label}",*) return 0 ;;
    *) return 1 ;;
  esac
}

verify_washington_ocr_artifacts() {
  local ocr_count
  local ocr_lines_count
  ocr_count="$(find "$DATA_DIR/ocr" -maxdepth 1 -type f | wc -l | tr -d ' ')"
  ocr_lines_count="$(find "$DATA_DIR/ocr_lines" -maxdepth 1 -type f | wc -l | tr -d ' ')"

  if [ "$ocr_count" != "40" ] || [ "$ocr_lines_count" != "20" ]; then
    echo "ERROR: Washington OCR artifacts are incomplete."
    echo "Expected ocr=40 and ocr_lines=20, got ocr=${ocr_count} and ocr_lines=${ocr_lines_count}."
    exit 1
  fi
}

list_live_washington_eval_jobs() {
  local current_user="${USER:-$(whoami)}"
  squeue -u "$current_user" -h -o "%A %j" | awk '$2 ~ /^washington_handwritten_m[234]_0shot$/ {print $1 ":" $2}'
}

list_pending_parent_matrix_jobs() {
  local current_user="${USER:-$(whoami)}"
  squeue -u "$current_user" -h -o "%A %j %T" | awk '$2 == "handwritten_matrix_0shot" {print $1 ":" $2 ":" $3}'
}

cleanup_step_artifacts() {
  local method="$1"
  local model_name="$2"
  local out_dir="$3"
  local eval_csv="$4"
  local checkpoint_dir="$5"

  local model_safe
  model_safe="$(to_checkpoint_safe "$model_name")"

  rm -rf "$out_dir"
  rm -f "$eval_csv"
  if [ "$checkpoint_dir" = "checkpoints" ]; then
    rm -f "${checkpoint_dir}/checkpoint_${method}_${DATASET_TAG}_${model_safe}_0shot"*.json
  else
    rm -f "${checkpoint_dir}/checkpoint_${method}_${DATASET_TAG}_"*.json
  fi
}

baseline_out_dir() {
  local method="$1"
  local model_suffix="$2"
  printf '%s\n' "${DATASET_TAG}_predictions_${method}_0shot_${model_suffix}"
}

baseline_eval_csv() {
  local method="$1"
  local model_suffix="$2"
  printf '%s\n' "${DATASET_TAG}_eval_${method}_0shot_${model_suffix}.csv"
}

variant_out_dir() {
  local method="$1"
  local prompt_variant="$2"
  local model_suffix="$3"
  printf '%s\n' "${DATASET_TAG}_predictions_${method}_0shot_prompt_${prompt_variant}_${model_suffix}"
}

variant_eval_csv() {
  local method="$1"
  local prompt_variant="$2"
  local model_suffix="$3"
  printf '%s\n' "${DATASET_TAG}_eval_${method}_0shot_prompt_${prompt_variant}_${model_suffix}.csv"
}

suffix_out_dir() {
  local method="$1"
  local suffix="$2"
  local model_suffix="$3"
  printf '%s\n' "${DATASET_TAG}_predictions_${method}_0shot_${suffix}_${model_suffix}"
}

suffix_eval_csv() {
  local method="$1"
  local suffix="$2"
  local model_suffix="$3"
  printf '%s\n' "${DATASET_TAG}_eval_${method}_0shot_${suffix}_${model_suffix}.csv"
}

prompt_compare_result_suffix() {
  local suffix="$1"
  if [ -n "$suffix" ]; then
    printf '%s\n' "${suffix}_prompt_${PROMPT_COMPARE_VARIANT}"
    return 0
  fi
  printf '%s\n' ""
}

step_checkpoint_dir() {
  local result_suffix="$1"
  local prompt_variant="$2"

  if [ -n "$result_suffix" ]; then
    printf '%s\n' "checkpoints/${DATASET_TAG}_${result_suffix}"
    return 0
  fi
  if [ "$prompt_variant" != "baseline" ]; then
    printf '%s\n' "checkpoints/${DATASET_TAG}_prompt_${prompt_variant}"
    return 0
  fi
  printf '%s\n' "checkpoints"
}

LAST_CHAIN_TAIL_JOB_ID=""

submit_washington_chain() {
  local chain_label="$1"
  local model_name="$2"
  local model_local="$3"

  local model_suffix
  local chain_env_name
  local chain_pkgs_dir
  local chain_pylaia_env
  local tail_job_id=""
  local job_id=""
  local export_vars=""
  local checkpoint_dir="checkpoints"
  declare -a chain_job_ids=()
  declare -a base_exports=()

  model_suffix="$(to_model_suffix "$model_name")"
  chain_env_name="${CONDA_ENV_NAME_BASE}-${chain_label}-washington"
  chain_pkgs_dir="${CONDA_PKGS_DIRS_BASE}-${chain_label}-washington"
  chain_pylaia_env="${CONDA_ENV_NAME_BASE}-${chain_label}-washington-pylaia"

  base_exports=(
    "REPO_ROOT=$REPO_ROOT"
    "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_washington_ocr_dependent_0shot.sbatch"
    "MODEL_NAME=$model_name"
    "MODEL_LOCAL=$model_local"
    "MODEL_SUFFIX=$model_suffix"
    "DATA_DIR=$DATA_DIR"
    "DATASET_TAG=$DATASET_TAG"
    "N_SHOTS=0"
    "SHOTS_SEED=42"
    "TARGET_CHAINS=$chain_label"
    "RUN_BASELINE=1"
    "RUN_PROMPT_COMPARE=0"
    "SKIP_PREFLIGHT_CLEANUP=1"
    "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE"
    "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE"
    "CONDA_ENV_NAME=$chain_env_name"
    "CONDA_PKGS_DIRS=$chain_pkgs_dir"
    "PYLAIA_CONDA_ENV=$chain_pylaia_env"
  )
  if [ -n "$IDS" ]; then
    base_exports+=("IDS=$IDS")
  fi
  if [ -n "$RESULT_SUFFIX" ]; then
    checkpoint_dir="$(step_checkpoint_dir "$RESULT_SUFFIX" "baseline")"
  fi

  export_vars=("${base_exports[@]}" "CHECKPOINT_DIR=$checkpoint_dir")
  if [ -n "$RESULT_SUFFIX" ]; then
    export_vars+=(
      "OUT_DIR=$(suffix_out_dir m2 "$RESULT_SUFFIX" "$model_suffix")"
      "EVAL_CSV=$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$model_suffix")"
    )
  fi
  job_id="$(
    submit_job \
      "$(afterok_dependency "$tail_job_id")" \
      "$(make_export_list "${export_vars[@]}")" \
      "$JOBS_DIR/eval/m2/washington_handwritten_0shot.sbatch"
  )"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars=("${base_exports[@]}" "CHECKPOINT_DIR=$checkpoint_dir")
  if [ -n "$RESULT_SUFFIX" ]; then
    export_vars+=(
      "OUT_DIR=$(suffix_out_dir m3 "$RESULT_SUFFIX" "$model_suffix")"
      "EVAL_CSV=$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$model_suffix")"
    )
  fi
  job_id="$(
    submit_job \
      "$(afterok_dependency "$tail_job_id")" \
      "$(make_export_list "${export_vars[@]}")" \
      "$JOBS_DIR/eval/m3/washington_handwritten_0shot.sbatch"
  )"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  export_vars=("${base_exports[@]}" "CHECKPOINT_DIR=$checkpoint_dir" "SKIP_OCR=1")
  if [ -n "$RESULT_SUFFIX" ]; then
    export_vars+=(
      "OUT_DIR=$(suffix_out_dir m4 "$RESULT_SUFFIX" "$model_suffix")"
      "EVAL_CSV=$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$model_suffix")"
    )
  fi
  job_id="$(
    submit_job \
      "$(afterok_dependency "$tail_job_id")" \
      "$(make_export_list "${export_vars[@]}")" \
      "$JOBS_DIR/eval/m4/washington_handwritten_0shot.sbatch"
  )"
  chain_job_ids+=("$job_id")
  tail_job_id="$job_id"

  LAST_CHAIN_TAIL_JOB_ID="$tail_job_id"
  echo "${chain_label} Washington chain (${model_name}) job IDs: ${chain_job_ids[*]}"
}

submit_openai_prompt_compare() {
  local dependency_job_id="$1"
  local model_name="$2"
  local model_suffix
  local checkpoint_dir
  local compare_env_name
  local compare_pkgs_dir
  local compare_pylaia_env
  local compare_result_suffix
  local export_vars=()
  local dependency=""
  local job_id=""

  model_suffix="$(to_model_suffix "$model_name")"
  compare_result_suffix="$(prompt_compare_result_suffix "$RESULT_SUFFIX")"
  checkpoint_dir="$(step_checkpoint_dir "$compare_result_suffix" "$PROMPT_COMPARE_VARIANT")"
  compare_env_name="${CONDA_ENV_NAME_BASE}-openai-washington-prompt"
  compare_pkgs_dir="${CONDA_PKGS_DIRS_BASE}-openai-washington-prompt"
  compare_pylaia_env="${CONDA_ENV_NAME_BASE}-openai-washington-prompt-pylaia"

  export_vars=(
    "REPO_ROOT=$REPO_ROOT"
    "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_washington_ocr_dependent_0shot.sbatch"
    "MODEL_NAME=$model_name"
    "MODEL_SUFFIX=$model_suffix"
    "DATA_DIR=$DATA_DIR"
    "DATASET_TAG=$DATASET_TAG"
    "N_SHOTS=0"
    "SHOTS_SEED=42"
    "TARGET_CHAINS=openai"
    "RUN_BASELINE=0"
    "RUN_PROMPT_COMPARE=1"
    "SKIP_PREFLIGHT_CLEANUP=1"
    "PROMPT_COMPARE_VARIANT=$PROMPT_COMPARE_VARIANT"
    "PROMPT_VARIANT=$PROMPT_COMPARE_VARIANT"
    "SKIP_OCR=1"
    "CHECKPOINT_DIR=$checkpoint_dir"
    "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE"
    "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE"
    "CONDA_ENV_NAME=$compare_env_name"
    "CONDA_PKGS_DIRS=$compare_pkgs_dir"
    "PYLAIA_CONDA_ENV=$compare_pylaia_env"
  )
  if [ -n "$IDS" ]; then
    export_vars+=("IDS=$IDS")
  fi
  if [ -n "$compare_result_suffix" ]; then
    export_vars+=(
      "OUT_DIR=$(suffix_out_dir m4 "$compare_result_suffix" "$model_suffix")"
      "EVAL_CSV=$(suffix_eval_csv m4 "$compare_result_suffix" "$model_suffix")"
    )
  else
    export_vars+=(
      "OUT_DIR=$(variant_out_dir m4 "$PROMPT_COMPARE_VARIANT" "$model_suffix")"
      "EVAL_CSV=$(variant_eval_csv m4 "$PROMPT_COMPARE_VARIANT" "$model_suffix")"
    )
  fi

  dependency="$(afterok_dependency "$dependency_job_id")"
  job_id="$(
    submit_job \
      "$dependency" \
      "$(make_export_list "${export_vars[@]}")" \
      "$JOBS_DIR/eval/m4/washington_handwritten_0shot.sbatch"
  )"
  echo "openai Washington prompt-compare job (${model_name}, ${PROMPT_COMPARE_VARIANT}): ${job_id}"
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

verify_washington_ocr_artifacts

if [ "$SKIP_PREFLIGHT_CLEANUP" != "1" ]; then
  live_washington_jobs="$(list_live_washington_eval_jobs || true)"
  if [ -n "$live_washington_jobs" ]; then
    echo "ERROR: Live Washington M2/M3/M4 jobs already exist:"
    printf '%s\n' "$live_washington_jobs"
    echo "Cancel the stale Washington jobs before launching this rerun."
    exit 1
  fi

  parent_matrix_jobs="$(list_pending_parent_matrix_jobs || true)"
  if [ -n "$parent_matrix_jobs" ]; then
    echo "ERROR: Pending parent handwritten matrix jobs already exist:"
    printf '%s\n' "$parent_matrix_jobs"
    echo "Cancel the stale matrix retry jobs before launching this rerun."
    exit 1
  fi
fi

OPENAI_CHAIN_MODEL="$(resolve_openai_model "$OPENAI_MODEL_NAME" "$OPENAI_MODEL_FALLBACK")"

if [ "$SKIP_PREFLIGHT_CLEANUP" != "1" ]; then
  if [ "$RUN_BASELINE" = "1" ]; then
    if target_chain_enabled "hf"; then
      hf_suffix="$(to_model_suffix "$HF_MODEL_NAME")"
      if [ -n "$RESULT_SUFFIX" ]; then
        baseline_checkpoint_dir="$(step_checkpoint_dir "$RESULT_SUFFIX" "baseline")"
        cleanup_step_artifacts m2 "$HF_MODEL_NAME" "$(suffix_out_dir m2 "$RESULT_SUFFIX" "$hf_suffix")" "$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$hf_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m3 "$HF_MODEL_NAME" "$(suffix_out_dir m3 "$RESULT_SUFFIX" "$hf_suffix")" "$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$hf_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m4 "$HF_MODEL_NAME" "$(suffix_out_dir m4 "$RESULT_SUFFIX" "$hf_suffix")" "$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$hf_suffix")" "$baseline_checkpoint_dir"
      else
        cleanup_step_artifacts m2 "$HF_MODEL_NAME" "$(baseline_out_dir m2 "$hf_suffix")" "$(baseline_eval_csv m2 "$hf_suffix")" "checkpoints"
        cleanup_step_artifacts m3 "$HF_MODEL_NAME" "$(baseline_out_dir m3 "$hf_suffix")" "$(baseline_eval_csv m3 "$hf_suffix")" "checkpoints"
        cleanup_step_artifacts m4 "$HF_MODEL_NAME" "$(baseline_out_dir m4 "$hf_suffix")" "$(baseline_eval_csv m4 "$hf_suffix")" "checkpoints"
      fi
    fi
    if target_chain_enabled "gemini"; then
      gemini_suffix="$(to_model_suffix "$GEMINI_MODEL_NAME")"
      if [ -n "$RESULT_SUFFIX" ]; then
        baseline_checkpoint_dir="$(step_checkpoint_dir "$RESULT_SUFFIX" "baseline")"
        cleanup_step_artifacts m2 "$GEMINI_MODEL_NAME" "$(suffix_out_dir m2 "$RESULT_SUFFIX" "$gemini_suffix")" "$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$gemini_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m3 "$GEMINI_MODEL_NAME" "$(suffix_out_dir m3 "$RESULT_SUFFIX" "$gemini_suffix")" "$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$gemini_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m4 "$GEMINI_MODEL_NAME" "$(suffix_out_dir m4 "$RESULT_SUFFIX" "$gemini_suffix")" "$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$gemini_suffix")" "$baseline_checkpoint_dir"
      else
        cleanup_step_artifacts m2 "$GEMINI_MODEL_NAME" "$(baseline_out_dir m2 "$gemini_suffix")" "$(baseline_eval_csv m2 "$gemini_suffix")" "checkpoints"
        cleanup_step_artifacts m3 "$GEMINI_MODEL_NAME" "$(baseline_out_dir m3 "$gemini_suffix")" "$(baseline_eval_csv m3 "$gemini_suffix")" "checkpoints"
        cleanup_step_artifacts m4 "$GEMINI_MODEL_NAME" "$(baseline_out_dir m4 "$gemini_suffix")" "$(baseline_eval_csv m4 "$gemini_suffix")" "checkpoints"
      fi
    fi
    if target_chain_enabled "openai"; then
      openai_name="$OPENAI_MODEL_NAME"
      openai_fallback_name="$OPENAI_MODEL_FALLBACK"
      openai_suffix="$(to_model_suffix "$openai_name")"
      openai_fallback_suffix="$(to_model_suffix "$openai_fallback_name")"
      if [ -n "$RESULT_SUFFIX" ]; then
        baseline_checkpoint_dir="$(step_checkpoint_dir "$RESULT_SUFFIX" "baseline")"
        cleanup_step_artifacts m2 "$openai_name" "$(suffix_out_dir m2 "$RESULT_SUFFIX" "$openai_suffix")" "$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$openai_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m3 "$openai_name" "$(suffix_out_dir m3 "$RESULT_SUFFIX" "$openai_suffix")" "$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$openai_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m4 "$openai_name" "$(suffix_out_dir m4 "$RESULT_SUFFIX" "$openai_suffix")" "$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$openai_suffix")" "$baseline_checkpoint_dir"
        if [ "$openai_fallback_name" != "$openai_name" ]; then
          cleanup_step_artifacts m2 "$openai_fallback_name" "$(suffix_out_dir m2 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$baseline_checkpoint_dir"
          cleanup_step_artifacts m3 "$openai_fallback_name" "$(suffix_out_dir m3 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$baseline_checkpoint_dir"
          cleanup_step_artifacts m4 "$openai_fallback_name" "$(suffix_out_dir m4 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$openai_fallback_suffix")" "$baseline_checkpoint_dir"
        fi
      else
        cleanup_step_artifacts m2 "$openai_name" "$(baseline_out_dir m2 "$openai_suffix")" "$(baseline_eval_csv m2 "$openai_suffix")" "checkpoints"
        cleanup_step_artifacts m3 "$openai_name" "$(baseline_out_dir m3 "$openai_suffix")" "$(baseline_eval_csv m3 "$openai_suffix")" "checkpoints"
        cleanup_step_artifacts m4 "$openai_name" "$(baseline_out_dir m4 "$openai_suffix")" "$(baseline_eval_csv m4 "$openai_suffix")" "checkpoints"
        if [ "$openai_fallback_name" != "$openai_name" ]; then
          cleanup_step_artifacts m2 "$openai_fallback_name" "$(baseline_out_dir m2 "$openai_fallback_suffix")" "$(baseline_eval_csv m2 "$openai_fallback_suffix")" "checkpoints"
          cleanup_step_artifacts m3 "$openai_fallback_name" "$(baseline_out_dir m3 "$openai_fallback_suffix")" "$(baseline_eval_csv m3 "$openai_fallback_suffix")" "checkpoints"
          cleanup_step_artifacts m4 "$openai_fallback_name" "$(baseline_out_dir m4 "$openai_fallback_suffix")" "$(baseline_eval_csv m4 "$openai_fallback_suffix")" "checkpoints"
        fi
      fi
    fi
    if target_chain_enabled "mistral"; then
      mistral_suffix="$(to_model_suffix "$MISTRAL_MODEL_NAME")"
      if [ -n "$RESULT_SUFFIX" ]; then
        baseline_checkpoint_dir="$(step_checkpoint_dir "$RESULT_SUFFIX" "baseline")"
        cleanup_step_artifacts m2 "$MISTRAL_MODEL_NAME" "$(suffix_out_dir m2 "$RESULT_SUFFIX" "$mistral_suffix")" "$(suffix_eval_csv m2 "$RESULT_SUFFIX" "$mistral_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m3 "$MISTRAL_MODEL_NAME" "$(suffix_out_dir m3 "$RESULT_SUFFIX" "$mistral_suffix")" "$(suffix_eval_csv m3 "$RESULT_SUFFIX" "$mistral_suffix")" "$baseline_checkpoint_dir"
        cleanup_step_artifacts m4 "$MISTRAL_MODEL_NAME" "$(suffix_out_dir m4 "$RESULT_SUFFIX" "$mistral_suffix")" "$(suffix_eval_csv m4 "$RESULT_SUFFIX" "$mistral_suffix")" "$baseline_checkpoint_dir"
      else
        cleanup_step_artifacts m2 "$MISTRAL_MODEL_NAME" "$(baseline_out_dir m2 "$mistral_suffix")" "$(baseline_eval_csv m2 "$mistral_suffix")" "checkpoints"
        cleanup_step_artifacts m3 "$MISTRAL_MODEL_NAME" "$(baseline_out_dir m3 "$mistral_suffix")" "$(baseline_eval_csv m3 "$mistral_suffix")" "checkpoints"
        cleanup_step_artifacts m4 "$MISTRAL_MODEL_NAME" "$(baseline_out_dir m4 "$mistral_suffix")" "$(baseline_eval_csv m4 "$mistral_suffix")" "checkpoints"
      fi
    fi
  fi

  if [ "$RUN_PROMPT_COMPARE" = "1" ] && target_chain_enabled "openai"; then
    openai_compare_suffix="$(to_model_suffix "$OPENAI_CHAIN_MODEL")"
    compare_result_suffix="$(prompt_compare_result_suffix "$RESULT_SUFFIX")"
    prompt_checkpoint_dir="$(step_checkpoint_dir "$compare_result_suffix" "$PROMPT_COMPARE_VARIANT")"
    if [ -n "$compare_result_suffix" ]; then
      cleanup_step_artifacts m4 "$OPENAI_CHAIN_MODEL" "$(suffix_out_dir m4 "$compare_result_suffix" "$openai_compare_suffix")" "$(suffix_eval_csv m4 "$compare_result_suffix" "$openai_compare_suffix")" "$prompt_checkpoint_dir"
    else
      cleanup_step_artifacts m4 "$OPENAI_CHAIN_MODEL" "$(variant_out_dir m4 "$PROMPT_COMPARE_VARIANT" "$openai_compare_suffix")" "$(variant_eval_csv m4 "$PROMPT_COMPARE_VARIANT" "$openai_compare_suffix")" "$prompt_checkpoint_dir"
    fi
  fi
fi

OPENAI_BASELINE_TAIL_JOB_ID=""

if [ "$RUN_BASELINE" = "1" ]; then
  if target_chain_enabled "hf"; then
    submit_washington_chain "hf" "$HF_MODEL_NAME" "$HF_MODEL_LOCAL"
  fi
  if target_chain_enabled "gemini"; then
    submit_washington_chain "gemini" "$GEMINI_MODEL_NAME" ""
  fi
  if target_chain_enabled "openai"; then
    submit_washington_chain "openai" "$OPENAI_CHAIN_MODEL" ""
    OPENAI_BASELINE_TAIL_JOB_ID="$LAST_CHAIN_TAIL_JOB_ID"
  fi
  if target_chain_enabled "mistral"; then
    submit_washington_chain "mistral" "$MISTRAL_MODEL_NAME" ""
  fi
fi

if [ "$RUN_PROMPT_COMPARE" = "1" ] && target_chain_enabled "openai"; then
  submit_openai_prompt_compare "$OPENAI_BASELINE_TAIL_JOB_ID" "$OPENAI_CHAIN_MODEL"
fi

echo ""
echo "=== Submission Complete ==="
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
