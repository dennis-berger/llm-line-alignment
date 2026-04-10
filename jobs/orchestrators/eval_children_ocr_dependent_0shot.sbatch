#!/bin/bash
#SBATCH --job-name=children_ocr_dependent_0shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting children OCR-dependent 0-shot rerun ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
DATA_DIR="${DATA_DIR:-datasets/children_handwritten}"
DATASET_TAG="${DATASET_TAG:-children_handwritten}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_ROOT="${RUN_ROOT:-cluster_runs/thesis_handwritten_completion_${RUN_DATE}}"

HF_MODEL_NAME="${HF_MODEL_NAME:-hf/Qwen/Qwen3-VL-8B-Instruct}"
HF_MODEL_LOCAL="${HF_MODEL_LOCAL:-./.hf/Qwen3-VL-8B-Instruct}"
OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-openai/gpt-5.4-2026-03-05}"
OPENAI_MODEL_FALLBACK="${OPENAI_MODEL_FALLBACK:-openai/gpt-5.4}"
MISTRAL_MODEL_NAME="${MISTRAL_MODEL_NAME:-mistral/mistral-large-2512}"
TARGET_CHAINS="${TARGET_CHAINS:-hf,openai,mistral}"
TARGET_METHODS="${TARGET_METHODS:-m2,m3,m4}"

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

target_method_enabled() {
  local method="$1"
  case ",${TARGET_METHODS}," in
    *,"${method}",*) return 0 ;;
    *) return 1 ;;
  esac
}

chain_has_credentials() {
  local chain_label="$1"
  case "$chain_label" in
    hf)
      return 0
      ;;
    openai)
      [ -n "${OPENAI_API_KEY:-}" ]
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

verify_children_ocr_provenance() {
  DATA_DIR="$DATA_DIR" python - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
meta_paths = sorted((data_dir / "ocr").glob("*.meta.json"))
if len(meta_paths) < 63:
    print(
        f"ERROR: Expected at least 63 OCR provenance files under {data_dir / 'ocr'}, found {len(meta_paths)}.",
        file=sys.stderr,
    )
    sys.exit(1)

bad_paths: list[str] = []
for meta_path in meta_paths:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload.get("segmenter") != "none":
        bad_paths.append(str(meta_path))
        continue
    if payload.get("recognizer") != "pylaia":
        bad_paths.append(str(meta_path))
        continue
    timestamp = str(payload.get("created_at") or payload.get("timestamp_utc") or "")
    if timestamp < "2026-04-03":
        bad_paths.append(str(meta_path))
        continue
    recognizer_model = str(payload.get("recognizer_model", ""))
    if "outputs/pylaia/children_handwritten/fold_" not in recognizer_model:
        bad_paths.append(str(meta_path))
        continue

if bad_paths:
    print("ERROR: Children OCR provenance does not look like the fixed April 3, 2026 cross-fit OCR.", file=sys.stderr)
    for path in bad_paths[:5]:
        print(f"  {path}", file=sys.stderr)
    sys.exit(1)
PY
}

children_artifact_root() {
  local method="$1"
  local model_suffix="$2"
  printf '%s\n' "$RUN_ROOT/$DATASET_TAG/$method/$model_suffix"
}

LAST_CHAIN_TAIL_JOB_ID=""

submit_children_chain() {
  local chain_label="$1"
  local model_name="$2"
  local model_local="$3"
  local run_m4="$4"

  local model_suffix
  local chain_env_name
  local chain_pkgs_dir
  local chain_pylaia_env
  local tail_job_id=""
  local job_id=""
  local export_vars=()
  declare -a chain_job_ids=()
  declare -a base_exports=()

  model_suffix="$(to_model_suffix "$model_name")"
  chain_env_name="${CONDA_ENV_NAME_BASE}-${chain_label}-children-rerun"
  chain_pkgs_dir="${CONDA_PKGS_DIRS_BASE}-${chain_label}-children-rerun"
  chain_pylaia_env="${CONDA_ENV_NAME_BASE}-${chain_label}-children-rerun-pylaia"

  base_exports=(
    "REPO_ROOT=$REPO_ROOT"
    "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_children_ocr_dependent_0shot.sbatch"
    "MODEL_NAME=$model_name"
    "MODEL_LOCAL=$model_local"
    "MODEL_SUFFIX=$model_suffix"
    "DATA_DIR=$DATA_DIR"
    "DATASET_TAG=$DATASET_TAG"
    "N_SHOTS=0"
    "SHOTS_SEED=42"
    "TARGET_CHAINS=$chain_label"
    "TARGET_METHODS=$TARGET_METHODS"
    "OCR_DIR=$DATA_DIR/ocr"
    "OCR_LINES_DIR=$DATA_DIR/ocr_lines"
    "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE"
    "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE"
    "CONDA_ENV_NAME=$chain_env_name"
    "CONDA_PKGS_DIRS=$chain_pkgs_dir"
    "PYLAIA_CONDA_ENV=$chain_pylaia_env"
  )

  if target_method_enabled "m2"; then
    export_vars=(
      "${base_exports[@]}"
      "OUT_DIR=$(children_artifact_root m2 "$model_suffix")/predictions"
      "EVAL_CSV=$(children_artifact_root m2 "$model_suffix")/evaluation.csv"
      "CHECKPOINT_DIR=$(children_artifact_root m2 "$model_suffix")/checkpoints"
    )
    job_id="$(
      submit_job \
        "$(afterok_dependency "$tail_job_id")" \
        "$(make_export_list "${export_vars[@]}")" \
        "$JOBS_DIR/eval/m2/children_handwritten_0shot.sbatch"
    )"
    chain_job_ids+=("$job_id")
    tail_job_id="$job_id"
  fi

  if target_method_enabled "m3"; then
    export_vars=(
      "${base_exports[@]}"
      "OUT_DIR=$(children_artifact_root m3 "$model_suffix")/predictions"
      "EVAL_CSV=$(children_artifact_root m3 "$model_suffix")/evaluation.csv"
      "CHECKPOINT_DIR=$(children_artifact_root m3 "$model_suffix")/checkpoints"
    )
    job_id="$(
      submit_job \
        "$(afterok_dependency "$tail_job_id")" \
        "$(make_export_list "${export_vars[@]}")" \
        "$JOBS_DIR/eval/m3/children_handwritten_0shot.sbatch"
    )"
    chain_job_ids+=("$job_id")
    tail_job_id="$job_id"
  fi

  if [ "$run_m4" = "1" ] && target_method_enabled "m4"; then
    export_vars=(
      "${base_exports[@]}"
      "DATASET_KEY=children_handwritten"
      "SEGMENTER=none"
      "EXISTING_LINES_DIR=$DATA_DIR/line_images"
      "SKIP_OCR=1"
      "OUT_DIR=$(children_artifact_root m4 "$model_suffix")/predictions"
      "EVAL_CSV=$(children_artifact_root m4 "$model_suffix")/evaluation.csv"
      "CHECKPOINT_DIR=$(children_artifact_root m4 "$model_suffix")/checkpoints"
    )
    job_id="$(
      submit_job \
        "$(afterok_dependency "$tail_job_id")" \
        "$(make_export_list "${export_vars[@]}")" \
        "$JOBS_DIR/eval/m4/children_handwritten_0shot.sbatch"
    )"
    chain_job_ids+=("$job_id")
    tail_job_id="$job_id"
  fi

  LAST_CHAIN_TAIL_JOB_ID="$tail_job_id"
  echo "${chain_label} children rerun chain (${model_name}) job IDs: ${chain_job_ids[*]}"
}

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "WARNING: OPENAI_API_KEY is not set in the submission environment."
fi
if [ -z "${MISTRAL_API_KEY:-}" ]; then
  echo "WARNING: MISTRAL_API_KEY is not set in the submission environment."
fi

verify_children_ocr_provenance
OPENAI_CHAIN_MODEL="$(resolve_openai_model "$OPENAI_MODEL_NAME" "$OPENAI_MODEL_FALLBACK")"

if target_chain_enabled "hf"; then
  submit_children_chain "hf" "$HF_MODEL_NAME" "$HF_MODEL_LOCAL" "0"
fi
if target_chain_enabled "openai" && chain_has_credentials "openai"; then
  submit_children_chain "openai" "$OPENAI_CHAIN_MODEL" "" "1"
elif target_chain_enabled "openai"; then
  echo "Skipping openai children rerun chain because OPENAI_API_KEY is missing."
fi
if target_chain_enabled "mistral" && chain_has_credentials "mistral"; then
  submit_children_chain "mistral" "$MISTRAL_MODEL_NAME" "" "1"
elif target_chain_enabled "mistral"; then
  echo "Skipping mistral children rerun chain because MISTRAL_API_KEY is missing."
fi

echo ""
echo "Run root: $RUN_ROOT"
echo "=== Submission Complete ==="
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
