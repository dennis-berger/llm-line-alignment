#!/bin/bash
#SBATCH --job-name=thesis_hw_completion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "=== Submitting thesis handwritten completion plan ==="
date

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p logs

JOBS_DIR="$REPO_ROOT/jobs"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_ROOT="${RUN_ROOT:-cluster_runs/thesis_handwritten_completion_${RUN_DATE}}"
TARGET_TASKS="${TARGET_TASKS:-children_nntp,children_ocr,m5}"

CHILDREN_DATA_DIR="${CHILDREN_DATA_DIR:-datasets/children_handwritten}"
CHILDREN_NNTP_MANIFEST_DIR="${CHILDREN_NNTP_MANIFEST_DIR:-outputs/manifests/children_handwritten_pylaia_cv}"
CHILDREN_NNTP_ASSETS_ROOT="${CHILDREN_NNTP_ASSETS_ROOT:-outputs/pylaia/children_handwritten}"
CHILDREN_NNTP_OVERWRITE="${CHILDREN_NNTP_OVERWRITE:-0}"
CHILDREN_NNTP_FOLD="${CHILDREN_NNTP_FOLD:-all}"
CHILDREN_NNTP_STOP_AFTER="${CHILDREN_NNTP_STOP_AFTER:-evaluate}"

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

target_task_enabled() {
  local task="$1"
  case ",${TARGET_TASKS}," in
    *,"${task}",*) return 0 ;;
    *) return 1 ;;
  esac
}

verify_children_ocr_provenance() {
  DATA_DIR="$CHILDREN_DATA_DIR" python - <<'PY'
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

verify_children_nntp_assets() {
  local fold

  if [ ! -d "$CHILDREN_NNTP_MANIFEST_DIR" ]; then
    echo "ERROR: Missing children NNTP manifest dir: $CHILDREN_NNTP_MANIFEST_DIR" >&2
    exit 1
  fi
  if [ ! -d "$CHILDREN_NNTP_ASSETS_ROOT" ]; then
    echo "ERROR: Missing children NNTP assets root: $CHILDREN_NNTP_ASSETS_ROOT" >&2
    exit 1
  fi

  for fold in fold_a fold_b fold_c; do
    if [ ! -f "$CHILDREN_NNTP_MANIFEST_DIR/$fold/test_ids.txt" ]; then
      echo "ERROR: Missing held-out IDs file for $fold." >&2
      exit 1
    fi
    if [ ! -f "$CHILDREN_NNTP_ASSETS_ROOT/$fold/best.ckpt" ]; then
      echo "ERROR: Missing PyLaia checkpoint for $fold." >&2
      exit 1
    fi
    if [ ! -f "$CHILDREN_NNTP_ASSETS_ROOT/$fold/syms.txt" ]; then
      echo "ERROR: Missing PyLaia symbols file for $fold." >&2
      exit 1
    fi
    if [ ! -f "$CHILDREN_NNTP_ASSETS_ROOT/$fold/model" ]; then
      echo "ERROR: Missing PyLaia model file for $fold." >&2
      exit 1
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

verify_children_ocr_provenance
verify_children_nntp_assets

CHILDREN_NNTP_JOB_ID=""
CHILDREN_OCR_JOB_ID=""
M5_SWEEP_JOB_ID=""

if target_task_enabled "children_nntp"; then
  CHILDREN_NNTP_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_thesis_handwritten_completion.sbatch" \
        "DATA_DIR=$CHILDREN_DATA_DIR" \
        "MANIFEST_DIR=$CHILDREN_NNTP_MANIFEST_DIR" \
        "ASSETS_ROOT=$CHILDREN_NNTP_ASSETS_ROOT" \
        "WORK_ROOT=$RUN_ROOT/children_handwritten/nntp/work" \
        "PRED_STEM=$RUN_ROOT/children_handwritten/nntp/predictions/children_handwritten_predictions_nntp" \
        "EVAL_STEM=$RUN_ROOT/children_handwritten/nntp/eval/children_handwritten_eval_nntp" \
        "SUMMARY_CSV=$RUN_ROOT/children_handwritten/nntp/children_handwritten_eval_nntp_cv.csv" \
        "FOLD=$CHILDREN_NNTP_FOLD" \
        "STOP_AFTER=$CHILDREN_NNTP_STOP_AFTER" \
        "OVERWRITE=$CHILDREN_NNTP_OVERWRITE" \
        "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" \
        "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE" \
        "CONDA_ENV_NAME=${CONDA_ENV_NAME_BASE}-children-nntp-thesis" \
        "CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS_BASE}-children-nntp-thesis" \
        "PYLAIA_CONDA_ENV=${CONDA_ENV_NAME_BASE}-children-nntp-thesis-pylaia")" \
      "$JOBS_DIR/eval/nntp/children_handwritten.sbatch"
  )"
fi

if target_task_enabled "children_ocr"; then
  CHILDREN_OCR_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_thesis_handwritten_completion.sbatch" \
        "DATA_DIR=$CHILDREN_DATA_DIR" \
        "RUN_DATE=$RUN_DATE" \
        "RUN_ROOT=$RUN_ROOT" \
        "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" \
        "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE")" \
      "$JOBS_DIR/orchestrators/eval_children_ocr_dependent_0shot.sbatch"
  )"
fi

if target_task_enabled "m5"; then
  M5_SWEEP_JOB_ID="$(
    submit_job \
      "" \
      "$(make_export_list \
        "REPO_ROOT=$REPO_ROOT" \
        "RESUBMIT_SCRIPT=$REPO_ROOT/jobs/orchestrators/eval_thesis_handwritten_completion.sbatch" \
        "RUN_DATE=$RUN_DATE" \
        "RUN_ROOT=$RUN_ROOT" \
        "CONDA_ENV_NAME_BASE=$CONDA_ENV_NAME_BASE" \
        "CONDA_PKGS_DIRS_BASE=$CONDA_PKGS_DIRS_BASE")" \
      "$JOBS_DIR/orchestrators/eval_handwritten_m5_context100_api_models.sbatch"
  )"
fi

echo ""
echo "Run root: $RUN_ROOT"
echo "Children NNTP job: $CHILDREN_NNTP_JOB_ID"
echo "Children OCR rerun job: $CHILDREN_OCR_JOB_ID"
echo "M5 sweep job: $M5_SWEEP_JOB_ID"
echo "=== Submission Complete ==="
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Done."
