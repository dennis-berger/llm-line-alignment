#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_eval_env.sh"
common_eval_bootstrap "run_eval_m1.py"

DATA_DIR="${DATA_DIR:?Must set DATA_DIR to a dataset directory.}"
DATASET_TAG="${DATASET_TAG:-$(basename "$DATA_DIR")}"
N_SHOTS="${N_SHOTS:-0}"
SHOTS_SEED="${SHOTS_SEED:-42}"
OUT_DIR="${OUT_DIR:-${DATASET_TAG}_predictions_m1_${N_SHOTS}shot_${MODEL_SUFFIX}}"
EVAL_CSV="${EVAL_CSV:-${DATASET_TAG}_eval_m1_${N_SHOTS}shot_${MODEL_SUFFIX}.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
TRANSCRIPTION_DIR="${TRANSCRIPTION_DIR:-}"
IDS="${IDS:-}"

RUN_ARGS=(
  python run_eval_m1.py
  --data-dir "$DATA_DIR"
  --model "$MODEL"
  --device "$DEVICE"
  --n-shots "$N_SHOTS"
  --out-dir "$OUT_DIR"
  --eval-csv "$EVAL_CSV"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [ -n "$TRANSCRIPTION_DIR" ]; then
  RUN_ARGS+=(--transcription-dir "$TRANSCRIPTION_DIR")
fi
if [ "$N_SHOTS" -gt 0 ]; then
  RUN_ARGS+=(--shots-seed "$SHOTS_SEED")
fi
if [ -n "$IDS" ]; then
  RUN_ARGS+=(--ids "$IDS")
fi

set +e
"${RUN_ARGS[@]}"
PYTHON_EXIT=$?
set -e

common_eval_handle_exit "$PYTHON_EXIT"
