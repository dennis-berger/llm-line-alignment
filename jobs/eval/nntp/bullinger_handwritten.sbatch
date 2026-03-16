#!/bin/bash
#SBATCH --job-name=bullinger_nntp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -uo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

if ! command -v pylaia-htr-netout >/dev/null 2>&1; then
  echo "ERROR: pylaia-htr-netout not found on PATH."
  exit 1
fi
if ! command -v java >/dev/null 2>&1 || ! command -v javac >/dev/null 2>&1; then
  echo "ERROR: java and javac must be available on PATH."
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
java -version || true
javac -version || true

DATA_DIR="${DATA_DIR:-datasets/bullinger_handwritten}"
WORK_DIR="${WORK_DIR:-outputs/nntp/bullinger_handwritten}"
PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-dennis}"
PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/epoch=170-lowest_va_cer.ckpt}"
PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
NNTP_ROOT="${NNTP_ROOT:-../nntp}"
PRED_DIR="${PRED_DIR:-bullinger_handwritten_predictions_nntp}"
EVAL_CSV="${EVAL_CSV:-bullinger_handwritten_eval_nntp.csv}"
STOP_AFTER="${STOP_AFTER:-evaluate}"
IDS="${IDS:-}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=(
  --data-dir "$DATA_DIR"
  --work-dir "$WORK_DIR"
  --pylaia-root "$PYLAIA_ROOT"
  --pylaia-checkpoint "$PYLAIA_CHECKPOINT"
  --pylaia-syms "$PYLAIA_SYMS"
  --nntp-root "$NNTP_ROOT"
  --pred-dir "$PRED_DIR"
  --eval-csv "$EVAL_CSV"
  --stop-after "$STOP_AFTER"
)

if [ -n "$IDS" ]; then
  EXTRA_ARGS+=(--ids "$IDS")
fi
if [ "$OVERWRITE" = "1" ]; then
  EXTRA_ARGS+=(--overwrite)
fi

python scripts/run_nntp_eval.py "${EXTRA_ARGS[@]}"

PYTHON_EXIT=$?
if [ "$PYTHON_EXIT" -ne 0 ]; then
  echo "Python script failed with exit code $PYTHON_EXIT"
  exit "$PYTHON_EXIT"
fi

echo "Done."
