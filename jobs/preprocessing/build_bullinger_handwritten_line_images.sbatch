#!/bin/bash
#SBATCH --job-name=build_bullinger_lines
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
set -x

echo "=== Node ==="
hostname
echo "============"

[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

BUILD_ARGS=(
  python scripts/build_bullinger_handwritten_line_images.py
  --data-dir datasets/bullinger_handwritten
  --out-dir datasets/bullinger_handwritten/line_images
  --log-level INFO
)

if [ "${OVERWRITE:-0}" = "1" ]; then
  BUILD_ARGS+=(--overwrite)
fi

"${BUILD_ARGS[@]}"

echo "Done."
