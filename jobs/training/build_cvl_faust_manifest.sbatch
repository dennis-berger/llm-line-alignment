#!/bin/bash
#SBATCH --job-name=build_cvl_faust_manifest
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
set -x

mkdir -p logs

# Make conda available
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

python scripts/build_cvl_faust_manifest.py \
  --cvl-root cvl-database-1-1 \
  --out-dir datasets/cvl_faust \
  --splits trainset,testset \
  --text-number 6 \
  --val-ratio 0.1 \
  --seed 13

echo "Done."
