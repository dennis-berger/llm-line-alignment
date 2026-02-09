#!/bin/bash
#SBATCH --job-name=ocr_children_handwritten
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -euo pipefail
set -x

echo "=== Node ==="
hostname
nvidia-smi || true
echo "============"

# Make conda available
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate bullinger-mwe

# HF cache in project dir
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Generate OCR for children_handwritten dataset
# Using kraken segmenter to segment handwritten pages into lines
python scripts/make_ocr_outputs.py \
  --dataset children_handwritten \
  --data-dir datasets/children_handwritten \
  --segmenter kraken \
  --recognizer trocr_handwritten \
  --recognizer-model outputs/models/trocr-cvl-faust \
  --device cuda \
  --batch-size 8 \
  --cache-dir outputs/cache/children_handwritten/lines \
  --log-level INFO \
  --overwrite

echo "Done."
