#!/bin/bash
#SBATCH --job-name=train_trocr_cvl_faust
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
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

mkdir -p logs

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

# Train TrOCR on CVL Faust line data
python scripts/train_trocr_from_tsv.py \
  --train-tsv datasets/cvl_faust/train.tsv \
  --val-tsv datasets/cvl_faust/val.tsv \
  --model-id microsoft/trocr-base-handwritten \
  --output-dir outputs/models/trocr-cvl-faust \
  --epochs 8 \
  --batch-size 8 \
  --grad-accum 1 \
  --lr 5e-5 \
  --warmup-steps 500 \
  --max-target-length 128 \
  --max-new-tokens 128 \
  --fp16

echo "Done."
