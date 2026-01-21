#!/bin/bash
#SBATCH --job-name=children_handwritten_m2_1shot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

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

# Prefer local snapshot if present, else hub
MODEL="./.hf/Qwen3-VL-8B-Instruct"
[ -d "$MODEL" ] || MODEL="Qwen/Qwen3-VL-8B-Instruct"

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python run_eval_m2.py \
  --data-dir datasets/children_handwritten \
  --hf-model "$MODEL" \
  --hf-device cuda \
  --n-shots 1 \
  --shots-seed 42 \
  --out-dir children_handwritten_predictions_m2_1shot \
  --eval-csv children_handwritten_eval_m2_1shot.csv \
  --max-new-tokens 800

echo "Done."
