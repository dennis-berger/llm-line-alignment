#!/bin/bash
#SBATCH --job-name=bullinger_m2_1shot
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

# Ensure OpenAI package is available (for API models)
pip install -q openai>=1.0.0

# HF cache in project dir
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# Reduce CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use orchestrator-provided model or default
# MODEL_LOCAL and MODEL_NAME can be set by orchestrator via --export
MODEL="${MODEL_LOCAL:-./.hf/Qwen3-VL-8B-Instruct}"
[ -d "$MODEL" ] || MODEL="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_SUFFIX="${MODEL_SUFFIX:-qwen3-vl-8b-instruct}"

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python run_eval_m2.py \
  --data-dir datasets/bullinger_handwritten \
  --model "$MODEL" \
  --device cuda \
  --n-shots 1 \
  --shots-seed 42 \
  --out-dir bullinger_handwritten_predictions_m2_1shot_${MODEL_SUFFIX} \
  --eval-csv bullinger_handwritten_eval_m2_1shot_${MODEL_SUFFIX}.csv \
  --max-new-tokens 800
