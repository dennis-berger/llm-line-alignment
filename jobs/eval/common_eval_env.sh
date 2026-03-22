#!/bin/bash

common_eval_bootstrap() {
  local entrypoint="${1:?Missing eval entrypoint}"

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

  REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
  export REPO_ROOT
  cd "$REPO_ROOT"
  if [ ! -f "$entrypoint" ]; then
    echo "ERROR: could not find $entrypoint in REPO_ROOT='$REPO_ROOT'"
    echo "Submit the job from the repository root or pass REPO_ROOT=/absolute/path/to/llm-line-alignment"
    exit 1
  fi
  mkdir -p logs

  CONDA_BASE="$(conda info --base)"
  CONDA_EXE="$CONDA_BASE/bin/conda"
  CONDA_INSTALL_BIN="conda"
  if command -v mamba >/dev/null 2>&1; then
    CONDA_INSTALL_BIN="mamba"
  fi
  export CONDA_BASE CONDA_EXE CONDA_INSTALL_BIN

  CONDA_ENV_NAME="${CONDA_ENV_NAME:-bullinger-mwe}"
  ENV_CREATED=0
  if ! conda env list | sed 's/\*//g' | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "Creating conda environment '$CONDA_ENV_NAME'..."
    "$CONDA_INSTALL_BIN" create -y -n "$CONDA_ENV_NAME" python=3.11 pip
    ENV_CREATED=1
  fi
  conda activate "$CONDA_ENV_NAME"

  if [ "$ENV_CREATED" = "1" ]; then
    echo "Installing repo Python requirements into '$CONDA_ENV_NAME'..."
    python -m pip install -q -r requirements.txt
  elif ! python - <<'PY'
import importlib

REQUIRED_MODULES = [
    "PIL",
    "torch",
    "torchvision",
    "transformers",
    "openai",
    "mistralai",
]

for module_name in REQUIRED_MODULES:
    importlib.import_module(module_name)

from google import genai  # noqa: F401
PY
  then
    echo "Installing repo Python requirements into '$CONDA_ENV_NAME' because one or more packages are missing..."
    python -m pip install -q -r requirements.txt
  fi

  python -m pip install -q "openai>=1.0.0" "google-genai>=0.3.0" "mistralai>=1.0.0"

  export HF_HOME="${HF_HOME:-$PWD/.hf}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  mkdir -p "$TRANSFORMERS_CACHE"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  if [ -n "${MODEL:-}" ]; then
    :
  else
    MODEL="${MODEL_LOCAL:-./.hf/Qwen3-VL-8B-Instruct}"
    [ -d "$MODEL" ] || MODEL="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
  fi
  export MODEL
  export MODEL_SUFFIX="${MODEL_SUFFIX:-qwen3-vl-8b-instruct}"
  export DEVICE="${DEVICE:-cuda}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-800}"

  which python
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
}


common_eval_handle_exit() {
  local python_exit="${1:?Missing Python exit code}"

  if [ "$python_exit" -eq 75 ]; then
    echo "Daily quota exhausted. Attempting to resubmit job for tomorrow..."
    if [ -z "${RESUBMIT_SCRIPT:-}" ]; then
      echo "Auto-resubmit skipped because RESUBMIT_SCRIPT is not set."
      exit 75
    fi
    if sbatch --begin=now+24hours "$RESUBMIT_SCRIPT"; then
      echo "Job resubmitted successfully. Will resume tomorrow."
    else
      echo "Auto-resubmit failed. To resume manually, run:"
      echo "  sbatch $RESUBMIT_SCRIPT"
    fi
    exit 75
  fi

  if [ "$python_exit" -ne 0 ]; then
    echo "Python script failed with exit code $python_exit"
    exit "$python_exit"
  fi

  echo "Done."
}
