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

  conda_env_exists() {
    local env_name="${1:?Missing conda env name}"
    local prefix=""
    prefix="$(conda_env_prefix "$env_name" 2>/dev/null || true)"
    [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]
  }

  conda_env_prefix() {
    local env_name="${1:?Missing conda env name}"
    local listed_prefix=""
    listed_prefix="$(conda env list | sed 's/\*//g' | awk -v env="$env_name" 'NF && $1 == env {print $NF; exit}')"
    if [ -n "$listed_prefix" ]; then
      printf '%s\n' "$listed_prefix"
      return 0
    fi
    if [ -d "$HOME/.conda/envs/$env_name" ]; then
      printf '%s\n' "$HOME/.conda/envs/$env_name"
      return 0
    fi
    if [ -d "$CONDA_BASE/envs/$env_name" ]; then
      printf '%s\n' "$CONDA_BASE/envs/$env_name"
      return 0
    fi
    return 1
  }

  CONDA_ENV_NAME="${CONDA_ENV_NAME:-bullinger-mwe}"
  ENV_CREATED=0
  if ! conda_env_exists "$CONDA_ENV_NAME"; then
    echo "Creating conda environment '$CONDA_ENV_NAME'..."
    "$CONDA_INSTALL_BIN" create -y -n "$CONDA_ENV_NAME" python=3.11 pip
    ENV_CREATED=1
  fi
  CONDA_ENV_PREFIX="$(conda_env_prefix "$CONDA_ENV_NAME")"
  export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
  hash -r

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
  local default_device="cuda"
  case "$MODEL" in
    openai/*|gemini/*|mistral/*)
      default_device="auto"
      ;;
  esac
  export DEVICE="${DEVICE:-$default_device}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1600}"

  which python
  if [ "$default_device" = "auto" ]; then
    echo "Skipping CUDA validation for API model '$MODEL'."
  else
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
  fi
  if [ "$DEVICE" = "cuda" ] && ! python - <<'PY'
import torch

raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    python - <<'PY'
import torch

print("ERROR: DEVICE=cuda but torch.cuda.is_available() is False")
print("torch version:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
PY
    exit 1
  fi
  if [ "$DEVICE" = "cuda" ]; then
    if ! python - <<'PY'
import torch

try:
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    probe = torch.tensor([0, 1], device="cuda")
    torch.isin(probe, probe).all().item()
except Exception as exc:  # pragma: no cover - shell bootstrap probe
    print("ERROR: CUDA kernel probe failed:", exc)
    raise SystemExit(1)

print("CUDA device:", device_name)
print("CUDA capability:", capability)
PY
    then
      exit 1
    fi
  fi
}


common_eval_handle_exit() {
  local python_exit="${1:?Missing Python exit code}"

  if [ "$python_exit" -eq 75 ]; then
    echo "Daily quota exhausted. Attempting to resubmit job for tomorrow..."
    if [ -z "${RESUBMIT_SCRIPT:-}" ]; then
      echo "Auto-resubmit skipped because RESUBMIT_SCRIPT is not set."
      exit 75
    fi
    local resubmit_args=(sbatch --begin=now+24hours --export=ALL)
    if [ -n "${SBATCH_EXCLUDE:-}" ]; then
      resubmit_args+=(--exclude="$SBATCH_EXCLUDE")
    fi
    resubmit_args+=("$RESUBMIT_SCRIPT")
    if "${resubmit_args[@]}"; then
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
