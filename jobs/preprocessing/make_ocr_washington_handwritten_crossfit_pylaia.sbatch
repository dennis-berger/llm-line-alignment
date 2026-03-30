#!/bin/bash
#SBATCH --job-name=ocr_washington_crossfit_pylaia
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO_ROOT/jobs/eval/common_eval_env.sh"
common_eval_bootstrap "scripts/run_washington_crossfit_ocr.py"

# PyLaia uses an older Torch stack that does not understand expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF

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

PYLAIA_CONDA_ENV="${PYLAIA_CONDA_ENV:-pylaia-htr}"
PYLAIA_PYTHON_VERSION="${PYLAIA_PYTHON_VERSION:-3.10}"
PYLAIA_PIP_SPEC="${PYLAIA_PIP_SPEC:-pylaia==1.1.2}"
PYLAIA_SOURCE_URL="${PYLAIA_SOURCE_URL:-https://github.com/jpuigcerver/PyLaia/archive/refs/heads/master.tar.gz}"
PYLAIA_CONDA_PREFIX="$(conda_env_prefix "$PYLAIA_CONDA_ENV" 2>/dev/null || printf '%s\n' "$HOME/.conda/envs/$PYLAIA_CONDA_ENV")"
if ! conda_env_exists "$PYLAIA_CONDA_ENV"; then
  echo "Creating dedicated PyLaia environment '$PYLAIA_CONDA_ENV' with Python $PYLAIA_PYTHON_VERSION..."
  "$CONDA_INSTALL_BIN" create -y -p "$PYLAIA_CONDA_PREFIX" python="$PYLAIA_PYTHON_VERSION" pip
fi
PYLAIA_PYTHON="$PYLAIA_CONDA_PREFIX/bin/python"

if ! "$PYLAIA_PYTHON" - <<'PY'
from importlib.metadata import version
version("pylaia")
PY
then
  "$PYLAIA_PYTHON" -m pip install "pip<24.1" setuptools wheel
  "$PYLAIA_PYTHON" -m pip install "$PYLAIA_PIP_SPEC"
fi
"$PYLAIA_PYTHON" - <<'PY'
from pathlib import Path
import shutil
import sysconfig
import tarfile
import tempfile
import urllib.request

source_url = "https://github.com/jpuigcerver/PyLaia/archive/refs/heads/master.tar.gz"
purelib = Path(sysconfig.get_paths()["purelib"])
laia_dir = purelib / "laia"
scripts_dir = laia_dir / "scripts"
models_dir = laia_dir / "models"
htr_dir = scripts_dir / "htr"
needs_patch = not (htr_dir / "netout.py").exists() or not (models_dir / "htr").exists()

if needs_patch:
    tar_path = Path(tempfile.gettempdir()) / "pylaia-source.tar.gz"
    urllib.request.urlretrieve(source_url, tar_path)
    extract_root = Path(tempfile.mkdtemp(prefix="pylaia-src-"))
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_root)
    src_laia = next(extract_root.glob("PyLaia-*/laia"))
    shutil.copytree(src_laia, laia_dir, dirs_exist_ok=True)

for pkg_dir in (scripts_dir, models_dir):
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_path = pkg_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Patched by llm-line-alignment to make PyLaia subpackages importable."""\n',
            encoding="utf-8",
        )
PY
PYLAIA_ENV_BIN="$("$PYLAIA_PYTHON" - <<'PY'
from pathlib import Path
import sys
print(Path(sys.executable).resolve().parent)
PY
)"
export PATH="$PYLAIA_ENV_BIN:$PATH"
if ! command -v pylaia-htr-netout >/dev/null 2>&1; then
  echo "ERROR: pylaia-htr-netout not found in '$PYLAIA_CONDA_ENV'."
  exit 1
fi

DATA_DIR="${DATA_DIR:-datasets/washington_handwritten}"
MANIFEST_DIR="${MANIFEST_DIR:-outputs/manifests/washington_handwritten_pylaia_cv}"
ASSETS_ROOT="${ASSETS_ROOT:-outputs/pylaia}"
CACHE_DIR="${CACHE_DIR:-outputs/cache/washington_handwritten/lines}"
PYLAIA_WORK_DIR="${PYLAIA_WORK_DIR:-outputs/cache/washington_handwritten/pylaia}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FOLD="${FOLD:-all}"
FORCE_OCR="${FORCE_OCR:-0}"

RUN_ARGS=(
  python scripts/run_washington_crossfit_ocr.py
  --data-dir "$DATA_DIR"
  --manifest-dir "$MANIFEST_DIR"
  --assets-root "$ASSETS_ROOT"
  --cache-dir "$CACHE_DIR"
  --pylaia-work-dir "$PYLAIA_WORK_DIR"
  --batch-size "$BATCH_SIZE"
  --fold "$FOLD"
)
if [ "$FORCE_OCR" = "1" ]; then
  RUN_ARGS+=(--overwrite)
fi
"${RUN_ARGS[@]}"

echo "Done."
