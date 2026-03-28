#!/bin/bash
#SBATCH --job-name=ocr_children_crossfit_pylaia
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../eval/common_eval_env.sh"
common_eval_bootstrap "scripts/run_children_crossfit_ocr.py"

PYLAIA_CONDA_ENV="${PYLAIA_CONDA_ENV:-pylaia-htr}"
PYLAIA_PYTHON_VERSION="${PYLAIA_PYTHON_VERSION:-3.10}"
PYLAIA_PIP_SPEC="${PYLAIA_PIP_SPEC:-pylaia==1.1.2}"
PYLAIA_SOURCE_URL="${PYLAIA_SOURCE_URL:-https://github.com/jpuigcerver/PyLaia/archive/refs/heads/master.tar.gz}"
if ! conda env list | sed 's/\*//g' | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$PYLAIA_CONDA_ENV"; then
  echo "Creating dedicated PyLaia environment '$PYLAIA_CONDA_ENV' with Python $PYLAIA_PYTHON_VERSION..."
  "$CONDA_INSTALL_BIN" create -y -n "$PYLAIA_CONDA_ENV" python="$PYLAIA_PYTHON_VERSION" pip
fi
if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
version("pylaia")
PY
then
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "pip<24.1" setuptools wheel
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "$PYLAIA_PIP_SPEC"
fi
"$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
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
PYLAIA_ENV_BIN="$("$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
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

DATA_DIR="${DATA_DIR:-datasets/children_handwritten}"
MANIFEST_DIR="${MANIFEST_DIR:-outputs/manifests/children_handwritten_pylaia_cv}"
ASSETS_ROOT="${ASSETS_ROOT:-outputs/pylaia/children_handwritten}"
CACHE_DIR="${CACHE_DIR:-outputs/cache/children_handwritten/lines}"
PYLAIA_WORK_DIR="${PYLAIA_WORK_DIR:-outputs/cache/children_handwritten/pylaia}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FOLD="${FOLD:-all}"
FORCE_OCR="${FORCE_OCR:-0}"

RUN_ARGS=(
  python scripts/run_children_crossfit_ocr.py
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
