#!/bin/bash
#SBATCH --job-name=children_nntp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load your conda module or install Miniconda."
  exit 1
fi
eval "$(conda shell.bash hook)"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
if [ ! -f "scripts/run_children_nntp_cv.py" ]; then
  echo "ERROR: could not find scripts/run_children_nntp_cv.py in REPO_ROOT='$REPO_ROOT'"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
CONDA_EXE="$CONDA_BASE/bin/conda"
CONDA_INSTALL_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_INSTALL_BIN="mamba"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-bullinger-mwe}"
ENV_CREATED=0
if ! conda env list | sed 's/*//g' | awk 'NF && $1 !~ /^#/' | grep -Fxq "$CONDA_ENV_NAME"; then
  "$CONDA_INSTALL_BIN" create -y -n "$CONDA_ENV_NAME" python=3.11 pip
  ENV_CREATED=1
fi
conda activate "$CONDA_ENV_NAME"

if [ "$ENV_CREATED" = "1" ]; then
  python -m pip install -q -r requirements.txt
elif ! python - <<'PY'
import importlib
for module_name in ["PIL", "torch", "transformers", "openai", "mistralai"]:
    importlib.import_module(module_name)
from google import genai  # noqa: F401
PY
then
  python -m pip install -q -r requirements.txt
fi

PYLAIA_CONDA_ENV="${PYLAIA_CONDA_ENV:-pylaia-htr}"
PYLAIA_PYTHON_VERSION="${PYLAIA_PYTHON_VERSION:-3.10}"
PYLAIA_PIP_SPEC="${PYLAIA_PIP_SPEC:-pylaia==1.1.2}"
if ! conda env list | sed 's/\*//g' | awk 'NF && $1 !~ /^#/' | grep -Fxq "$PYLAIA_CONDA_ENV"; then
  "$CONDA_INSTALL_BIN" create -y -n "$PYLAIA_CONDA_ENV" python="$PYLAIA_PYTHON_VERSION" pip
fi
if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
version("pylaia")
PY
then
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "pip<24.1" "setuptools<81" wheel
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
"$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install -q "setuptools<81" wheel
if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
import pkg_resources
import laia
import laia.models.htr
import laia.scripts.htr.netout

print("setuptools pkg_resources OK:", pkg_resources.__file__)
print("PyLaia package version:", version("pylaia"))
print("PyLaia import OK:", laia.__file__)
print("PyLaia models import OK:", laia.models.htr.__file__)
print("PyLaia netout import OK:", laia.scripts.htr.netout.__file__)
PY
then
  echo "Reinstalling PyLaia bootstrap dependencies in '$PYLAIA_CONDA_ENV'..."
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install --force-reinstall "pip<24.1" "setuptools<81" wheel
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install --force-reinstall "$PYLAIA_PIP_SPEC"
fi
if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
import pkg_resources
import laia
import laia.models.htr
import laia.scripts.htr.netout

print("setuptools pkg_resources OK:", pkg_resources.__file__)
print("PyLaia package version:", version("pylaia"))
print("PyLaia import OK:", laia.__file__)
print("PyLaia models import OK:", laia.models.htr.__file__)
print("PyLaia netout import OK:", laia.scripts.htr.netout.__file__)
PY
then
  echo "ERROR: PyLaia import validation failed in '$PYLAIA_CONDA_ENV'."
  exit 1
fi
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
if ! command -v java >/dev/null 2>&1 || ! command -v javac >/dev/null 2>&1; then
  "$CONDA_INSTALL_BIN" install -y -c conda-forge openjdk
  hash -r
fi

DATA_DIR="${DATA_DIR:-datasets/children_handwritten}"
MANIFEST_DIR="${MANIFEST_DIR:-outputs/manifests/children_handwritten_pylaia_cv}"
ASSETS_ROOT="${ASSETS_ROOT:-outputs/pylaia/children_handwritten}"
WORK_ROOT="${WORK_ROOT:-outputs/nntp/children_handwritten}"
PRED_STEM="${PRED_STEM:-children_handwritten_predictions_nntp}"
EVAL_STEM="${EVAL_STEM:-children_handwritten_eval_nntp}"
SUMMARY_CSV="${SUMMARY_CSV:-children_handwritten_eval_nntp_cv.csv}"
FOLD="${FOLD:-all}"
STOP_AFTER="${STOP_AFTER:-evaluate}"
OVERWRITE="${OVERWRITE:-0}"
PYLAIA_GPUS="${PYLAIA_GPUS:-0}"
PYLAIA_AUTO_SELECT_GPUS="${PYLAIA_AUTO_SELECT_GPUS:-0}"

RUN_ARGS=(
  python scripts/run_children_nntp_cv.py
  --data-dir "$DATA_DIR"
  --manifest-dir "$MANIFEST_DIR"
  --assets-root "$ASSETS_ROOT"
  --work-root "$WORK_ROOT"
  --pred-stem "$PRED_STEM"
  --eval-stem "$EVAL_STEM"
  --summary-csv "$SUMMARY_CSV"
  --fold "$FOLD"
  --stop-after "$STOP_AFTER"
  --pylaia-gpus "$PYLAIA_GPUS"
)
if [ "$OVERWRITE" = "1" ]; then
  RUN_ARGS+=(--overwrite)
fi
if [ "$PYLAIA_AUTO_SELECT_GPUS" = "1" ]; then
  RUN_ARGS+=(--pylaia-auto-select-gpus)
fi
"${RUN_ARGS[@]}"

echo "Done."
