#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_eval_env.sh"
common_eval_bootstrap "run_eval_m4.py"

DATASET_KEY="${DATASET_KEY:?Must set DATASET_KEY to a supported dataset name.}"
DATA_DIR="${DATA_DIR:?Must set DATA_DIR to a dataset directory.}"
DATASET_TAG="${DATASET_TAG:-$(basename "$DATA_DIR")}"
SEGMENTER="${SEGMENTER:-kraken}"
EXISTING_LINES_DIR="${EXISTING_LINES_DIR:-}"
TRANSCRIPTION_DIR="${TRANSCRIPTION_DIR:-}"
OCR_LINES_DIR="${OCR_LINES_DIR:-$DATA_DIR/ocr_lines}"
OCR_CACHE_DIR="${OCR_CACHE_DIR:-outputs/cache/$DATASET_TAG/lines}"
OCR_BATCH_SIZE="${OCR_BATCH_SIZE:-8}"
SKIP_OCR="${SKIP_OCR:-0}"
FORCE_OCR="${FORCE_OCR:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
IDS="${IDS:-}"
N_SHOTS="${N_SHOTS:-0}"
SHOTS_SEED="${SHOTS_SEED:-42}"

SHOT_SUFFIX="${N_SHOTS}shot"
OUT_DIR="${OUT_DIR:-${DATASET_TAG}_predictions_m4_${SHOT_SUFFIX}_${MODEL_SUFFIX}}"
EVAL_CSV="${EVAL_CSV:-${DATASET_TAG}_eval_m4_${SHOT_SUFFIX}_${MODEL_SUFFIX}.csv}"

if [ "$SKIP_OCR" != "1" ]; then
  PYLAIA_ROOT="${PYLAIA_ROOT:?Must set PYLAIA_ROOT to the PyLaia asset directory.}"
  PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/weights.ckpt}"
  PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
  PYLAIA_WORK_DIR="${PYLAIA_WORK_DIR:-outputs/cache/$DATASET_TAG/pylaia}"
  PYLAIA_GPUS="${PYLAIA_GPUS:-0}"
  PYLAIA_AUTO_SELECT_GPUS="${PYLAIA_AUTO_SELECT_GPUS:-0}"
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
    echo "Installing PyLaia into '$PYLAIA_CONDA_ENV' from PyPI package '$PYLAIA_PIP_SPEC'..."
    "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "pip<24.1" setuptools wheel
    "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "$PYLAIA_PIP_SPEC"
  fi

  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<PY
from pathlib import Path
import shutil
import sysconfig
import tarfile
import tempfile
import urllib.request

source_url = r"$PYLAIA_SOURCE_URL"
purelib = Path(sysconfig.get_paths()["purelib"])
laia_dir = purelib / "laia"
scripts_dir = laia_dir / "scripts"
models_dir = laia_dir / "models"
htr_dir = scripts_dir / "htr"
needs_patch = not (htr_dir / "netout.py").exists() or not (models_dir / "htr").exists()

if needs_patch:
    print("PyLaia source tree incomplete; restoring laia/ from source tarball...")
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

  if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
import laia
import laia.models.htr
import laia.scripts.htr.netout

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
    echo "ERROR: pylaia-htr-netout not found even after preparing '$PYLAIA_CONDA_ENV'."
    exit 1
  fi

  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
import torch

print("PyLaia CUDA available:", torch.cuda.is_available())
print("PyLaia CUDA device count:", torch.cuda.device_count())
PY

  if [ "$SEGMENTER" = "kraken" ] && ! python -c "import kraken" 2>/dev/null; then
    echo "Installing kraken into '$CONDA_ENV_NAME'..."
    python -m pip install -q kraken
  fi

  OCR_ARGS=(
    python scripts/make_ocr_outputs.py
    --dataset "$DATASET_KEY"
    --data-dir "$DATA_DIR"
    --segmenter "$SEGMENTER"
    --recognizer pylaia
    --cache-dir "$OCR_CACHE_DIR"
    --batch-size "$OCR_BATCH_SIZE"
    --pylaia-root "$PYLAIA_ROOT"
    --pylaia-checkpoint "$PYLAIA_CHECKPOINT"
    --pylaia-syms "$PYLAIA_SYMS"
    --pylaia-work-dir "$PYLAIA_WORK_DIR"
    --pylaia-gpus "$PYLAIA_GPUS"
    --log-level INFO
  )
  if [ -n "$EXISTING_LINES_DIR" ]; then
    OCR_ARGS+=(--existing-lines-dir "$EXISTING_LINES_DIR")
  fi
  if [ -n "$IDS" ]; then
    OCR_ARGS+=(--ids "$IDS")
  fi
  if [ "$FORCE_OCR" = "1" ]; then
    OCR_ARGS+=(--overwrite)
  fi
  "${OCR_ARGS[@]}"
fi

if [ "$SKIP_EVAL" = "1" ]; then
  echo "Skipping M4 evaluation because SKIP_EVAL=1."
  exit 0
fi

RUN_ARGS=(
  python run_eval_m4.py
  --data-dir "$DATA_DIR"
  --model "$MODEL"
  --device "$DEVICE"
  --n-shots "$N_SHOTS"
  --out-dir "$OUT_DIR"
  --eval-csv "$EVAL_CSV"
  --ocr-lines-dir "$OCR_LINES_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
)
if [ -n "$TRANSCRIPTION_DIR" ]; then
  RUN_ARGS+=(--transcription-dir "$TRANSCRIPTION_DIR")
fi
if [ "$N_SHOTS" -gt 0 ]; then
  RUN_ARGS+=(--shots-seed "$SHOTS_SEED")
fi
if [ -n "$IDS" ]; then
  RUN_ARGS+=(--ids "$IDS")
fi

set +e
"${RUN_ARGS[@]}"
PYTHON_EXIT=$?
set -e

common_eval_handle_exit "$PYTHON_EXIT"
