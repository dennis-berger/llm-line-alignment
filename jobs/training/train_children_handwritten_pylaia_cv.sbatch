#!/bin/bash
#SBATCH --job-name=children_pylaia_cv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=diufrd205
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail
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
cd "$REPO_ROOT"
if [ ! -f "scripts/build_children_pylaia_manifests.py" ]; then
  echo "ERROR: could not find scripts/build_children_pylaia_manifests.py in REPO_ROOT='$REPO_ROOT'"
  exit 1
fi
mkdir -p logs
CONDA_BASE="$(conda info --base)"
CONDA_EXE="$CONDA_BASE/bin/conda"

CONDA_INSTALL_BIN="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_INSTALL_BIN="mamba"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-bullinger-mwe}"
ENV_CREATED=0
if ! conda env list | sed 's/*//g' | awk 'NF && $1 !~ /^#/' | grep -Fxq "$CONDA_ENV_NAME"; then
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
    "transformers",
    "pytesseract",
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

PYLAIA_CONDA_ENV="${PYLAIA_CONDA_ENV:-pylaia-htr}"
PYLAIA_PYTHON_VERSION="${PYLAIA_PYTHON_VERSION:-3.10}"
PYLAIA_PIP_SPEC="${PYLAIA_PIP_SPEC:-pylaia==1.1.2}"
PYLAIA_SOURCE_URL="${PYLAIA_SOURCE_URL:-https://github.com/jpuigcerver/PyLaia/archive/refs/heads/master.tar.gz}"
if ! conda env list | sed 's/*//g' | awk 'NF && $1 !~ /^#/' | grep -Fxq "$PYLAIA_CONDA_ENV"; then
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

if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" pylaia-htr-train-ctc --help >/dev/null 2>&1; then
  echo "ERROR: pylaia-htr-train-ctc is unavailable in '$PYLAIA_CONDA_ENV'."
  exit 1
fi
if ! "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" pylaia-htr-dataset-validate --help >/dev/null 2>&1; then
  echo "ERROR: pylaia-htr-dataset-validate is unavailable in '$PYLAIA_CONDA_ENV'."
  exit 1
fi

DATA_DIR="${DATA_DIR:-datasets/children_handwritten}"
MANIFEST_OUT_DIR="${MANIFEST_OUT_DIR:-outputs/manifests/children_handwritten_pylaia_cv}"
SYMS_OUT="${SYMS_OUT:-$MANIFEST_OUT_DIR/children.syms.txt}"
TRAIN_FOLD="${TRAIN_FOLD:-fold_a}"
PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-iam}"
PYLAIA_MODEL="${PYLAIA_MODEL:-$PYLAIA_ROOT/model}"
PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/weights.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pylaia/children_handwritten}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$OUTPUT_ROOT/$TRAIN_FOLD}"
TRAIN_OVERWRITE="${TRAIN_OVERWRITE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-40}"
CHECKPOINT_K="${CHECKPOINT_K:-1}"
PYLAIA_GPUS="${PYLAIA_GPUS:-1}"
PYLAIA_AUTO_SELECT_GPUS="${PYLAIA_AUTO_SELECT_GPUS:-0}"
PYLAIA_NUM_WORKERS="${PYLAIA_NUM_WORKERS:-6}"

if [ "$TRAIN_FOLD" != "fold_a" ] && [ "$TRAIN_FOLD" != "fold_b" ] && [ "$TRAIN_FOLD" != "fold_c" ]; then
  echo "ERROR: TRAIN_FOLD must be fold_a, fold_b, or fold_c, got '$TRAIN_FOLD'"
  exit 1
fi

PYLAIA_MODEL_ABS="$(python - <<PY
from pathlib import Path
print(Path("$PYLAIA_MODEL").resolve())
PY
)"
PYLAIA_CHECKPOINT_ABS="$(python - <<PY
from pathlib import Path
print(Path("$PYLAIA_CHECKPOINT").resolve())
PY
)"
PYLAIA_IMAGE_HEIGHT="$(python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$REPO_ROOT") / "src"))
from linealign.nntp import infer_pylaia_input_height

height = infer_pylaia_input_height(Path("$PYLAIA_MODEL_ABS"))
if height is None:
    raise SystemExit("Could not infer PyLaia input height from model")
print(height)
PY
)"
echo "Using PyLaia input height: $PYLAIA_IMAGE_HEIGHT"

python scripts/build_children_pylaia_manifests.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$MANIFEST_OUT_DIR" \
  --syms-out "$SYMS_OUT" \
  --fixed-height "$PYLAIA_IMAGE_HEIGHT" \
  --fold "$TRAIN_FOLD"

MANIFEST_DIR="$MANIFEST_OUT_DIR/$TRAIN_FOLD"
TRAIN_TXT="$MANIFEST_DIR/train.txt"
VAL_TXT="$MANIFEST_DIR/val.txt"
TEST_TXT="$MANIFEST_DIR/test.txt"
mkdir -p "$EXPERIMENT_DIR"
cp "$SYMS_OUT" "$EXPERIMENT_DIR/syms.txt"

SYMS_COUNT="$(python - <<PY
from pathlib import Path
print(len(Path("$SYMS_OUT").read_text(encoding="utf-8").splitlines()))
PY
)"
python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$REPO_ROOT") / "src"))
from linealign.nntp import clone_pylaia_model_with_num_outputs

clone_pylaia_model_with_num_outputs(
    Path("$PYLAIA_MODEL_ABS"),
    $SYMS_COUNT,
    Path("$EXPERIMENT_DIR/model"),
)
PY

PYLAIA_IMG_DIRS="$(python - <<PY
import json
from pathlib import Path

meta = json.loads((Path("$MANIFEST_DIR") / "manifest_meta.json").read_text(encoding="utf-8"))
print("[" + ",".join(meta["pylaia_img_dirs"]) + "]")
PY
)"

PYLAIA_TORCH_INFO="$("$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
import json
import torch

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    )
)
PY
)"
echo "PyLaia torch info: $PYLAIA_TORCH_INFO"
PYLAIA_CUDA_AVAILABLE="$("$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
import torch

print(1 if torch.cuda.is_available() else 0)
PY
)"
if [ "$PYLAIA_GPUS" -gt 0 ] && [ "$PYLAIA_CUDA_AVAILABLE" != "1" ]; then
  echo "WARNING: PyLaia environment cannot see CUDA. Falling back to CPU training."
  PYLAIA_GPUS=0
  PYLAIA_AUTO_SELECT_GPUS=0
fi
if [ "$PYLAIA_GPUS" -eq 0 ]; then
  PYLAIA_AUTO_SELECT_GPUS=0
fi
echo "PyLaia training settings: fold=$TRAIN_FOLD gpus=$PYLAIA_GPUS auto_select_gpus=$PYLAIA_AUTO_SELECT_GPUS num_workers=$PYLAIA_NUM_WORKERS"

if [ "$TRAIN_OVERWRITE" = "1" ]; then
  rm -rf "$EXPERIMENT_DIR"
  mkdir -p "$EXPERIMENT_DIR"
  cp "$SYMS_OUT" "$EXPERIMENT_DIR/syms.txt"
  python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$REPO_ROOT") / "src"))
from linealign.nntp import clone_pylaia_model_with_num_outputs

clone_pylaia_model_with_num_outputs(
    Path("$PYLAIA_MODEL_ABS"),
    $SYMS_COUNT,
    Path("$EXPERIMENT_DIR/model"),
)
PY
fi

if [ -f "$EXPERIMENT_DIR/best.ckpt" ]; then
  echo "Reusing existing trained checkpoint: $EXPERIMENT_DIR/best.ckpt"
else
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" pylaia-htr-dataset-validate \
    "$EXPERIMENT_DIR/syms.txt" \
    "$PYLAIA_IMG_DIRS" \
    "$TRAIN_TXT" \
    "$VAL_TXT" \
    "$TEST_TXT" \
    --common.model_filename "$EXPERIMENT_DIR/model" \
    --fixed_input_height "$PYLAIA_IMAGE_HEIGHT" \
    --statistics_output "$EXPERIMENT_DIR/dataset_statistics.md"

  TRAIN_ARGS=(
    "$EXPERIMENT_DIR/syms.txt"
    "$PYLAIA_IMG_DIRS"
    "$TRAIN_TXT"
    "$VAL_TXT"
    --common.experiment_dirname "$EXPERIMENT_DIR"
    --common.model_filename "$EXPERIMENT_DIR/model"
    --common.checkpoint "$PYLAIA_CHECKPOINT_ABS"
    --train.pretrain true
    --data.batch_size "$BATCH_SIZE"
    --data.num_workers "$PYLAIA_NUM_WORKERS"
    --trainer.max_epochs "$MAX_EPOCHS"
    --train.early_stopping_patience "$EARLY_STOPPING_PATIENCE"
    --train.checkpoint_k "$CHECKPOINT_K"
    --trainer.gpus "$PYLAIA_GPUS"
  )
  if [ "$PYLAIA_AUTO_SELECT_GPUS" = "1" ]; then
    TRAIN_ARGS+=(--trainer.auto_select_gpus true)
  fi

  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" pylaia-htr-train-ctc "${TRAIN_ARGS[@]}"

  BEST_CKPT="$(find "$EXPERIMENT_DIR" -type f -name '*lowest_*.ckpt' ! -path '*/pretrained/*' | sort | head -n 1)"
  if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$(find "$EXPERIMENT_DIR" -type f -name '*.ckpt' ! -name '*-last.ckpt' ! -path '*/pretrained/*' | sort | head -n 1)"
  fi
  if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$(find "$EXPERIMENT_DIR" -type f -name '*-last.ckpt' | sort | head -n 1)"
  fi
  if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint produced under $EXPERIMENT_DIR"
    exit 1
  fi
  cp "$BEST_CKPT" "$EXPERIMENT_DIR/best.ckpt"
fi

echo "Prepared fold assets under $EXPERIMENT_DIR"
echo "Done."
