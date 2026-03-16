#!/bin/bash
#SBATCH --job-name=bullinger_nntp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
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
if [ ! -f "scripts/run_nntp_eval.py" ]; then
  echo "ERROR: could not find scripts/run_nntp_eval.py in REPO_ROOT='$REPO_ROOT'"
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
  echo "Installing PyLaia into '$PYLAIA_CONDA_ENV' from the official GitHub source..."
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "pip<24.1" setuptools wheel
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "$PYLAIA_SOURCE_URL"
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
        print("Patched missing", init_path)
    else:
        print("Found", init_path)
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

if ! command -v java >/dev/null 2>&1 || ! command -v javac >/dev/null 2>&1; then
  echo "Installing OpenJDK into '$CONDA_ENV_NAME' because java/javac are missing..."
  "$CONDA_INSTALL_BIN" install -y -c conda-forge openjdk
  hash -r
fi
if ! command -v java >/dev/null 2>&1 || ! command -v javac >/dev/null 2>&1; then
  echo "ERROR: java and javac are still unavailable after installing OpenJDK."
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo "Using PyLaia module runner from env: $PYLAIA_CONDA_ENV"
java -version || true
javac -version || true

DATA_DIR="${DATA_DIR:-datasets/bullinger_handwritten}"
WORK_DIR="${WORK_DIR:-outputs/nntp/bullinger_handwritten}"
PYLAIA_ROOT="${PYLAIA_ROOT:-third_party/pylaia-dennis}"
PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/epoch=170-lowest_va_cer.ckpt}"
PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
NNTP_ROOT="${NNTP_ROOT:-../nntp}"
PRED_DIR="${PRED_DIR:-bullinger_handwritten_predictions_nntp}"
EVAL_CSV="${EVAL_CSV:-bullinger_handwritten_eval_nntp.csv}"
STOP_AFTER="${STOP_AFTER:-evaluate}"
IDS="${IDS:-}"
OVERWRITE="${OVERWRITE:-0}"
PYLAIA_GPUS="${PYLAIA_GPUS:-0}"
PYLAIA_AUTO_SELECT_GPUS="${PYLAIA_AUTO_SELECT_GPUS:-0}"
PYLAIA_NUM_WORKERS="${PYLAIA_NUM_WORKERS:-6}"

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
  echo "WARNING: PyLaia environment cannot see CUDA. Falling back to CPU netout."
  PYLAIA_GPUS=0
  PYLAIA_AUTO_SELECT_GPUS=0
fi
if [ "$PYLAIA_GPUS" -eq 0 ]; then
  PYLAIA_AUTO_SELECT_GPUS=0
fi
echo "PyLaia runtime settings: gpus=$PYLAIA_GPUS auto_select_gpus=$PYLAIA_AUTO_SELECT_GPUS num_workers=$PYLAIA_NUM_WORKERS"

PYLAIA_RUNTIME_ROOT="$WORK_DIR/pylaia_runtime"
mkdir -p "$PYLAIA_RUNTIME_ROOT"
cp "$PYLAIA_SYMS" "$PYLAIA_RUNTIME_ROOT/syms.txt"
"$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<PY
import torch
from pathlib import Path

src_model = Path(r"$PYLAIA_ROOT") / "model"
dst_model = Path(r"$PYLAIA_RUNTIME_ROOT") / "model"
checkpoint_path = Path(r"$PYLAIA_CHECKPOINT")

model_obj = torch.load(src_model, map_location="cpu", weights_only=False)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)

linear_weight = state_dict.get("model.linear.weight")
if linear_weight is None:
    linear_weight = state_dict["linear.weight"]

checkpoint_outputs = int(linear_weight.shape[0])
current_outputs = int(model_obj["kwargs"]["num_output_labels"])
if current_outputs != checkpoint_outputs:
    print(f"Patching PyLaia model num_output_labels: {current_outputs} -> {checkpoint_outputs}")
    model_obj["kwargs"]["num_output_labels"] = checkpoint_outputs
else:
    print(f"PyLaia model already matches checkpoint outputs: {current_outputs}")

torch.save(model_obj, dst_model)
PY
PYLAIA_ROOT="$PYLAIA_RUNTIME_ROOT"
PYLAIA_SYMS="$PYLAIA_RUNTIME_ROOT/syms.txt"

WRAPPER_DIR="$WORK_DIR/job-bin"
mkdir -p "$WRAPPER_DIR"
cat > "$WRAPPER_DIR/run_pylaia_netout.py" <<'EOF'
import os
import sys

from laia.scripts.htr import common_main
from laia.scripts.htr.netout import get_args, run


def main() -> None:
    args = get_args(sys.argv[1:])
    args = common_main(args)
    args["num_workers"] = int(os.environ.get("PYLAIA_NUM_WORKERS", "6"))
    run(**args)


if __name__ == "__main__":
    main()
EOF
cat > "$WRAPPER_DIR/pylaia-htr-netout" <<EOF
#!/bin/bash
exec "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" env PYLAIA_NUM_WORKERS="$PYLAIA_NUM_WORKERS" python "$WRAPPER_DIR/run_pylaia_netout.py" "\$@"
EOF
chmod +x "$WRAPPER_DIR/pylaia-htr-netout"
export PATH="$WRAPPER_DIR:$PATH"
which pylaia-htr-netout

EXTRA_ARGS=(
  --data-dir "$DATA_DIR"
  --work-dir "$WORK_DIR"
  --pylaia-root "$PYLAIA_ROOT"
  --pylaia-checkpoint "$PYLAIA_CHECKPOINT"
  --pylaia-syms "$PYLAIA_SYMS"
  --nntp-root "$NNTP_ROOT"
  --pylaia-gpus "$PYLAIA_GPUS"
  --pred-dir "$PRED_DIR"
  --eval-csv "$EVAL_CSV"
  --stop-after "$STOP_AFTER"
)

if [ -n "$IDS" ]; then
  EXTRA_ARGS+=(--ids "$IDS")
fi
if [ "$PYLAIA_AUTO_SELECT_GPUS" = "1" ]; then
  EXTRA_ARGS+=(--pylaia-auto-select-gpus)
fi
if [ "$OVERWRITE" = "1" ]; then
  EXTRA_ARGS+=(--overwrite)
fi

python scripts/run_nntp_eval.py "${EXTRA_ARGS[@]}"

PYTHON_EXIT=$?
if [ "$PYTHON_EXIT" -ne 0 ]; then
  echo "Python script failed with exit code $PYTHON_EXIT"
  exit "$PYTHON_EXIT"
fi

echo "Done."
