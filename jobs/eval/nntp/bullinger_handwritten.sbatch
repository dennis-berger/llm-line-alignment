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

set -uo pipefail
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
if ! conda env list | sed 's/*//g' | awk 'NF && $1 !~ /^#/' | grep -Fxq "$PYLAIA_CONDA_ENV"; then
  echo "Creating dedicated PyLaia environment '$PYLAIA_CONDA_ENV' with Python $PYLAIA_PYTHON_VERSION..."
  "$CONDA_INSTALL_BIN" create -y -n "$PYLAIA_CONDA_ENV" python="$PYLAIA_PYTHON_VERSION" pip
fi

PYLAIA_ENV_PREFIX="$("$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
PYLAIA_NETOUT_BIN="$PYLAIA_ENV_PREFIX/bin/pylaia-htr-netout"

if [ ! -x "$PYLAIA_NETOUT_BIN" ]; then
  echo "Installing PyLaia into '$PYLAIA_CONDA_ENV' from the official GitHub source..."
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "pip<24.1" setuptools wheel
  "$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -m pip install "https://github.com/jpuigcerver/PyLaia/archive/refs/heads/master.tar.gz"
  PYLAIA_ENV_PREFIX="$("$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python -c 'import sys; print(sys.prefix)')"
  PYLAIA_NETOUT_BIN="$PYLAIA_ENV_PREFIX/bin/pylaia-htr-netout"
fi
if [ ! -x "$PYLAIA_NETOUT_BIN" ]; then
  echo "ERROR: pylaia-htr-netout is still unavailable after installing PyLaia in '$PYLAIA_CONDA_ENV'."
  exit 1
fi

"$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from pathlib import Path
import sysconfig

scripts_dir = Path(sysconfig.get_paths()["purelib"]) / "laia" / "scripts"
scripts_dir.mkdir(parents=True, exist_ok=True)
init_path = scripts_dir / "__init__.py"
if not init_path.exists():
    init_path.write_text(
        '"""Patched by llm-line-alignment to make PyLaia console scripts importable."""\n',
        encoding="utf-8",
    )
    print("Patched missing", init_path)
else:
    print("Found", init_path)
PY

"$CONDA_EXE" run --no-capture-output -n "$PYLAIA_CONDA_ENV" python - <<'PY'
from importlib.metadata import version
import laia
import laia.scripts.htr.netout

print("PyLaia package version:", version("pylaia"))
print("PyLaia import OK:", laia.__file__)
print("PyLaia netout import OK:", laia.scripts.htr.netout.__file__)
PY

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
echo "Using PyLaia netout binary: $PYLAIA_NETOUT_BIN"
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

WRAPPER_DIR="$WORK_DIR/job-bin"
mkdir -p "$WRAPPER_DIR"
cat > "$WRAPPER_DIR/pylaia-htr-netout" <<EOF
#!/bin/bash
exec "$PYLAIA_NETOUT_BIN" "\$@"
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
  --pred-dir "$PRED_DIR"
  --eval-csv "$EVAL_CSV"
  --stop-after "$STOP_AFTER"
)

if [ -n "$IDS" ]; then
  EXTRA_ARGS+=(--ids "$IDS")
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
