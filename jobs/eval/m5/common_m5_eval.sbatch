#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_eval_env.sh"
common_eval_bootstrap "run_eval_m5.py"
EVAL_PYTHON="$(command -v python)"

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
IDS="${IDS:-}"
N_SHOTS="${N_SHOTS:-0}"
SHOTS_SEED="${SHOTS_SEED:-42}"
SHOTS_DATASET_SCOPE="${SHOTS_DATASET_SCOPE:-same}"
LINE_IMAGE_MODE="${LINE_IMAGE_MODE:-separate}"
USE_OCR_TEXT="${USE_OCR_TEXT:-0}"
PROMPT_VARIANT="${PROMPT_VARIANT:-baseline}"
TRACE_DIR="${TRACE_DIR:-}"

SHOT_SUFFIX="${N_SHOTS}shot"
MODE_SUFFIX="${LINE_IMAGE_MODE}"
OCR_HINT_SUFFIX=""
if [ "$USE_OCR_TEXT" = "1" ]; then
  OCR_HINT_SUFFIX="_ocrtext"
fi
PROMPT_SUFFIX=""
if [ "$PROMPT_VARIANT" != "baseline" ]; then
  PROMPT_SUFFIX="_prompt_${PROMPT_VARIANT}"
fi
OUT_DIR="${OUT_DIR:-${DATASET_TAG}_predictions_m5_${MODE_SUFFIX}${OCR_HINT_SUFFIX}_${SHOT_SUFFIX}${PROMPT_SUFFIX}_${MODEL_SUFFIX}}"
EVAL_CSV="${EVAL_CSV:-${DATASET_TAG}_eval_m5_${MODE_SUFFIX}${OCR_HINT_SUFFIX}_${SHOT_SUFFIX}${PROMPT_SUFFIX}_${MODEL_SUFFIX}.csv}"

if [ "$SKIP_OCR" != "1" ]; then
  PYLAIA_ROOT="${PYLAIA_ROOT:?Must set PYLAIA_ROOT to the PyLaia asset directory.}"
  PYLAIA_CHECKPOINT="${PYLAIA_CHECKPOINT:-$PYLAIA_ROOT/weights.ckpt}"
  PYLAIA_SYMS="${PYLAIA_SYMS:-$PYLAIA_ROOT/syms.txt}"
  PYLAIA_WORK_DIR="${PYLAIA_WORK_DIR:-outputs/cache/$DATASET_TAG/pylaia}"
  PYLAIA_GPUS="${PYLAIA_GPUS:-0}"
  PYLAIA_AUTO_SELECT_GPUS="${PYLAIA_AUTO_SELECT_GPUS:-0}"
  PYLAIA_NUM_WORKERS="${PYLAIA_NUM_WORKERS:-1}"
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
  PYLAIA_ENV_BIN="$PYLAIA_CONDA_PREFIX/bin"

  if ! "$PYLAIA_PYTHON" - <<'PY'
from importlib.metadata import version
version("pylaia")
PY
  then
    echo "Installing PyLaia into '$PYLAIA_CONDA_ENV' from PyPI package '$PYLAIA_PIP_SPEC'..."
    "$PYLAIA_PYTHON" -m pip install "pip<24.1" setuptools wheel
    "$PYLAIA_PYTHON" -m pip install "$PYLAIA_PIP_SPEC"
  fi

  "$PYLAIA_PYTHON" - <<PY
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

  if ! "$PYLAIA_PYTHON" - <<'PY'
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

  export PATH="$PYLAIA_ENV_BIN:$PATH"
  if ! command -v pylaia-htr-netout >/dev/null 2>&1; then
    echo "ERROR: pylaia-htr-netout not found even after preparing '$PYLAIA_CONDA_ENV'."
    exit 1
  fi

  "$PYLAIA_PYTHON" - <<'PY'
import torch

print("PyLaia CUDA available:", torch.cuda.is_available())
print("PyLaia CUDA device count:", torch.cuda.device_count())
PY

  WRAPPER_DIR="${PYLAIA_WORK_DIR}/job-bin"
  mkdir -p "$WRAPPER_DIR"
  cat > "$WRAPPER_DIR/run_pylaia_netout.py" <<'PY'
import os
import sys

from laia.scripts.htr import common_main
from laia.scripts.htr.netout import get_args, run


def main() -> None:
    args = get_args(sys.argv[1:])
    args = common_main(args)
    args["num_workers"] = max(1, int(os.environ.get("PYLAIA_NUM_WORKERS", "1")))
    run(**args)


if __name__ == "__main__":
    main()
PY
  cat > "$WRAPPER_DIR/pylaia-htr-netout" <<EOF
#!/bin/bash
exec "$PYLAIA_PYTHON" "$WRAPPER_DIR/run_pylaia_netout.py" "\$@"
EOF
  chmod +x "$WRAPPER_DIR/pylaia-htr-netout"
  export PYLAIA_NUM_WORKERS
  export PATH="$WRAPPER_DIR:$PYLAIA_ENV_BIN:$PATH"
  hash -r

  if ! command -v pylaia-htr-netout >/dev/null 2>&1; then
    echo "ERROR: pylaia-htr-netout not found even after preparing '$PYLAIA_CONDA_ENV'."
    exit 1
  fi

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

RUN_ARGS=(
  "$EVAL_PYTHON" run_eval_m5.py
  --data-dir "$DATA_DIR"
  --model "$MODEL"
  --device "$DEVICE"
  --n-shots "$N_SHOTS"
  --shots-dataset-scope "$SHOTS_DATASET_SCOPE"
  --out-dir "$OUT_DIR"
  --eval-csv "$EVAL_CSV"
  --ocr-lines-dir "$OCR_LINES_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --line-image-mode "$LINE_IMAGE_MODE"
  --prompt-variant "$PROMPT_VARIANT"
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
if [ "$USE_OCR_TEXT" = "1" ]; then
  RUN_ARGS+=(--use-ocr-text)
fi
if [ -n "$TRACE_DIR" ]; then
  RUN_ARGS+=(--trace-dir "$TRACE_DIR")
fi

set +e
"${RUN_ARGS[@]}"
PYTHON_EXIT=$?
set -e

common_eval_handle_exit "$PYTHON_EXIT"
