#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Official BOP19 evaluation wrapper
#
# Important isolation:
#
# GigaPose / MegaPose inference may configure NVIDIA/Panda3D EGL.
# BOP19 VSD evaluation uses a separate, deterministic headless renderer:
#
#   VisPy
#   Mesa surfaceless EGL
#   llvmpipe
#
# BOP Toolkit also launches nested commands with bare "python".
# Therefore the directory containing --python is prepended to PATH so every
# child Python process uses exactly the requested evaluation environment.
# =============================================================================


WORKSPACE=""
BOP_TOOLKIT_DIR=""
PYTHON_BIN=""
RESULT_CSV=""
OUTPUT_DIR=""

NUM_WORKERS="${BOP_EVAL_NUM_WORKERS:-1}"

MESA_VENDOR_FILE="${BOP_EGL_VENDOR_JSON:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"


die () {
    echo "ERROR: $*" >&2
    exit 1
}


log () {
    printf '[%s] %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$*"
}


usage () {
    cat <<'USAGE'
Usage: scripts/evaluate_bop19.sh [options]

Required:
  --workspace PATH
  --bop-toolkit PATH
  --python PATH
  --result-csv CSV
  --output-dir PATH

Options:
  --num-workers N       default: 1
  -h, --help

Environment overrides:
  BOP_EVAL_NUM_WORKERS
  BOP_EGL_VENDOR_JSON
USAGE
}


while [[ $# -gt 0 ]]; do

    case "$1" in

        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;

        --bop-toolkit)
            BOP_TOOLKIT_DIR="$2"
            shift 2
            ;;

        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;

        --result-csv)
            RESULT_CSV="$2"
            shift 2
            ;;

        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;

        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            die "unknown argument: $1"
            ;;
    esac

done


# =============================================================================
# Required inputs
# =============================================================================

[[ -n "$WORKSPACE" ]] || \
    die "--workspace is required"

[[ -n "$BOP_TOOLKIT_DIR" ]] || \
    die "--bop-toolkit is required"

[[ -n "$PYTHON_BIN" ]] || \
    die "--python is required"

[[ -n "$RESULT_CSV" ]] || \
    die "--result-csv is required"

[[ -n "$OUTPUT_DIR" ]] || \
    die "--output-dir is required"

[[ -d "$WORKSPACE/datasets" ]] || \
    die "BOP datasets directory missing: $WORKSPACE/datasets"

[[ -d "$BOP_TOOLKIT_DIR" ]] || \
    die "BOP Toolkit directory missing: $BOP_TOOLKIT_DIR"

[[ -f "$BOP_TOOLKIT_DIR/scripts/eval_bop19_pose.py" ]] || \
    die "eval_bop19_pose.py missing"

[[ -x "$PYTHON_BIN" ]] || \
    die "Python is not executable: $PYTHON_BIN"

[[ -s "$RESULT_CSV" ]] || \
    die "result CSV missing or empty: $RESULT_CSV"

[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || \
    die "--num-workers must be a positive integer"


# =============================================================================
# Pin all nested BOP Toolkit Python processes to --python
# =============================================================================

PYTHON_DIR="$(
    cd "$(dirname "$PYTHON_BIN")"
    pwd
)"

export PATH="$PYTHON_DIR:$PATH"

hash -r

RESOLVED_CHILD_PYTHON="$(
    command -v python
)"

[[ "$RESOLVED_CHILD_PYTHON" == "$PYTHON_BIN" ]] || {
    die \
        "nested python mismatch: expected $PYTHON_BIN, " \
        "got $RESOLVED_CHILD_PYTHON"
}


# =============================================================================
# Isolate BOP evaluation from GigaPose/Panda3D/NVIDIA EGL state
# =============================================================================

unset DISPLAY || true
unset LD_PRELOAD || true

unset GIGAPOSE_FORCE_NVIDIA_EGL || true

unset CUDA_VISIBLE_DEVICES || true
unset EGL_VISIBLE_DEVICES || true

unset __EGL_VENDOR_LIBRARY_FILENAMES || true


# =============================================================================
# Stable VisPy renderer: Mesa surfaceless EGL / llvmpipe
# =============================================================================

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP_BACKEND=egl

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe

[[ -f "$MESA_VENDOR_FILE" ]] || {
    die \
        "Mesa EGL vendor JSON missing: $MESA_VENDOR_FILE"
}

export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_VENDOR_FILE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONUNBUFFERED=1

# BOP Toolkit config.py reads BOP_PATH when available.
export BOP_PATH="$WORKSPACE/datasets"

export PYTHONPATH="$BOP_TOOLKIT_DIR:$BOP_TOOLKIT_DIR/bop_toolkit_lib:${PYTHONPATH:-}"


# =============================================================================
# Fail-fast child-Python and package smoke
# =============================================================================

"$PYTHON_BIN" - <<'PY'
import sys

import png
import OpenGL
import vispy

print(
    "BOP evaluation Python:",
    sys.executable,
)

print(
    "PyPNG:",
    png.__file__,
)

print(
    "PyOpenGL:",
    OpenGL.__version__,
)

print(
    "VisPy:",
    vispy.__version__,
)
PY


# =============================================================================
# Fail-fast VisPy / Mesa surfaceless EGL smoke
# =============================================================================

"$PYTHON_BIN" - <<'PY'
from OpenGL.GL import (
    GL_RENDERER,
    GL_VENDOR,
    glGetString,
)

from vispy import app


def decode(value):
    if value is None:
        return "<none>"

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


app.use_app("egl")

canvas = app.Canvas(
    show=False,
    size=(32, 32),
)

canvas.set_current()

try:

    renderer = decode(
        glGetString(GL_RENDERER)
    )

    vendor = decode(
        glGetString(GL_VENDOR)
    )

    print(
        "BOP EGL renderer:",
        renderer,
    )

    print(
        "BOP EGL vendor:",
        vendor,
    )

    if (
        "llvmpipe" not in renderer.lower()
        and
        "mesa" not in vendor.lower()
    ):
        raise RuntimeError(
            "Expected Mesa/llvmpipe BOP evaluation renderer"
        )

finally:

    canvas.close()

print(
    "BOP VisPy/Mesa EGL smoke: PASS"
)
PY


# =============================================================================
# Prepare isolated results/eval directories
# =============================================================================

RESULTS_DIR="$OUTPUT_DIR/results"
EVAL_DIR="$OUTPUT_DIR/eval"

rm -rf "$OUTPUT_DIR"

mkdir -p \
    "$RESULTS_DIR" \
    "$EVAL_DIR"

RESULT_NAME="$(
    basename "$RESULT_CSV"
)"

RESULT_COPY="$RESULTS_DIR/$RESULT_NAME"

cp -f \
    "$RESULT_CSV" \
    "$RESULT_COPY"


SOURCE_SHA="$(
    sha256sum "$RESULT_CSV" |
    awk '{print $1}'
)"

COPY_SHA="$(
    sha256sum "$RESULT_COPY" |
    awk '{print $1}'
)"

[[ "$SOURCE_SHA" == "$COPY_SHA" ]] || \
    die "BOP result copy SHA256 mismatch"


# =============================================================================
# Official BOP19
# =============================================================================

log \
    "running official BOP19 evaluation for $RESULT_NAME"

log \
    "python=$PYTHON_BIN"

log \
    "nested_python=$(command -v python)"

log \
    "renderer=vispy / Mesa surfaceless / llvmpipe"

log \
    "num_workers=$NUM_WORKERS"


"$PYTHON_BIN" \
    "$BOP_TOOLKIT_DIR/scripts/eval_bop19_pose.py" \
    --result_filenames="$RESULT_NAME" \
    --results_path="$RESULTS_DIR" \
    --eval_path="$EVAL_DIR" \
    --targets_filename=test_targets_bop19.json \
    --renderer_type=vispy \
    --num_workers="$NUM_WORKERS"


# =============================================================================
# Verify official score artifact
# =============================================================================

RESULT_STEM="${RESULT_NAME%.csv}"

SCORES="$EVAL_DIR/$RESULT_STEM/scores_bop19.json"

[[ -s "$SCORES" ]] || {
    die \
        "official scores_bop19.json was not produced: $SCORES"
}

log \
    "BOP19 evaluation complete: $EVAL_DIR"

log \
    "scores: $SCORES"
