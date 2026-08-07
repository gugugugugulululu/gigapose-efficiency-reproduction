#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

WORKSPACE="${BOP_WORKSPACE:-}"
BOP_TOOLKIT="${BOP_TOOLKIT:-}"
PYTHON_BIN="${GP_PYTHON:-}"
RESULT_CSV=""
OUTPUT_DIR=""
DATASET="lmo"
RENDERER="vispy"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --bop-toolkit) BOP_TOOLKIT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --result-csv) RESULT_CSV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$WORKSPACE" ]] || die "set --workspace or BOP_WORKSPACE"
[[ -n "$BOP_TOOLKIT" ]] || BOP_TOOLKIT="$WORKSPACE/code/bop_toolkit"
[[ -n "$RESULT_CSV" ]] || die "--result-csv is required"
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$(dirname "$RESULT_CSV")/bop19_eval"
PYTHON_BIN="$(resolve_python "$PYTHON_BIN")"
require_file "$RESULT_CSV"
require_file "$BOP_TOOLKIT/scripts/eval_bop19_pose.py"

mkdir -p "$OUTPUT_DIR/results" "$OUTPUT_DIR/eval"
RESULT_NAME="$(basename "$RESULT_CSV")"
cp -f "$RESULT_CSV" "$OUTPUT_DIR/results/$RESULT_NAME"

export BOP_PATH="$WORKSPACE/datasets"
export BOP_DATASETS_DIR="$WORKSPACE/datasets"
export BOP_RESULTS_DIR="$OUTPUT_DIR/results"
export BOP_EVAL_PATH="$OUTPUT_DIR/eval"
export PYTHONPATH="$BOP_TOOLKIT:$BOP_TOOLKIT/bop_toolkit_lib:${PYTHONPATH:-}"

log "running official BOP19 evaluation for $RESULT_NAME"
"$PYTHON_BIN" -u "$BOP_TOOLKIT/scripts/eval_bop19_pose.py" \
  --renderer_type="$RENDERER" \
  --result_filenames="$RESULT_NAME" \
  --results_path="$OUTPUT_DIR/results" \
  --eval_path="$OUTPUT_DIR/eval" \
  --targets_filename="test_targets_bop19.json"

log "BOP19 evaluation complete: $OUTPUT_DIR/eval"
