#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

WORKSPACE="${BOP_WORKSPACE:-}"
GIGAPOSE_REPO="${GIGAPOSE_REPO:-}"
CHECKPOINT="${GIGAPOSE_CHECKPOINT:-}"
MEGAPOSE_ROOT="${MEGAPOSE_MODELS_ROOT:-}"
BOP_TOOLKIT_DIR="${BOP_TOOLKIT:-}"
PYTHON_BIN="${GP_PYTHON:-}"
ENV_SCRIPT="${GP_ENV_SCRIPT:-}"
COARSE_MAIN=""
COARSE_MULTI=""
OUTPUT_DIR=""
RUN_ID="lmo_combined_top3_$(date +%Y%m%d_%H%M%S)"
BATCH_IMAGES=8
BATCH_OBJECTS=8
N_ITER=5
RUN_EVAL=1

usage() {
  cat <<'USAGE'
Usage: scripts/run_lmo_top3.sh [options]

Required:
  --workspace PATH
  --coarse-main CSV
  --coarse-multi CSV

Options:
  --gigapose-repo PATH
  --checkpoint PATH
  --megapose-models-root PATH
  --bop-toolkit PATH
  --python PATH
  --env-script PATH
  --output-dir PATH
  --run-id NAME
  --batch-images N       default: 8
  --batch-objects N      default: 8
  --iterations N         default: 5
  --skip-eval
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --gigapose-repo) GIGAPOSE_REPO="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --megapose-models-root) MEGAPOSE_ROOT="$2"; shift 2 ;;
    --bop-toolkit) BOP_TOOLKIT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --env-script) ENV_SCRIPT="$2"; shift 2 ;;
    --coarse-main) COARSE_MAIN="$2"; shift 2 ;;
    --coarse-multi) COARSE_MULTI="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --batch-images) BATCH_IMAGES="$2"; shift 2 ;;
    --batch-objects) BATCH_OBJECTS="$2"; shift 2 ;;
    --iterations) N_ITER="$2"; shift 2 ;;
    --skip-eval) RUN_EVAL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$WORKSPACE" ]] || die "--workspace is required"
[[ -n "$COARSE_MAIN" ]] || die "--coarse-main is required"
[[ -n "$COARSE_MULTI" ]] || die "--coarse-multi is required"
[[ -n "$GIGAPOSE_REPO" ]] || GIGAPOSE_REPO="$WORKSPACE/code/gigapose"
[[ -n "$CHECKPOINT" ]] || CHECKPOINT="$WORKSPACE/pretrained/gigaPose_v1.ckpt"
[[ -n "$MEGAPOSE_ROOT" ]] || MEGAPOSE_ROOT="$WORKSPACE/pretrained/megapose-models"
[[ -n "$BOP_TOOLKIT_DIR" ]] || BOP_TOOLKIT_DIR="$WORKSPACE/code/bop_toolkit"
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$WORKSPACE/results/reproduction/$RUN_ID"
PYTHON_BIN="$(resolve_python "$PYTHON_BIN")"
source_optional_env "$ENV_SCRIPT"

require_file "$COARSE_MAIN"
require_file "$COARSE_MULTI"
require_file "$GIGAPOSE_REPO/refine.py"
require_file "$CHECKPOINT"
require_dir "$MEGAPOSE_ROOT"
mkdir -p "$OUTPUT_DIR"/{inputs,logs,predictions,reports}

export PYTHONPATH="$ROOT_DIR/src:$GIGAPOSE_REPO:$GIGAPOSE_REPO/src:$BOP_TOOLKIT_DIR:$BOP_TOOLKIT_DIR/bop_toolkit_lib:${PYTHONPATH:-}"

log "validating accelerated coarse inputs"
"$PYTHON_BIN" -m gigapose_efficiency.cli validate \
  --csv "$COARSE_MAIN" --rows 1445 --targets 1445 --images 200 --hypotheses 1 --lmo-objects \
  > "$OUTPUT_DIR/reports/coarse_main_validation.json"
"$PYTHON_BIN" -m gigapose_efficiency.cli validate \
  --csv "$COARSE_MULTI" --rows 7225 --targets 1445 --images 200 --hypotheses 5 --lmo-objects \
  > "$OUTPUT_DIR/reports/coarse_multi_validation.json"

TOP3_MULTI="$OUTPUT_DIR/inputs/top3_MultiHypothesis.csv"
"$PYTHON_BIN" -m gigapose_efficiency.cli prune-topk \
  --input "$COARSE_MULTI" \
  --output "$TOP3_MULTI" \
  --top-k 3 \
  --expected-targets 1445 \
  --expected-rows 4335 \
  > "$OUTPUT_DIR/reports/top3_validation.json"

STAGE_RUN_ID="${RUN_ID}_upstream"
STAGE_RESULT="$WORKSPACE/results/large_${STAGE_RUN_ID}"
STAGE_PRED="$STAGE_RESULT/predictions"
mkdir -p "$STAGE_PRED"
STAGE_MAIN="$STAGE_PRED/large-pbrreal-rgb-mmodel_lmo-test_${STAGE_RUN_ID}.csv"
STAGE_MULTI="$STAGE_PRED/large-pbrreal-rgb-mmodel_lmo-test_${STAGE_RUN_ID}MultiHypothesis.csv"
cp -f "$COARSE_MAIN" "$STAGE_MAIN"
cp -f "$TOP3_MULTI" "$STAGE_MULTI"

REFINE_LOG="$OUTPUT_DIR/logs/refine.log"
log "running MegaPose K=3, I=$N_ITER, batch=${BATCH_IMAGES}x${BATCH_OBJECTS}"
set +e
(
  cd "$GIGAPOSE_REPO"
  "$PYTHON_BIN" -u refine.py \
    test_dataset_name=lmo \
    machine.root_dir="$WORKSPACE" \
    model=large \
    model.checkpoint_path="$CHECKPOINT" \
    run_id="$STAGE_RUN_ID" \
    use_multiple=true \
    ++model.refiner.model_name=megapose-1.0-RGB-multi-hypothesis \
    ++model.refiner.models_root="$MEGAPOSE_ROOT" \
    model.refiner.n_iterations="$N_ITER" \
    model.refiner.batch_size_images="$BATCH_IMAGES" \
    model.refiner.batch_size_objects="$BATCH_OBJECTS" \
    machine.trainer.num_sanity_val_steps=0 \
    ++machine.trainer.strategy=auto \
    ++machine.trainer.precision=32
) > "$REFINE_LOG" 2>&1
REFINE_RC=$?
set -e

REFINED_DIR="$STAGE_RESULT/refined_multiple_predictions"
REFINED_FOUND="$(find_latest_refined_csv "$REFINED_DIR" || true)"
[[ -n "$REFINED_FOUND" && -s "$REFINED_FOUND" ]] || {
  tail -n 120 "$REFINE_LOG" || true
  die "refined CSV was not produced (refine exit code $REFINE_RC)"
}
if [[ "$REFINE_RC" -ne 0 ]]; then
  log "warning: refine.py returned $REFINE_RC but produced a CSV; continuing after validation"
fi

# BOP Toolkit expects:
#   <method>_<dataset>-<split>.csv
#
# Method names must therefore not contain "_" because "_" separates the
# method name from the dataset/split component.
BOP_METHOD="${RUN_ID//_/-}"
FINAL_CSV="$OUTPUT_DIR/predictions/${BOP_METHOD}_lmo-test.csv"
cp -f "$REFINED_FOUND" "$FINAL_CSV"
"$PYTHON_BIN" -m gigapose_efficiency.cli validate \
  --csv "$FINAL_CSV" --rows 1445 --targets 1445 --images 200 --hypotheses 1 --lmo-objects \
  > "$OUTPUT_DIR/reports/final_validation.json"
"$PYTHON_BIN" -m gigapose_efficiency.cli manifest \
  --file "$COARSE_MAIN" --file "$COARSE_MULTI" --file "$TOP3_MULTI" --file "$FINAL_CSV" \
  --output "$OUTPUT_DIR/reports/sha256_manifest.json" >/dev/null

if [[ "$RUN_EVAL" -eq 1 ]]; then
  bash "$SCRIPT_DIR/evaluate_bop19.sh" \
    --workspace "$WORKSPACE" \
    --bop-toolkit "$BOP_TOOLKIT_DIR" \
    --python "$PYTHON_BIN" \
    --result-csv "$FINAL_CSV" \
    --output-dir "$OUTPUT_DIR/bop19"
fi

log "Combined Top-3 complete"
log "final CSV: $FINAL_CSV"
log "refine log: $REFINE_LOG"
