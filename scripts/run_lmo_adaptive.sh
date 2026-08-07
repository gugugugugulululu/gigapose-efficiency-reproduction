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
POLICY_CSV="$ROOT_DIR/configs/lmo/object_policy.csv"
OUTPUT_DIR=""
RUN_ID="lmo_combined_adaptive_$(date +%Y%m%d_%H%M%S)"
BATCH_IMAGES=8
BATCH_OBJECTS=8
GROUP_TIME_MODE="total"
RUN_EVAL=1
PATCH_EMPTY_BATCH=1

usage() {
  cat <<'USAGE'
Usage: scripts/run_lmo_adaptive.sh [options]

Required:
  --workspace PATH
  --coarse-main CSV
  --coarse-multi CSV

Options:
  --policy CSV             default: configs/lmo/object_policy.csv
  --gigapose-repo PATH
  --checkpoint PATH
  --megapose-models-root PATH
  --bop-toolkit PATH
  --python PATH
  --env-script PATH
  --output-dir PATH
  --run-id NAME
  --batch-images N         default: 8
  --batch-objects N        default: 8
  --group-time-mode MODE   total|increment, default: total
  --no-empty-batch-patch
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
    --policy) POLICY_CSV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --batch-images) BATCH_IMAGES="$2"; shift 2 ;;
    --batch-objects) BATCH_OBJECTS="$2"; shift 2 ;;
    --group-time-mode) GROUP_TIME_MODE="$2"; shift 2 ;;
    --no-empty-batch-patch) PATCH_EMPTY_BATCH=0; shift ;;
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
require_file "$POLICY_CSV"
require_file "$GIGAPOSE_REPO/refine.py"
require_file "$GIGAPOSE_REPO/src/models/refiner.py"
require_file "$CHECKPOINT"
require_dir "$MEGAPOSE_ROOT"
mkdir -p "$OUTPUT_DIR"/{groups,logs,predictions,reports}

export PYTHONPATH="$ROOT_DIR/src:$GIGAPOSE_REPO:$GIGAPOSE_REPO/src:$BOP_TOOLKIT_DIR:$BOP_TOOLKIT_DIR/bop_toolkit_lib:${PYTHONPATH:-}"

log "validating accelerated coarse inputs"
"$PYTHON_BIN" -m gigapose_efficiency.cli validate \
  --csv "$COARSE_MAIN" --rows 1445 --targets 1445 --images 200 --hypotheses 1 --lmo-objects \
  > "$OUTPUT_DIR/reports/coarse_main_validation.json"
"$PYTHON_BIN" -m gigapose_efficiency.cli validate \
  --csv "$COARSE_MULTI" --rows 7225 --targets 1445 --images 200 --hypotheses 5 --lmo-objects \
  > "$OUTPUT_DIR/reports/coarse_multi_validation.json"

if [[ "$PATCH_EMPTY_BATCH" -eq 1 ]]; then
  log "ensuring MegaPose empty-batch support"
  "$PYTHON_BIN" -m gigapose_efficiency.cli patch-empty-batch \
    --refiner "$GIGAPOSE_REPO/src/models/refiner.py" \
    > "$OUTPUT_DIR/reports/empty_batch_patch.json"
  "$PYTHON_BIN" -m py_compile "$GIGAPOSE_REPO/src/models/refiner.py"
fi

PREPARED_DIR="$OUTPUT_DIR/groups/prepared"
"$PYTHON_BIN" -m gigapose_efficiency.cli build-adaptive-groups \
  --main "$COARSE_MAIN" \
  --multi "$COARSE_MULTI" \
  --policy "$POLICY_CSV" \
  --output-dir "$PREPARED_DIR" \
  --expected-targets 1445 \
  --expected-total-hypotheses 5268 \
  > "$OUTPUT_DIR/reports/adaptive_groups.json"

GROUP_TSV="$OUTPUT_DIR/reports/group_commands.tsv"

# Do not redirect Python stdout into the TSV.
#
# The restored formal GigaPose runtime may emit process-bootstrap messages
# to stdout when Python starts. If stdout is used as a machine-readable TSV
# transport, such messages become bogus group rows and can leave GROUP_MAIN
# and GROUP_MULTI empty.
#
# Instead, Python writes the TSV directly to the requested path.
"$PYTHON_BIN" - "$PREPARED_DIR/manifest.json" "$GROUP_TSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

groups = manifest["groups"]

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.writer(
        handle,
        delimiter="\t",
        lineterminator="\n",
    )

    for group in groups:
        writer.writerow([
            group["group"],
            str(group["k"]),
            str(group["i"]),
            str(group["targets"]),
            group["main_csv"],
            group["multi_csv"],
        ])
PY

# Fail fast before any GPU work.
#
# LM-O object-adaptive policy currently contains exactly five K/I groups,
# and every row must contain six tab-separated fields:
#
#   group, K, I, targets, main_csv, multi_csv
"$PYTHON_BIN" - "$GROUP_TSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])

with path.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(
        csv.reader(
            handle,
            delimiter="\t",
        )
    )

if len(rows) != 5:
    raise RuntimeError(
        f"group_commands.tsv rows={len(rows)}, expected=5"
    )

expected_groups = {
    "k3_i3",
    "k3_i4",
    "k3_i5",
    "k4_i4",
    "k5_i4",
}

actual_groups = set()

for index, row in enumerate(rows, start=1):
    if len(row) != 6:
        raise RuntimeError(
            f"group_commands.tsv line {index} has "
            f"{len(row)} fields, expected=6: {row!r}"
        )

    (
        group_name,
        k_value,
        i_value,
        targets,
        main_csv,
        multi_csv,
    ) = row

    if not group_name:
        raise RuntimeError(
            f"line {index}: empty group_name"
        )

    if not main_csv:
        raise RuntimeError(
            f"line {index}: empty main_csv"
        )

    if not multi_csv:
        raise RuntimeError(
            f"line {index}: empty multi_csv"
        )

    if not Path(main_csv).is_file():
        raise RuntimeError(
            f"line {index}: missing main_csv: {main_csv}"
        )

    if not Path(multi_csv).is_file():
        raise RuntimeError(
            f"line {index}: missing multi_csv: {multi_csv}"
        )

    int(k_value)
    int(i_value)
    int(targets)

    actual_groups.add(group_name)

if actual_groups != expected_groups:
    raise RuntimeError(
        "unexpected adaptive groups: "
        f"{sorted(actual_groups)}"
    )

print("GROUP COMMAND TSV: PASS")

for row in rows:
    print("\t".join(row))
PY

REFINED_INPUTS=()
RUNTIME_GROUP_ARGS=()

while IFS=$'\t' read -r GROUP_NAME K_VALUE I_VALUE TARGETS GROUP_MAIN GROUP_MULTI; do
  [[ -n "$GROUP_NAME" ]] || continue
  GROUP_OUTPUT="$OUTPUT_DIR/groups/$GROUP_NAME"
  mkdir -p "$GROUP_OUTPUT"/{logs,predictions,reports}
  GROUP_RUN_ID="${RUN_ID}_${GROUP_NAME}"
  STAGE_RESULT="$WORKSPACE/results/large_${GROUP_RUN_ID}"
  STAGE_PRED="$STAGE_RESULT/predictions"
  mkdir -p "$STAGE_PRED"
  STAGE_MAIN="$STAGE_PRED/large-pbrreal-rgb-mmodel_lmo-test_${GROUP_RUN_ID}.csv"
  STAGE_MULTI="$STAGE_PRED/large-pbrreal-rgb-mmodel_lmo-test_${GROUP_RUN_ID}MultiHypothesis.csv"
  cp -f "$GROUP_MAIN" "$STAGE_MAIN"
  cp -f "$GROUP_MULTI" "$STAGE_MULTI"

  REFINE_LOG="$GROUP_OUTPUT/logs/refine.log"
  log "running $GROUP_NAME: K=$K_VALUE I=$I_VALUE targets=$TARGETS"
  set +e
  (
    cd "$GIGAPOSE_REPO"
    "$PYTHON_BIN" -u refine.py \
      test_dataset_name=lmo \
      machine.root_dir="$WORKSPACE" \
      model=large \
      model.checkpoint_path="$CHECKPOINT" \
      run_id="$GROUP_RUN_ID" \
      use_multiple=true \
      ++model.refiner.model_name=megapose-1.0-RGB-multi-hypothesis \
      ++model.refiner.models_root="$MEGAPOSE_ROOT" \
      model.refiner.n_iterations="$I_VALUE" \
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
    tail -n 160 "$REFINE_LOG" || true
    die "$GROUP_NAME did not produce a refined CSV (exit code $REFINE_RC)"
  }
  if [[ "$REFINE_RC" -ne 0 ]]; then
    log "warning: $GROUP_NAME refine.py returned $REFINE_RC but produced a CSV"
  fi

  GROUP_FINAL="$GROUP_OUTPUT/predictions/${GROUP_NAME}_refined.csv"
  cp -f "$REFINED_FOUND" "$GROUP_FINAL"
  "$PYTHON_BIN" -m gigapose_efficiency.cli validate \
    --csv "$GROUP_FINAL" --rows "$TARGETS" --targets "$TARGETS" --hypotheses 1 \
    > "$GROUP_OUTPUT/reports/final_validation.json"

  REFINED_INPUTS+=(--input "$GROUP_FINAL")
  RUNTIME_GROUP_ARGS+=(--group "$GROUP_MAIN" "$GROUP_FINAL")
done < "$GROUP_TSV"

RUNTIME_JSON="$OUTPUT_DIR/reports/reconstructed_runtime.json"
IMAGE_TIMES="$OUTPUT_DIR/reports/reconstructed_image_times.csv"
"$PYTHON_BIN" -m gigapose_efficiency.cli reconstruct-runtime \
  --source-main "$COARSE_MAIN" \
  "${RUNTIME_GROUP_ARGS[@]}" \
  --group-time-mode "$GROUP_TIME_MODE" \
  --output-json "$RUNTIME_JSON" \
  --output-image-times-csv "$IMAGE_TIMES" \
  > "$OUTPUT_DIR/reports/reconstructed_runtime_stdout.json"

# BOP Toolkit expects:
#   <method>_<dataset>-<split>.csv
#
# Replace underscores in RUN_ID so the method name does not interfere
# with BOP Toolkit filename parsing.
BOP_METHOD="${RUN_ID//_/-}"
FINAL_CSV="$OUTPUT_DIR/predictions/${BOP_METHOD}_lmo-test.csv"
"$PYTHON_BIN" -m gigapose_efficiency.cli merge-groups \
  "${REFINED_INPUTS[@]}" \
  --output "$FINAL_CSV" \
  --expected-targets 1445 \
  --image-times-csv "$IMAGE_TIMES" \
  > "$OUTPUT_DIR/reports/final_validation.json"

"$PYTHON_BIN" -m gigapose_efficiency.cli manifest \
  --file "$COARSE_MAIN" --file "$COARSE_MULTI" --file "$POLICY_CSV" --file "$FINAL_CSV" \
  --output "$OUTPUT_DIR/reports/sha256_manifest.json" >/dev/null

if [[ "$RUN_EVAL" -eq 1 ]]; then
  bash "$SCRIPT_DIR/evaluate_bop19.sh" \
    --workspace "$WORKSPACE" \
    --bop-toolkit "$BOP_TOOLKIT_DIR" \
    --python "$PYTHON_BIN" \
    --result-csv "$FINAL_CSV" \
    --output-dir "$OUTPUT_DIR/bop19"
fi

log "Combined Object-adaptive complete"
log "final CSV: $FINAL_CSV"
log "runtime report: $RUNTIME_JSON"
