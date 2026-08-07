set -Eeuo pipefail

# =============================================================================
# FIXED LM-O CSM + INT8 + FP16 + IST CACHE
#
# Fix:
#   Do NOT rely on sitecustomize Module.to/cuda/eval hook.
#   Instead, append an explicit patch to GigaPose.py:
#       on_test_start()
#           -> locate self.ae_net.dinov2_model
#           -> apply TorchAO INT8 weight-only
#           -> wrap forward_features with CUDA FP16 autocast
#
# Also keeps:
#   - CSM-v1 aggressive filtered detections
#   - IST query cache
#   - coarse-only, no refinement
#   - official BOP19 evaluation
# =============================================================================

ROOT="/content/drive/MyDrive/bop_workspace"
CACHE="$ROOT/runtime_cache"
PY="/content/gp_py"
ENV="/content/drive/MyDrive/mamba_root/envs/gp"
PYVER="python3.12"

SOURCE_REPO="$ROOT/code/gigapose"
BOPTK_DRIVE="$ROOT/code/bop_toolkit"

EXPECTED_COMMIT="388e8bddd8a5443e284a7f70ad103d03f3f461c5"

CHECKPOINT="$ROOT/pretrained/gigaPose_v1.ckpt"
EXPECTED_CHECKPOINT_SHA256="0f60a23b03ddc41d2135c916ed1e66fb16f814f612dbde0305ae5a2c0f45c932"

RAW_DET_JSON="$ROOT/datasets/cnos-fastsam/cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json"
EXPECTED_RAW_DET_SHA256="1a03d3c7a1d57a9c7e6e1bc162f99281b5044ca50428c619477ec4ab11fa375a"

CSM_SOURCE_RUN_ID="lmo_csm_v1_aggressive_coarse_20260708_180153"
CSM_FILTERED_JSON="$ROOT/results/cascaded_segmentation_masking/${CSM_SOURCE_RUN_ID}/filtered_detections/${CSM_SOURCE_RUN_ID}_filtered_detections.json"
EXPECTED_CSM_SHA256="ad206cd3c6ae0d238a3c61e29ff3e3bc1b93824d5c8449c6835b66569316db70"

IST_CACHE_SOURCE="$ROOT/results/ist_query_feature_reuse/lmo/lmo_ist_query_cache_full_20260711_171044/source/gigaPose_ist_cache_used.py"

TORCHAO_SITE="$CACHE/torchao_site_py312"

# Baselines for summary
FP32_BASELINE_AR_PERCENT="29.6623"
FP32_BASELINE_TIME="2.681836"

IST_CACHE_AR_PERCENT="29.6623"
IST_CACHE_TIME="1.379860"

CSM_V1_FP16_AR_PERCENT="29.4842"
CSM_V1_FP16_TIME="1.699353"

CSM_IST_UNVERIFIED_TIME="0.934197"

RUN_ID="lmo_csmv1_int8_fp16_istcache_fixed_$(date +%Y%m%d_%H%M%S)"
EXP_REPO="/content/gigapose_${RUN_ID}"

RESULT_DIR="$ROOT/results/large_${RUN_ID}"
PRED_DIR="$RESULT_DIR/predictions"

OUT_DIR="$ROOT/results/combined_csm_int8_fp16_istcache_fixed/lmo/${RUN_ID}"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
SOURCE_DIR="$OUT_DIR/source"
PROFILE_DIR="$OUT_DIR/profiles"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR" "$SOURCE_DIR" "$PROFILE_DIR"

MASTER_LOG="$LOG_DIR/${RUN_ID}_master.log"
INFER_LOG="$LOG_DIR/${RUN_ID}_inference.log"
EVAL_LOG="$LOG_DIR/${RUN_ID}_bop19_eval.log"

IST_PROFILE_JSONL="$PROFILE_DIR/${RUN_ID}_ist_profile.jsonl"
PATCH_REPORT_JSON="$REPORT_DIR/${RUN_ID}_explicit_int8_fp16_patch_report.json"
VALIDATION_JSON="$REPORT_DIR/${RUN_ID}_validation.json"
SUMMARY_JSON="$REPORT_DIR/${RUN_ID}_summary.json"
SUMMARY_TXT="$REPORT_DIR/${RUN_ID}_summary.txt"

BACKUP_DET_JSON="$RAW_DET_JSON.backup_${RUN_ID}"

LOCAL_BOPTK="/content/bop_toolkit_${RUN_ID}"
LOCAL_RESULTS_DIR="/content/bop_results_${RUN_ID}"
EVAL_SITE="/content/bop_eval_site_${RUN_ID}"
BIN_DIR="/content/bop_eval_bin_${RUN_ID}"
EVAL_DIR="$OUT_DIR/bop19_eval/${RUN_ID}_bop19_eval"

mkdir -p "$LOCAL_RESULTS_DIR" "$EVAL_SITE" "$BIN_DIR" "$EVAL_DIR"

BOP_RESULT_FILENAME="csmint8fp16istcache-fixed_lmo-test.csv"
BOP_RESULT_CSV="$LOCAL_RESULTS_DIR/$BOP_RESULT_FILENAME"

exec > >(tee -a "$MASTER_LOG") 2>&1

restore_raw_detection() {
  if [[ -s "$BACKUP_DET_JSON" ]]; then
    cp -f "$BACKUP_DET_JSON" "$RAW_DET_JSON"
    echo
    echo "Restored raw detection JSON:"
    sha256sum "$RAW_DET_JSON" || true
  fi
}
trap restore_raw_detection EXIT

echo "================================================================================"
echo "FIXED LM-O CSM + INT8 + FP16 + IST CACHE"
echo "================================================================================"
echo "RUN_ID=$RUN_ID"
echo "OUT_DIR=$OUT_DIR"
echo "RESULT_DIR=$RESULT_DIR"

# =============================================================================
# 1. Basic checks
# =============================================================================

echo
echo "================================================================================"
echo "STEP 1 — BASIC CHECKS"
echo "================================================================================"

for P in \
  "$SOURCE_REPO/.git" \
  "$BOPTK_DRIVE" \
  "$CACHE/gp_py" \
  "$CACHE/gp_env_drive.sh" \
  "$CACHE/nvidia_egl/enable_nvidia_egl.sh" \
  "$CHECKPOINT" \
  "$RAW_DET_JSON" \
  "$CSM_FILTERED_JSON" \
  "$IST_CACHE_SOURCE" \
  "$TORCHAO_SITE" \
  "$ROOT/datasets/lmo/test_targets_bop19.json" \
  "$ROOT/datasets/lmo/models_eval/models_info.json"
do
  test -e "$P" || {
    echo "ERROR: missing required path:"
    echo "$P"
    exit 1
  }
done

CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
RAW_DET_SHA="$(sha256sum "$RAW_DET_JSON" | awk '{print $1}')"
CSM_SHA="$(sha256sum "$CSM_FILTERED_JSON" | awk '{print $1}')"

echo "Checkpoint SHA: $CHECKPOINT_SHA"
echo "Raw det SHA:    $RAW_DET_SHA"
echo "CSM det SHA:    $CSM_SHA"

[[ "$CHECKPOINT_SHA" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  echo "ERROR: checkpoint SHA mismatch"
  exit 1
}

[[ "$RAW_DET_SHA" == "$EXPECTED_RAW_DET_SHA256" ]] || {
  echo "ERROR: active detection JSON is not raw original"
  exit 1
}

[[ "$CSM_SHA" == "$EXPECTED_CSM_SHA256" ]] || {
  echo "ERROR: CSM filtered detection SHA mismatch"
  exit 1
}

ACTIVE="$(
  pgrep -af "python.*(test.py|refine.py|eval_bop19_pose.py|eval_calc_errors.py|eval_calc_scores.py)" || true
)"
if [[ -n "$ACTIVE" ]]; then
  echo "ERROR: another inference/eval process is active:"
  echo "$ACTIVE"
  exit 1
fi

echo "Basic checks passed"

# =============================================================================
# 2. Create clean worktree, copy IST-cache source, append explicit INT8/FP16 patch
# =============================================================================

echo
echo "================================================================================"
echo "STEP 2 — CREATE WORKTREE AND PATCH GIGAPOSE EXPLICITLY"
echo "================================================================================"

git -C "$SOURCE_REPO" worktree remove --force "$EXP_REPO" 2>/dev/null || true
rm -rf "$EXP_REPO"
git -C "$SOURCE_REPO" worktree prune
git -C "$SOURCE_REPO" worktree add --detach "$EXP_REPO" "$EXPECTED_COMMIT"

GIGAPOSE_FILE="$EXP_REPO/src/models/gigaPose.py"

cp -f "$IST_CACHE_SOURCE" "$GIGAPOSE_FILE"

for MARKER in \
  "GP_IST_QUERY_CACHE" \
  "GP_IST_QUERY_CACHE_PROFILE_JSONL" \
  "_profiled_ist_query_forward"
do
  grep -Fq "$MARKER" "$GIGAPOSE_FILE" || {
    echo "ERROR: IST-cache marker missing:"
    echo "$MARKER"
    exit 1
  }
done

cat >> "$GIGAPOSE_FILE" <<'PY'

# =============================================================================
# GP_COMBO_EXPLICIT_INT8_FP16_PATCH
#
# This explicit patch is appended after class GigaPose is defined.
# It applies INT8/FP16 at on_test_start(), after checkpoint loading and device move.
# =============================================================================

def _gp_combo_write_report(report):
    import json
    import os
    from pathlib import Path

    path = os.environ.get("GP_COMBO_EXPLICIT_REPORT_JSON", "")
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gp_combo_count_linear(module):
    import torch
    return sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))


def _gp_combo_first_param_device_dtype(module):
    for p in module.parameters(recurse=True):
        return str(p.device), str(p.dtype)
    return None, None


def _gp_combo_find_dinov2_target(model):
    candidates = []

    try:
        target = model.ae_net.dinov2_model
        candidates.append(("self.ae_net.dinov2_model", target))
    except Exception:
        pass

    try:
        for name, mod in model.named_modules():
            lname = name.lower()
            if "dinov2" in lname or "dino" in lname:
                candidates.append((name, mod))
    except Exception:
        pass

    if not candidates:
        return None, None

    # Prefer target with most Linear modules.
    candidates = sorted(candidates, key=lambda x: _gp_combo_count_linear(x[1]), reverse=True)
    return candidates[0]


def _gp_combo_apply_int8_fp16(model):
    import os
    import time
    import traceback
    import torch

    report = {
        "patch_name": "GP_COMBO_EXPLICIT_INT8_FP16_PATCH",
        "pid": os.getpid(),
        "time": time.time(),
        "enable_int8": os.environ.get("GP_COMBO_EXPLICIT_INT8", "0") == "1",
        "enable_fp16": os.environ.get("GP_COMBO_EXPLICIT_FP16", "0") == "1",
        "on_test_start_called": True,
        "target_found": False,
        "target_name": None,
        "target_class": None,
        "target_device_before": None,
        "target_dtype_before": None,
        "target_device_after": None,
        "target_dtype_after": None,
        "linear_count_before": None,
        "linear_count_after": None,
        "int8_applied": False,
        "int8_error": None,
        "fp16_wrapped": False,
        "fp16_forward_features_calls": 0,
        "fp16_forward_calls": 0,
        "errors": [],
    }

    # Avoid double patching.
    if getattr(model, "_gp_combo_explicit_patch_done", False):
        report["already_patched"] = True
        existing = getattr(model, "_gp_combo_explicit_patch_report", report)
        _gp_combo_write_report(existing)
        return

    target_name, target = _gp_combo_find_dinov2_target(model)

    if target is None:
        report["errors"].append("Could not locate DINOv2 target module.")
        model._gp_combo_explicit_patch_report = report
        _gp_combo_write_report(report)
        raise RuntimeError("GP combo explicit patch failed: DINOv2 target not found")

    report["target_found"] = True
    report["target_name"] = target_name
    report["target_class"] = target.__class__.__name__
    report["linear_count_before"] = _gp_combo_count_linear(target)
    dev, dtype = _gp_combo_first_param_device_dtype(target)
    report["target_device_before"] = dev
    report["target_dtype_before"] = dtype

    # 1) INT8 weight-only quantization.
    if report["enable_int8"]:
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(target, int8_weight_only())
            report["int8_applied"] = True
        except Exception as e:
            report["int8_error"] = repr(e)
            report["errors"].append(traceback.format_exc()[-3000:])

    # 2) FP16 autocast wrapping for forward_features.
    if report["enable_fp16"]:
        if hasattr(target, "forward_features") and callable(getattr(target, "forward_features")):
            original_forward_features = target.forward_features

            def _gp_combo_fp16_forward_features(*args, **kwargs):
                rep = getattr(model, "_gp_combo_explicit_patch_report", report)
                rep["fp16_forward_features_calls"] = int(rep.get("fp16_forward_features_calls", 0)) + 1
                model._gp_combo_explicit_patch_report = rep

                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = original_forward_features(*args, **kwargs)
                else:
                    out = original_forward_features(*args, **kwargs)

                _gp_combo_write_report(rep)
                return out

            target.forward_features = _gp_combo_fp16_forward_features
            report["fp16_wrapped"] = True

        elif hasattr(target, "forward") and callable(getattr(target, "forward")):
            original_forward = target.forward

            def _gp_combo_fp16_forward(*args, **kwargs):
                rep = getattr(model, "_gp_combo_explicit_patch_report", report)
                rep["fp16_forward_calls"] = int(rep.get("fp16_forward_calls", 0)) + 1
                model._gp_combo_explicit_patch_report = rep

                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = original_forward(*args, **kwargs)
                else:
                    out = original_forward(*args, **kwargs)

                _gp_combo_write_report(rep)
                return out

            target.forward = _gp_combo_fp16_forward
            report["fp16_wrapped"] = True

        else:
            report["errors"].append("Target has no callable forward_features or forward.")

    report["linear_count_after"] = _gp_combo_count_linear(target)
    dev, dtype = _gp_combo_first_param_device_dtype(target)
    report["target_device_after"] = dev
    report["target_dtype_after"] = dtype

    model._gp_combo_explicit_patch_done = True
    model._gp_combo_explicit_patch_report = report
    _gp_combo_write_report(report)

    if report["enable_int8"] and not report["int8_applied"]:
        raise RuntimeError("GP combo explicit patch failed: INT8 was requested but not applied")

    if report["enable_fp16"] and not report["fp16_wrapped"]:
        raise RuntimeError("GP combo explicit patch failed: FP16 was requested but not wrapped")


_GP_COMBO_ORIG_ON_TEST_START = getattr(GigaPose, "on_test_start", None)

def _gp_combo_on_test_start(self):
    if _GP_COMBO_ORIG_ON_TEST_START is not None:
        _GP_COMBO_ORIG_ON_TEST_START(self)
    _gp_combo_apply_int8_fp16(self)

GigaPose.on_test_start = _gp_combo_on_test_start
PY

grep -Fq "GP_COMBO_EXPLICIT_INT8_FP16_PATCH" "$GIGAPOSE_FILE" || {
  echo "ERROR: explicit combo patch not appended"
  exit 1
}

"$PY" -m py_compile "$GIGAPOSE_FILE"

cp -f "$GIGAPOSE_FILE" "$SOURCE_DIR/gigaPose_combo_explicit_used.py"
git -C "$EXP_REPO" diff > "$SOURCE_DIR/source.diff"
git -C "$EXP_REPO" diff --stat

# =============================================================================
# 3. Restore environment
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3 — RESTORE RUNTIME ENVIRONMENT"
echo "================================================================================"

cp -f "$CACHE/gp_py" "$PY"
chmod +x "$PY"

unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV LD_PRELOAD || true
unset DISPLAY || true
unset LIBGL_ALWAYS_SOFTWARE || true
unset GALLIUM_DRIVER || true
unset MESA_LOADER_DRIVER_OVERRIDE || true

source "$CACHE/gp_env_drive.sh"
source "$CACHE/nvidia_egl/enable_nvidia_egl.sh"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP=egl

export BOP_DATASETS_DIR="$ROOT/datasets"
export BOP_RESULTS_DIR="$ROOT/results"
export BOP_LOGS_DIR="$ROOT/logs"
export TORCH_HOME="$ROOT/.cache/torch"

export PYTHONPATH="$TORCHAO_SITE:$EXP_REPO:$EXP_REPO/src:$BOPTK_DRIVE:$BOPTK_DRIVE/bop_toolkit_lib:$CACHE/gp_site_py312:$ENV/lib/$PYVER/site-packages"

# Explicit combo env
export GP_COMBO_EXPLICIT_INT8="1"
export GP_COMBO_EXPLICIT_FP16="1"
export GP_COMBO_EXPLICIT_REPORT_JSON="$PATCH_REPORT_JSON"

# IST cache env
export GP_IST_QUERY_CACHE="1"
export GP_IST_QUERY_CACHE_PROFILE_JSONL="$IST_PROFILE_JSONL"

"$PY" -V
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

"$PY" - <<'PY'
import inspect
from pathlib import Path
import torch
from src.models.gigaPose import GigaPose

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("GigaPose source:", Path(inspect.getsourcefile(GigaPose)).resolve())
print("has patched on_test_start:", hasattr(GigaPose, "on_test_start"))
PY

# =============================================================================
# 4. Swap CSM detections and run inference
# =============================================================================

echo
echo "================================================================================"
echo "STEP 4 — SWAP CSM DETECTIONS AND RUN INFERENCE"
echo "================================================================================"

cp -f "$RAW_DET_JSON" "$BACKUP_DET_JSON"
cp -f "$CSM_FILTERED_JSON" "$RAW_DET_JSON"

echo "Active detection JSON:"
sha256sum "$RAW_DET_JSON"

INFER_START="$(date +%s)"

set +e
(
  cd "$EXP_REPO"

  timeout --signal=TERM --kill-after=120s 6h \
  "$PY" -u test.py \
    test_dataset_name=lmo \
    machine.root_dir="$ROOT" \
    model=large \
    model.checkpoint_path="$CHECKPOINT" \
    model.refiner=null \
    run_id="$RUN_ID" \
    use_multiple=true \
    machine.trainer.num_sanity_val_steps=0 \
    ++machine.trainer.strategy=auto \
    ++machine.trainer.precision=32 \
    ++machine.trainer.limit_test_batches=1.0
) > "$INFER_LOG" 2>&1
INFER_RC=$?
set -e

INFER_END="$(date +%s)"
INFER_WALL_SECONDS=$((INFER_END - INFER_START))

echo "INFER_RC=$INFER_RC"
echo "INFER_WALL_SECONDS=$INFER_WALL_SECONDS"

tail -n 240 "$INFER_LOG" || true

restore_raw_detection
rm -f "$BACKUP_DET_JSON" || true
trap - EXIT

if [[ "$INFER_RC" -ne 0 ]]; then
  echo "ERROR: inference failed"
  exit "$INFER_RC"
fi

if grep -q "Traceback (most recent call last)" "$INFER_LOG"; then
  echo "ERROR: traceback detected in inference log"
  exit 1
fi

MAIN_CSV="$(find "$PRED_DIR" -maxdepth 1 -type f -name "*.csv" ! -name "*MultiHypothesis.csv" | sort | tail -n 1 || true)"
MULTI_CSV="$(find "$PRED_DIR" -maxdepth 1 -type f -name "*MultiHypothesis.csv" | sort | tail -n 1 || true)"

test -s "$MAIN_CSV" || { echo "ERROR: main CSV missing"; exit 1; }
test -s "$MULTI_CSV" || { echo "ERROR: multi CSV missing"; exit 1; }
test -s "$IST_PROFILE_JSONL" || { echo "ERROR: IST profile missing"; exit 1; }
test -s "$PATCH_REPORT_JSON" || { echo "ERROR: explicit patch report missing"; exit 1; }

echo "MAIN_CSV=$MAIN_CSV"
echo "MULTI_CSV=$MULTI_CSV"
echo "IST_PROFILE_JSONL=$IST_PROFILE_JSONL"
echo "PATCH_REPORT_JSON=$PATCH_REPORT_JSON"

sha256sum "$MAIN_CSV" "$MULTI_CSV" "$IST_PROFILE_JSONL" "$PATCH_REPORT_JSON"

# =============================================================================
# 5. Validate output, IST cache, explicit INT8/FP16 patch
# =============================================================================

echo
echo "================================================================================"
echo "STEP 5 — VALIDATE OUTPUT + PATCH ACTIVITY"
echo "================================================================================"

export MAIN_CSV MULTI_CSV IST_PROFILE_JSONL PATCH_REPORT_JSON VALIDATION_JSON
export RUN_ID CSM_SOURCE_RUN_ID CSM_FILTERED_JSON INFER_WALL_SECONDS

"$PY" - <<'PY'
from pathlib import Path
import os
import json
import pandas as pd

main_csv = Path(os.environ["MAIN_CSV"])
multi_csv = Path(os.environ["MULTI_CSV"])
profile_jsonl = Path(os.environ["IST_PROFILE_JSONL"])
report_json = Path(os.environ["PATCH_REPORT_JSON"])
validation_json = Path(os.environ["VALIDATION_JSON"])

main = pd.read_csv(main_csv)
multi = pd.read_csv(multi_csv)

required = {"scene_id", "im_id", "obj_id", "score", "R", "t", "time"}
group_cols = ["scene_id", "im_id", "obj_id"]

for name, df in [("main", main), ("multi", multi)]:
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {name} missing columns: {sorted(missing)}")

    images = df[["scene_id", "im_id"]].drop_duplicates().shape[0]
    targets = df[group_cols].drop_duplicates().shape[0]
    objects = sorted(df["obj_id"].astype(int).unique().tolist())
    time_img = float(df.groupby(["scene_id", "im_id"])["time"].first().mean())

    print()
    print("=" * 80)
    print(name)
    print("rows:", len(df))
    print("images:", images)
    print("targets:", targets)
    print("objects:", objects)
    print(f"time/img: {time_img:.6f}")
    print("zero scores:", int((df["score"] == 0).sum()))
    print("textual_nan_R:", int(df["R"].astype(str).str.contains("nan", case=False).sum()))
    print("textual_nan_t:", int(df["t"].astype(str).str.contains("nan", case=False).sum()))

    if images != 200:
        raise SystemExit(f"ERROR: {name} expected 200 images, got {images}")
    if targets != 1445:
        raise SystemExit(f"ERROR: {name} expected 1445 targets, got {targets}")
    if set(objects) != {1,5,6,8,9,10,11,12}:
        raise SystemExit(f"ERROR: {name} wrong object IDs: {objects}")
    if df["R"].astype(str).str.contains("nan", case=False).any():
        raise SystemExit(f"ERROR: textual nan in {name} R")
    if df["t"].astype(str).str.contains("nan", case=False).any():
        raise SystemExit(f"ERROR: textual nan in {name} t")

if len(main) != 1445:
    raise SystemExit(f"ERROR: expected 1445 main rows, got {len(main)}")
if len(multi) != 7225:
    raise SystemExit(f"ERROR: expected 7225 multi rows, got {len(multi)}")

sizes = multi.groupby(group_cols).size()
if sizes.min() != 5 or sizes.max() != 5:
    raise SystemExit(f"ERROR: expected 5 hypotheses per target, got min={sizes.min()} max={sizes.max()}")

profile_rows = []
for line in profile_jsonl.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        profile_rows.append(json.loads(line))

profile = pd.DataFrame(profile_rows)

if len(profile) != 200:
    raise SystemExit(f"ERROR: expected 200 IST profile rows, got {len(profile)}")

ist_calls = sorted(profile["ist_query_forward_calls"].astype(int).unique().tolist())
hyp_values = sorted(profile["hypotheses"].astype(int).unique().tolist())
cache_values = sorted(profile["cache_enabled"].astype(bool).unique().tolist())

print()
print("IST profile")
print("rows:", len(profile))
print("cache_enabled:", cache_values)
print("IST query forwards/image:", ist_calls)
print("hypotheses:", hyp_values)

if cache_values != [True]:
    raise SystemExit(f"ERROR: IST cache not enabled: {cache_values}")
if ist_calls != [1]:
    raise SystemExit(f"ERROR: expected IST forwards/image [1], got {ist_calls}")
if hyp_values != [5]:
    raise SystemExit(f"ERROR: expected hypotheses [5], got {hyp_values}")

report = json.loads(report_json.read_text(encoding="utf-8"))

print()
print("Explicit patch report")
for k in [
    "patch_name",
    "on_test_start_called",
    "target_found",
    "target_name",
    "target_class",
    "target_device_before",
    "target_dtype_before",
    "target_device_after",
    "target_dtype_after",
    "linear_count_before",
    "linear_count_after",
    "enable_int8",
    "int8_applied",
    "int8_error",
    "enable_fp16",
    "fp16_wrapped",
    "fp16_forward_features_calls",
    "fp16_forward_calls",
    "errors",
]:
    print(f"{k}: {report.get(k)}")

if report.get("on_test_start_called") is not True:
    raise SystemExit("ERROR: explicit on_test_start patch was not called")
if report.get("target_found") is not True:
    raise SystemExit("ERROR: DINOv2 target not found")
if report.get("enable_int8") is not True:
    raise SystemExit("ERROR: INT8 not enabled")
if report.get("enable_fp16") is not True:
    raise SystemExit("ERROR: FP16 not enabled")
if report.get("int8_applied") is not True:
    raise SystemExit(f"ERROR: INT8 was not applied: {report.get('int8_error')}")
if report.get("fp16_wrapped") is not True:
    raise SystemExit("ERROR: FP16 was not wrapped")
if int(report.get("fp16_forward_features_calls") or 0) <= 0 and int(report.get("fp16_forward_calls") or 0) <= 0:
    raise SystemExit("ERROR: no FP16 forward calls were recorded")

time_img = float(main.groupby(["scene_id", "im_id"])["time"].first().mean())

validation = {
    "run_id": os.environ["RUN_ID"],
    "csm_source_run_id": os.environ["CSM_SOURCE_RUN_ID"],
    "csm_filtered_json": os.environ["CSM_FILTERED_JSON"],
    "main_rows": int(len(main)),
    "multi_rows": int(len(multi)),
    "images": int(main[["scene_id", "im_id"]].drop_duplicates().shape[0]),
    "targets": int(main[group_cols].drop_duplicates().shape[0]),
    "objects": sorted(main["obj_id"].astype(int).unique().tolist()),
    "time_per_image_s": time_img,
    "zero_scores_main": int((main["score"] == 0).sum()),
    "zero_scores_multi": int((multi["score"] == 0).sum()),
    "ist_profile_rows": int(len(profile)),
    "ist_query_forward_calls_values": ist_calls,
    "hypotheses_values": hyp_values,
    "cache_enabled_values": cache_values,
    "explicit_patch_report": report,
    "inference_wall_seconds": int(os.environ["INFER_WALL_SECONDS"]),
}

validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print()
print("Validation passed")
PY

# =============================================================================
# 6. Prepare BOP eval environment
# =============================================================================

echo
echo "================================================================================"
echo "STEP 6 — PREPARE BOP EVAL ENVIRONMENT"
echo "================================================================================"

unset GP_COMBO_EXPLICIT_INT8 || true
unset GP_COMBO_EXPLICIT_FP16 || true
unset GP_COMBO_EXPLICIT_REPORT_JSON || true
unset GP_IST_QUERY_CACHE || true
unset GP_IST_QUERY_CACHE_PROFILE_JSONL || true

rm -rf "$LOCAL_BOPTK" "$EVAL_SITE" "$BIN_DIR"
cp -a "$BOPTK_DRIVE" "$LOCAL_BOPTK"

mkdir -p "$EVAL_SITE" "$BIN_DIR"

for pkg in joblib transforms3d; do
  if [[ -d "$CACHE/gp_site_py312/$pkg" ]]; then
    cp -a "$CACHE/gp_site_py312/$pkg" "$EVAL_SITE/"
    echo "copied $pkg"
  fi
done

for item in OpenGL vispy freetype hsluv.py; do
  if [[ -e "$CACHE/bop_eval_site_py312/$item" ]]; then
    cp -a "$CACHE/bop_eval_site_py312/$item" "$EVAL_SITE/"
    echo "copied $item"
  fi
done

cat > "$BIN_DIR/python" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$EVAL_SITE:$LOCAL_BOPTK:$LOCAL_BOPTK/scripts:$LOCAL_BOPTK/bop_toolkit_lib:$CACHE/bop_eval_site_py312:$CACHE/gp_site_py312:$ENV/lib/$PYVER/site-packages:\${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV/lib:\${LD_LIBRARY_PATH:-}"

if [[ -s "$ENV/lib/libstdc++.so.6" ]] && [[ -s "$ENV/lib/libgcc_s.so.1" ]]; then
  export LD_PRELOAD="$ENV/lib/libstdc++.so.6:$ENV/lib/libgcc_s.so.1"
fi

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP=egl

unset DISPLAY
unset LIBGL_ALWAYS_SOFTWARE
unset GALLIUM_DRIVER
unset MESA_LOADER_DRIVER_OVERRIDE

exec "$PY" "\$@"
EOF

chmod +x "$BIN_DIR/python"
ln -sfn "$BIN_DIR/python" "$BIN_DIR/python3"

export PATH="$BIN_DIR:$PATH"
export PYTHONPATH="$EVAL_SITE:$LOCAL_BOPTK:$LOCAL_BOPTK/scripts:$LOCAL_BOPTK/bop_toolkit_lib:$CACHE/bop_eval_site_py312:$CACHE/gp_site_py312:$ENV/lib/$PYVER/site-packages:${PYTHONPATH:-}"

CONFIG="$LOCAL_BOPTK/bop_toolkit_lib/config.py"

export ROOT LOCAL_RESULTS_DIR EVAL_DIR CONFIG

"$PY" - <<'PY'
from pathlib import Path
import os
import re

config = Path(os.environ["CONFIG"])
text = config.read_text(encoding="utf-8")

values = {
    "datasets_path": str(Path(os.environ["ROOT"]) / "datasets"),
    "results_path": os.environ["LOCAL_RESULTS_DIR"],
    "eval_path": os.environ["EVAL_DIR"],
}

for name, value in values.items():
    pattern = rf"(?m)^{re.escape(name)}\s*=.*$"
    repl = f'{name} = r"{value}"'
    if re.search(pattern, text):
        text = re.sub(pattern, repl, text)
    else:
        text += f"\n{repl}\n"

config.write_text(text, encoding="utf-8")

print("Patched config.py")
for k, v in values.items():
    print(f"  {k} = {v}")
PY

python - <<'PY'
from pathlib import Path
from bop_toolkit_lib import config

print("BOP config.datasets_path:", config.datasets_path)
assert Path(config.datasets_path, "lmo", "models_eval", "models_info.json").exists()
assert Path(config.datasets_path, "lmo", "test_targets_bop19.json").exists()
print("BOP dataset paths verified")
PY

python - <<'PY'
from vispy import app, gloo

canvas = app.Canvas(show=False, app="egl")
print("GL renderer:", gloo.gl.glGetParameter(gloo.gl.GL_RENDERER))
print("GL vendor:", gloo.gl.glGetParameter(gloo.gl.GL_VENDOR))
print("GL version:", gloo.gl.glGetParameter(gloo.gl.GL_VERSION))
canvas.close()
print("NVIDIA EGL renderer verified")
PY

# =============================================================================
# 7. Prepare BOP result CSV
# =============================================================================

echo
echo "================================================================================"
echo "STEP 7 — PREPARE BOP RESULT CSV"
echo "================================================================================"

cp -f "$MAIN_CSV" "$BOP_RESULT_CSV"

echo "BOP_RESULT_FILENAME=$BOP_RESULT_FILENAME"
echo "BOP_RESULT_CSV=$BOP_RESULT_CSV"
sha256sum "$BOP_RESULT_CSV"

export BOP_RESULT_FILENAME BOP_RESULT_CSV

python - <<'PY'
from pathlib import Path
import os
import pandas as pd
from bop_toolkit_lib import inout

fn = os.environ["BOP_RESULT_FILENAME"]
csv_path = Path(os.environ["BOP_RESULT_CSV"])

if "lmo-test" not in fn:
    raise SystemExit("ERROR: filename does not contain lmo-test")
if "tless" in fn.lower():
    raise SystemExit("ERROR: wrong T-LESS filename")

df = pd.read_csv(csv_path)
print("BOP CSV rows:", len(df))
print("BOP CSV images:", df[["scene_id", "im_id"]].drop_duplicates().shape[0])
print("BOP CSV time/img:", df.groupby(["scene_id", "im_id"])["time"].first().mean())

if len(df) != 1445:
    raise SystemExit(f"ERROR: expected 1445 rows, got {len(df)}")

if hasattr(inout, "parse_result_filename"):
    print("Parsed filename:", inout.parse_result_filename(fn))

print("BOP result CSV precheck passed")
PY

# =============================================================================
# 8. Official BOP19 evaluation
# =============================================================================

echo
echo "================================================================================"
echo "STEP 8 — RUN OFFICIAL BOP19 EVALUATION"
echo "================================================================================"

EVAL_START="$(date +%s)"

set +e
timeout --signal=TERM --kill-after=120s 120m \
python "$LOCAL_BOPTK/scripts/eval_bop19_pose.py" \
  --result_filenames="$BOP_RESULT_FILENAME" \
  --results_path="$LOCAL_RESULTS_DIR" \
  --eval_path="$EVAL_DIR" \
  --targets_filename=test_targets_bop19.json \
  --renderer_type=vispy \
  --num_workers=1 \
  > "$EVAL_LOG" 2>&1
EVAL_RC=$?
set -e

EVAL_END="$(date +%s)"
EVAL_WALL_SECONDS=$((EVAL_END - EVAL_START))

tail -n 220 "$EVAL_LOG" || true

echo
echo "EVAL_RC=$EVAL_RC"
echo "EVAL_WALL_SECONDS=$EVAL_WALL_SECONDS"

if [[ "$EVAL_RC" -ne 0 ]]; then
  echo "ERROR: BOP19 evaluation failed"
  exit "$EVAL_RC"
fi

if grep -q "Traceback (most recent call last)" "$EVAL_LOG"; then
  echo "ERROR: Traceback detected in eval log"
  exit 1
fi

# =============================================================================
# 9. Parse metrics and write summary
# =============================================================================

echo
echo "================================================================================"
echo "STEP 9 — WRITE SUMMARY"
echo "================================================================================"

export EVAL_LOG SUMMARY_JSON SUMMARY_TXT MAIN_CSV MULTI_CSV VALIDATION_JSON PATCH_REPORT_JSON
export RUN_ID CSM_SOURCE_RUN_ID CSM_FILTERED_JSON INFER_WALL_SECONDS EVAL_WALL_SECONDS
export FP32_BASELINE_AR_PERCENT FP32_BASELINE_TIME
export IST_CACHE_AR_PERCENT IST_CACHE_TIME
export CSM_V1_FP16_AR_PERCENT CSM_V1_FP16_TIME
export CSM_IST_UNVERIFIED_TIME

"$PY" - <<'PY'
from pathlib import Path
import os
import re
import json
import pandas as pd

eval_log = Path(os.environ["EVAL_LOG"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_txt = Path(os.environ["SUMMARY_TXT"])
main_csv = Path(os.environ["MAIN_CSV"])

text = eval_log.read_text(encoding="utf-8", errors="replace")

def find_metric(key):
    vals = re.findall(rf"{re.escape(key)}:\s*([-+0-9.eE]+)", text)
    if not vals:
        return None
    return float(vals[-1])

ar_vsd = find_metric("bop19_average_recall_vsd")
ar_mssd = find_metric("bop19_average_recall_mssd")
ar_mspd = find_metric("bop19_average_recall_mspd")
ar = find_metric("bop19_average_recall")
time_img = find_metric("bop19_average_time_per_image")

if ar is None:
    raise SystemExit("ERROR: could not parse bop19_average_recall")
if ar_vsd is None or ar_mssd is None or ar_mspd is None:
    raise SystemExit("ERROR: could not parse VSD/MSSD/MSPD")
if time_img is None:
    df = pd.read_csv(main_csv)
    time_img = float(df.groupby(["scene_id", "im_id"])["time"].first().mean())

ar_percent = ar * 100.0
vsd_percent = ar_vsd * 100.0
mssd_percent = ar_mssd * 100.0
mspd_percent = ar_mspd * 100.0

fp32_ar = float(os.environ["FP32_BASELINE_AR_PERCENT"])
fp32_time = float(os.environ["FP32_BASELINE_TIME"])

ist_ar = float(os.environ["IST_CACHE_AR_PERCENT"])
ist_time = float(os.environ["IST_CACHE_TIME"])

csm_fp16_ar = float(os.environ["CSM_V1_FP16_AR_PERCENT"])
csm_fp16_time = float(os.environ["CSM_V1_FP16_TIME"])

csm_ist_unverified_time = float(os.environ["CSM_IST_UNVERIFIED_TIME"])

def comp(base_ar, base_time):
    return {
        "baseline_AR_percent": base_ar,
        "baseline_time_per_image_s": base_time,
        "AR_change_percentage_points": ar_percent - base_ar,
        "runtime_reduction_percent": (base_time - time_img) / base_time * 100.0,
        "speedup": base_time / time_img,
    }

validation = json.loads(Path(os.environ["VALIDATION_JSON"]).read_text(encoding="utf-8"))
patch_report = json.loads(Path(os.environ["PATCH_REPORT_JSON"]).read_text(encoding="utf-8"))

summary = {
    "status": "verified_csm_int8_fp16_istcache_bop19_complete",
    "run_id": os.environ["RUN_ID"],
    "dataset": "LM-O",
    "method": "CSM-v1 + INT8 + FP16 + IST query cache",
    "csm_source_run_id": os.environ["CSM_SOURCE_RUN_ID"],
    "csm_filtered_json": os.environ["CSM_FILTERED_JSON"],
    "setting": {
        "csm": True,
        "int8": True,
        "fp16": True,
        "ist_query_cache": True,
        "refinement": False,
        "hypotheses": 5,
    },
    "official_bop19": {
        "AR_percent": ar_percent,
        "AR_VSD_percent": vsd_percent,
        "AR_MSSD_percent": mssd_percent,
        "AR_MSPD_percent": mspd_percent,
        "time_per_image_s": time_img,
    },
    "comparison_vs_fp32_baseline": comp(fp32_ar, fp32_time),
    "comparison_vs_ist_cache_only": comp(ist_ar, ist_time),
    "comparison_vs_csm_v1_fp16": comp(csm_fp16_ar, csm_fp16_time),
    "comparison_vs_previous_csm_ist_unverified_time_only": {
        "previous_unverified_time_s": csm_ist_unverified_time,
        "new_time_s": time_img,
        "runtime_change_percent": (csm_ist_unverified_time - time_img) / csm_ist_unverified_time * 100.0,
    },
    "validation": validation,
    "explicit_patch_report": patch_report,
    "files": {
        "main_csv": os.environ["MAIN_CSV"],
        "multi_csv": os.environ["MULTI_CSV"],
        "eval_log": os.environ["EVAL_LOG"],
        "validation_json": os.environ["VALIDATION_JSON"],
        "patch_report_json": os.environ["PATCH_REPORT_JSON"],
        "summary_json": os.environ["SUMMARY_JSON"],
        "summary_txt": os.environ["SUMMARY_TXT"],
    },
    "wall_seconds": {
        "inference_wall_seconds": int(os.environ["INFER_WALL_SECONDS"]),
        "eval_wall_seconds": int(os.environ["EVAL_WALL_SECONDS"]),
    },
}

summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = []
lines.append("=" * 100)
lines.append("LM-O CSM-v1 + INT8 + FP16 + IST Query Cache — Official BOP19 Summary")
lines.append("=" * 100)
lines.append("")
lines.append(f"Run ID: {os.environ['RUN_ID']}")
lines.append(f"CSM source: {os.environ['CSM_SOURCE_RUN_ID']}")
lines.append("")
lines.append("Official BOP19 metrics")
lines.append(f"  AR:        {ar_percent:.4f}%")
lines.append(f"  AR_VSD:    {vsd_percent:.4f}%")
lines.append(f"  AR_MSSD:   {mssd_percent:.4f}%")
lines.append(f"  AR_MSPD:   {mspd_percent:.4f}%")
lines.append(f"  Time/img:  {time_img:.6f} s/image")
lines.append("")
lines.append("Comparison vs FP32 coarse baseline")
c = comp(fp32_ar, fp32_time)
lines.append(f"  Baseline AR:        {fp32_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {fp32_time:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Comparison vs IST cache only")
c = comp(ist_ar, ist_time)
lines.append(f"  Baseline AR:        {ist_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {ist_time:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Comparison vs CSM-v1 + FP16")
c = comp(csm_fp16_ar, csm_fp16_time)
lines.append(f"  Baseline AR:        {csm_fp16_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {csm_fp16_time:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Patch validation")
lines.append(f"  Target:                       {patch_report.get('target_name')}")
lines.append(f"  INT8 applied:                 {patch_report.get('int8_applied')}")
lines.append(f"  FP16 wrapped:                 {patch_report.get('fp16_wrapped')}")
lines.append(f"  FP16 forward_features calls:  {patch_report.get('fp16_forward_features_calls')}")
lines.append(f"  IST forwards/image:           {validation.get('ist_query_forward_calls_values')}")
lines.append("")
lines.append("Saved")
lines.append(f"  Summary JSON: {summary_json}")
lines.append(f"  Summary TXT:  {summary_txt}")
lines.append("=" * 100)

summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo
echo "================================================================================"
echo "FIXED CSM + INT8 + FP16 + IST CACHE COMPLETE"
echo "================================================================================"
cat "$SUMMARY_TXT"
