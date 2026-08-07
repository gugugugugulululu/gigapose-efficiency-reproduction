set -Eeuo pipefail

# =============================================================================
# BOP19-only resume for:
#   CSM-v1 + INT8 + FP16 + IST cache + top-3 MegaPose 5iter refinement
#
# This does NOT rerun refinement.
# It only evaluates the already generated refined_bop_top1 CSV.
# =============================================================================

ROOT="/content/drive/MyDrive/bop_workspace"
CACHE="$ROOT/runtime_cache"
PY="/content/gp_py"
ENV="/content/drive/MyDrive/mamba_root/envs/gp"
PYVER="python3.12"
BOPTK_DRIVE="$ROOT/code/bop_toolkit"

COARSE_RUN_ID="lmo_csmv1_int8_fp16_istcache_fixed_20260711_212937"
REFINE_ID="lmo_csmv1_int8_fp16_istcache_fixed_20260711_212937_top3_megapose_3hyp5iter_refine_20260712_005007"

OUT_DIR="$ROOT/results/full_pipeline_csm_int8_fp16_istcache_refine/lmo/${REFINE_ID}"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
WORK_DIR="$OUT_DIR/work"

REFINED_BOP_CSV="$WORK_DIR/${REFINE_ID}_refined_bop_top1.csv"
VALIDATION_JSON="$REPORT_DIR/${REFINE_ID}_validation.json"

LOCAL_BOPTK="/content/bop_toolkit_${REFINE_ID}_resume"
LOCAL_RESULTS_DIR="/content/bop_results_${REFINE_ID}_resume"
EVAL_SITE="/content/bop_eval_site_${REFINE_ID}_resume"
BIN_DIR="/content/bop_eval_bin_${REFINE_ID}_resume"
EVAL_DIR="$OUT_DIR/bop19_eval/${REFINE_ID}_bop19_eval_resume"

BOP_RESULT_FILENAME="csmint8fp16istcache-top3-5iter-refine_lmo-test.csv"
BOP_RESULT_CSV="$LOCAL_RESULTS_DIR/$BOP_RESULT_FILENAME"

EVAL_LOG="$LOG_DIR/${REFINE_ID}_bop19_eval_resume.log"
SUMMARY_JSON="$REPORT_DIR/${REFINE_ID}_bop19_resume_summary.json"
SUMMARY_TXT="$REPORT_DIR/${REFINE_ID}_bop19_resume_summary.txt"

REFINE_BASELINE_AR_PERCENT="60.1400"
REFINE_BASELINE_TIME="8.378500"

NO_CSM_TOP3_AR_PERCENT="59.7744"
NO_CSM_TOP3_TIME="6.358132"

CSM_FULL_5HYP5ITER_AR_PERCENT="59.9082"
CSM_FULL_5HYP5ITER_TIME="8.123629"

COARSE_AR_PERCENT="29.2353"
COARSE_TIME="0.926764"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$LOCAL_RESULTS_DIR" "$EVAL_SITE" "$BIN_DIR" "$EVAL_DIR"

echo "================================================================================"
echo "BOP19 RESUME — CSM-v1 + INT8 + FP16 + IST CACHE + 3x5 REFINEMENT"
echo "================================================================================"
echo "REFINE_ID=$REFINE_ID"
echo "REFINED_BOP_CSV=$REFINED_BOP_CSV"

for P in \
  "$CACHE/gp_py" \
  "$CACHE/gp_env_drive.sh" \
  "$CACHE/nvidia_egl/enable_nvidia_egl.sh" \
  "$BOPTK_DRIVE" \
  "$REFINED_BOP_CSV" \
  "$VALIDATION_JSON" \
  "$ROOT/datasets/lmo/test_targets_bop19.json" \
  "$ROOT/datasets/lmo/models_eval/models_info.json"
do
  test -e "$P" || {
    echo "ERROR: missing required path:"
    echo "$P"
    exit 1
  }
done

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
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP=egl

echo
echo "================================================================================"
echo "STEP 1 — VALIDATE REFINED CSV"
echo "================================================================================"

export REFINED_BOP_CSV VALIDATION_JSON

"$PY" - <<'PY'
from pathlib import Path
import os, json
import pandas as pd

csv_path = Path(os.environ["REFINED_BOP_CSV"])
val_path = Path(os.environ["VALIDATION_JSON"])

df = pd.read_csv(csv_path)
val = json.loads(val_path.read_text())

print("CSV:", csv_path)
print("rows:", len(df))
print("images:", df[["scene_id", "im_id"]].drop_duplicates().shape[0])
print("targets:", df[["scene_id", "im_id", "obj_id"]].drop_duplicates().shape[0])
print("objects:", sorted(df["obj_id"].astype(int).unique().tolist()))
print("time/img:", df.groupby(["scene_id", "im_id"])["time"].first().mean())
print("zero scores:", int((df["score"] == 0).sum()))
print("nan R:", int(df["R"].astype(str).str.contains("nan", case=False).sum()))
print("nan t:", int(df["t"].astype(str).str.contains("nan", case=False).sum()))

print()
print("validation output_mode:", val.get("output_mode"))
print("validation raw rows:", val.get("raw_rows"))
print("validation bop rows:", val.get("bop_rows"))

if len(df) != 1445:
    raise SystemExit(f"ERROR: expected 1445 rows, got {len(df)}")

sizes = df.groupby(["scene_id", "im_id", "obj_id"]).size()
if sizes.min() != 1 or sizes.max() != 1:
    raise SystemExit("ERROR: BOP CSV should contain exactly one row per target")

if df["R"].astype(str).str.contains("nan", case=False).any():
    raise SystemExit("ERROR: NaN in R")
if df["t"].astype(str).str.contains("nan", case=False).any():
    raise SystemExit("ERROR: NaN in t")

print("Refined BOP CSV precheck passed")
PY

echo
echo "================================================================================"
echo "STEP 2 — PREPARE BOP TOOLKIT"
echo "================================================================================"

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
import os, re

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

print("Patched BOP config:")
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

echo
echo "================================================================================"
echo "STEP 3 — RUN OFFICIAL BOP19 EVALUATION"
echo "================================================================================"

cp -f "$REFINED_BOP_CSV" "$BOP_RESULT_CSV"

echo "BOP_RESULT_FILENAME=$BOP_RESULT_FILENAME"
echo "BOP_RESULT_CSV=$BOP_RESULT_CSV"
sha256sum "$BOP_RESULT_CSV"

# Critical fix: export these two variables before Python reads os.environ.
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
print("Filename:", fn)
print("BOP CSV rows:", len(df))
print("BOP CSV images:", df[["scene_id", "im_id"]].drop_duplicates().shape[0])
print("BOP CSV time/img:", df.groupby(["scene_id", "im_id"])["time"].first().mean())

if len(df) != 1445:
    raise SystemExit(f"ERROR: expected 1445 rows, got {len(df)}")

if hasattr(inout, "parse_result_filename"):
    print("Parsed filename:", inout.parse_result_filename(fn))

print("BOP CSV precheck passed")
PY

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

echo
echo "================================================================================"
echo "STEP 4 — SUMMARY"
echo "================================================================================"

export EVAL_LOG SUMMARY_JSON SUMMARY_TXT VALIDATION_JSON REFINE_ID COARSE_RUN_ID
export REFINE_BASELINE_AR_PERCENT REFINE_BASELINE_TIME
export NO_CSM_TOP3_AR_PERCENT NO_CSM_TOP3_TIME
export CSM_FULL_5HYP5ITER_AR_PERCENT CSM_FULL_5HYP5ITER_TIME
export COARSE_AR_PERCENT COARSE_TIME EVAL_WALL_SECONDS

"$PY" - <<'PY'
from pathlib import Path
import os, re, json

eval_log = Path(os.environ["EVAL_LOG"])
summary_json = Path(os.environ["SUMMARY_JSON"])
summary_txt = Path(os.environ["SUMMARY_TXT"])
validation_json = Path(os.environ["VALIDATION_JSON"])

text = eval_log.read_text(encoding="utf-8", errors="replace")

def find_metric(key):
    vals = re.findall(rf"{re.escape(key)}:\s*([-+0-9.eE]+)", text)
    return float(vals[-1]) if vals else None

ar_vsd = find_metric("bop19_average_recall_vsd")
ar_mssd = find_metric("bop19_average_recall_mssd")
ar_mspd = find_metric("bop19_average_recall_mspd")
ar = find_metric("bop19_average_recall")
time_img = find_metric("bop19_average_time_per_image")

if None in [ar_vsd, ar_mssd, ar_mspd, ar, time_img]:
    raise SystemExit("ERROR: failed to parse BOP19 metrics")

ar_p = ar * 100.0
vsd_p = ar_vsd * 100.0
mssd_p = ar_mssd * 100.0
mspd_p = ar_mspd * 100.0

validation = json.loads(validation_json.read_text(encoding="utf-8"))

ref_base_ar = float(os.environ["REFINE_BASELINE_AR_PERCENT"])
ref_base_t = float(os.environ["REFINE_BASELINE_TIME"])

top3_ar = float(os.environ["NO_CSM_TOP3_AR_PERCENT"])
top3_t = float(os.environ["NO_CSM_TOP3_TIME"])

csm_full_ar = float(os.environ["CSM_FULL_5HYP5ITER_AR_PERCENT"])
csm_full_t = float(os.environ["CSM_FULL_5HYP5ITER_TIME"])

coarse_ar = float(os.environ["COARSE_AR_PERCENT"])
coarse_t = float(os.environ["COARSE_TIME"])

def comp(base_ar, base_t):
    return {
        "baseline_AR_percent": base_ar,
        "baseline_time_per_image_s": base_t,
        "AR_change_percentage_points": ar_p - base_ar,
        "runtime_reduction_percent": (base_t - time_img) / base_t * 100.0,
        "speedup": base_t / time_img,
    }

summary = {
    "status": "csm_int8_fp16_istcache_top3_5iter_refine_bop19_complete",
    "refine_id": os.environ["REFINE_ID"],
    "coarse_run_id": os.environ["COARSE_RUN_ID"],
    "method": "CSM-v1 + INT8 + FP16 + IST cache coarse + top-3 MegaPose 5-iteration refinement",
    "official_bop19": {
        "AR_percent": ar_p,
        "AR_VSD_percent": vsd_p,
        "AR_MSSD_percent": mssd_p,
        "AR_MSPD_percent": mspd_p,
        "time_per_image_s": time_img,
    },
    "comparison_vs_original_5hyp5iter_refine": comp(ref_base_ar, ref_base_t),
    "comparison_vs_nocsm_top3_5iter_refine": comp(top3_ar, top3_t),
    "comparison_vs_csm_v1_5hyp5iter_refine": comp(csm_full_ar, csm_full_t),
    "coarse_reference": {
        "coarse_AR_percent": coarse_ar,
        "coarse_time_per_image_s": coarse_t,
    },
    "validation": validation,
    "files": {
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "eval_log": str(eval_log),
        "validation_json": str(validation_json),
    },
    "eval_wall_seconds": int(os.environ["EVAL_WALL_SECONDS"]),
}

summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = []
lines.append("=" * 100)
lines.append("LM-O Full Pipeline — CSM-v1 + INT8 + FP16 + IST cache + 3hyp×5iter refinement")
lines.append("=" * 100)
lines.append(f"Refine ID:  {os.environ['REFINE_ID']}")
lines.append(f"Coarse ID:  {os.environ['COARSE_RUN_ID']}")
lines.append("")
lines.append("Official BOP19 metrics")
lines.append(f"  AR:        {ar_p:.4f}%")
lines.append(f"  AR_VSD:    {vsd_p:.4f}%")
lines.append(f"  AR_MSSD:   {mssd_p:.4f}%")
lines.append(f"  AR_MSPD:   {mspd_p:.4f}%")
lines.append(f"  Time/img:  {time_img:.6f} s/image")
lines.append("")
lines.append("Comparison vs original MegaPose 5hyp×5iter baseline")
c = comp(ref_base_ar, ref_base_t)
lines.append(f"  Baseline AR:        {ref_base_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {ref_base_t:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Comparison vs No-CSM top-3 3hyp×5iter refinement")
c = comp(top3_ar, top3_t)
lines.append(f"  Baseline AR:        {top3_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {top3_t:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Comparison vs CSM-v1 5hyp×5iter refinement")
c = comp(csm_full_ar, csm_full_t)
lines.append(f"  Baseline AR:        {csm_full_ar:.4f}%")
lines.append(f"  Baseline Time/img:  {csm_full_t:.6f} s/image")
lines.append(f"  AR change:          {c['AR_change_percentage_points']:+.4f} pp")
lines.append(f"  Runtime reduction:  {c['runtime_reduction_percent']:.2f}%")
lines.append(f"  Speedup:            {c['speedup']:.4f}x")
lines.append("")
lines.append("Saved")
lines.append(f"  Summary JSON:      {summary_json}")
lines.append(f"  Summary TXT:       {summary_txt}")
lines.append(f"  Validation JSON:   {validation_json}")
lines.append("=" * 100)

summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo
echo "================================================================================"
echo "BOP19 RESUME COMPLETE"
echo "================================================================================"
cat "$SUMMARY_TXT"


