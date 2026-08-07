set -Eeuo pipefail

# =============================================================================
# LM-O full pipeline:
#   CSM-v1 + INT8 + FP16 + IST cache coarse
#   + score-based top-3 pose hypotheses
#   + MegaPose 3hyp × 5iter refinement
#   + official BOP19 evaluation
#
# FIXED:
#   1) Restore /content/gp_py before Step 2.
#   2) Patch MegaPose dataclass mutable defaults for Python 3.12.
#   3) Patch NumPy 2.x / Panda3D compatibility:
#        - ndarray.tostring() -> ndarray.tobytes()
#        - ref.setPos(*tWR[:3]) -> flatten + scalar float list
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

COARSE_RUN_ID="lmo_csmv1_int8_fp16_istcache_fixed_20260711_212937"
COARSE_RESULT_DIR="$ROOT/results/large_${COARSE_RUN_ID}"
COARSE_PRED_DIR="$COARSE_RESULT_DIR/predictions"

COARSE_MAIN_CSV="$COARSE_PRED_DIR/large-pbrreal-rgb-mmodel_lmo-test_${COARSE_RUN_ID}.csv"
COARSE_MULTI_CSV="$COARSE_PRED_DIR/large-pbrreal-rgb-mmodel_lmo-test_${COARSE_RUN_ID}MultiHypothesis.csv"

COARSE_AR_PERCENT="29.2353"
COARSE_TIME="0.926764"

REFINE_BASELINE_AR_PERCENT="60.1400"
REFINE_BASELINE_TIME="8.378500"

NO_CSM_TOP3_AR_PERCENT="59.7744"
NO_CSM_TOP3_TIME="6.358132"

CSM_FULL_5HYP5ITER_AR_PERCENT="59.9082"
CSM_FULL_5HYP5ITER_TIME="8.123629"

TOPK=3
N_ITER=5

REFINE_ID="${COARSE_RUN_ID}_top3_megapose_3hyp5iter_refine_$(date +%Y%m%d_%H%M%S)"
EXP_REPO="/content/gigapose_${REFINE_ID}"

OUT_DIR="$ROOT/results/full_pipeline_csm_int8_fp16_istcache_refine/lmo/${REFINE_ID}"
LOG_DIR="$OUT_DIR/logs"
REPORT_DIR="$OUT_DIR/reports"
WORK_DIR="$OUT_DIR/work"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR" "$WORK_DIR"

MASTER_LOG="$LOG_DIR/${REFINE_ID}_master.log"
REFINE_LOG="$LOG_DIR/${REFINE_ID}_refine.log"
EVAL_LOG="$LOG_DIR/${REFINE_ID}_bop19_eval.log"

TOP3_MULTI_CSV="$WORK_DIR/${REFINE_ID}_top3_input_MultiHypothesis.csv"
BACKUP_MULTI_CSV="$WORK_DIR/original_${COARSE_RUN_ID}MultiHypothesis.csv"

REFINED_RAW_CSV="$WORK_DIR/${REFINE_ID}_refined_raw.csv"
REFINED_BOP_CSV="$WORK_DIR/${REFINE_ID}_refined_bop_top1.csv"

VALIDATION_JSON="$REPORT_DIR/${REFINE_ID}_validation.json"
SUMMARY_JSON="$REPORT_DIR/${REFINE_ID}_summary.json"
SUMMARY_TXT="$REPORT_DIR/${REFINE_ID}_summary.txt"

LOCAL_BOPTK="/content/bop_toolkit_${REFINE_ID}"
LOCAL_RESULTS_DIR="/content/bop_results_${REFINE_ID}"
EVAL_SITE="/content/bop_eval_site_${REFINE_ID}"
BIN_DIR="/content/bop_eval_bin_${REFINE_ID}"
EVAL_DIR="$OUT_DIR/bop19_eval/${REFINE_ID}_bop19_eval"

mkdir -p "$LOCAL_RESULTS_DIR" "$EVAL_SITE" "$BIN_DIR" "$EVAL_DIR"

BOP_RESULT_FILENAME="csmint8fp16istcache-top3-5iter-refine_lmo-test.csv"
BOP_RESULT_CSV="$LOCAL_RESULTS_DIR/$BOP_RESULT_FILENAME"

exec > >(tee -a "$MASTER_LOG") 2>&1

restore_multi() {
  if [[ -s "$BACKUP_MULTI_CSV" ]]; then
    cp -f "$BACKUP_MULTI_CSV" "$COARSE_MULTI_CSV"
    echo
    echo "Restored original MultiHypothesis CSV:"
    sha256sum "$COARSE_MULTI_CSV" || true
  fi
}
trap restore_multi EXIT

echo "================================================================================"
echo "CSM-v1 + INT8 + FP16 + IST CACHE + 3x5 REFINEMENT"
echo "================================================================================"
echo "COARSE_RUN_ID=$COARSE_RUN_ID"
echo "REFINE_ID=$REFINE_ID"
echo "OUT_DIR=$OUT_DIR"

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
  "$COARSE_MAIN_CSV" \
  "$COARSE_MULTI_CSV" \
  "$ROOT/datasets/lmo/test_targets_bop19.json" \
  "$ROOT/datasets/lmo/models_eval/models_info.json"
do
  test -e "$P" || {
    echo "ERROR: missing required path:"
    echo "$P"
    exit 1
  }
done

ACTIVE="$(
  pgrep -af "python.*(test.py|refine.py|eval_bop19_pose.py|eval_calc_errors.py|eval_calc_scores.py)" || true
)"
if [[ -n "$ACTIVE" ]]; then
  echo "ERROR: another inference/eval process is active:"
  echo "$ACTIVE"
  exit 1
fi

echo "Coarse main CSV:"
sha256sum "$COARSE_MAIN_CSV"
echo "Coarse multi CSV:"
sha256sum "$COARSE_MULTI_CSV"

# =============================================================================
# 1.5 Restore Python wrapper early
# =============================================================================

echo
echo "================================================================================"
echo "STEP 1.5 — RESTORE PYTHON WRAPPER EARLY"
echo "================================================================================"

cp -f "$CACHE/gp_py" "$PY"
chmod +x "$PY"

unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV LD_PRELOAD || true

source "$CACHE/gp_env_drive.sh"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export PYTHONPATH="$CACHE/gp_site_py312:$ENV/lib/$PYVER/site-packages:${PYTHONPATH:-}"

"$PY" -V
echo "Python wrapper ready: $PY"

# =============================================================================
# 2. Validate coarse CSV and create top-3 MultiHypothesis input
# =============================================================================

echo
echo "================================================================================"
echo "STEP 2 — CREATE TOP-3 MULTIHYPOTHESIS INPUT"
echo "================================================================================"

export COARSE_MAIN_CSV COARSE_MULTI_CSV TOP3_MULTI_CSV TOPK REFINE_ID

"$PY" - <<'PY'
from pathlib import Path
import os
import json
import pandas as pd

main_csv = Path(os.environ["COARSE_MAIN_CSV"])
multi_csv = Path(os.environ["COARSE_MULTI_CSV"])
top3_csv = Path(os.environ["TOP3_MULTI_CSV"])
topk = int(os.environ["TOPK"])

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
    if set(objects) != {1, 5, 6, 8, 9, 10, 11, 12}:
        raise SystemExit(f"ERROR: {name} wrong objects: {objects}")

if len(main) != 1445:
    raise SystemExit(f"ERROR: main expected 1445 rows, got {len(main)}")
if len(multi) != 7225:
    raise SystemExit(f"ERROR: multi expected 7225 rows, got {len(multi)}")

sizes = multi.groupby(group_cols).size()
if sizes.min() != 5 or sizes.max() != 5:
    raise SystemExit(f"ERROR: expected 5 hypotheses per target, got min={sizes.min()}, max={sizes.max()}")

multi = multi.copy()
multi["_orig_hyp_rank"] = multi.groupby(group_cols, sort=False).cumcount() + 1

multi_sorted = multi.sort_values(
    group_cols + ["score", "_orig_hyp_rank"],
    ascending=[True, True, True, False, True],
).reset_index(drop=True)

top = (
    multi_sorted
    .groupby(group_cols, sort=False, group_keys=False)
    .head(topk)
    .copy()
)

top["_kept_rank"] = top.groupby(group_cols, sort=False).cumcount() + 1

sizes_top = top.groupby(group_cols).size()
if len(sizes_top) != 1445:
    raise SystemExit(f"ERROR: top-{topk} targets expected 1445, got {len(sizes_top)}")
if sizes_top.min() != topk or sizes_top.max() != topk:
    raise SystemExit(f"ERROR: top-{topk} hyp/target invalid min={sizes_top.min()} max={sizes_top.max()}")

drop_cols = [c for c in ["_orig_hyp_rank", "_kept_rank"] if c in top.columns]
top_out = top.drop(columns=drop_cols)

top3_csv.parent.mkdir(parents=True, exist_ok=True)
top_out.to_csv(top3_csv, index=False)

summary = {
    "refine_id": os.environ["REFINE_ID"],
    "topk": topk,
    "input_multi_rows": int(len(multi)),
    "output_topk_rows": int(len(top_out)),
    "targets": int(len(sizes_top)),
    "hypotheses_per_target_min": int(sizes_top.min()),
    "hypotheses_per_target_max": int(sizes_top.max()),
    "mean_input_score": float(multi["score"].mean()),
    "mean_topk_score": float(top_out["score"].mean()),
    "median_topk_score": float(top_out["score"].median()),
    "topk_csv": str(top3_csv),
}

Path(str(top3_csv) + ".summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print()
print("=" * 80)
print("Top-k summary")
for k, v in summary.items():
    print(f"{k}: {v}")
PY

echo
echo "Top-3 CSV:"
sha256sum "$TOP3_MULTI_CSV"

cp -f "$COARSE_MULTI_CSV" "$BACKUP_MULTI_CSV"
cp -f "$TOP3_MULTI_CSV" "$COARSE_MULTI_CSV"

echo
echo "Active MultiHypothesis CSV replaced by top-3:"
sha256sum "$COARSE_MULTI_CSV"

# =============================================================================
# 3. Prepare clean repo and environment for MegaPose refinement
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3 — PREPARE REFINEMENT ENVIRONMENT"
echo "================================================================================"

git -C "$SOURCE_REPO" worktree remove --force "$EXP_REPO" 2>/dev/null || true
rm -rf "$EXP_REPO"
git -C "$SOURCE_REPO" worktree prune
git -C "$SOURCE_REPO" worktree add --detach "$EXP_REPO" "$EXPECTED_COMMIT"

# =============================================================================
# 3.5 Patch MegaPose dataclass mutable defaults for Python 3.12
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3.5 — PATCH MEGAPOSE DATACLASS DEFAULTS FOR PYTHON 3.12"
echo "================================================================================"

TRAINING_CONFIG="$EXP_REPO/src/megapose/training/training_config.py"

test -s "$TRAINING_CONFIG" || {
  echo "ERROR: missing training_config.py:"
  echo "$TRAINING_CONFIG"
  exit 1
}

export TRAINING_CONFIG

"$PY" - <<'PY'
from pathlib import Path
import os
import re

path = Path(os.environ["TRAINING_CONFIG"])
text = path.read_text(encoding="utf-8")
original = text

if "from dataclasses import" in text:
    text = re.sub(
        r"from dataclasses import ([^\n]+)",
        lambda m: (
            m.group(0)
            if "field" in [x.strip() for x in m.group(1).split(",")]
            else "from dataclasses import " + m.group(1).strip() + ", field"
        ),
        text,
        count=1,
    )
else:
    text = "from dataclasses import field\n" + text

pattern = re.compile(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*\s*:\s*([A-Za-z_][A-Za-z0-9_]*Config))\s*=\s*\3\(\)\s*(#.*)?$",
    re.MULTILINE,
)

def repl(m):
    indent = m.group(1)
    lhs = m.group(2)
    cls = m.group(3)
    comment = m.group(4) or ""
    return f"{indent}{lhs} = field(default_factory={cls}){comment}"

text = pattern.sub(repl, text)

text = re.sub(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*\s*:\s*list(?:\[[^\]]+\])?)\s*=\s*\[\]\s*(#.*)?$",
    lambda m: f"{m.group(1)}{m.group(2)} = field(default_factory=list){m.group(3) or ''}",
    text,
    flags=re.MULTILINE,
)

text = re.sub(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*\s*:\s*dict(?:\[[^\]]+\])?)\s*=\s*\{\}\s*(#.*)?$",
    lambda m: f"{m.group(1)}{m.group(2)} = field(default_factory=dict){m.group(3) or ''}",
    text,
    flags=re.MULTILINE,
)

path.write_text(text, encoding="utf-8")

print("Patched file:", path)
print("Changed:", text != original)
print()
print("Patched default_factory lines:")
for line in text.splitlines():
    if "field(default_factory=" in line:
        print(line)
PY

"$PY" -m py_compile "$TRAINING_CONFIG"
echo "Dataclass patch OK"

# =============================================================================
# 3.55 Patch MegaPose / Panda3D compatibility for NumPy 2.x + Python 3.12
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3.55 — PATCH PANDa3D / NUMPY 2.x COMPATIBILITY"
echo "================================================================================"

PANDA_UTILS="$EXP_REPO/src/megapose/panda3d_renderer/utils.py"
MULTIVIEW_FILE="$EXP_REPO/src/megapose/lib3d/multiview.py"

for P in "$PANDA_UTILS" "$MULTIVIEW_FILE"; do
  test -s "$P" || {
    echo "ERROR: missing file:"
    echo "$P"
    exit 1
  }
done

export PANDA_UTILS MULTIVIEW_FILE

"$PY" - <<'PY'
from pathlib import Path
import os

panda_utils = Path(os.environ["PANDA_UTILS"])
text = panda_utils.read_text(encoding="utf-8")
original = text

text = text.replace(".tostring()", ".tobytes()")

panda_utils.write_text(text, encoding="utf-8")

print("Patched Panda3D utils:", panda_utils)
print("  changed:", text != original)
print("  tostring remaining:", ".tostring()" in text)
print("  tobytes count:", text.count(".tobytes()"))

multiview = Path(os.environ["MULTIVIEW_FILE"])
text = multiview.read_text(encoding="utf-8")
original = text

if "import numpy as np" not in text:
    lines = text.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1
    lines.insert(insert_idx, "import numpy as np")
    text = "\n".join(lines) + "\n"

text = text.replace(
    "ref.setPos(*tWR[:3])",
    "ref.setPos(*np.asarray(tWR[:3], dtype=float).reshape(-1)[:3].tolist())"
)

text = text.replace(
    "ref.setPos(*tWR)",
    "ref.setPos(*np.asarray(tWR, dtype=float).reshape(-1)[:3].tolist())"
)

multiview.write_text(text, encoding="utf-8")

print("Patched multiview:", multiview)
print("  changed:", text != original)
print("  safe setPos present:", "np.asarray(tWR[:3], dtype=float).reshape(-1)[:3].tolist()" in text)
PY

"$PY" -m py_compile "$PANDA_UTILS"
"$PY" -m py_compile "$MULTIVIEW_FILE"

echo "Panda3D / NumPy compatibility patch OK"

# =============================================================================
# 3.6 Restore runtime env for refinement
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3.6 — RESTORE REFINEMENT RUNTIME ENV"
echo "================================================================================"

cp -f "$CACHE/gp_py" "$PY"
chmod +x "$PY"

unset PYTHONPATH PYTHONHOME PYTHONNOUSERSITE CONDA_PREFIX CONDA_DEFAULT_ENV LD_PRELOAD || true
unset GP_COMBO_EXPLICIT_INT8 GP_COMBO_EXPLICIT_FP16 GP_COMBO_EXPLICIT_REPORT_JSON || true
unset GIGAPOSE_COMBO_INT8 GIGAPOSE_COMBO_FP16 GIGAPOSE_COMBO_HOOK_REPORT_JSON || true
unset GP_IST_QUERY_CACHE GP_IST_QUERY_CACHE_PROFILE_JSONL || true

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

unset DISPLAY || true
unset LIBGL_ALWAYS_SOFTWARE || true
unset GALLIUM_DRIVER || true
unset MESA_LOADER_DRIVER_OVERRIDE || true

export BOP_DATASETS_DIR="$ROOT/datasets"
export BOP_RESULTS_DIR="$ROOT/results"
export BOP_LOGS_DIR="$ROOT/logs"
export TORCH_HOME="$ROOT/.cache/torch"

export PYTHONPATH="$EXP_REPO:$EXP_REPO/src:$BOPTK_DRIVE:$BOPTK_DRIVE/bop_toolkit_lib:$CACHE/gp_site_py312:$ENV/lib/$PYVER/site-packages"

"$PY" -V
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

export EXP_REPO
"$PY" - <<'PY'
from pathlib import Path
import os
import torch

repo = Path(os.environ["EXP_REPO"])
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("refine.py exists:", (repo / "refine.py").exists())
print("refine.py path:", repo / "refine.py")
PY

# =============================================================================
# 4. Run MegaPose 3hyp × 5iter refinement
# =============================================================================

echo
echo "================================================================================"
echo "STEP 4 — RUN MEGAPOSE 3HYP × 5ITER REFINEMENT"
echo "================================================================================"

MARKER="$WORK_DIR/refine_started.marker"
touch "$MARKER"

REFINE_START="$(date +%s)"

set +e
(
  cd "$EXP_REPO"

  timeout --signal=TERM --kill-after=120s 8h \
  "$PY" -u refine.py \
    test_dataset_name=lmo \
    machine.root_dir="$ROOT" \
    model=large \
    model.checkpoint_path="$CHECKPOINT" \
    run_id="$COARSE_RUN_ID" \
    use_multiple=true \
    model.refiner.n_iterations="$N_ITER" \
    machine.trainer.num_sanity_val_steps=0 \
    ++machine.trainer.strategy=auto \
    ++machine.trainer.precision=32 \
    ++machine.trainer.limit_test_batches=1.0
) > "$REFINE_LOG" 2>&1
REFINE_RC=$?

set -e

REFINE_END="$(date +%s)"
REFINE_WALL_SECONDS=$((REFINE_END - REFINE_START))

echo "REFINE_RC=$REFINE_RC"
echo "REFINE_WALL_SECONDS=$REFINE_WALL_SECONDS"

tail -n 260 "$REFINE_LOG" || true

restore_multi
rm -f "$BACKUP_MULTI_CSV" || true
trap - EXIT

REFINED_DIR="$COARSE_RESULT_DIR/refined_multiple_predictions"
test -d "$REFINED_DIR" || {
  echo "ERROR: refined_multiple_predictions directory not found:"
  echo "$REFINED_DIR"
  exit 1
}

REFINED_FOUND="$(
  find "$REFINED_DIR" -type f -name "*.csv" -newer "$MARKER" -printf "%T@ %p\n" 2>/dev/null \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2- || true
)"

if [[ -z "$REFINED_FOUND" ]]; then
  REFINED_FOUND="$(
    find "$REFINED_DIR" -type f -name "*.csv" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- || true
  )"
fi

test -s "$REFINED_FOUND" || {
  echo "ERROR: refined CSV not found"
  exit 1
}

cp -f "$REFINED_FOUND" "$REFINED_RAW_CSV"

echo
echo "REFINED_FOUND=$REFINED_FOUND"
echo "REFINED_RAW_CSV=$REFINED_RAW_CSV"
sha256sum "$REFINED_RAW_CSV"

if [[ "$REFINE_RC" -ne 0 ]]; then
  echo
  echo "WARNING: refine.py returned non-zero, but refined CSV exists."
  echo "Continuing to validation and BOP eval."
fi

# =============================================================================
# 5. Validate refined output and prepare BOP top-1 CSV
# =============================================================================

echo
echo "================================================================================"
echo "STEP 5 — VALIDATE REFINED OUTPUT"
echo "================================================================================"

export REFINED_RAW_CSV REFINED_BOP_CSV VALIDATION_JSON REFINE_RC REFINE_WALL_SECONDS
export REFINE_ID COARSE_RUN_ID TOPK N_ITER COARSE_AR_PERCENT COARSE_TIME

"$PY" - <<'PY'
from pathlib import Path
import os
import json
import pandas as pd

raw_csv = Path(os.environ["REFINED_RAW_CSV"])
bop_csv = Path(os.environ["REFINED_BOP_CSV"])
validation_json = Path(os.environ["VALIDATION_JSON"])

df = pd.read_csv(raw_csv)

required = {"scene_id", "im_id", "obj_id", "score", "R", "t", "time"}
group_cols = ["scene_id", "im_id", "obj_id"]

missing = required - set(df.columns)
if missing:
    raise SystemExit(f"ERROR: refined CSV missing columns: {sorted(missing)}")

for c in group_cols:
    df[c] = df[c].astype(int)

images = df[["scene_id", "im_id"]].drop_duplicates().shape[0]
targets = df[group_cols].drop_duplicates().shape[0]
objects = sorted(df["obj_id"].astype(int).unique().tolist())
time_img_raw = float(df.groupby(["scene_id", "im_id"])["time"].first().mean())
zero_scores = int((df["score"] == 0).sum())
nan_r = int(df["R"].astype(str).str.contains("nan", case=False).sum())
nan_t = int(df["t"].astype(str).str.contains("nan", case=False).sum())

print("raw rows:", len(df))
print("images:", images)
print("targets:", targets)
print("objects:", objects)
print(f"raw time/img: {time_img_raw:.6f}")
print("zero scores:", zero_scores)
print("textual_nan_R:", nan_r)
print("textual_nan_t:", nan_t)

if images != 200:
    raise SystemExit(f"ERROR: expected 200 images, got {images}")
if targets != 1445:
    raise SystemExit(f"ERROR: expected 1445 targets, got {targets}")
if set(objects) != {1, 5, 6, 8, 9, 10, 11, 12}:
    raise SystemExit(f"ERROR: wrong object IDs: {objects}")
if nan_r or nan_t:
    raise SystemExit("ERROR: textual NaN in refined output")

sizes = df.groupby(group_cols).size()
hyp_min = int(sizes.min())
hyp_max = int(sizes.max())

if len(df) == 1445 and hyp_min == 1 and hyp_max == 1:
    bop = df.copy()
    output_mode = "already_top1"
else:
    bop = (
        df.sort_values(group_cols + ["score"], ascending=[True, True, True, False])
          .groupby(group_cols, sort=False, group_keys=False)
          .head(1)
          .copy()
          .reset_index(drop=True)
    )
    output_mode = "selected_top1_by_refined_score"

if len(bop) != 1445:
    raise SystemExit(f"ERROR: BOP top1 rows expected 1445, got {len(bop)}")

bop_sizes = bop.groupby(group_cols).size()
if bop_sizes.min() != 1 or bop_sizes.max() != 1:
    raise SystemExit("ERROR: BOP CSV still has duplicate targets")

bop.to_csv(bop_csv, index=False)
time_img_bop = float(bop.groupby(["scene_id", "im_id"])["time"].first().mean())

validation = {
    "refine_id": os.environ["REFINE_ID"],
    "coarse_run_id": os.environ["COARSE_RUN_ID"],
    "topk": int(os.environ["TOPK"]),
    "n_iter": int(os.environ["N_ITER"]),
    "refine_rc": int(os.environ["REFINE_RC"]),
    "refine_wall_seconds": int(os.environ["REFINE_WALL_SECONDS"]),
    "raw_rows": int(len(df)),
    "bop_rows": int(len(bop)),
    "images": int(images),
    "targets": int(targets),
    "objects": objects,
    "raw_hypotheses_per_target_min": hyp_min,
    "raw_hypotheses_per_target_max": hyp_max,
    "output_mode": output_mode,
    "raw_time_per_image_s": time_img_raw,
    "bop_time_per_image_s": time_img_bop,
    "zero_scores_raw": zero_scores,
    "textual_nan_R_raw": nan_r,
    "textual_nan_t_raw": nan_t,
    "coarse_AR_percent": float(os.environ["COARSE_AR_PERCENT"]),
    "coarse_time_per_image_s": float(os.environ["COARSE_TIME"]),
    "refined_raw_csv": str(raw_csv),
    "refined_bop_csv": str(bop_csv),
}

validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print()
print("BOP CSV:", bop_csv)
print("BOP rows:", len(bop))
print(f"BOP time/img: {time_img_bop:.6f}")
print("output_mode:", output_mode)
print("Validation JSON:", validation_json)
PY

echo
echo "REFINED_BOP_CSV=$REFINED_BOP_CSV"
sha256sum "$REFINED_BOP_CSV"

# =============================================================================
# 6. Prepare BOP toolkit environment
# =============================================================================

echo
echo "================================================================================"
echo "STEP 6 — PREPARE BOP EVAL ENVIRONMENT"
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

# =============================================================================
# 7. Run official BOP19 evaluation
# =============================================================================

echo
echo "================================================================================"
echo "STEP 7 — RUN OFFICIAL BOP19 EVALUATION"
echo "================================================================================"

cp -f "$REFINED_BOP_CSV" "$BOP_RESULT_CSV"

echo "BOP_RESULT_FILENAME=$BOP_RESULT_FILENAME"
echo "BOP_RESULT_CSV=$BOP_RESULT_CSV"
sha256sum "$BOP_RESULT_CSV"

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

# =============================================================================
# 8. Parse metrics and write summary
# =============================================================================

echo
echo "================================================================================"
echo "STEP 8 — WRITE SUMMARY"
echo "================================================================================"

export EVAL_LOG SUMMARY_JSON SUMMARY_TXT VALIDATION_JSON REFINE_ID COARSE_RUN_ID
export REFINE_BASELINE_AR_PERCENT REFINE_BASELINE_TIME
export NO_CSM_TOP3_AR_PERCENT NO_CSM_TOP3_TIME
export CSM_FULL_5HYP5ITER_AR_PERCENT CSM_FULL_5HYP5ITER_TIME
export COARSE_AR_PERCENT COARSE_TIME EVAL_WALL_SECONDS

"$PY" - <<'PY'
from pathlib import Path
import os
import re
import json

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
    "setting": {
        "coarse": "CSM-v1 + INT8 + FP16 + IST cache",
        "refinement": "MegaPose",
        "pose_hypotheses": 3,
        "refinement_iterations": 5,
    },
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
echo "FULL PIPELINE REFINEMENT COMPLETE"
echo "================================================================================"
cat "$SUMMARY_TXT"
