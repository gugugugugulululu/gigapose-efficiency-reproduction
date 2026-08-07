set -Eeuo pipefail

# =============================================================================
# LM-O FRESH K3-5 / I3-5 PER-OBJECT POLICY RERUN
#
# Frozen policy:
#   obj 01 ape          -> K=4, I=4
#   obj 05 can          -> K=5, I=4
#   obj 06 cat          -> K=3, I=4
#   obj 08 driller      -> K=3, I=5
#   obj 09 duck         -> K=4, I=4
#   obj 10 eggbox       -> K=4, I=4
#   obj 11 glue         -> K=3, I=5
#   obj 12 holepuncher  -> K=3, I=3
#
# Actual execution groups:
#   K=3, I=3 -> obj [12]          -> 200 targets
#   K=3, I=4 -> obj [6]           -> 171 targets
#   K=3, I=5 -> obj [8,11]        -> 340 targets
#   K=4, I=4 -> obj [1,9,10]      -> 535 targets
#   K=5, I=4 -> obj [5]           -> 199 targets
#
# Total:
#   1445 targets
#   5268 physically retained initial hypotheses
#
# Workflow:
#   original LM-O coarse MultiHypothesis CSV
#       -> physical Top-K truncation per object
#       -> five fresh MegaPose refinement jobs
#       -> merge 1445 refined predictions
#       -> reconstruct measured policy runtime
#       -> Official BOP19 evaluation
#
# Runtime:
#   source coarse runtime counted once
#   + measured refinement increment from each active policy group
#
# The runtime is reconstructed from measured CSV timings.
# It is not one integrated single-process online wall-clock run.
# =============================================================================


# =============================================================================
# 0. FIXED PATHS
# =============================================================================

ROOT="/content/drive/MyDrive/bop_workspace"

BUNDLE="$ROOT/runtime_cache/object_adaptive_h345"
RESTORE_SCRIPT="$BUNDLE/restore.sh"
ENV_SCRIPT="$BUNDLE/env.sh"

REPO="/content/gigapose_lmo_histartifact_exactcfg_5h5i_b8o2_w1_p32true_20260715_034423"
BOPTK="$ROOT/code/bop_toolkit"

CHECKPOINT="$ROOT/pretrained/gigaPose_v1.ckpt"
MEGAPOSE_ROOT="$ROOT/pretrained/megapose-models"

SOURCE_RUN_ID="lmo_clean_coarse_wds0257_20260713_002914"
SOURCE_RESULT_DIR="$ROOT/results/large_${SOURCE_RUN_ID}"

SOURCE_MAIN="$SOURCE_RESULT_DIR/predictions/large-pbrreal-rgb-mmodel_lmo-test_${SOURCE_RUN_ID}.csv"
SOURCE_MULTI="$SOURCE_RESULT_DIR/predictions/large-pbrreal-rgb-mmodel_lmo-test_${SOURCE_RUN_ID}MultiHypothesis.csv"

# New ID: no K=1 result will be reused.
ADAPT_ID="lmo_real_objpolicy_k345_i345_rerun_v1"

OUT_DIR="$ROOT/results/adaptive_ki_grouped/$ADAPT_ID"
GROUP_DIR="$OUT_DIR/groups"
PRED_DIR="$OUT_DIR/predictions"
REPORT_DIR="$OUT_DIR/reports"
LOG_DIR="$OUT_DIR/logs"

POLICY_CSV="$REPORT_DIR/${ADAPT_ID}_frozen_policy.csv"
MANIFEST_JSON="$REPORT_DIR/${ADAPT_ID}_manifest.json"
MANIFEST_TSV="$REPORT_DIR/${ADAPT_ID}_manifest.tsv"

MERGED_CSV="$PRED_DIR/${ADAPT_ID}_merged_lmo-test.csv"
MERGE_REPORT="$REPORT_DIR/${ADAPT_ID}_runtime_report.json"

SCORES_JSON="$REPORT_DIR/${ADAPT_ID}_scores_bop19.json"
FINAL_JSON="$REPORT_DIR/${ADAPT_ID}_final_summary.json"
FINAL_TXT="$REPORT_DIR/${ADAPT_ID}_final_summary.txt"

MASTER_LOG="$LOG_DIR/${ADAPT_ID}_master.log"
EVAL_LOG="$LOG_DIR/${ADAPT_ID}_official_bop19.log"

mkdir -p \
    "$GROUP_DIR" \
    "$PRED_DIR" \
    "$REPORT_DIR" \
    "$LOG_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1


echo "================================================================================"
echo "LM-O FRESH K3-5 / I3-5 OBJECT-POLICY RERUN"
echo "================================================================================"

echo "ADAPT_ID:"
echo "  $ADAPT_ID"

echo "OUT_DIR:"
echo "  $OUT_DIR"

echo
echo "This run starts from the original coarse prediction files."
echo "No refined pose from the K=1 policy or KxI grid is reused."


# =============================================================================
# 1. RESTORE VERIFIED RUNTIME
# =============================================================================

echo
echo "================================================================================"
echo "STEP 1 - RESTORE VERIFIED RUNTIME"
echo "================================================================================"

test -s "$RESTORE_SCRIPT" || {
    echo "ERROR: restore script missing:"
    echo "  $RESTORE_SCRIPT"
    exit 1
}

test -s "$ENV_SCRIPT" || {
    echo "ERROR: environment script missing:"
    echo "  $ENV_SCRIPT"
    exit 1
}

bash "$RESTORE_SCRIPT"

# shellcheck disable=SC1090
source "$ENV_SCRIPT"

PY="/content/gp_adaptive_py"
BARE_PY="/content/gp_exec/bin/python"

export CUDA_VISIBLE_DEVICES=0
export EGL_VISIBLE_DEVICES=0

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP=egl

export BOP_PATH="$ROOT/datasets"
export BOP_DATASETS_DIR="$ROOT/datasets"
export BOP_DATASETS_PATH="$ROOT/datasets"

export PYTHONPATH="$REPO:$REPO/src:$BOPTK:$BOPTK/bop_toolkit_lib:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


required_paths=(
    "$PY"
    "$BARE_PY"

    "$REPO/refine.py"
    "$REPO/src/models/refiner.py"

    "$BOPTK/scripts/eval_bop19_pose.py"
    "$BOPTK/bop_toolkit_lib"

    "$CHECKPOINT"
    "$MEGAPOSE_ROOT"

    "$SOURCE_MAIN"
    "$SOURCE_MULTI"

    "$ROOT/datasets/lmo/test_targets_bop19.json"
    "$ROOT/datasets/lmo/models_eval/models_info.json"
)

for path in "${required_paths[@]}"; do
    test -e "$path" || {
        echo "ERROR: required path missing:"
        echo "  $path"
        exit 1
    }

    echo "OK: $path"
done


"$PY" -u - <<'PY'
import sys

import numpy
import pandas
import torch

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

gpu_name = torch.cuda.get_device_name(0)

print("GPU:", gpu_name)

if "T4" not in gpu_name:
    raise RuntimeError(
        f"Expected Tesla T4, found {gpu_name}"
    )

if torch.__version__ != "2.5.1+cu121":
    raise RuntimeError(
        f"Unexpected PyTorch: {torch.__version__}"
    )

if torch.version.cuda != "12.1":
    raise RuntimeError(
        f"Unexpected CUDA build: {torch.version.cuda}"
    )

if numpy.__version__ != "2.0.2":
    raise RuntimeError(
        f"Unexpected NumPy: {numpy.__version__}"
    )

if pandas.__version__ != "2.2.2":
    raise RuntimeError(
        f"Unexpected Pandas: {pandas.__version__}"
    )

print("RUNTIME CHECK: PASS")
PY


# =============================================================================
# 2. ENSURE EMPTY-BATCH SUPPORT
#
# Some object groups may have no target in a particular image.
# =============================================================================

echo
echo "================================================================================"
echo "STEP 2 - ENSURE EMPTY-BATCH SUPPORT"
echo "================================================================================"

REFINER_PY="$REPO/src/models/refiner.py"

REFINER_PY="$REFINER_PY" \
"$PY" -u - <<'PY'
from pathlib import Path
import datetime
import os
import re


path = Path(os.environ["REFINER_PY"])
text = path.read_text(encoding="utf-8")

markers = [
    "# === K345-I345 POLICY EMPTY-BATCH GUARD ===",
    "# === OBJECT-POLICY EMPTY-BATCH GUARD V2 ===",
    "# === OBJECT-POLICY EMPTY-BATCH GUARD V1 ===",
    "# === REAL ADAPTIVE-KI EMPTY-BATCH GUARD ===",
    "# === ADAPTIVE-KI EMPTY-BATCH GUARD ===",
    "# === MIXED-K EMPTY-BATCH GUARD ===",
]

if any(marker in text for marker in markers):
    print("Compatible empty-batch guard already exists.")

else:
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)"
        r"def test_step\([^\n]*\):[ \t]*$",
        re.MULTILINE,
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(
            "Could not locate test_step() in refiner.py"
        )

    indent = match.group("indent") + "    "

    guard = (
        "\n"
        f"{indent}# === K345-I345 POLICY EMPTY-BATCH GUARD ===\n"
        f"{indent}try:\n"
        f"{indent}    _policy_tco_init = batch.TCO_init\n"
        f"{indent}except AttributeError:\n"
        f"{indent}    print(\n"
        f"{indent}        "
        "\"[K345-I345] skip empty batch: no TCO_init\",\n"
        f"{indent}        flush=True,\n"
        f"{indent}    )\n"
        f"{indent}    return 0\n"
        "\n"
        f"{indent}try:\n"
        f"{indent}    _policy_pose_count = len(_policy_tco_init)\n"
        f"{indent}except TypeError:\n"
        f"{indent}    _policy_pose_count = None\n"
        "\n"
        f"{indent}if _policy_pose_count == 0:\n"
        f"{indent}    print(\n"
        f"{indent}        "
        "\"[K345-I345] skip empty batch: zero poses\",\n"
        f"{indent}        flush=True,\n"
        f"{indent}    )\n"
        f"{indent}    return 0\n"
    )

    stamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = path.with_name(
        f"{path.name}.before_k345_i345_{stamp}.bak"
    )

    backup.write_text(
        text,
        encoding="utf-8",
    )

    patched = (
        text[:match.end()]
        + guard
        + text[match.end():]
    )

    path.write_text(
        patched,
        encoding="utf-8",
    )

    print("Patched:", path)
    print("Backup:", backup)

print("EMPTY-BATCH SUPPORT: PASS")
PY

"$PY" -m py_compile "$REFINER_PY"

echo "REFINER COMPILATION: PASS"


# =============================================================================
# 3. CREATE FROZEN POLICY AND BUILD FRESH GROUP INPUTS
# =============================================================================

echo
echo "================================================================================"
echo "STEP 3 - BUILD FRESH K3-5 / I3-5 POLICY INPUTS"
echo "================================================================================"

ROOT="$ROOT" \
ADAPT_ID="$ADAPT_ID" \
GROUP_DIR="$GROUP_DIR" \
POLICY_CSV="$POLICY_CSV" \
MANIFEST_JSON="$MANIFEST_JSON" \
MANIFEST_TSV="$MANIFEST_TSV" \
SOURCE_MAIN="$SOURCE_MAIN" \
SOURCE_MULTI="$SOURCE_MULTI" \
"$PY" -u - <<'PY'
from pathlib import Path
import hashlib
import json
import os

import pandas as pd


root = Path(os.environ["ROOT"])
adapt_id = os.environ["ADAPT_ID"]

group_dir = Path(os.environ["GROUP_DIR"])

policy_csv = Path(
    os.environ["POLICY_CSV"]
)

manifest_json = Path(
    os.environ["MANIFEST_JSON"]
)

manifest_tsv = Path(
    os.environ["MANIFEST_TSV"]
)

source_main_path = Path(
    os.environ["SOURCE_MAIN"]
)

source_multi_path = Path(
    os.environ["SOURCE_MULTI"]
)


target_keys = [
    "scene_id",
    "im_id",
    "obj_id",
]


# -------------------------------------------------------------------------
# Frozen K3-5 / I3-5 policy.
# -------------------------------------------------------------------------

policy_rows = [
    {
        "obj_id": 1,
        "object_name": "ape",
        "selected_k": 4,
        "selected_iterations": 4,
        "expected_targets": 175,
    },
    {
        "obj_id": 5,
        "object_name": "can",
        "selected_k": 5,
        "selected_iterations": 4,
        "expected_targets": 199,
    },
    {
        "obj_id": 6,
        "object_name": "cat",
        "selected_k": 3,
        "selected_iterations": 4,
        "expected_targets": 171,
    },
    {
        "obj_id": 8,
        "object_name": "driller",
        "selected_k": 3,
        "selected_iterations": 5,
        "expected_targets": 200,
    },
    {
        "obj_id": 9,
        "object_name": "duck",
        "selected_k": 4,
        "selected_iterations": 4,
        "expected_targets": 180,
    },
    {
        "obj_id": 10,
        "object_name": "eggbox",
        "selected_k": 4,
        "selected_iterations": 4,
        "expected_targets": 180,
    },
    {
        "obj_id": 11,
        "object_name": "glue",
        "selected_k": 3,
        "selected_iterations": 5,
        "expected_targets": 140,
    },
    {
        "obj_id": 12,
        "object_name": "holepuncher",
        "selected_k": 3,
        "selected_iterations": 3,
        "expected_targets": 200,
    },
]

policy = pd.DataFrame(policy_rows)

policy_csv.parent.mkdir(
    parents=True,
    exist_ok=True,
)

policy.to_csv(
    policy_csv,
    index=False,
)


main = pd.read_csv(
    source_main_path
)

multi = pd.read_csv(
    source_multi_path
)


for column in target_keys:
    main[column] = main[column].astype(int)
    multi[column] = multi[column].astype(int)


if len(main) != 1445:
    raise RuntimeError(
        f"Expected 1445 source rows, found {len(main)}"
    )

if main.duplicated(target_keys).any():
    raise RuntimeError(
        "Duplicate target keys in source main CSV"
    )


expected_target_counts = {
    int(row.obj_id): int(row.expected_targets)
    for row in policy.itertuples()
}

actual_target_counts = (
    main.groupby("obj_id")
    .size()
    .astype(int)
    .to_dict()
)

if actual_target_counts != expected_target_counts:
    raise RuntimeError(
        "LM-O target-count mismatch.\n"
        f"Expected: {expected_target_counts}\n"
        f"Actual:   {actual_target_counts}"
    )


source_hypothesis_counts = (
    multi.groupby(
        target_keys,
        sort=False,
    )
    .size()
)

if len(source_hypothesis_counts) != 1445:
    raise RuntimeError(
        f"Expected 1445 MultiHypothesis targets, "
        f"found {len(source_hypothesis_counts)}"
    )

if int(source_hypothesis_counts.min()) != 5:
    raise RuntimeError(
        "Some source targets have fewer than five hypotheses"
    )

if int(source_hypothesis_counts.max()) != 5:
    raise RuntimeError(
        "Some source targets have more than five hypotheses"
    )


multi = multi.copy()
multi["_original_order"] = range(len(multi))

groups = []


for (
    k,
    iterations,
), policy_group in policy.groupby(
    [
        "selected_k",
        "selected_iterations",
    ],
    sort=True,
):
    k = int(k)
    iterations = int(iterations)

    objects = sorted(
        policy_group["obj_id"]
        .astype(int)
        .tolist()
    )

    object_names = (
        policy_group
        .sort_values("obj_id")["object_name"]
        .tolist()
    )

    run_id = (
        f"{adapt_id}_k{k}_i{iterations}"
    )

    result_dir = (
        root
        / "results"
        / f"large_{run_id}"
    )

    input_dir = (
        result_dir
        / "predictions"
    )

    stable_dir = (
        group_dir
        / f"k{k}_i{iterations}"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stable_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    main_out = (
        main.loc[
            main["obj_id"].isin(objects)
        ]
        .copy()
        .sort_values(target_keys)
        .reset_index(drop=True)
    )


    multi_candidates = (
        multi.loc[
            multi["obj_id"].isin(objects)
        ]
        .copy()
        .sort_values("_original_order")
    )


    # Physical Top-K enforcement.
    multi_out = (
        multi_candidates
        .groupby(
            target_keys,
            sort=False,
            group_keys=False,
        )
        .head(k)
        .drop(columns=["_original_order"])
        .reset_index(drop=True)
    )


    retained_counts = (
        multi_out.groupby(
            target_keys,
            sort=False,
        )
        .size()
    )

    if len(retained_counts) != len(main_out):
        raise RuntimeError(
            f"K={k}, I={iterations}: "
            "main/multi target mismatch"
        )

    if not retained_counts.eq(k).all():
        raise RuntimeError(
            f"K={k}, I={iterations}: "
            "physical Top-K truncation failed"
        )


    expected_group_targets = int(
        policy_group["expected_targets"].sum()
    )

    if len(main_out) != expected_group_targets:
        raise RuntimeError(
            f"K={k}, I={iterations}: "
            f"expected {expected_group_targets} targets, "
            f"found {len(main_out)}"
        )


    main_file = (
        input_dir
        / (
            "large-pbrreal-rgb-mmodel_"
            "lmo-test_"
            f"{run_id}.csv"
        )
    )

    multi_file = (
        input_dir
        / (
            "large-pbrreal-rgb-mmodel_"
            "lmo-test_"
            f"{run_id}"
            "MultiHypothesis.csv"
        )
    )

    stable_csv = (
        stable_dir
        / f"{run_id}_refined.csv"
    )

    process_wall_file = (
        stable_dir
        / f"{run_id}_process_wall_seconds.txt"
    )


    main_out.to_csv(
        main_file,
        index=False,
    )

    multi_out.to_csv(
        multi_file,
        index=False,
    )


    image_count = (
        main_out[
            ["scene_id", "im_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )


    groups.append(
        {
            "k": k,
            "iterations": iterations,
            "objects": objects,
            "object_names": object_names,
            "target_count": int(len(main_out)),
            "hypothesis_rows": int(len(multi_out)),
            "images_with_targets": int(image_count),
            "run_id": run_id,
            "result_dir": str(result_dir),
            "input_main": str(main_file),
            "input_multi": str(multi_file),
            "stable_csv": str(stable_csv),
            "process_wall_file": str(process_wall_file),
        }
    )


expected_groups = {
    (3, 3): [12],
    (3, 4): [6],
    (3, 5): [8, 11],
    (4, 4): [1, 9, 10],
    (5, 4): [5],
}

actual_groups = {
    (
        int(group["k"]),
        int(group["iterations"]),
    ): group["objects"]
    for group in groups
}

if actual_groups != expected_groups:
    raise RuntimeError(
        "Unexpected execution grouping.\n"
        f"Expected: {expected_groups}\n"
        f"Actual:   {actual_groups}"
    )


total_targets = sum(
    group["target_count"]
    for group in groups
)

total_hypotheses = sum(
    group["hypothesis_rows"]
    for group in groups
)

if total_targets != 1445:
    raise RuntimeError(
        f"Expected 1445 grouped targets, "
        f"found {total_targets}"
    )

if total_hypotheses != 5268:
    raise RuntimeError(
        f"Expected 5268 retained hypotheses, "
        f"found {total_hypotheses}"
    )


manifest = {
    "adapt_id": adapt_id,
    "dataset": "lmo",
    "images": 200,
    "targets": 1445,
    "source_hypotheses": 7225,
    "retained_hypotheses": 5268,
    "execution": (
        "fresh MegaPose refinement from original "
        "clean coarse predictions"
    ),
    "policy_type": (
        "frozen per-object K3-5 and I3-5 policy"
    ),
    "source_main": str(source_main_path),
    "source_main_sha256": hashlib.sha256(
        source_main_path.read_bytes()
    ).hexdigest(),
    "source_multi": str(source_multi_path),
    "source_multi_sha256": hashlib.sha256(
        source_multi_path.read_bytes()
    ).hexdigest(),
    "policy_csv": str(policy_csv),
    "policy": policy_rows,
    "groups": groups,
}


manifest_json.write_text(
    json.dumps(
        manifest,
        indent=2,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)


with manifest_tsv.open(
    "w",
    encoding="utf-8",
) as stream:
    stream.write(
        "\t".join(
            [
                "k",
                "iterations",
                "objects",
                "target_count",
                "images",
                "run_id",
                "stable_csv",
                "process_wall_file",
            ]
        )
        + "\n"
    )

    for group in groups:
        stream.write(
            "\t".join(
                [
                    str(group["k"]),
                    str(group["iterations"]),
                    ",".join(
                        str(value)
                        for value in group["objects"]
                    ),
                    str(group["target_count"]),
                    str(group["images_with_targets"]),
                    group["run_id"],
                    group["stable_csv"],
                    group["process_wall_file"],
                ]
            )
            + "\n"
        )


print("=" * 100)
print("FROZEN K3-5 / I3-5 POLICY")
print("=" * 100)

for row in policy.itertuples():
    print(
        f"obj {int(row.obj_id):02d} "
        f"{row.object_name:<12} "
        f"targets={int(row.expected_targets):3d} "
        f"-> K={int(row.selected_k)}, "
        f"I={int(row.selected_iterations)}"
    )

print()
print("Execution groups:")

for group in groups:
    print(
        f"  K={group['k']}, "
        f"I={group['iterations']}: "
        f"objects={group['objects']} "
        f"{group['object_names']}, "
        f"targets={group['target_count']}, "
        f"hypotheses={group['hypothesis_rows']}, "
        f"images={group['images_with_targets']}"
    )

print()
print("Total targets:", total_targets)
print("Retained hypotheses:", total_hypotheses)
print("FRESH POLICY INPUT GENERATION: PASS")
PY


# =============================================================================
# 4. GROUP OUTPUT VALIDATOR
# =============================================================================

validate_group_output() {
    local K="$1"
    local ITERATIONS="$2"
    local OBJECTS="$3"
    local EXPECTED_ROWS="$4"
    local CSV_PATH="$5"

    K="$K" \
    ITERATIONS="$ITERATIONS" \
    OBJECTS="$OBJECTS" \
    EXPECTED_ROWS="$EXPECTED_ROWS" \
    CSV_PATH="$CSV_PATH" \
    "$PY" -u - <<'PY'
from pathlib import Path
import os

import numpy as np
import pandas as pd


k = int(os.environ["K"])
iterations = int(os.environ["ITERATIONS"])

expected_objects = sorted(
    int(value)
    for value in os.environ["OBJECTS"].split(",")
)

expected_rows = int(
    os.environ["EXPECTED_ROWS"]
)

path = Path(
    os.environ["CSV_PATH"]
)

if not path.is_file():
    raise FileNotFoundError(path)

frame = pd.read_csv(path)

required_columns = {
    "scene_id",
    "im_id",
    "obj_id",
    "score",
    "R",
    "t",
    "time",
}

missing_columns = (
    required_columns
    - set(frame.columns)
)

if missing_columns:
    raise RuntimeError(
        f"Missing output columns: "
        f"{sorted(missing_columns)}"
    )

if len(frame) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"found {len(frame)}"
    )

target_keys = [
    "scene_id",
    "im_id",
    "obj_id",
]

if frame.duplicated(target_keys).any():
    raise RuntimeError(
        "Duplicate target keys in grouped output"
    )

actual_objects = sorted(
    frame["obj_id"]
    .astype(int)
    .unique()
    .tolist()
)

if actual_objects != expected_objects:
    raise RuntimeError(
        f"Expected objects {expected_objects}, "
        f"found {actual_objects}"
    )

if frame[
    ["score", "R", "t", "time"]
].isna().any().any():
    raise RuntimeError(
        "NaN values detected"
    )

times = pd.to_numeric(
    frame["time"],
    errors="raise",
)

if not np.isfinite(
    times.to_numpy()
).all():
    raise RuntimeError(
        "Non-finite runtime values"
    )

if float(times.min()) < 0:
    raise RuntimeError(
        "Negative runtime values"
    )

images = (
    frame[
        ["scene_id", "im_id"]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    f"K={k}, I={iterations}: "
    f"rows={len(frame)}, "
    f"images={images}, "
    f"objects={actual_objects}"
)

print("GROUP OUTPUT VALIDATION: PASS")
PY
}


# =============================================================================
# 5. RUN OR RESUME FIVE FRESH POLICY GROUPS
# =============================================================================

echo
echo "================================================================================"
echo "STEP 4 - RUN FRESH K3-5 / I3-5 POLICY REFINEMENT"
echo "================================================================================"

while IFS=$'\t' read -r \
    K \
    ITERATIONS \
    OBJECTS \
    TARGET_COUNT \
    IMAGE_COUNT \
    RUN_ID \
    STABLE_CSV \
    PROCESS_WALL_FILE
do
    echo
    echo "================================================================================"
    echo "POLICY GROUP"
    echo "================================================================================"

    echo "K=$K"
    echo "Iterations=$ITERATIONS"
    echo "Objects=$OBJECTS"
    echo "Targets=$TARGET_COUNT"
    echo "Images with targets=$IMAGE_COUNT"
    echo "RUN_ID=$RUN_ID"


    RESULT_DIR="$ROOT/results/large_${RUN_ID}"
    REFINED_DIR="$RESULT_DIR/refined_multiple_predictions"

    REFINE_LOG="$LOG_DIR/${RUN_ID}_refine.log"
    CONFIG_LOG="$LOG_DIR/${RUN_ID}_resolved_config.log"


    # -------------------------------------------------------------------------
    # Existing stable output.
    # -------------------------------------------------------------------------

    if [[ -s "$STABLE_CSV" ]] &&
       validate_group_output \
           "$K" \
           "$ITERATIONS" \
           "$OBJECTS" \
           "$TARGET_COUNT" \
           "$STABLE_CSV"
    then
        echo
        echo "Valid completed output exists."
        echo "Skipping this group."
        continue
    fi


    # -------------------------------------------------------------------------
    # Recover output produced before a possible notebook interruption.
    # -------------------------------------------------------------------------

    mapfile -t RECOVERABLE_CSVS < <(
        {
            find "$REFINED_DIR" \
                -maxdepth 1 \
                -type f \
                -name "*.csv" \
                -size +0c \
                -print 2>/dev/null \
            || true
        } |
        sort
    )


    if [[ "${#RECOVERABLE_CSVS[@]}" -eq 1 ]]; then
        RECOVERABLE_CSV="${RECOVERABLE_CSVS[0]}"

        echo
        echo "Found recoverable output:"
        echo "  $RECOVERABLE_CSV"

        set +e

        validate_group_output \
            "$K" \
            "$ITERATIONS" \
            "$OBJECTS" \
            "$TARGET_COUNT" \
            "$RECOVERABLE_CSV"

        RECOVER_RC="$?"

        set -e

        if [[ "$RECOVER_RC" -eq 0 ]]; then
            mkdir -p "$(dirname "$STABLE_CSV")"

            cp -f \
                "$RECOVERABLE_CSV" \
                "$STABLE_CSV"

            if [[ ! -s "$PROCESS_WALL_FILE" ]]; then
                printf '%s\n' \
                    "0" \
                    >"$PROCESS_WALL_FILE"
            fi

            sync

            echo
            echo "RECOVERED COMPLETED GROUP:"
            echo "  $STABLE_CSV"

            continue
        fi
    fi


    # -------------------------------------------------------------------------
    # Fresh group execution.
    # -------------------------------------------------------------------------

    echo
    echo "Starting fresh MegaPose refinement."

    rm -rf "$REFINED_DIR"

    mkdir -p "$REFINED_DIR"

    rm -f \
        "$STABLE_CSV" \
        "$PROCESS_WALL_FILE" \
        "$REFINE_LOG" \
        "$CONFIG_LOG"


    ARGS=(
        "test_dataset_name=lmo"

        "machine.root_dir=$ROOT"

        "model=large"

        "model.checkpoint_path=$CHECKPOINT"

        "run_id=$RUN_ID"

        "use_multiple=true"

        "++model.refiner.model_name=megapose-1.0-RGB-multi-hypothesis"

        "++model.refiner.models_root=$MEGAPOSE_ROOT"

        "++model.refiner.n_pose_hypotheses=$K"

        "model.refiner.n_iterations=$ITERATIONS"

        "++model.refiner.batch_size_images=8"

        "++model.refiner.batch_size_objects=2"

        "machine.num_workers=1"

        "machine.trainer.num_sanity_val_steps=0"

        "++machine.trainer.accelerator=gpu"

        "++machine.trainer.devices=1"

        "++machine.trainer.strategy=auto"

        "++machine.trainer.precision=32-true"

        "++machine.trainer.limit_test_batches=1.0"
    )


    echo
    echo "Resolving Hydra configuration..."

    (
        cd "$REPO"

        "$PY" -u refine.py \
            "${ARGS[@]}" \
            --cfg job \
            --resolve
    ) >"$CONFIG_LOG" 2>&1


    grep -Eq \
        "n_pose_hypotheses:[[:space:]]*$K" \
        "$CONFIG_LOG" || {
            echo "ERROR: resolved K mismatch"
            tail -n 180 "$CONFIG_LOG"
            exit 1
        }


    grep -Eq \
        "n_iterations:[[:space:]]*$ITERATIONS" \
        "$CONFIG_LOG" || {
            echo "ERROR: resolved iteration mismatch"
            tail -n 180 "$CONFIG_LOG"
            exit 1
        }


    echo "RESOLVED CONFIGURATION: PASS"


    START_SECONDS="$(date +%s)"

    set +e

    (
        cd "$REPO"

        "$PY" -u refine.py \
            "${ARGS[@]}"
    ) 2>&1 |
    tee "$REFINE_LOG"

    REFINE_RC="${PIPESTATUS[0]}"

    set -e


    END_SECONDS="$(date +%s)"

    # Correct Bash arithmetic syntax.
    PROCESS_WALL_SECONDS=$((END_SECONDS - START_SECONDS))

    printf '%s\n' \
        "$PROCESS_WALL_SECONDS" \
        >"$PROCESS_WALL_FILE"


    echo
    echo "Refinement exit code:"
    echo "  $REFINE_RC"

    echo "Process wall seconds:"
    echo "  $PROCESS_WALL_SECONDS"


    if [[ "$REFINE_RC" -ne 0 ]]; then
        echo "ERROR: grouped refinement failed"

        tail -n 350 "$REFINE_LOG" || true

        exit "$REFINE_RC"
    fi


    mapfile -t GENERATED_CSVS < <(
        find "$REFINED_DIR" \
            -maxdepth 1 \
            -type f \
            -name "*.csv" \
            -size +0c \
            -print |
        sort
    )


    if [[ "${#GENERATED_CSVS[@]}" -ne 1 ]]; then
        echo "ERROR: expected exactly one generated CSV"

        printf '  %s\n' \
            "${GENERATED_CSVS[@]:-}"

        exit 1
    fi


    cp -f \
        "${GENERATED_CSVS[0]}" \
        "$STABLE_CSV"

    sync


    validate_group_output \
        "$K" \
        "$ITERATIONS" \
        "$OBJECTS" \
        "$TARGET_COUNT" \
        "$STABLE_CSV"


    echo
    echo "GROUP COMPLETE:"
    echo "  $STABLE_CSV"

done < <(
    tail -n +2 "$MANIFEST_TSV"
)


# =============================================================================
# 6. MERGE 1445 POSES AND RECONSTRUCT POLICY RUNTIME
#
# Each group CSV normally reports:
#   source coarse time + that group's refinement time
#
# Therefore:
#   combined policy time
#     = source coarse time once
#     + sum(group output time - source coarse time)
# =============================================================================

echo
echo "================================================================================"
echo "STEP 5 - MERGE POSES AND RECONSTRUCT POLICY RUNTIME"
echo "================================================================================"

MANIFEST_JSON="$MANIFEST_JSON" \
SOURCE_MAIN="$SOURCE_MAIN" \
MERGED_CSV="$MERGED_CSV" \
MERGE_REPORT="$MERGE_REPORT" \
"$PY" -u - <<'PY'
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


manifest_path = Path(
    os.environ["MANIFEST_JSON"]
)

source_main_path = Path(
    os.environ["SOURCE_MAIN"]
)

merged_path = Path(
    os.environ["MERGED_CSV"]
)

report_path = Path(
    os.environ["MERGE_REPORT"]
)


manifest = json.loads(
    manifest_path.read_text(
        encoding="utf-8"
    )
)

source = pd.read_csv(
    source_main_path
)


target_keys = [
    "scene_id",
    "im_id",
    "obj_id",
]

image_keys = [
    "scene_id",
    "im_id",
]


for column in target_keys:
    source[column] = source[column].astype(int)

source["time"] = pd.to_numeric(
    source["time"],
    errors="raise",
)


source_target_keys = set(
    source[target_keys].itertuples(
        index=False,
        name=None,
    )
)


all_images = (
    source[image_keys]
    .drop_duplicates()
    .sort_values(image_keys)
    .reset_index(drop=True)
)

if len(all_images) != 200:
    raise RuntimeError(
        f"Expected 200 LM-O images, "
        f"found {len(all_images)}"
    )

all_image_index = pd.MultiIndex.from_frame(
    all_images
)


# One source coarse time per image.
coarse_time = (
    source.groupby(
        image_keys,
        sort=False,
    )["time"]
    .max()
    .reindex(all_image_index)
)

if coarse_time.isna().any():
    raise RuntimeError(
        "Missing source coarse runtime"
    )

if not np.isfinite(
    coarse_time.to_numpy()
).all():
    raise RuntimeError(
        "Invalid source coarse runtime"
    )

if float(coarse_time.min()) < 0:
    raise RuntimeError(
        "Negative source coarse runtime"
    )


combined_time = coarse_time.copy()

refined_frames = []
group_reports = []

sum_process_wall_seconds = 0


for group in manifest["groups"]:
    output_path = Path(
        group["stable_csv"]
    )

    if not output_path.is_file():
        raise FileNotFoundError(output_path)

    frame = pd.read_csv(output_path)

    for column in target_keys:
        frame[column] = frame[column].astype(int)

    frame["time"] = pd.to_numeric(
        frame["time"],
        errors="raise",
    )

    expected_rows = int(
        group["target_count"]
    )

    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{output_path}: expected "
            f"{expected_rows} rows, "
            f"found {len(frame)}"
        )

    if frame.duplicated(target_keys).any():
        raise RuntimeError(
            f"Duplicate target keys: {output_path}"
        )

    refined_frames.append(frame)


    group_time_by_image = frame.groupby(
        image_keys,
        sort=False,
    )["time"]


    within_image_spread = group_time_by_image.agg(
        lambda values: float(
            values.max() - values.min()
        )
    )

    maximum_spread = float(
        within_image_spread.max()
    )

    if maximum_spread > 1e-3:
        raise RuntimeError(
            f"Inconsistent per-image group time "
            f"in {output_path}: {maximum_spread}"
        )


    group_total_time_active = (
        group_time_by_image.max()
    )

    active_coarse_time = coarse_time.reindex(
        group_total_time_active.index
    )

    if active_coarse_time.isna().any():
        raise RuntimeError(
            f"Missing coarse time for {output_path}"
        )


    raw_increment = (
        group_total_time_active
        - active_coarse_time
    )


    # Small negatives can occur because the source and refinement runs
    # were measured in separate processes.
    strongly_negative = raw_increment < -0.50

    if int(strongly_negative.sum()) > 0:
        examples = (
            raw_increment[
                strongly_negative
            ]
            .head(10)
            .to_dict()
        )

        raise RuntimeError(
            "Group output timing does not appear to "
            "include source coarse timing. "
            f"Examples: {examples}"
        )


    refinement_increment_active = (
        raw_increment.clip(lower=0.0)
    )

    refinement_increment_all = (
        refinement_increment_active
        .reindex(
            all_image_index,
            fill_value=0.0,
        )
    )

    if not np.isfinite(
        refinement_increment_all.to_numpy()
    ).all():
        raise RuntimeError(
            f"Invalid refinement increment: {output_path}"
        )


    combined_time = (
        combined_time
        + refinement_increment_all
    )


    process_wall_path = Path(
        group["process_wall_file"]
    )

    process_wall_seconds = None

    if process_wall_path.is_file():
        text = (
            process_wall_path
            .read_text(encoding="utf-8")
            .strip()
        )

        if text:
            process_wall_seconds = int(text)

            sum_process_wall_seconds += (
                process_wall_seconds
            )


    group_reports.append(
        {
            "k": int(group["k"]),
            "iterations": int(
                group["iterations"]
            ),
            "objects": group["objects"],
            "target_count": int(
                group["target_count"]
            ),
            "images_with_targets": int(
                len(group_total_time_active)
            ),
            "mean_group_total_time_active_images_s": (
                float(group_total_time_active.mean())
            ),
            "mean_refinement_increment_active_images_s": (
                float(refinement_increment_active.mean())
            ),
            "mean_refinement_increment_all_200_images_s": (
                float(refinement_increment_all.mean())
            ),
            "maximum_within_image_time_spread_s": (
                maximum_spread
            ),
            "process_wall_seconds": (
                process_wall_seconds
            ),
        }
    )


merged = pd.concat(
    refined_frames,
    ignore_index=True,
)

merged = (
    merged
    .sort_values(target_keys)
    .reset_index(drop=True)
)


if len(merged) != 1445:
    raise RuntimeError(
        f"Expected 1445 merged rows, "
        f"found {len(merged)}"
    )

if merged.duplicated(target_keys).any():
    raise RuntimeError(
        "Duplicate keys in merged prediction"
    )


merged_target_keys = set(
    merged[target_keys].itertuples(
        index=False,
        name=None,
    )
)

if merged_target_keys != source_target_keys:
    missing = sorted(
        source_target_keys
        - merged_target_keys
    )[:20]

    extra = sorted(
        merged_target_keys
        - source_target_keys
    )[:20]

    raise RuntimeError(
        "Merged target-key mismatch. "
        f"Missing={missing}; extra={extra}"
    )


combined_time_frame = (
    combined_time
    .rename("time")
    .reset_index()
)


# Replace the individual group time with full policy time.
merged = (
    merged
    .drop(columns=["time"])
    .merge(
        combined_time_frame,
        on=image_keys,
        how="left",
        validate="many_to_one",
    )
)


if merged["time"].isna().any():
    raise RuntimeError(
        "Missing merged policy runtime"
    )

if not np.isfinite(
    merged["time"].to_numpy()
).all():
    raise RuntimeError(
        "Invalid merged policy runtime"
    )

if float(merged["time"].min()) < 0:
    raise RuntimeError(
        "Negative merged policy runtime"
    )


merged_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

merged.to_csv(
    merged_path,
    index=False,
)


merged_sha256 = hashlib.sha256(
    merged_path.read_bytes()
).hexdigest()


report = {
    "status": (
        "k345_i345_policy_merge_complete"
    ),
    "rows": int(len(merged)),
    "images": int(len(all_images)),
    "targets": int(len(merged_target_keys)),
    "objects": sorted(
        merged["obj_id"]
        .astype(int)
        .unique()
        .tolist()
    ),
    "source_coarse_time_s_per_image": {
        "mean": float(coarse_time.mean()),
        "median": float(coarse_time.median()),
        "p90": float(
            coarse_time.quantile(0.90)
        ),
        "maximum": float(coarse_time.max()),
    },
    "groups": group_reports,
    "combined_measured_policy_compute_time_s_per_image": {
        "mean": float(combined_time.mean()),
        "median": float(combined_time.median()),
        "p90": float(
            combined_time.quantile(0.90)
        ),
        "maximum": float(combined_time.max()),
    },
    "sum_of_group_process_wall_seconds": int(
        sum_process_wall_seconds
    ),
    "timing_definition": (
        "Source coarse runtime counted once plus "
        "measured refinement increments from the "
        "five K3-5/I3-5 policy groups."
    ),
    "timing_warning": (
        "The five groups were executed as separate "
        "processes. This is reconstructed measured "
        "policy compute time per image, not one "
        "integrated online wall-clock process."
    ),
    "merged_csv": str(merged_path),
    "merged_sha256": merged_sha256,
}


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 100)
print("MERGED K3-5 / I3-5 POLICY RESULT")
print("=" * 100)

print("Rows:", len(merged))
print("Images:", len(all_images))
print("Targets:", len(merged_target_keys))

print(
    "Objects:",
    sorted(
        merged["obj_id"]
        .astype(int)
        .unique()
        .tolist()
    ),
)

print()
print(
    "Source coarse mean:",
    f"{coarse_time.mean():.6f} s/image",
)

print()
print("Group refinement increments:")

for group in group_reports:
    print(
        f"  K={group['k']}, "
        f"I={group['iterations']}, "
        f"objects={group['objects']}: "
        f"{group['mean_refinement_increment_all_200_images_s']:.6f} "
        "s/image"
    )

print()
print(
    "Combined measured policy compute mean:",
    f"{combined_time.mean():.6f} s/image",
)

print(
    "Combined median:",
    f"{combined_time.median():.6f} s/image",
)

print(
    "Combined p90:",
    f"{combined_time.quantile(0.90):.6f} s/image",
)

print(
    "Combined maximum:",
    f"{combined_time.max():.6f} s/image",
)

print(
    "Merged SHA256:",
    merged_sha256,
)

print("MERGE AND RUNTIME RECONSTRUCTION: PASS")
PY


# =============================================================================
# 7. PREPARE OFFICIAL BOP19 INPUT
# =============================================================================

echo
echo "================================================================================"
echo "STEP 6 - PREPARE OFFICIAL BOP19"
echo "================================================================================"

BOP_FILENAME="real-objpolicy-k345-i345-v1_lmo-test.csv"
RESULT_STEM="${BOP_FILENAME%.csv}"

LOCAL_RESULTS="/content/bop_results_${ADAPT_ID}"
LOCAL_EVAL="/content/bop_eval_${ADAPT_ID}"

LOCAL_RESULT="$LOCAL_RESULTS/$BOP_FILENAME"
SCORES_SOURCE="$LOCAL_EVAL/$RESULT_STEM/scores_bop19.json"

EVAL_BIN="/content/${ADAPT_ID}_eval_bin"

EVAL_PYTHONPATH="$BOPTK:$BOPTK/bop_toolkit_lib:$REPO:$REPO/src:${PYTHONPATH:-}"


rm -rf \
    "$LOCAL_RESULTS" \
    "$LOCAL_EVAL" \
    "$EVAL_BIN"

mkdir -p \
    "$LOCAL_RESULTS" \
    "$LOCAL_EVAL" \
    "$EVAL_BIN"


cp -f \
    "$MERGED_CSV" \
    "$LOCAL_RESULT"


cat > "$EVAL_BIN/python" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export PYTHONPATH="$EVAL_PYTHONPATH"

export BOP_PATH="$ROOT/datasets"
export BOP_DATASETS_DIR="$ROOT/datasets"
export BOP_DATASETS_PATH="$ROOT/datasets"

export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export VISPY_APP=egl

exec /content/gp_exec/bin/python "\$@"
EOF

chmod +x "$EVAL_BIN/python"
bash -n "$EVAL_BIN/python"


RESULT_PATH="$LOCAL_RESULT" \
"$PY" -u - <<'PY'
from pathlib import Path
import os

import numpy as np
import pandas as pd


path = Path(
    os.environ["RESULT_PATH"]
)

frame = pd.read_csv(path)

required_columns = {
    "scene_id",
    "im_id",
    "obj_id",
    "score",
    "R",
    "t",
    "time",
}

missing_columns = (
    required_columns
    - set(frame.columns)
)

if missing_columns:
    raise RuntimeError(
        f"Missing BOP columns: "
        f"{sorted(missing_columns)}"
    )

if len(frame) != 1445:
    raise RuntimeError(
        f"Expected 1445 BOP rows, "
        f"found {len(frame)}"
    )

target_keys = [
    "scene_id",
    "im_id",
    "obj_id",
]

if frame.duplicated(target_keys).any():
    raise RuntimeError(
        "Duplicate BOP target keys"
    )

times = pd.to_numeric(
    frame["time"],
    errors="raise",
)

if not np.isfinite(
    times.to_numpy()
).all():
    raise RuntimeError(
        "Invalid BOP runtime values"
    )

image_time = (
    frame[
        ["scene_id", "im_id", "time"]
    ]
    .drop_duplicates(
        ["scene_id", "im_id"]
    )
)

if len(image_time) != 200:
    raise RuntimeError(
        f"Expected 200 image-time records, "
        f"found {len(image_time)}"
    )

print("Rows:", len(frame))
print("Images:", len(image_time))

print(
    "Objects:",
    sorted(
        frame["obj_id"]
        .astype(int)
        .unique()
        .tolist()
    ),
)

print(
    "Mean policy time:",
    float(image_time["time"].mean()),
)

print("OFFICIAL BOP INPUT VALIDATION: PASS")
PY


# =============================================================================
# 8. RUN OFFICIAL BOP19
# =============================================================================

echo
echo "================================================================================"
echo "STEP 7 - RUN OFFICIAL BOP19"
echo "================================================================================"

set +e

PATH="$EVAL_BIN:$PATH" \
PYTHONPATH="$EVAL_PYTHONPATH" \
timeout \
    --signal=TERM \
    --kill-after=120s \
    90m \
"$EVAL_BIN/python" -u \
    "$BOPTK/scripts/eval_bop19_pose.py" \
    --renderer_type vispy \
    --result_filenames "$BOP_FILENAME" \
    --results_path "$LOCAL_RESULTS" \
    --eval_path "$LOCAL_EVAL" \
    --targets_filename test_targets_bop19.json \
    --num_workers 1 \
    2>&1 |
tee "$EVAL_LOG"

EVAL_RC="${PIPESTATUS[0]}"

set -e


echo
echo "Official BOP19 exit code:"
echo "  $EVAL_RC"


if [[ "$EVAL_RC" -eq 124 ]]; then
    echo "ERROR: Official BOP19 timed out"
    exit 124
fi

if [[ "$EVAL_RC" -ne 0 ]]; then
    echo "ERROR: Official BOP19 failed"

    tail -n 350 "$EVAL_LOG" || true

    exit "$EVAL_RC"
fi


test -s "$SCORES_SOURCE" || {
    echo "ERROR: scores_bop19.json missing:"
    echo "  $SCORES_SOURCE"
    exit 1
}


cp -f \
    "$SCORES_SOURCE" \
    "$SCORES_JSON"

echo "OFFICIAL BOP19: PASS"


# =============================================================================
# 9. CREATE FINAL AR AND RUNTIME REPORT
# =============================================================================

echo
echo "================================================================================"
echo "STEP 8 - FINAL AR AND RUNTIME REPORT"
echo "================================================================================"

SCORES_JSON="$SCORES_JSON" \
MERGE_REPORT="$MERGE_REPORT" \
MERGED_CSV="$MERGED_CSV" \
POLICY_CSV="$POLICY_CSV" \
FINAL_JSON="$FINAL_JSON" \
FINAL_TXT="$FINAL_TXT" \
"$PY" -u - <<'PY'
from pathlib import Path
import json
import os


scores_path = Path(
    os.environ["SCORES_JSON"]
)

runtime_report_path = Path(
    os.environ["MERGE_REPORT"]
)

merged_csv_path = Path(
    os.environ["MERGED_CSV"]
)

policy_csv_path = Path(
    os.environ["POLICY_CSV"]
)

final_json_path = Path(
    os.environ["FINAL_JSON"]
)

final_txt_path = Path(
    os.environ["FINAL_TXT"]
)


scores = json.loads(
    scores_path.read_text(
        encoding="utf-8"
    )
)

runtime_report = json.loads(
    runtime_report_path.read_text(
        encoding="utf-8"
    )
)


ar = float(
    scores["bop19_average_recall"]
)

ar_vsd = float(
    scores["bop19_average_recall_vsd"]
)

ar_mssd = float(
    scores["bop19_average_recall_mssd"]
)

ar_mspd = float(
    scores["bop19_average_recall_mspd"]
)

official_time = float(
    scores["bop19_average_time_per_image"]
)

reconstructed_time = float(
    runtime_report[
        "combined_measured_policy_compute_time_s_per_image"
    ]["mean"]
)


time_difference = abs(
    official_time
    - reconstructed_time
)

if time_difference > 1e-5:
    raise RuntimeError(
        "Official BOP time and reconstructed "
        "runtime do not match: "
        f"{time_difference}"
    )


summary = {
    "status": (
        "k345_i345_policy_official_bop19_complete"
    ),
    "dataset": "lmo",
    "images": 200,
    "targets": 1445,
    "execution": (
        "fresh MegaPose refinement from original "
        "clean coarse predictions"
    ),
    "policy": {
        "obj_01_ape": {
            "k": 4,
            "iterations": 4,
        },
        "obj_05_can": {
            "k": 5,
            "iterations": 4,
        },
        "obj_06_cat": {
            "k": 3,
            "iterations": 4,
        },
        "obj_08_driller": {
            "k": 3,
            "iterations": 5,
        },
        "obj_09_duck": {
            "k": 4,
            "iterations": 4,
        },
        "obj_10_eggbox": {
            "k": 4,
            "iterations": 4,
        },
        "obj_11_glue": {
            "k": 3,
            "iterations": 5,
        },
        "obj_12_holepuncher": {
            "k": 3,
            "iterations": 3,
        },
    },
    "official_bop19": {
        "AR": ar,
        "AR_percent": ar * 100.0,
        "AR_VSD": ar_vsd,
        "AR_VSD_percent": ar_vsd * 100.0,
        "AR_MSSD": ar_mssd,
        "AR_MSSD_percent": ar_mssd * 100.0,
        "AR_MSPD": ar_mspd,
        "AR_MSPD_percent": ar_mspd * 100.0,
        "time_s_per_image": official_time,
    },
    "runtime_definition": (
        "Source coarse runtime counted once plus "
        "measured refinement increments from five "
        "separately executed policy groups."
    ),
    "runtime_warning": (
        "This is reconstructed measured policy "
        "compute time per image, not one integrated "
        "online wall-clock process."
    ),
    "files": {
        "policy_csv": str(policy_csv_path),
        "merged_csv": str(merged_csv_path),
        "scores_json": str(scores_path),
        "runtime_report": str(
            runtime_report_path
        ),
    },
}


final_json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)


lines = [
    "=" * 105,
    (
        "LM-O FRESH K3-5 / I3-5 OBJECT-POLICY "
        "— OFFICIAL BOP19"
    ),
    "=" * 105,
    "",
    "Execution:",
    (
        "  Fresh MegaPose refinement from the "
        "original clean coarse predictions."
    ),
    (
        "  No refined pose from the K=1 policy "
        "or KxI grid was reused."
    ),
    "",
    "Policy:",
    "  obj 01 ape          -> K=4, I=4",
    "  obj 05 can          -> K=5, I=4",
    "  obj 06 cat          -> K=3, I=4",
    "  obj 08 driller      -> K=3, I=5",
    "  obj 09 duck         -> K=4, I=4",
    "  obj 10 eggbox       -> K=4, I=4",
    "  obj 11 glue         -> K=3, I=5",
    "  obj 12 holepuncher  -> K=3, I=3",
    "",
    "Official BOP19:",
    f"  AR:      {ar * 100.0:.6f}%",
    f"  AR_VSD:  {ar_vsd * 100.0:.6f}%",
    f"  AR_MSSD: {ar_mssd * 100.0:.6f}%",
    f"  AR_MSPD: {ar_mspd * 100.0:.6f}%",
    "",
    "Runtime:",
    (
        "  combined measured policy compute time:"
    ),
    f"  {official_time:.6f} s/image",
    "",
    "Timing definition:",
    (
        "  source coarse runtime counted once"
    ),
    (
        "  + K3/I3 refinement increment"
    ),
    (
        "  + K3/I4 refinement increment"
    ),
    (
        "  + K3/I5 refinement increment"
    ),
    (
        "  + K4/I4 refinement increment"
    ),
    (
        "  + K5/I4 refinement increment"
    ),
    "",
    "Timing warning:",
    (
        "  Five groups were executed separately."
    ),
    (
        "  This is reconstructed measured compute "
        "time, not one integrated online "
        "wall-clock process."
    ),
    "",
    f"Policy CSV:      {policy_csv_path}",
    f"Merged CSV:      {merged_csv_path}",
    f"Official scores: {scores_path}",
    f"Runtime report:  {runtime_report_path}",
    f"Final JSON:      {final_json_path}",
    "=" * 105,
]


final_txt_path.write_text(
    "\n".join(lines)
    + "\n",
    encoding="utf-8",
)


print(
    final_txt_path.read_text(
        encoding="utf-8"
    )
)

print("FINAL OFFICIAL AR AND RUNTIME REPORT: PASS")
PY


# =============================================================================
# 10. FINAL ARTIFACT VALIDATION
# =============================================================================

echo
echo "================================================================================"
echo "STEP 9 - FINAL ARTIFACT VALIDATION"
echo "================================================================================"

final_files=(
    "$POLICY_CSV"
    "$MANIFEST_JSON"
    "$MANIFEST_TSV"

    "$MERGED_CSV"
    "$MERGE_REPORT"

    "$SCORES_JSON"

    "$FINAL_JSON"
    "$FINAL_TXT"

    "$MASTER_LOG"
    "$EVAL_LOG"
)

for file in "${final_files[@]}"; do
    test -s "$file" || {
        echo "ERROR: final artifact missing:"
        echo "  $file"
        exit 1
    }

    echo "PASS:"
    echo "  $file"
done


echo
echo "SHA256:"

sha256sum \
    "$POLICY_CSV" \
    "$MANIFEST_JSON" \
    "$MERGED_CSV" \
    "$MERGE_REPORT" \
    "$SCORES_JSON" \
    "$FINAL_JSON" \
    "$FINAL_TXT"


echo
echo "================================================================================"
echo "LM-O K3-5 / I3-5 OBJECT-POLICY RERUN: COMPLETE"
echo "================================================================================"

echo "Final summary:"
echo "  $FINAL_TXT"

echo
echo "Official BOP19 scores:"
echo "  $SCORES_JSON"

echo
echo "Merged prediction:"
echo "  $MERGED_CSV"

echo
echo "Runtime report:"
echo "  $MERGE_REPORT"

echo
echo "Master log:"
echo "  $MASTER_LOG"

