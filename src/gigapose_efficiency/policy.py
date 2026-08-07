from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .bop_csv import load_bop_csv, target_columns, validate_bop_csv, write_csv, write_json
from .pruning import prune_topk

POLICY_COLUMNS = ("obj_id", "object_name", "k", "i", "expected_targets")


def load_policy(path: str | Path) -> pd.DataFrame:
    policy = pd.read_csv(path)
    missing = [column for column in POLICY_COLUMNS if column not in policy.columns]
    if missing:
        raise ValueError(f"policy is missing columns: {missing}")
    policy = policy.copy()
    for column in ("obj_id", "k", "i", "expected_targets"):
        policy[column] = pd.to_numeric(policy[column], errors="raise").astype("int64")
    if policy["obj_id"].duplicated().any():
        raise ValueError("policy contains duplicate obj_id values")
    if bool((policy["k"] < 1).any()) or bool((policy["i"] < 1).any()):
        raise ValueError("all K and I values must be >= 1")
    return policy.sort_values("obj_id").reset_index(drop=True)


def build_adaptive_groups(
    main_csv: str | Path,
    multi_csv: str | Path,
    policy_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_targets: int | None = None,
    expected_total_hypotheses: int | None = None,
) -> dict[str, object]:
    main = load_bop_csv(main_csv)
    multi = load_bop_csv(multi_csv)
    policy = load_policy(policy_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_bop_csv(main, expected_targets=expected_targets, expected_hypotheses_per_target=1)
    validate_bop_csv(multi, expected_targets=expected_targets)

    main_objects = set(main["obj_id"].unique().astype(int))
    policy_objects = set(policy["obj_id"].astype(int))
    if main_objects != policy_objects:
        raise ValueError(
            f"policy objects do not match input objects: policy={sorted(policy_objects)}, "
            f"input={sorted(main_objects)}"
        )

    group_records: list[dict[str, object]] = []
    retained_frames: list[pd.DataFrame] = []

    for (k_value, i_value), group_policy in policy.groupby(["k", "i"], sort=True):
        object_ids = sorted(group_policy["obj_id"].astype(int).tolist())
        group_name = f"k{k_value}_i{i_value}"
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        group_main = main[main["obj_id"].isin(object_ids)].copy()
        group_multi_full = multi[multi["obj_id"].isin(object_ids)].copy()
        group_multi = prune_topk(group_multi_full, int(k_value))

        expected_group_targets = int(group_policy["expected_targets"].sum())
        main_report = validate_bop_csv(
            group_main,
            expected_targets=expected_group_targets,
            expected_hypotheses_per_target=1,
        )
        multi_report = validate_bop_csv(
            group_multi,
            expected_rows=expected_group_targets * int(k_value),
            expected_targets=expected_group_targets,
            expected_hypotheses_per_target=int(k_value),
        )

        main_path = group_dir / "main.csv"
        multi_path = group_dir / "MultiHypothesis.csv"
        write_csv(group_main, main_path)
        write_csv(group_multi, multi_path)
        retained_frames.append(group_multi)

        group_records.append(
            {
                "group": group_name,
                "k": int(k_value),
                "i": int(i_value),
                "object_ids": object_ids,
                "targets": expected_group_targets,
                "hypotheses": int(len(group_multi)),
                "images": int(group_main[["scene_id", "im_id"]].drop_duplicates().shape[0]),
                "main_csv": str(main_path.resolve()),
                "multi_csv": str(multi_path.resolve()),
                "main_validation": main_report,
                "multi_validation": multi_report,
            }
        )

    retained = pd.concat(retained_frames, ignore_index=True)
    total_report = validate_bop_csv(retained, expected_targets=expected_targets)
    if expected_total_hypotheses is not None and len(retained) != expected_total_hypotheses:
        raise ValueError(
            f"expected {expected_total_hypotheses} retained hypotheses, got {len(retained)}"
        )

    manifest = {
        "policy_csv": str(Path(policy_csv).resolve()),
        "main_csv": str(Path(main_csv).resolve()),
        "multi_csv": str(Path(multi_csv).resolve()),
        "groups": group_records,
        "total": total_report,
    }
    write_json(manifest, output_dir / "manifest.json")
    policy.to_csv(output_dir / "policy_used.csv", index=False)
    return manifest


def merge_group_results(
    refined_csvs: list[str | Path],
    output_csv: str | Path,
    *,
    expected_targets: int | None = None,
    image_times_csv: str | Path | None = None,
) -> dict[str, object]:
    if not refined_csvs:
        raise ValueError("at least one refined CSV is required")
    frames = [load_bop_csv(path) for path in refined_csvs]
    merged = pd.concat(frames, ignore_index=True)
    keys = target_columns(merged)
    duplicates = merged.duplicated(keys, keep=False)
    if bool(duplicates.any()):
        examples = merged.loc[duplicates, keys].head(10).to_dict("records")
        raise ValueError(f"duplicate refined targets across groups: {examples}")
    if image_times_csv is not None:
        image_times = pd.read_csv(image_times_csv)
        required = {"scene_id", "im_id", "time"}
        if not required.issubset(image_times.columns):
            raise ValueError(f"image-times CSV must contain {sorted(required)}")
        merged = merged.drop(columns=["time"]).merge(
            image_times[["scene_id", "im_id", "time"]],
            on=["scene_id", "im_id"],
            how="left",
            validate="many_to_one",
        )
        if merged["time"].isna().any():
            raise ValueError("missing reconstructed time for one or more merged targets")

    report = validate_bop_csv(
        merged,
        expected_rows=expected_targets,
        expected_targets=expected_targets,
        expected_hypotheses_per_target=1,
    )
    merged = merged.sort_values(keys).reset_index(drop=True)
    write_csv(merged, output_csv)
    return report
