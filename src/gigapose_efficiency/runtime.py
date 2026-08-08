from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bop_csv import load_bop_csv, mean_image_time

IMAGE_KEYS = ["scene_id", "im_id"]


def _image_times(frame: pd.DataFrame, tolerance: float = 1e-6) -> pd.Series:
    grouped = frame.groupby(IMAGE_KEYS, sort=False)["time"]
    spread = grouped.max() - grouped.min()
    if bool((spread > tolerance).any()):
        raise ValueError(
            "time is not constant within an image; "
            f"maximum spread={float(spread.max()):.9f} s"
        )
    return grouped.first()


def reconstruct_grouped_runtime(
    source_main_csv: str | Path,
    group_pairs: list[tuple[str | Path, str | Path]],
    *,
    group_time_mode: str = "total",
) -> dict[str, object]:
    """Reconstruct policy runtime from measured object-group outputs.

    Args:
        source_main_csv: Full accelerated-coarse main CSV. Its per-image time is
            counted once.
        group_pairs: Pairs of (group source main CSV, group refined CSV).
        group_time_mode: ``total`` means refined time includes source coarse time,
            so the group increment is refined minus source. ``increment`` means the
            refined CSV already stores refinement-only time.
    """
    if group_time_mode not in {"total", "increment"}:
        raise ValueError("group_time_mode must be 'total' or 'increment'")

    source_full = load_bop_csv(source_main_csv)
    full_times = _image_times(source_full)
    total_images = int(len(full_times))
    coarse_mean = float(full_times.mean())

    sum_increment_over_all_images = 0.0
    per_image_total = full_times.copy()
    groups: list[dict[str, object]] = []

    for source_group_csv, refined_group_csv in group_pairs:
        source_group = load_bop_csv(source_group_csv)
        refined_group = load_bop_csv(refined_group_csv)
        source_times = _image_times(source_group)
        refined_times = _image_times(refined_group)
        # Image coverage is a set property, not an ordering property.
        #
        # MegaPose may emit the same image keys in a different order from the
        # group source CSV. MultiIndex.equals() is order-sensitive and therefore
        # incorrectly rejects valid group outputs whose coverage is identical.
        #
        # First detect genuine missing/extra image keys. Then explicitly align
        # refined timings to the source order before arithmetic.
        missing_source = refined_times.index.difference(source_times.index)
        missing_refined = source_times.index.difference(refined_times.index)

        if len(missing_source) or len(missing_refined):
            raise ValueError(
                f"group image coverage mismatch; missing source={list(missing_source[:5])}, "
                f"missing refined={list(missing_refined[:5])}"
            )

        refined_times = refined_times.reindex(source_times.index)

        if group_time_mode == "total":
            increments = refined_times - source_times
        else:
            increments = refined_times
        if bool((increments < -1e-6).any()):
            raise ValueError(
                f"negative refinement increment detected; minimum={float(increments.min()):.9f} s"
            )
        increments = increments.clip(lower=0.0)
        group_sum = float(increments.sum())
        sum_increment_over_all_images += group_sum
        for image_key, value in increments.items():
            per_image_total.loc[image_key] = float(per_image_total.loc[image_key]) + float(value)
        groups.append(
            {
                "source_group_csv": str(Path(source_group_csv).resolve()),
                "refined_group_csv": str(Path(refined_group_csv).resolve()),
                "active_images": int(len(increments)),
                "mean_increment_active_image_s": float(increments.mean()),
                "contribution_over_full_split_s_per_image": group_sum / total_images,
            }
        )

    refinement_contribution = sum_increment_over_all_images / total_images
    reconstructed = coarse_mean + refinement_contribution
    return {
        "total_images": total_images,
        "source_coarse_mean_s_per_image": coarse_mean,
        "refinement_group_contribution_s_per_image": refinement_contribution,
        "reconstructed_mean_s_per_image": reconstructed,
        "group_time_mode": group_time_mode,
        "groups": groups,
        "image_times": [
            {"scene_id": int(index[0]), "im_id": int(index[1]), "time": float(value)}
            for index, value in per_image_total.items()
        ],
    }
