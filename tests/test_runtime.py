from __future__ import annotations

from pathlib import Path

import pandas as pd

from gigapose_efficiency.runtime import reconstruct_grouped_runtime


def row(image: int, obj: int, time: float) -> dict[str, object]:
    return {
        "scene_id": 1,
        "im_id": image,
        "obj_id": obj,
        "score": 1.0,
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "0 0 1",
        "time": time,
    }


def test_reconstructed_runtime_counts_coarse_once(tmp_path: Path) -> None:
    source_full = pd.DataFrame([row(1, 1, 1.0), row(2, 1, 1.2)])
    source_group_a = pd.DataFrame([row(1, 1, 1.0)])
    refined_group_a = pd.DataFrame([row(1, 1, 1.5)])
    source_group_b = pd.DataFrame([row(2, 5, 1.2)])
    refined_group_b = pd.DataFrame([row(2, 5, 2.0)])

    paths = []
    for name, frame in (
        ("source_full", source_full),
        ("source_a", source_group_a),
        ("refined_a", refined_group_a),
        ("source_b", source_group_b),
        ("refined_b", refined_group_b),
    ):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    report = reconstruct_grouped_runtime(
        paths[0],
        [(paths[1], paths[2]), (paths[3], paths[4])],
        group_time_mode="total",
    )
    # coarse mean=(1.0+1.2)/2=1.1; increments=(0.5+0.8)/2=0.65
    assert abs(report["reconstructed_mean_s_per_image"] - 1.75) < 1e-9
    assert report["image_times"] == [
        {"scene_id": 1, "im_id": 1, "time": 1.5},
        {"scene_id": 1, "im_id": 2, "time": 2.0},
    ]
