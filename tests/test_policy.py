from __future__ import annotations

from pathlib import Path

import pandas as pd

from gigapose_efficiency.policy import build_adaptive_groups, merge_group_results


def row(scene: int, image: int, obj: int, score: float, time: float) -> dict[str, object]:
    return {
        "scene_id": scene,
        "im_id": image,
        "obj_id": obj,
        "score": score,
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "0 0 1",
        "time": time,
    }


def test_build_groups_and_merge(tmp_path: Path) -> None:
    main = pd.DataFrame([row(1, 1, 1, 1.0, 1.0), row(1, 1, 5, 1.0, 1.0)])
    multi_rows = []
    for obj in (1, 5):
        for score in (0.9, 0.8, 0.7, 0.6, 0.5):
            multi_rows.append(row(1, 1, obj, score, 1.0))
    multi = pd.DataFrame(multi_rows)
    policy = pd.DataFrame(
        [
            {"obj_id": 1, "object_name": "ape", "k": 3, "i": 4, "expected_targets": 1},
            {"obj_id": 5, "object_name": "can", "k": 5, "i": 4, "expected_targets": 1},
        ]
    )
    main_path = tmp_path / "main.csv"
    multi_path = tmp_path / "multi.csv"
    policy_path = tmp_path / "policy.csv"
    main.to_csv(main_path, index=False)
    multi.to_csv(multi_path, index=False)
    policy.to_csv(policy_path, index=False)

    manifest = build_adaptive_groups(
        main_path,
        multi_path,
        policy_path,
        tmp_path / "groups",
        expected_targets=2,
        expected_total_hypotheses=8,
    )
    assert manifest["total"]["rows"] == 8
    assert len(manifest["groups"]) == 2

    refined_paths = []
    for group in manifest["groups"]:
        group_main = pd.read_csv(group["main_csv"])
        group_main["time"] = 2.0
        refined = tmp_path / f"{group['group']}_refined.csv"
        group_main.to_csv(refined, index=False)
        refined_paths.append(refined)

    image_times = pd.DataFrame([{"scene_id": 1, "im_id": 1, "time": 3.0}])
    image_times_path = tmp_path / "image_times.csv"
    image_times.to_csv(image_times_path, index=False)
    report = merge_group_results(
        refined_paths,
        tmp_path / "merged.csv",
        expected_targets=2,
        image_times_csv=image_times_path,
    )
    assert report["rows"] == 2
    merged = pd.read_csv(tmp_path / "merged.csv")
    assert merged["time"].tolist() == [3.0, 3.0]
