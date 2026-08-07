from __future__ import annotations

import pandas as pd

from gigapose_efficiency.pruning import prune_topk


def make_multi() -> pd.DataFrame:
    rows = []
    for obj_id in (1, 5):
        for rank, score in enumerate((0.2, 0.9, 0.4, 0.8, 0.1)):
            rows.append(
                {
                    "scene_id": 1,
                    "im_id": 1,
                    "obj_id": obj_id,
                    "score": score,
                    "R": "1 0 0 0 1 0 0 0 1",
                    "t": f"{rank} 0 1",
                    "time": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_prune_top3_is_physical_and_score_sorted() -> None:
    output = prune_topk(make_multi(), 3)
    assert len(output) == 6
    counts = output.groupby(["scene_id", "im_id", "obj_id"]).size()
    assert counts.tolist() == [3, 3]
    for _, group in output.groupby(["scene_id", "im_id", "obj_id"]):
        assert group["score"].tolist() == [0.9, 0.8, 0.4]
