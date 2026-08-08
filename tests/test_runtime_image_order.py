from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gigapose_efficiency.runtime import (
    reconstruct_grouped_runtime,
)


def _write_bop_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def _row(
    scene_id: int,
    im_id: int,
    obj_id: int,
    time: float,
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "im_id": im_id,
        "obj_id": obj_id,
        "score": 1.0,
        "R": "1 0 0 0 1 0 0 0 1",
        "t": "0 0 1000",
        "time": time,
    }


def test_group_runtime_accepts_same_coverage_in_different_order(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.csv"
    source = tmp_path / "source.csv"
    refined = tmp_path / "refined.csv"

    _write_bop_csv(
        full,
        [
            _row(2, 10, 1, 1.0),
            _row(2, 20, 1, 1.0),
            _row(2, 30, 1, 1.0),
        ],
    )

    _write_bop_csv(
        source,
        [
            _row(2, 10, 1, 1.0),
            _row(2, 20, 1, 1.0),
            _row(2, 30, 1, 1.0),
        ],
    )

    # Same image coverage, deliberately shuffled.
    _write_bop_csv(
        refined,
        [
            _row(2, 30, 1, 1.4),
            _row(2, 10, 1, 1.2),
            _row(2, 20, 1, 1.3),
        ],
    )

    report = reconstruct_grouped_runtime(
        full,
        [(source, refined)],
        group_time_mode="total",
    )

    assert report["total_images"] == 3

    assert (
        report[
            "source_coarse_mean_s_per_image"
        ]
        == pytest.approx(1.0)
    )

    assert (
        report[
            "refinement_group_contribution_s_per_image"
        ]
        == pytest.approx(0.3)
    )

    assert (
        report[
            "reconstructed_mean_s_per_image"
        ]
        == pytest.approx(1.3)
    )

    image_times = {
        (
            int(item["scene_id"]),
            int(item["im_id"]),
        ): float(item["time"])
        for item in report["image_times"]
    }

    assert (
        image_times[(2, 10)]
        == pytest.approx(1.2)
    )

    assert (
        image_times[(2, 20)]
        == pytest.approx(1.3)
    )

    assert (
        image_times[(2, 30)]
        == pytest.approx(1.4)
    )


def test_group_runtime_rejects_real_coverage_mismatch(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.csv"
    source = tmp_path / "source.csv"
    refined = tmp_path / "refined.csv"

    _write_bop_csv(
        full,
        [
            _row(2, 10, 1, 1.0),
            _row(2, 20, 1, 1.0),
            _row(2, 30, 1, 1.0),
        ],
    )

    _write_bop_csv(
        source,
        [
            _row(2, 10, 1, 1.0),
            _row(2, 20, 1, 1.0),
            _row(2, 30, 1, 1.0),
        ],
    )

    # 30 is genuinely missing; 40 is extra.
    _write_bop_csv(
        refined,
        [
            _row(2, 10, 1, 1.2),
            _row(2, 20, 1, 1.3),
            _row(2, 40, 1, 1.4),
        ],
    )

    with pytest.raises(
        ValueError,
        match="group image coverage mismatch",
    ):
        reconstruct_grouped_runtime(
            full,
            [(source, refined)],
            group_time_mode="total",
        )
