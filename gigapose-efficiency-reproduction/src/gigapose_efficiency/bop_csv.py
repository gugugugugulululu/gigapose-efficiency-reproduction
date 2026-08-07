from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("scene_id", "im_id", "obj_id", "score", "R", "t", "time")
BASE_TARGET_COLUMNS = ("scene_id", "im_id", "obj_id")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bop_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing BOP columns: {missing}")
    for column in ("scene_id", "im_id", "obj_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    frame["time"] = pd.to_numeric(frame["time"], errors="raise")
    if "instance_id" in frame.columns:
        frame["instance_id"] = pd.to_numeric(frame["instance_id"], errors="raise").astype("int64")
    return frame


def target_columns(frame: pd.DataFrame) -> list[str]:
    columns = list(BASE_TARGET_COLUMNS)
    if "instance_id" in frame.columns:
        columns.append("instance_id")
    return columns


def count_targets(frame: pd.DataFrame) -> int:
    return int(frame[target_columns(frame)].drop_duplicates().shape[0])


def validate_pose_strings(frame: pd.DataFrame) -> None:
    for column, expected in (("R", 9), ("t", 3)):
        for row_index, value in frame[column].astype(str).items():
            tokens = value.replace(",", " ").split()
            if len(tokens) != expected:
                raise ValueError(
                    f"row {row_index}: {column} must contain {expected} numbers; got {len(tokens)}"
                )
            values = np.asarray([float(token) for token in tokens], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"row {row_index}: non-finite value in {column}")


def validate_bop_csv(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_targets: int | None = None,
    expected_images: int | None = None,
    expected_objects: Iterable[int] | None = None,
    expected_hypotheses_per_target: int | None = None,
) -> dict[str, object]:
    validate_pose_strings(frame)
    target_cols = target_columns(frame)
    target_count = count_targets(frame)
    image_count = int(frame[["scene_id", "im_id"]].drop_duplicates().shape[0])
    objects = sorted(frame["obj_id"].unique().astype(int).tolist())
    counts = frame.groupby(target_cols, sort=False).size()

    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(frame)}")
    if expected_targets is not None and target_count != expected_targets:
        raise ValueError(f"expected {expected_targets} targets, got {target_count}")
    if expected_images is not None and image_count != expected_images:
        raise ValueError(f"expected {expected_images} images, got {image_count}")
    if expected_objects is not None and objects != sorted(int(value) for value in expected_objects):
        raise ValueError(f"expected objects {sorted(expected_objects)}, got {objects}")
    if expected_hypotheses_per_target is not None:
        bad = counts[counts != expected_hypotheses_per_target]
        if not bad.empty:
            raise ValueError(
                f"expected {expected_hypotheses_per_target} hypotheses per target; "
                f"observed range {int(counts.min())}--{int(counts.max())}"
            )

    return {
        "rows": int(len(frame)),
        "targets": target_count,
        "images": image_count,
        "objects": objects,
        "hypotheses_min": int(counts.min()),
        "hypotheses_max": int(counts.max()),
        "mean_time_s_per_image": mean_image_time(frame),
    }


def mean_image_time(frame: pd.DataFrame, *, tolerance: float = 1e-6) -> float:
    grouped = frame.groupby(["scene_id", "im_id"], sort=False)["time"]
    spread = grouped.max() - grouped.min()
    if bool((spread > tolerance).any()):
        worst = float(spread.max())
        raise ValueError(
            "BOP time must be constant for all estimates in an image; "
            f"maximum within-image spread is {worst:.9f} s"
        )
    return float(grouped.first().mean())


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
