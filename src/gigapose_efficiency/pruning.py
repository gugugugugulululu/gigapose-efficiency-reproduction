from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bop_csv import load_bop_csv, target_columns, validate_bop_csv, write_csv


def prune_topk(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    keys = target_columns(frame)
    counts = frame.groupby(keys, sort=False).size()
    if int(counts.min()) < top_k:
        raise ValueError(
            f"some targets contain fewer than {top_k} hypotheses; "
            f"minimum is {int(counts.min())}"
        )
    ranked = frame.copy()
    ranked["__source_order"] = range(len(ranked))
    ranked = ranked.sort_values(
        keys + ["score", "__source_order"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    )
    output = ranked.groupby(keys, sort=False, as_index=False).head(top_k)
    output = output.sort_values(keys + ["score"], ascending=[True] * len(keys) + [False])
    return output.drop(columns="__source_order").reset_index(drop=True)


def prune_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int,
    expected_targets: int | None = None,
    expected_rows: int | None = None,
) -> dict[str, object]:
    frame = load_bop_csv(input_path)
    output = prune_topk(frame, top_k)
    report = validate_bop_csv(
        output,
        expected_rows=expected_rows,
        expected_targets=expected_targets,
        expected_hypotheses_per_target=top_k,
    )
    write_csv(output, output_path)
    return report
