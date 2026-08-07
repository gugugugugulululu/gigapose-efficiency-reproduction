from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bop_csv import load_bop_csv, sha256_file, validate_bop_csv, write_json
from .patches import apply_empty_batch_guard
from .policy import build_adaptive_groups, merge_group_results
from .pruning import prune_file
from .runtime import reconstruct_grouped_runtime

LMO_OBJECTS = [1, 5, 6, 8, 9, 10, 11, 12]


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gigapose-efficiency")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a BOP CSV")
    validate.add_argument("--csv", required=True)
    validate.add_argument("--rows", type=int)
    validate.add_argument("--targets", type=int)
    validate.add_argument("--images", type=int)
    validate.add_argument("--hypotheses", type=int)
    validate.add_argument("--lmo-objects", action="store_true")

    prune = sub.add_parser("prune-topk", help="physically retain top-K hypotheses per target")
    prune.add_argument("--input", required=True)
    prune.add_argument("--output", required=True)
    prune.add_argument("--top-k", required=True, type=int)
    prune.add_argument("--expected-targets", type=int)
    prune.add_argument("--expected-rows", type=int)

    groups = sub.add_parser("build-adaptive-groups", help="build physical K/I policy groups")
    groups.add_argument("--main", required=True)
    groups.add_argument("--multi", required=True)
    groups.add_argument("--policy", required=True)
    groups.add_argument("--output-dir", required=True)
    groups.add_argument("--expected-targets", type=int)
    groups.add_argument("--expected-total-hypotheses", type=int)

    merge = sub.add_parser("merge-groups", help="merge one-row-per-target refined group CSVs")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--expected-targets", type=int)
    merge.add_argument("--image-times-csv")

    runtime = sub.add_parser("reconstruct-runtime", help="reconstruct grouped policy compute time")
    runtime.add_argument("--source-main", required=True)
    runtime.add_argument(
        "--group",
        nargs=2,
        action="append",
        metavar=("SOURCE_GROUP_MAIN", "REFINED_GROUP"),
        required=True,
    )
    runtime.add_argument("--group-time-mode", choices=["total", "increment"], default="total")
    runtime.add_argument("--output-json")
    runtime.add_argument("--output-image-times-csv")

    patch = sub.add_parser("patch-empty-batch", help="patch MegaPose refiner test_step")
    patch.add_argument("--refiner", required=True)

    manifest = sub.add_parser("manifest", help="write SHA256 and validation metadata")
    manifest.add_argument("--file", action="append", required=True)
    manifest.add_argument("--output", required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "validate":
        frame = load_bop_csv(args.csv)
        report = validate_bop_csv(
            frame,
            expected_rows=args.rows,
            expected_targets=args.targets,
            expected_images=args.images,
            expected_objects=LMO_OBJECTS if args.lmo_objects else None,
            expected_hypotheses_per_target=args.hypotheses,
        )
        report["sha256"] = sha256_file(args.csv)
        _print(report)
        return

    if args.command == "prune-topk":
        _print(
            prune_file(
                args.input,
                args.output,
                top_k=args.top_k,
                expected_targets=args.expected_targets,
                expected_rows=args.expected_rows,
            )
        )
        return

    if args.command == "build-adaptive-groups":
        _print(
            build_adaptive_groups(
                args.main,
                args.multi,
                args.policy,
                args.output_dir,
                expected_targets=args.expected_targets,
                expected_total_hypotheses=args.expected_total_hypotheses,
            )
        )
        return

    if args.command == "merge-groups":
        _print(merge_group_results(
            args.input, args.output, expected_targets=args.expected_targets,
            image_times_csv=args.image_times_csv,
        ))
        return

    if args.command == "reconstruct-runtime":
        report = reconstruct_grouped_runtime(
            args.source_main,
            [(source, refined) for source, refined in args.group],
            group_time_mode=args.group_time_mode,
        )
        if args.output_json:
            write_json(report, args.output_json)
        if args.output_image_times_csv:
            import pandas as pd
            Path(args.output_image_times_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report["image_times"]).to_csv(args.output_image_times_csv, index=False)
        _print(report)
        return

    if args.command == "patch-empty-batch":
        backup = apply_empty_batch_guard(args.refiner)
        _print({"patched": backup is not None, "backup": str(backup) if backup else None})
        return

    if args.command == "manifest":
        payload = {
            "files": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                    "bytes": Path(path).stat().st_size,
                }
                for path in args.file
            ]
        }
        write_json(payload, args.output)
        _print(payload)
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
