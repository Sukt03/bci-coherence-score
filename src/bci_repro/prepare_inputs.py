from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._paths import relative_to_data_root, resolve_data_root


REQUIRED_LIGHT_FILES = [
    "expanded_metric_scores_clean_with_sbert.csv",
    "expanded_extra_model_885_clean_with_sbert(1).csv",
    "consensus_rank1_gt_generated/manifest.csv",
]

REQUIRED_LIGHT_DIRS = [
    "metric_selected_images_only",
    "consensus_rank1_gt_generated",
]

FINAL_JSONL_RUNS = [
    "internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl",
    "vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl",
    "vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl",
    "vlm_eval_runs/ovis_full_both_reasoning_20260530_083349/pair_scores.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify expected input artifacts for reproduction.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--verify-only", action="store_true", help="Kept for explicitness; this command only verifies.")
    parser.add_argument("--require-cached-runs", action="store_true")
    parser.add_argument("--expected-count", type=int, default=6885)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify(data_root: Path, require_cached_runs: bool, expected_count: int) -> dict[str, Any]:
    missing: list[str] = []
    for item in REQUIRED_LIGHT_DIRS:
        if not relative_to_data_root(item, data_root).is_dir():
            missing.append(item)
    for item in REQUIRED_LIGHT_FILES:
        if not relative_to_data_root(item, data_root).is_file():
            missing.append(item)

    run_counts: dict[str, int] = {}
    if require_cached_runs:
        for item in FINAL_JSONL_RUNS:
            path = relative_to_data_root(item, data_root)
            if not path.is_file():
                missing.append(item)
                continue
            run_counts[item] = count_jsonl(path)
            if run_counts[item] != expected_count:
                missing.append(f"{item} has {run_counts[item]} rows, expected {expected_count}")

    return {
        "data_root": str(data_root),
        "missing": missing,
        "ok": not missing,
        "run_counts": run_counts,
    }


def main() -> None:
    args = parse_args()
    report = verify(resolve_data_root(args.data_root), args.require_cached_runs, args.expected_count)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

