#!/usr/bin/env python3
"""Write compact comparison tables for distiller result directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Named run as label=distill_run_dir.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-question-csv", type=Path, default=None)
    return parser.parse_args()


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--run must be label=path, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty run label in {raw!r}")
    return label, Path(path)


def metric_block(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "test" in metrics:
        return "predictions", metrics
    for key in ("calibrated", "raw", "train_mean_baseline"):
        if key in metrics and "test" in metrics[key]:
            return key, metrics[key]
    raise ValueError("Could not find a test metric block")


def value_at(block: dict[str, Any], *keys: str) -> Any:
    cursor: Any = block
    for key in keys:
        if not isinstance(cursor, dict):
            return ""
        cursor = cursor.get(key)
    return "" if cursor is None else cursor


def row_for(label: str, path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_set, block = metric_block(metrics)
    return {
        "run": label,
        "path": str(path),
        "prediction_set": prediction_set,
        "test_mae": value_at(block, "test", "per_question", "mae"),
        "test_rmse": value_at(block, "test", "per_question", "rmse"),
        "test_pearson": value_at(block, "test", "per_question", "pearson"),
        "test_spearman": value_at(block, "test", "per_question", "spearman"),
        "test_nearest_label_accuracy": value_at(block, "test", "per_question", "nearest_label_accuracy"),
        "t_pas_mae": value_at(block, "test", "aggregates", "T_PAS", "mae"),
        "t_pas_pearson": value_at(block, "test", "aggregates", "T_PAS", "pearson"),
        "t_pas_spearman": value_at(block, "test", "aggregates", "T_PAS", "spearman"),
        "t_sas_mae": value_at(block, "test", "aggregates", "T_SAS", "mae"),
        "t_sas_pearson": value_at(block, "test", "aggregates", "T_SAS", "pearson"),
        "t_sas_spearman": value_at(block, "test", "aggregates", "T_SAS", "spearman"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = [row_for(label, path) for label, path in map(parse_named_path, args.run)]
    rows.sort(key=lambda row: float(row["test_mae"]) if row["test_mae"] != "" else float("inf"))
    fieldnames = [
        "run",
        "path",
        "prediction_set",
        "test_mae",
        "test_rmse",
        "test_pearson",
        "test_spearman",
        "test_nearest_label_accuracy",
        "t_pas_mae",
        "t_pas_pearson",
        "t_pas_spearman",
        "t_sas_mae",
        "t_sas_pearson",
        "t_sas_spearman",
    ]
    write_csv(args.output_csv, rows, fieldnames)
    print(f"Wrote {args.output_csv}")
    for row in rows:
        print(
            f"{row['run']}: MAE={row['test_mae']} "
            f"Pearson={row['test_pearson']} T-PAS_MAE={row['t_pas_mae']} T-SAS_MAE={row['t_sas_mae']}"
        )


if __name__ == "__main__":
    main()
