#!/usr/bin/env python3
"""Average saved distiller predictions and evaluate the ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_vlm_score_distiller import evaluate_aggregate_scores, evaluate_arrays, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--prediction-key", choices=["raw_predictions", "calibrated_predictions"], default="raw_predictions")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_npz(run_dir: Path, key: str) -> dict[str, Any]:
    path = run_dir / "predictions_all.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    pred = data[key]
    if pred.size == 0:
        raise ValueError(f"{key} is empty in {path}")
    return {
        "pred": pred.astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "weights": data["weights"].astype(np.float32),
        "train_indices": data["train_indices"].astype(np.int64).tolist(),
        "val_indices": data["val_indices"].astype(np.int64).tolist(),
        "test_indices": data["test_indices"].astype(np.int64).tolist(),
        "questions": [str(x) for x in data["questions"].tolist()],
    }


def same_array(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.allclose(left, right, equal_nan=True):
        raise ValueError(f"Input runs do not share identical {name}.")


def main() -> None:
    args = parse_args()
    loaded = [load_npz(run_dir, args.prediction_key) for run_dir in args.run_dirs]
    base = loaded[0]
    for item in loaded[1:]:
        same_array("targets", base["targets"], item["targets"])
        same_array("weights", base["weights"], item["weights"])
        if base["train_indices"] != item["train_indices"] or base["val_indices"] != item["val_indices"] or base["test_indices"] != item["test_indices"]:
            raise ValueError("Input runs do not share identical splits.")
        if base["questions"] != item["questions"]:
            raise ValueError("Input runs do not share identical questions.")

    predictions = np.mean(np.stack([item["pred"] for item in loaded], axis=0), axis=0).astype(np.float32)
    splits = {
        "train": base["train_indices"],
        "val": base["val_indices"],
        "test": base["test_indices"],
    }
    targets = base["targets"]
    weights = base["weights"]
    questions = base["questions"]

    metrics: dict[str, Any] = {
        "prediction_key": args.prediction_key,
        "members": [str(run_dir) for run_dir in args.run_dirs],
        "n_members": len(args.run_dirs),
    }
    question_rows: list[dict[str, Any]] = []
    for split, indices in splits.items():
        overall, rows = evaluate_arrays(predictions, targets, weights, indices, questions)
        aggregates = evaluate_aggregate_scores(predictions, targets, weights, indices, questions)
        metrics[split] = {"per_question": overall, "aggregates": aggregates}
        for row in rows:
            question_rows.append({"split": split, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(
        args.output_dir / "per_question_metrics.csv",
        question_rows,
        [
            "split",
            "question",
            "n",
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "nearest_label_accuracy",
            "mean_pred",
            "mean_target",
        ],
    )
    np.savez_compressed(
        args.output_dir / "predictions_all.npz",
        predictions=predictions,
        targets=targets,
        weights=weights,
        train_indices=np.array(splits["train"], dtype=np.int64),
        val_indices=np.array(splits["val"], dtype=np.int64),
        test_indices=np.array(splits["test"], dtype=np.int64),
        questions=np.array(questions, dtype=object),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
