#!/usr/bin/env python3
"""Validation-learned ordinal threshold calibration for distiller scores."""

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
    parser.add_argument("prediction_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-key", default=None)
    return parser.parse_args()


def load_prediction(path: Path, key: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if key is None:
        for candidate in ("predictions", "raw_predictions", "calibrated_predictions"):
            if candidate in data and data[candidate].size:
                key = candidate
                break
    if key is None or key not in data:
        raise ValueError(f"No prediction array found in {path}")
    pred = data[key].astype(np.float32)
    meta = {
        "prediction_key": key,
        "targets": data["targets"].astype(np.float32),
        "weights": data["weights"].astype(np.float32),
        "train_indices": data["train_indices"].astype(np.int64).tolist(),
        "val_indices": data["val_indices"].astype(np.int64).tolist(),
        "test_indices": data["test_indices"].astype(np.int64).tolist(),
        "questions": [str(x) for x in data["questions"].tolist()],
    }
    return pred, meta


def apply_thresholds(pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.float32)
    for q_idx, (low, high) in enumerate(thresholds):
        q = pred[:, q_idx]
        out[:, q_idx] = np.where(q < low, 0.0, np.where(q < high, 0.5, 1.0))
    return out


def fit_thresholds(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, val_indices: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lows = np.linspace(0.05, 0.55, 51)
    highs = np.linspace(0.45, 0.95, 51)
    thresholds = np.zeros((pred.shape[1], 2), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for q_idx in range(pred.shape[1]):
        active = weights[val_indices, q_idx] > 0
        p = pred[val_indices, q_idx][active]
        y = target[val_indices, q_idx][active]
        best = (0.25, 0.75)
        best_mae = float("inf")
        for low in lows:
            for high in highs:
                if high <= low + 0.05:
                    continue
                calibrated = np.where(p < low, 0.0, np.where(p < high, 0.5, 1.0))
                mae = float(np.mean(np.abs(calibrated - y)))
                if mae < best_mae:
                    best_mae = mae
                    best = (float(low), float(high))
        thresholds[q_idx] = best
        rows.append({"question_index": q_idx, "low": best[0], "high": best[1], "val_mae": best_mae})
    return thresholds, rows


def evaluate(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, splits: dict[str, list[int]], questions: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, indices in splits.items():
        overall, _ = evaluate_arrays(pred, target, weights, indices, questions)
        out[split] = {
            "per_question": overall,
            "aggregates": evaluate_aggregate_scores(pred, target, weights, indices, questions),
        }
    return out


def write_question_metrics(path: Path, pred: np.ndarray, target: np.ndarray, weights: np.ndarray, splits: dict[str, list[int]], questions: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for split, indices in splits.items():
        _, qrows = evaluate_arrays(pred, target, weights, indices, questions)
        for row in qrows:
            rows.append({"split": split, **row})
    write_csv(
        path,
        rows,
        ["split", "question", "n", "mae", "rmse", "pearson", "spearman", "nearest_label_accuracy", "mean_pred", "mean_target"],
    )


def main() -> None:
    args = parse_args()
    pred, meta = load_prediction(args.prediction_npz, args.prediction_key)
    target = meta["targets"]
    weights = meta["weights"]
    splits = {
        "train": meta["train_indices"],
        "val": meta["val_indices"],
        "test": meta["test_indices"],
    }
    questions = meta["questions"]
    thresholds, threshold_rows = fit_thresholds(pred, target, weights, splits["val"])
    calibrated = apply_thresholds(pred, thresholds)
    metrics = {
        "source": str(args.prediction_npz),
        "prediction_key": meta["prediction_key"],
        "thresholds": [
            {"question": questions[row["question_index"]], **row}
            for row in threshold_rows
        ],
        **evaluate(calibrated, target, weights, splits, questions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "thresholds.csv", metrics["thresholds"], ["question", "question_index", "low", "high", "val_mae"])
    write_question_metrics(args.output_dir / "per_question_metrics.csv", calibrated, target, weights, splits, questions)
    np.savez_compressed(
        args.output_dir / "predictions_all.npz",
        predictions=calibrated,
        targets=target,
        weights=weights,
        train_indices=np.array(splits["train"], dtype=np.int64),
        val_indices=np.array(splits["val"], dtype=np.int64),
        test_indices=np.array(splits["test"], dtype=np.int64),
        questions=np.array(questions, dtype=object),
        thresholds=thresholds,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "test": metrics["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
