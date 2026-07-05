#!/usr/bin/env python3
"""Select candidate + ordinal thresholds per question using validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ordinal_calibrate_predictions import apply_thresholds
from stack_distiller_predictions import assert_same_meta, load_candidate_arrays
from train_vlm_score_distiller import evaluate_aggregate_scores, evaluate_arrays, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-calibrated", action="store_true")
    return parser.parse_args()


def fit_best_candidate_threshold(
    candidates: list[tuple[str, np.ndarray]],
    target: np.ndarray,
    weights: np.ndarray,
    val_indices: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lows = np.linspace(0.05, 0.55, 51)
    highs = np.linspace(0.45, 0.95, 51)
    out = np.zeros_like(candidates[0][1], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for q_idx in range(target.shape[1]):
        active = weights[val_indices, q_idx] > 0
        y = target[val_indices, q_idx][active]
        best: dict[str, Any] | None = None
        for name, pred in candidates:
            p = pred[val_indices, q_idx][active]
            for low in lows:
                for high in highs:
                    if high <= low + 0.05:
                        continue
                    calibrated = np.where(p < low, 0.0, np.where(p < high, 0.5, 1.0))
                    mae = float(np.mean(np.abs(calibrated - y)))
                    if best is None or mae < best["val_mae"]:
                        best = {"candidate": name, "candidate_index": len(rows), "low": float(low), "high": float(high), "val_mae": mae, "pred": pred}
        assert best is not None
        thresholds = np.zeros((target.shape[1], 2), dtype=np.float32)
        thresholds[q_idx] = [best["low"], best["high"]]
        q_out = np.where(best["pred"][:, q_idx] < best["low"], 0.0, np.where(best["pred"][:, q_idx] < best["high"], 0.5, 1.0))
        out[:, q_idx] = q_out.astype(np.float32)
        rows.append(
            {
                "question_index": q_idx,
                "candidate": best["candidate"],
                "low": best["low"],
                "high": best["high"],
                "val_mae": best["val_mae"],
            }
        )
    return out, rows


def evaluate(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, splits: dict[str, list[int]], questions: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, indices in splits.items():
        overall, _ = evaluate_arrays(pred, target, weights, indices, questions)
        out[split] = {
            "per_question": overall,
            "aggregates": evaluate_aggregate_scores(pred, target, weights, indices, questions),
        }
    return out


def main() -> None:
    args = parse_args()
    candidates: list[tuple[str, np.ndarray]] = []
    base_meta = None
    for run_dir in args.run_dirs:
        run_candidates, meta = load_candidate_arrays(run_dir, args.include_calibrated)
        if base_meta is None:
            base_meta = meta
        else:
            assert_same_meta(base_meta, meta)
        candidates.extend(run_candidates)
    assert base_meta is not None
    target = base_meta["targets"]
    weights = base_meta["weights"]
    splits = {
        "train": base_meta["train_indices"],
        "val": base_meta["val_indices"],
        "test": base_meta["test_indices"],
    }
    questions = base_meta["questions"]
    pred, selection_rows = fit_best_candidate_threshold(candidates, target, weights, splits["val"])
    metrics = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "candidate_names": [name for name, _ in candidates],
        "selection": [
            {"question": questions[row["question_index"]], **row}
            for row in selection_rows
        ],
        **evaluate(pred, target, weights, splits, questions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "selection.csv", metrics["selection"], ["question", "question_index", "candidate", "low", "high", "val_mae"])
    np.savez_compressed(
        args.output_dir / "predictions_all.npz",
        predictions=pred,
        targets=target,
        weights=weights,
        train_indices=np.array(splits["train"], dtype=np.int64),
        val_indices=np.array(splits["val"], dtype=np.int64),
        test_indices=np.array(splits["test"], dtype=np.int64),
        questions=np.array(questions, dtype=object),
    )
    rows: list[dict[str, Any]] = []
    for split, indices in splits.items():
        _, qrows = evaluate_arrays(pred, target, weights, indices, questions)
        for row in qrows:
            rows.append({"split": split, **row})
    write_csv(
        args.output_dir / "per_question_metrics.csv",
        rows,
        ["split", "question", "n", "mae", "rmse", "pearson", "spearman", "nearest_label_accuracy", "mean_pred", "mean_target"],
    )
    print(json.dumps({"output_dir": str(args.output_dir), "test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
