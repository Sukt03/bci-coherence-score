#!/usr/bin/env python3
"""Validation-only stacking for VLM-score distiller predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from train_vlm_score_distiller import evaluate_aggregate_scores, evaluate_arrays, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-calibrated", action="store_true")
    return parser.parse_args()


def load_candidate_arrays(run_dir: Path, include_calibrated: bool) -> tuple[list[tuple[str, np.ndarray]], dict[str, Any]]:
    path = run_dir / "predictions_all.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    candidates: list[tuple[str, np.ndarray]] = []
    if "raw_predictions" in data and data["raw_predictions"].size:
        candidates.append((f"{run_dir.name}:raw", data["raw_predictions"].astype(np.float32)))
    if "predictions" in data and data["predictions"].size:
        candidates.append((f"{run_dir.name}:predictions", data["predictions"].astype(np.float32)))
    if include_calibrated and "calibrated_predictions" in data and data["calibrated_predictions"].size:
        candidates.append((f"{run_dir.name}:calibrated", data["calibrated_predictions"].astype(np.float32)))
    meta = {
        "targets": data["targets"].astype(np.float32),
        "weights": data["weights"].astype(np.float32),
        "train_indices": data["train_indices"].astype(np.int64).tolist(),
        "val_indices": data["val_indices"].astype(np.int64).tolist(),
        "test_indices": data["test_indices"].astype(np.int64).tolist(),
        "questions": [str(x) for x in data["questions"].tolist()],
    }
    return candidates, meta


def assert_same_meta(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("targets", "weights"):
        if left[key].shape != right[key].shape or not np.allclose(left[key], right[key], equal_nan=True):
            raise ValueError(f"Candidate metadata mismatch for {key}")
    for key in ("train_indices", "val_indices", "test_indices", "questions"):
        if left[key] != right[key]:
            raise ValueError(f"Candidate metadata mismatch for {key}")


def mae_for(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, indices: list[int], q_idx: int) -> float:
    active = weights[indices, q_idx] > 0
    if not np.any(active):
        return float("inf")
    return float(np.mean(np.abs(pred[indices, q_idx][active] - target[indices, q_idx][active])))


def split_validation(indices: list[int], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    arr = np.array(indices, dtype=np.int64)
    rng.shuffle(arr)
    cut = max(1, int(round(len(arr) * 0.7)))
    return arr[:cut].tolist(), arr[cut:].tolist()


def selected_expert(
    candidate_names: list[str],
    candidate_preds: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    val_indices: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.zeros_like(candidate_preds[0])
    selected: dict[str, Any] = {}
    for q_idx in range(target.shape[1]):
        maes = [mae_for(pred, target, weights, val_indices, q_idx) for pred in candidate_preds]
        best = int(np.argmin(maes))
        out[:, q_idx] = candidate_preds[best, :, q_idx]
        selected[str(q_idx)] = {"candidate": candidate_names[best], "val_mae": maes[best]}
    return out, {"selected_by_question": selected}


def softmax_weighted(
    candidate_names: list[str],
    candidate_preds: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    val_indices: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    temps = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
    best_pred = None
    best_info: dict[str, Any] = {}
    best_val = float("inf")
    for temp in temps:
        out = np.zeros_like(candidate_preds[0])
        question_weights: dict[str, list[float]] = {}
        for q_idx in range(target.shape[1]):
            maes = np.array([mae_for(pred, target, weights, val_indices, q_idx) for pred in candidate_preds])
            scores = -maes / temp
            scores = scores - np.max(scores)
            alpha = np.exp(scores)
            alpha = alpha / alpha.sum()
            out[:, q_idx] = np.einsum("c,c n -> n", alpha, candidate_preds[:, :, q_idx])
            question_weights[str(q_idx)] = alpha.tolist()
        val_mae = overall_mae(out, target, weights, val_indices)
        if val_mae < best_val:
            best_val = val_mae
            best_pred = out
            best_info = {"temperature": temp, "val_mae": val_mae, "candidate_names": candidate_names, "question_weights": question_weights}
    assert best_pred is not None
    return best_pred, best_info


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(X_aug.shape[1], dtype=np.float64) * alpha
    reg[-1, -1] = 0.0
    return np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)


def apply_ridge(candidate_preds: np.ndarray, coef: np.ndarray, q_idx: int) -> np.ndarray:
    X = candidate_preds[:, :, q_idx].T.astype(np.float64)
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    return np.clip(X_aug @ coef, 0.0, 1.0).astype(np.float32)


def ridge_stacked(
    candidate_names: list[str],
    candidate_preds: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    val_indices: list[int],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    stack_train, stack_hold = split_validation(val_indices, seed)
    alphas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    out = np.zeros_like(candidate_preds[0])
    info: dict[str, Any] = {"candidate_names": candidate_names, "questions": {}}
    for q_idx in range(target.shape[1]):
        train_active = weights[stack_train, q_idx] > 0
        hold_active = weights[stack_hold, q_idx] > 0
        if train_active.sum() < candidate_preds.shape[0] + 2 or hold_active.sum() < 3:
            maes = [mae_for(pred, target, weights, val_indices, q_idx) for pred in candidate_preds]
            best = int(np.argmin(maes))
            out[:, q_idx] = candidate_preds[best, :, q_idx]
            info["questions"][str(q_idx)] = {"fallback": candidate_names[best], "val_mae": maes[best]}
            continue
        X_train = candidate_preds[:, stack_train, q_idx].T[train_active].astype(np.float64)
        y_train = target[stack_train, q_idx][train_active].astype(np.float64)
        X_hold = candidate_preds[:, stack_hold, q_idx].T[hold_active].astype(np.float64)
        y_hold = target[stack_hold, q_idx][hold_active].astype(np.float64)
        best_alpha = None
        best_coef = None
        best_hold = float("inf")
        for alpha in alphas:
            try:
                coef = fit_ridge(X_train, y_train, alpha)
            except np.linalg.LinAlgError:
                continue
            pred_hold = np.clip(np.concatenate([X_hold, np.ones((X_hold.shape[0], 1))], axis=1) @ coef, 0.0, 1.0)
            hold_mae = float(np.mean(np.abs(pred_hold - y_hold)))
            if hold_mae < best_hold:
                best_hold = hold_mae
                best_alpha = alpha
                best_coef = coef
        if best_coef is None:
            best_coef = fit_ridge(X_train, y_train, 1e-2)
            best_alpha = 1e-2
        all_active = weights[val_indices, q_idx] > 0
        X_all = candidate_preds[:, val_indices, q_idx].T[all_active].astype(np.float64)
        y_all = target[val_indices, q_idx][all_active].astype(np.float64)
        final_coef = fit_ridge(X_all, y_all, float(best_alpha))
        out[:, q_idx] = apply_ridge(candidate_preds, final_coef, q_idx)
        info["questions"][str(q_idx)] = {
            "alpha": best_alpha,
            "holdout_mae": best_hold,
            "coef": final_coef.tolist(),
        }
    return out, info


def overall_mae(pred: np.ndarray, target: np.ndarray, weights: np.ndarray, indices: list[int]) -> float:
    mask = weights[indices] > 0
    return float(np.mean(np.abs(pred[indices][mask] - target[indices][mask])))


def evaluate_prediction(
    pred: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    splits: dict[str, list[int]],
    questions: list[str],
) -> dict[str, Any]:
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
    all_candidates: list[tuple[str, np.ndarray]] = []
    base_meta = None
    for run_dir in args.run_dirs:
        candidates, meta = load_candidate_arrays(run_dir, args.include_calibrated)
        if base_meta is None:
            base_meta = meta
        else:
            assert_same_meta(base_meta, meta)
        all_candidates.extend(candidates)
    assert base_meta is not None
    # Keep candidate names unique if the same directory contributed multiple arrays.
    names = [name for name, _ in all_candidates]
    preds = np.stack([pred for _, pred in all_candidates], axis=0).astype(np.float32)
    target = base_meta["targets"]
    weights = base_meta["weights"]
    splits = {
        "train": base_meta["train_indices"],
        "val": base_meta["val_indices"],
        "test": base_meta["test_indices"],
    }
    questions = base_meta["questions"]

    methods = {
        "selected_expert": selected_expert(names, preds, target, weights, splits["val"]),
        "softmax_weighted": softmax_weighted(names, preds, target, weights, splits["val"]),
        "ridge_stacked": ridge_stacked(names, preds, target, weights, splits["val"], args.seed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "candidate_names": names,
        "run_dirs": [str(path) for path in args.run_dirs],
        "include_calibrated": args.include_calibrated,
        "methods": {},
    }
    rows: list[dict[str, Any]] = []
    for method_name, (pred, info) in methods.items():
        method_dir = args.output_dir / method_name
        method_dir.mkdir(exist_ok=True)
        metrics = evaluate_prediction(pred, target, weights, splits, questions)
        (method_dir / "metrics.json").write_text(json.dumps({"info": info, **metrics}, indent=2), encoding="utf-8")
        write_question_metrics(method_dir / "per_question_metrics.csv", pred, target, weights, splits, questions)
        np.savez_compressed(
            method_dir / "predictions_all.npz",
            predictions=pred,
            targets=target,
            weights=weights,
            train_indices=np.array(splits["train"], dtype=np.int64),
            val_indices=np.array(splits["val"], dtype=np.int64),
            test_indices=np.array(splits["test"], dtype=np.int64),
            questions=np.array(questions, dtype=object),
        )
        summary["methods"][method_name] = metrics
        test = metrics["test"]
        row = {
            "method": method_name,
            "test_mae": test["per_question"]["mae"],
            "test_rmse": test["per_question"]["rmse"],
            "test_pearson": test["per_question"]["pearson"],
            "test_spearman": test["per_question"]["spearman"],
            "test_nearest_label_accuracy": test["per_question"]["nearest_label_accuracy"],
            "t_pas_mae": test["aggregates"].get("T_PAS", {}).get("mae"),
            "t_pas_pearson": test["aggregates"].get("T_PAS", {}).get("pearson"),
            "t_sas_mae": test["aggregates"].get("T_SAS", {}).get("mae"),
            "t_sas_pearson": test["aggregates"].get("T_SAS", {}).get("pearson"),
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["test_mae"]))
    write_csv(
        args.output_dir / "stacking_comparison.csv",
        rows,
        ["method", "test_mae", "test_rmse", "test_pearson", "test_spearman", "test_nearest_label_accuracy", "t_pas_mae", "t_pas_pearson", "t_sas_mae", "t_sas_pearson"],
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
