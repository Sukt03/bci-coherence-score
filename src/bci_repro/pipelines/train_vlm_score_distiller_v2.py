#!/usr/bin/env python3
"""Train improved VLM-score distillers with multi-encoder and ordinal heads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_vlm_score_distiller import (
    aggregate_scores,
    build_dataset,
    configured_runs,
    evaluate_aggregate_scores,
    evaluate_arrays,
    pair_features,
    pearson,
    precompute_embeddings,
    spearman,
    split_indices,
    train_mean_baseline_predictions,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="VLM run as label=pair_scores.jsonl.")
    parser.add_argument(
        "--encoder-model",
        action="append",
        default=[],
        help="Frozen image encoder. May be passed multiple times for fusion.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(f"distill_runs/fusion_ordinal_{stamp}"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("distill_runs/embedding_cache"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-by", choices=["concept", "pair_id", "subject", "method"], default="concept")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--ce-weight", type=float, default=0.35)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--architecture", choices=["mlp", "ordinal_prototype"], default="ordinal_prototype")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    return parser.parse_args()


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, indices: list[int]) -> None:
        self.features = torch.from_numpy(features[indices]).float()
        self.targets = torch.from_numpy(np.nan_to_num(targets[indices], nan=0.0)).float()
        self.weights = torch.from_numpy(weights[indices]).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx], self.weights[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.input(features))
        pred = torch.sigmoid(self.output(hidden))
        return {"pred": pred}


class QuestionOrdinalPrototype(nn.Module):
    """Question-conditioned ordinal prototype head.

    The trunk embeds each image pair once. Each question owns three answer
    prototypes in the same latent space; logits are dot products between the
    pair embedding and question-answer prototypes. This keeps heads coupled but
    still question-aware.
    """

    def __init__(self, input_dim: int, n_questions: int, hidden_dim: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.prototypes = nn.Parameter(torch.randn(n_questions, 3, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_questions, 3))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.value_head = nn.Linear(hidden_dim, n_questions)
        self.register_buffer("class_values", torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.norm(self.blocks(self.input(features)))
        hidden = F.normalize(hidden, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = torch.clamp(self.temperature.exp(), min=1.0, max=50.0)
        logits = torch.einsum("bd,qkd->bqk", hidden, prototypes) * scale + self.bias
        probs = F.softmax(logits, dim=-1)
        ordinal_pred = (probs * self.class_values.to(probs.device)).sum(dim=-1)
        value_pred = torch.sigmoid(self.value_head(hidden))
        pred = 0.75 * ordinal_pred + 0.25 * value_pred
        return {"pred": pred, "logits": logits, "ordinal_pred": ordinal_pred, "value_pred": value_pred}


def model_for(args: argparse.Namespace, input_dim: int, output_dim: int) -> nn.Module:
    if args.architecture == "mlp":
        return MLPRegressor(input_dim, output_dim, args.hidden_dim, args.blocks, args.dropout)
    return QuestionOrdinalPrototype(input_dim, output_dim, args.hidden_dim, args.blocks, args.dropout)


def target_classes(target: torch.Tensor) -> torch.Tensor:
    return torch.round(target * 2.0).long().clamp(0, 2)


def weighted_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, weights: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    pred = outputs["pred"]
    active = weights > 0
    if not torch.any(active):
        return torch.zeros((), device=pred.device)
    mse = (((pred - target) ** 2) * weights)[active].sum() / weights[active].sum().clamp_min(1e-6)
    if "logits" not in outputs:
        return mse
    logits = outputs["logits"]
    classes = target_classes(target)
    ce = F.cross_entropy(logits[active], classes[active], reduction="none")
    ce = (ce * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)
    return args.mse_weight * mse + args.ce_weight * ce


def concatenate_encoder_features(
    examples: list[Any],
    encoder_models: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], float]:
    all_features: list[np.ndarray] = []
    cache_paths: list[str] = []
    started = time.time()
    for encoder_model in encoder_models:
        reference, generated, cache_path = precompute_embeddings(
            examples,
            encoder_model,
            args.embedding_cache_dir,
            args.encoder_batch_size,
            args.device,
            args.dtype,
            args.force_recompute_embeddings,
        )
        all_features.append(pair_features(reference, generated))
        cache_paths.append(str(cache_path))
    features = np.concatenate(all_features, axis=1).astype(np.float32)
    return features, cache_paths, time.time() - started


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    splits: dict[str, list[int]],
    args: argparse.Namespace,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    train_loader = DataLoader(
        FeatureDataset(features, targets, weights, splits["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FeatureDataset(features, targets, weights, splits["val"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = model_for(args, features.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    rows: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_features, batch_targets, batch_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(batch_features), batch_targets, batch_weights, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for batch_features, batch_targets, batch_weights in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_weights = batch_weights.to(device)
                val_losses.append(float(weighted_loss(model(batch_features), batch_targets, batch_weights, args).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - started,
            "best_epoch": best_epoch,
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"lr={row['lr']:.2e} elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


def predict_all(model: nn.Module, features: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
            outputs.append(model(batch)["pred"].cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def fit_linear_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    val_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    val_pred = predictions[val_indices]
    val_target = targets[val_indices]
    val_weights = weights[val_indices]
    slopes = np.ones(predictions.shape[1], dtype=np.float32)
    intercepts = np.zeros(predictions.shape[1], dtype=np.float32)
    for q_idx in range(predictions.shape[1]):
        active = val_weights[:, q_idx] > 0
        if active.sum() < 3:
            continue
        x = val_pred[:, q_idx][active].astype(np.float64)
        y = val_target[:, q_idx][active].astype(np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 1e-12:
            intercepts[q_idx] = float(y_mean)
            continue
        slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        intercept = float(y_mean - slope * x_mean)
        slopes[q_idx] = float(np.clip(slope, 0.25, 3.0))
        intercepts[q_idx] = float(np.clip(intercept, -0.5, 0.5))
    return slopes, intercepts


def apply_linear_calibration(predictions: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    return np.clip(predictions * slopes.reshape(1, -1) + intercepts.reshape(1, -1), 0.0, 1.0).astype(np.float32)


def write_question_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "split",
            "prediction_set",
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


def evaluate_prediction_set(
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    splits: dict[str, list[int]],
    questions: list[str],
    question_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, indices in splits.items():
        overall, rows = evaluate_arrays(predictions, targets, weights, indices, questions)
        aggregates = evaluate_aggregate_scores(predictions, targets, weights, indices, questions)
        out[split] = {"per_question": overall, "aggregates": aggregates}
        for row in rows:
            question_rows.append({"split": split, "prediction_set": name, **row})
    return out


def write_test_predictions(
    path: Path,
    examples: list[Any],
    questions: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    splits: dict[str, list[int]],
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray | None,
) -> None:
    fieldnames = [
        "pair_id",
        "method",
        "routing",
        "subject",
        "concept",
        "question",
        "target",
        "prediction_raw",
        "prediction_calibrated",
        "weight",
    ]
    rows: list[dict[str, Any]] = []
    for idx in splits["test"]:
        example = examples[idx]
        for q_idx, question in enumerate(questions):
            if weights[idx, q_idx] <= 0:
                continue
            rows.append(
                {
                    "pair_id": example.pair_id,
                    "method": example.method,
                    "routing": example.routing,
                    "subject": example.subject,
                    "concept": example.concept,
                    "question": question,
                    "target": float(targets[idx, q_idx]),
                    "prediction_raw": float(raw_predictions[idx, q_idx]),
                    "prediction_calibrated": (
                        float(calibrated_predictions[idx, q_idx]) if calibrated_predictions is not None else ""
                    ),
                    "weight": float(weights[idx, q_idx]),
                }
            )
    write_csv(path, rows, fieldnames)


def main() -> None:
    args = parse_args()
    encoder_models = args.encoder_model or ["google/siglip-base-patch16-224"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = vars(args).copy()
    config["encoder_model"] = encoder_models
    config["output_dir"] = str(args.output_dir)
    config["embedding_cache_dir"] = str(args.embedding_cache_dir)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    started = time.time()
    examples, questions, targets, weights, dataset_summary = build_dataset(configured_runs(args.run), args.max_pairs)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    (args.output_dir / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    with (args.output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example)) + "\n")
    print(f"loaded {len(examples)} pairs, {len(questions)} questions", flush=True)

    features, cache_paths, embedding_elapsed = concatenate_encoder_features(examples, encoder_models, args)
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    splits = split_indices(examples, args.split_by, split_seed, args.train_frac, args.val_frac)
    split_summary = {
        split: {"n_pairs": len(indices), "n_targets": int((weights[indices] > 0).sum())}
        for split, indices in splits.items()
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(f"feature_dim={features.shape[1]} split_summary={split_summary}", flush=True)

    train_started = time.time()
    model, log_rows = train_model(features, targets, weights, splits, args)
    training_elapsed = time.time() - train_started
    write_csv(
        args.output_dir / "training_log.csv",
        log_rows,
        ["epoch", "train_loss", "val_loss", "lr", "elapsed_s", "best_epoch"],
    )

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    predictions = predict_all(model.to(device), features, device, args.batch_size)
    calibrated_predictions = None
    calibration: dict[str, Any] | None = None
    if not args.no_calibration:
        slopes, intercepts = fit_linear_calibration(predictions, targets, weights, splits["val"])
        calibrated_predictions = apply_linear_calibration(predictions, slopes, intercepts)
        calibration = {"slopes": slopes.tolist(), "intercepts": intercepts.tolist()}
        (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    baseline_predictions = train_mean_baseline_predictions(targets, weights, splits["train"])
    question_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "architecture": args.architecture,
        "encoder_models": encoder_models,
        "embedding_cache": cache_paths,
        "feature_dim": int(features.shape[1]),
        "embedding_elapsed_s": embedding_elapsed,
        "training_elapsed_s": training_elapsed,
        "total_elapsed_s": time.time() - started,
        "splits": split_summary,
        "pair_count": len(examples),
        "question_count": len(questions),
    }
    metrics["raw"] = evaluate_prediction_set("raw", predictions, targets, weights, splits, questions, question_rows)
    if calibrated_predictions is not None:
        metrics["calibrated"] = evaluate_prediction_set(
            "calibrated", calibrated_predictions, targets, weights, splits, questions, question_rows
        )
    metrics["train_mean_baseline"] = evaluate_prediction_set(
        "train_mean_baseline", baseline_predictions, targets, weights, splits, questions, question_rows
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_question_metrics(args.output_dir / "per_question_metrics.csv", question_rows)
    write_test_predictions(
        args.output_dir / "test_predictions.csv",
        examples,
        questions,
        targets,
        weights,
        splits,
        predictions,
        calibrated_predictions,
    )
    np.savez_compressed(
        args.output_dir / "predictions_all.npz",
        raw_predictions=predictions,
        calibrated_predictions=calibrated_predictions if calibrated_predictions is not None else np.array([], dtype=np.float32),
        targets=targets,
        weights=weights,
        train_indices=np.array(splits["train"], dtype=np.int64),
        val_indices=np.array(splits["val"], dtype=np.int64),
        test_indices=np.array(splits["test"], dtype=np.int64),
        questions=np.array(questions, dtype=object),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "questions": questions,
            "input_dim": int(features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "architecture": args.architecture,
            "encoder_models": encoder_models,
            "calibration": calibration,
        },
        args.output_dir / "best_model.pt",
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
