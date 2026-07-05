#!/usr/bin/env python3
"""Train a lightweight image-pair model to distill consensus VLM scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True

ANSWER_VALUES = {"no": 0.0, "somewhat": 0.5, "yes": 1.0}
DEFAULT_RUNS = {
    "internvl3": Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl"),
    "sail": Path("vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl"),
    "ola": Path("vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl"),
}
PERCEPTUAL_PREFIX = "perceptual."
SEMANTIC_PREFIX = "semantic."


@dataclass
class PairExample:
    pair_id: str
    reference_path: str
    generated_path: str
    method: str
    routing: str
    subject: str
    concept: str


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="VLM run as label=pair_scores.jsonl.")
    parser.add_argument("--encoder-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--output-dir", type=Path, default=Path(f"distill_runs/siglip_pair_mlp_{stamp}"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("distill_runs/embedding_cache"))
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by", choices=["concept", "pair_id", "subject", "method"], default="concept")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def configured_runs(raw_runs: list[str]) -> dict[str, Path]:
    if not raw_runs:
        return DEFAULT_RUNS
    runs: dict[str, Path] = {}
    for raw in raw_runs:
        if "=" not in raw:
            raise ValueError(f"--run must be label=path, got {raw!r}")
        label, path = raw.split("=", 1)
        runs[label.strip()] = Path(path)
    if len(runs) < 2:
        raise ValueError("Need at least two VLM runs.")
    return runs


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id")
            if not pair_id:
                raise ValueError(f"Missing pair_id in {path}:{line_no}")
            if pair_id in rows:
                raise ValueError(f"Duplicate pair_id {pair_id!r} in {path}")
            rows[pair_id] = row
    return rows


def normalize_answer(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("answer")
    if value is None:
        return None
    answer = str(value).strip().lower()
    return answer if answer in ANSWER_VALUES else None


def extract_answers(row: dict[str, Any]) -> dict[str, str]:
    normalized = row.get("normalized_response") or {}
    answers: dict[str, str] = {}
    for section in ("perceptual", "semantic"):
        section_values = normalized.get(section) or {}
        if not isinstance(section_values, dict):
            continue
        for question_key, value in section_values.items():
            answer = normalize_answer(value)
            if answer is not None:
                answers[f"{section}.{question_key}"] = answer
    return answers


def agreement_weight(scores: list[float]) -> float:
    rounded = [round(score, 3) for score in scores]
    if len(set(rounded)) == 1:
        return 1.0
    counts = {score: rounded.count(score) for score in set(rounded)}
    if max(counts.values()) >= 2:
        return 0.75
    return 0.40


def build_dataset(
    run_paths: dict[str, Path],
    max_pairs: int | None,
) -> tuple[list[PairExample], list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    loaded = {label: read_jsonl(path) for label, path in run_paths.items()}
    labels = list(loaded)
    common_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if max_pairs is not None:
        common_ids = common_ids[:max_pairs]
    if not common_ids:
        raise ValueError("No common pair IDs across VLM runs.")

    per_pair_answers: dict[str, dict[str, dict[str, str]]] = {}
    question_set: set[str] = set()
    missing_paths: list[str] = []
    examples: list[PairExample] = []

    for pair_id in common_ids:
        base = loaded[labels[0]][pair_id]
        reference_path = str(base.get("reference_path") or "")
        generated_path = str(base.get("generated_path") or "")
        if not Path(reference_path).exists():
            missing_paths.append(reference_path)
        if not Path(generated_path).exists():
            missing_paths.append(generated_path)
        answers = {label: extract_answers(loaded[label][pair_id]) for label in labels}
        common_questions = set.intersection(*(set(values) for values in answers.values()))
        question_set.update(common_questions)
        per_pair_answers[pair_id] = answers
        examples.append(
            PairExample(
                pair_id=pair_id,
                reference_path=reference_path,
                generated_path=generated_path,
                method=str(base.get("method") or ""),
                routing=str(base.get("routing") or ""),
                subject=str(base.get("subject") or ""),
                concept=str(base.get("concept") or ""),
            )
        )

    if missing_paths:
        sample = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"{len(missing_paths)} image paths are missing. First paths:\n{sample}")

    questions = sorted(question_set)
    question_to_idx = {question: idx for idx, question in enumerate(questions)}
    targets = np.full((len(examples), len(questions)), np.nan, dtype=np.float32)
    weights = np.zeros((len(examples), len(questions)), dtype=np.float32)

    for row_idx, example in enumerate(examples):
        answers_by_label = per_pair_answers[example.pair_id]
        common_questions = set.intersection(*(set(values) for values in answers_by_label.values()))
        for question in common_questions:
            scores = [ANSWER_VALUES[answers_by_label[label][question]] for label in labels]
            col_idx = question_to_idx[question]
            targets[row_idx, col_idx] = float(median(scores))
            weights[row_idx, col_idx] = agreement_weight(scores)

    summary = {
        "vlm_runs": {label: str(path) for label, path in run_paths.items()},
        "vlm_labels": labels,
        "n_pairs": len(examples),
        "n_questions": len(questions),
        "questions": questions,
        "target_coverage": {
            question: int(np.isfinite(targets[:, idx]).sum())
            for question, idx in question_to_idx.items()
        },
        "mean_target_by_question": {
            question: float(np.nanmean(targets[:, idx]))
            for question, idx in question_to_idx.items()
        },
    }
    return examples, questions, targets, weights, summary


def dataset_fingerprint(examples: list[PairExample], encoder_model: str) -> str:
    digest = hashlib.sha1()
    digest.update(encoder_model.encode("utf-8"))
    for example in examples:
        digest.update(example.pair_id.encode("utf-8"))
        digest.update(example.reference_path.encode("utf-8"))
        digest.update(example.generated_path.encode("utf-8"))
    return digest.hexdigest()[:16]


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def model_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def image_features(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        try:
            features = model.get_image_features(**inputs)
        except TypeError:
            features = model.get_image_features(pixel_values=inputs["pixel_values"])
    else:
        outputs = model(**inputs)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(features.float(), dim=-1)


def encode_paths(
    paths: list[str],
    encoder_model: str,
    batch_size: int,
    device: str,
    dtype_name: str,
) -> np.ndarray:
    torch_dtype = model_dtype(dtype_name) if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(encoder_model)
    model = AutoModel.from_pretrained(encoder_model, torch_dtype=torch_dtype)
    model = model.to(device).eval()

    encoded: list[np.ndarray] = []
    start_time = time.time()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [load_image(path) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            tensor_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            }
            features = image_features(model, tensor_inputs)
            encoded.append(features.cpu().numpy().astype(np.float32))
            done = min(start + batch_size, len(paths))
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"encoded {done}/{len(paths)} images ({done / elapsed:.1f} img/s)", flush=True)
    return np.concatenate(encoded, axis=0)


def precompute_embeddings(
    examples: list[PairExample],
    encoder_model: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
    dtype_name: str,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, Path]:
    fingerprint = dataset_fingerprint(examples, encoder_model)
    cache_path = cache_dir / f"{slugify(encoder_model)}_{fingerprint}.npz"
    if cache_path.exists() and not force:
        cache = np.load(cache_path, allow_pickle=True)
        return cache["reference_embeddings"].astype(np.float32), cache["generated_embeddings"].astype(np.float32), cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_paths = [example.reference_path for example in examples]
    generated_paths = [example.generated_path for example in examples]
    print(f"precomputing reference embeddings with {encoder_model}", flush=True)
    reference_embeddings = encode_paths(reference_paths, encoder_model, batch_size, device, dtype_name)
    print("precomputing generated embeddings", flush=True)
    generated_embeddings = encode_paths(generated_paths, encoder_model, batch_size, device, dtype_name)
    np.savez_compressed(
        cache_path,
        pair_ids=np.array([example.pair_id for example in examples], dtype=object),
        reference_embeddings=reference_embeddings,
        generated_embeddings=generated_embeddings,
        encoder_model=encoder_model,
    )
    return reference_embeddings, generated_embeddings, cache_path


def pair_features(reference_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> np.ndarray:
    cosine = np.sum(reference_embeddings * generated_embeddings, axis=1, keepdims=True)
    return np.concatenate(
        [
            reference_embeddings,
            generated_embeddings,
            np.abs(reference_embeddings - generated_embeddings),
            reference_embeddings * generated_embeddings,
            cosine,
        ],
        axis=1,
    ).astype(np.float32)


def split_indices(examples: list[PairExample], split_by: str, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    group_to_indices: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        group = getattr(example, split_by)
        group_to_indices.setdefault(group, []).append(idx)

    rng = random.Random(seed)
    groups = list(group_to_indices)
    rng.shuffle(groups)
    n_groups = len(groups)
    train_cut = int(round(n_groups * train_frac))
    val_cut = int(round(n_groups * (train_frac + val_frac)))
    split_groups = {
        "train": set(groups[:train_cut]),
        "val": set(groups[train_cut:val_cut]),
        "test": set(groups[val_cut:]),
    }
    splits = {
        split: sorted(idx for group in groups_for_split for idx in group_to_indices[group])
        for split, groups_for_split in split_groups.items()
    }
    if not splits["val"] or not splits["test"]:
        raise ValueError(f"Split by {split_by!r} produced empty val/test split.")
    return splits


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, indices: list[int]) -> None:
        self.features = torch.from_numpy(features[indices]).float()
        self.targets = torch.from_numpy(np.nan_to_num(targets[indices], nan=0.0)).float()
        self.weights = torch.from_numpy(weights[indices]).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx], self.weights[idx]


class PairScoreMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    active = weights > 0
    if not torch.any(active):
        return torch.zeros((), device=pred.device)
    loss = ((pred - target) ** 2) * weights
    return loss[active].sum() / weights[active].sum().clamp_min(1e-6)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return None
    return float((x * y).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def nearest_label_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_label = np.where(pred < 0.25, 0.0, np.where(pred < 0.75, 0.5, 1.0))
    return float(np.mean(pred_label == target))


def evaluate_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    indices: list[int],
    questions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred = predictions[indices]
    target = targets[indices]
    weight = weights[indices]
    mask = weight > 0
    diff = pred[mask] - target[mask]
    overall = {
        "n_pairs": len(indices),
        "n_targets": int(mask.sum()),
        "mse": float(np.mean(diff**2)),
        "rmse": float(math.sqrt(float(np.mean(diff**2)))),
        "mae": float(np.mean(np.abs(diff))),
        "pearson": pearson(pred[mask], target[mask]),
        "spearman": spearman(pred[mask], target[mask]),
        "nearest_label_accuracy": nearest_label_accuracy(pred[mask], target[mask]),
        "mean_pred": float(np.mean(pred[mask])),
        "mean_target": float(np.mean(target[mask])),
    }

    question_rows: list[dict[str, Any]] = []
    for q_idx, question in enumerate(questions):
        q_mask = mask[:, q_idx]
        if not np.any(q_mask):
            continue
        q_pred = pred[:, q_idx][q_mask]
        q_target = target[:, q_idx][q_mask]
        q_diff = q_pred - q_target
        question_rows.append(
            {
                "question": question,
                "n": int(q_mask.sum()),
                "mae": float(np.mean(np.abs(q_diff))),
                "rmse": float(math.sqrt(float(np.mean(q_diff**2)))),
                "pearson": pearson(q_pred, q_target),
                "spearman": spearman(q_pred, q_target),
                "nearest_label_accuracy": nearest_label_accuracy(q_pred, q_target),
                "mean_pred": float(np.mean(q_pred)),
                "mean_target": float(np.mean(q_target)),
            }
        )
    return overall, question_rows


def aggregate_scores(values: np.ndarray, mask: np.ndarray, questions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    perceptual_indices = [idx for idx, question in enumerate(questions) if question.startswith(PERCEPTUAL_PREFIX)]
    semantic_indices = [idx for idx, question in enumerate(questions) if question.startswith(SEMANTIC_PREFIX)]
    t_pas = np.full(values.shape[0], np.nan, dtype=np.float32)
    t_sas = np.full(values.shape[0], np.nan, dtype=np.float32)
    if perceptual_indices:
        p_mask = mask[:, perceptual_indices]
        p_values = np.where(p_mask, values[:, perceptual_indices], np.nan)
        valid = np.any(p_mask, axis=1)
        t_pas[valid] = np.nanmean(p_values[valid], axis=1)
    if semantic_indices:
        s_mask = mask[:, semantic_indices]
        s_values = np.where(s_mask, values[:, semantic_indices], np.nan)
        valid = np.any(s_mask, axis=1)
        t_sas[valid] = np.nanmean(s_values[valid], axis=1)
    return t_pas, t_sas


def evaluate_aggregate_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    indices: list[int],
    questions: list[str],
) -> dict[str, dict[str, Any]]:
    mask = weights > 0
    pred_t_pas, pred_t_sas = aggregate_scores(predictions, mask, questions)
    target_t_pas, target_t_sas = aggregate_scores(targets, mask, questions)
    out: dict[str, dict[str, Any]] = {}
    for name, pred_score, target_score in [
        ("T_PAS", pred_t_pas, target_t_pas),
        ("T_SAS", pred_t_sas, target_t_sas),
    ]:
        idx = np.array(indices)
        valid = np.isfinite(pred_score[idx]) & np.isfinite(target_score[idx])
        if not np.any(valid):
            continue
        p = pred_score[idx][valid]
        t = target_score[idx][valid]
        diff = p - t
        out[name] = {
            "n": int(valid.sum()),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(math.sqrt(float(np.mean(diff**2)))),
            "pearson": pearson(p, t),
            "spearman": spearman(p, t),
            "mean_pred": float(np.mean(p)),
            "mean_target": float(np.mean(t)),
        }
    return out


def train_mean_baseline_predictions(targets: np.ndarray, weights: np.ndarray, train_indices: list[int]) -> np.ndarray:
    train_targets = targets[train_indices]
    train_weights = weights[train_indices]
    means = np.zeros(targets.shape[1], dtype=np.float32)
    global_mean = float(np.nanmean(train_targets[train_weights > 0]))
    for q_idx in range(targets.shape[1]):
        mask = train_weights[:, q_idx] > 0
        means[q_idx] = float(np.mean(train_targets[:, q_idx][mask])) if np.any(mask) else global_mean
    return np.tile(means.reshape(1, -1), (targets.shape[0], 1))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_all(model: nn.Module, features: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).float().to(device)
            outputs.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    splits: dict[str, list[int]],
    args: argparse.Namespace,
) -> tuple[PairScoreMLP, list[dict[str, Any]]]:
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    train_ds = FeatureDataset(features, targets, weights, splits["train"])
    val_ds = FeatureDataset(features, targets, weights, splits["val"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PairScoreMLP(features.shape[1], targets.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    log_rows: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_features, batch_targets, batch_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_features)
            loss = masked_weighted_mse(pred, batch_targets, batch_weights)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for batch_features, batch_targets, batch_weights in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_weights = batch_weights.to(device)
                pred = model(batch_features)
                val_losses.append(float(masked_weighted_mse(pred, batch_targets, batch_weights).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_s": elapsed,
            "best_epoch": best_epoch,
        }
        log_rows.append(row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} elapsed={elapsed:.1f}s",
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
    return model, log_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_paths = configured_runs(args.run)
    run_config = vars(args).copy()
    run_config["run"] = args.run
    run_config["output_dir"] = str(args.output_dir)
    run_config["embedding_cache_dir"] = str(args.embedding_cache_dir)
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    started = time.time()
    examples, questions, targets, weights, dataset_summary = build_dataset(run_paths, args.max_pairs)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    with (args.output_dir / "questions.json").open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)
    with (args.output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example)) + "\n")
    print(f"loaded {len(examples)} pairs and {len(questions)} questions", flush=True)

    embed_started = time.time()
    reference_embeddings, generated_embeddings, cache_path = precompute_embeddings(
        examples,
        args.encoder_model,
        args.embedding_cache_dir,
        args.encoder_batch_size,
        args.device,
        args.dtype,
        args.force_recompute_embeddings,
    )
    embed_elapsed = time.time() - embed_started
    features = pair_features(reference_embeddings, generated_embeddings)
    splits = split_indices(examples, args.split_by, args.seed, args.train_frac, args.val_frac)
    split_summary = {
        split: {
            "n_pairs": len(indices),
            "n_targets": int((weights[indices] > 0).sum()),
        }
        for split, indices in splits.items()
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(f"split summary: {split_summary}", flush=True)

    train_started = time.time()
    model, log_rows = train_model(features, targets, weights, splits, args)
    train_elapsed = time.time() - train_started
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "questions": questions,
            "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim,
            "encoder_model": args.encoder_model,
        },
        args.output_dir / "best_model.pt",
    )
    write_csv(
        args.output_dir / "training_log.csv",
        log_rows,
        ["epoch", "train_loss", "val_loss", "elapsed_s", "best_epoch"],
    )

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    predictions = predict_all(model.to(device), features, device, args.batch_size)
    baseline_predictions = train_mean_baseline_predictions(targets, weights, splits["train"])
    metrics: dict[str, Any] = {
        "encoder_model": args.encoder_model,
        "embedding_cache": str(cache_path),
        "embedding_elapsed_s": embed_elapsed,
        "training_elapsed_s": train_elapsed,
        "total_elapsed_s": time.time() - started,
        "splits": split_summary,
        "question_count": len(questions),
        "pair_count": len(examples),
    }
    all_question_rows: list[dict[str, Any]] = []
    for split, indices in splits.items():
        overall, question_rows = evaluate_arrays(predictions, targets, weights, indices, questions)
        aggregates = evaluate_aggregate_scores(predictions, targets, weights, indices, questions)
        baseline_overall, _ = evaluate_arrays(baseline_predictions, targets, weights, indices, questions)
        baseline_aggregates = evaluate_aggregate_scores(baseline_predictions, targets, weights, indices, questions)
        metrics[split] = {
            "per_question": overall,
            "aggregates": aggregates,
            "train_mean_baseline": {
                "per_question": baseline_overall,
                "aggregates": baseline_aggregates,
            },
        }
        for row in question_rows:
            row = {"split": split, **row}
            all_question_rows.append(row)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(
        args.output_dir / "per_question_metrics.csv",
        all_question_rows,
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

    test_rows: list[dict[str, Any]] = []
    for idx in splits["test"]:
        example = examples[idx]
        for q_idx, question in enumerate(questions):
            if weights[idx, q_idx] <= 0:
                continue
            test_rows.append(
                {
                    "pair_id": example.pair_id,
                    "method": example.method,
                    "routing": example.routing,
                    "subject": example.subject,
                    "concept": example.concept,
                    "question": question,
                    "target": float(targets[idx, q_idx]),
                    "prediction": float(predictions[idx, q_idx]),
                    "weight": float(weights[idx, q_idx]),
                }
            )
    write_csv(
        args.output_dir / "test_predictions.csv",
        test_rows,
        ["pair_id", "method", "routing", "subject", "concept", "question", "target", "prediction", "weight"],
    )
    np.savez_compressed(
        args.output_dir / "predictions_all.npz",
        raw_predictions=predictions,
        targets=targets,
        weights=weights,
        train_indices=np.array(splits["train"], dtype=np.int64),
        val_indices=np.array(splits["val"], dtype=np.int64),
        test_indices=np.array(splits["test"], dtype=np.int64),
        questions=np.array(questions, dtype=object),
    )

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
