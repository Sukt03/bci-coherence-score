#!/usr/bin/env python3
"""Compare semantic similarity of VLM annotation reasoning fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ANSWER_VALUES = {"no": 0.0, "somewhat": 0.5, "yes": 1.0}
DEFAULT_RUNS = {
    "internvl3": Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl"),
    "sail": Path("vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl"),
    "ola": Path("vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Model label and JSONL path as label=path. Defaults to InternVL3, SAIL, and OLA full runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vlm_eval_runs/reasoning_semantics_internvl3_sail_ola"),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sbert", "clip", "tfidf"],
        default="auto",
        help="Embedding backend. auto tries transformer sentence embeddings, cached CLIP, then TF-IDF.",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=200)
    return parser.parse_args()


def configured_runs(raw_runs: list[str]) -> dict[str, Path]:
    if not raw_runs:
        return DEFAULT_RUNS
    runs: dict[str, Path] = {}
    for raw in raw_runs:
        if "=" not in raw:
            raise ValueError(f"--run must be label=path, got: {raw}")
        label, path = raw.split("=", 1)
        runs[label.strip()] = Path(path)
    if len(runs) < 2:
        raise ValueError("Need at least two runs.")
    return runs


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
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


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_and_reasoning(record: dict[str, Any]) -> dict[str, dict[str, str]]:
    normalized = record.get("normalized_response") or {}
    out: dict[str, dict[str, str]] = {}
    for section in ("perceptual", "semantic"):
        section_values = normalized.get(section) or {}
        if not isinstance(section_values, dict):
            continue
        for question_key, value in section_values.items():
            if not isinstance(value, dict):
                continue
            answer = clean_text(value.get("answer")).lower()
            reasoning = clean_text(value.get("reasoning"))
            if answer in ANSWER_VALUES and reasoning:
                out[f"{section}.{question_key}"] = {"answer": answer, "reasoning": reasoning}
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def embed_with_clip(texts: list[str], model_id: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
    model = CLIPTextModel.from_pretrained(model_id, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            pooled = outputs.pooler_output
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def embed_with_sbert(texts: list[str], model_id: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def tokenize_for_tfidf(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 2]


def embed_with_tfidf(texts: list[str]) -> np.ndarray:
    docs = [tokenize_for_tfidf(text) for text in texts]
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    terms = [term for term, count in df.items() if count >= 2]
    vocab = {term: idx for idx, term in enumerate(terms)}
    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32)
    idf = np.zeros(len(vocab), dtype=np.float32)
    n_docs = len(docs)
    for term, idx in vocab.items():
        idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_idx, doc in enumerate(docs):
        counts = Counter(term for term in doc if term in vocab)
        if not counts:
            continue
        length = sum(counts.values())
        for term, count in counts.items():
            matrix[row_idx, vocab[term]] = (count / length) * idf[vocab[term]]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def get_embeddings(
    texts: list[str],
    backend: str,
    sbert_model: str,
    clip_model: str,
    batch_size: int,
) -> tuple[str, np.ndarray]:
    if backend in {"auto", "sbert"}:
        try:
            return "sbert", embed_with_sbert(texts, sbert_model, batch_size)
        except Exception as exc:
            if backend == "sbert":
                raise
            print(f"SBERT backend unavailable, trying CLIP/TF-IDF: {type(exc).__name__}: {exc}")
    if backend in {"auto", "clip"}:
        try:
            return "clip", embed_with_clip(texts, clip_model, batch_size)
        except Exception as exc:
            if backend == "clip":
                raise
            print(f"CLIP backend unavailable, falling back to TF-IDF: {type(exc).__name__}: {exc}")
    return "tfidf", embed_with_tfidf(texts)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    arr = np.array(values, dtype=np.float32)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return {"n": 0, "mean": None, "median": None, **quantiles([])}
    return {
        "n": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        **quantiles(finite),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = configured_runs(args.run)
    labels = list(runs)
    loaded = {label: load_jsonl(path) for label, path in runs.items()}
    common_pair_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if not common_pair_ids:
        raise ValueError("No common pair IDs.")

    items: list[dict[str, Any]] = []
    all_texts: dict[str, str] = {}
    for pair_id in common_pair_ids:
        per_model = {label: answer_and_reasoning(loaded[label][pair_id]) for label in labels}
        common_questions = sorted(set.intersection(*(set(values) for values in per_model.values())))
        base = loaded[labels[0]][pair_id]
        for question in common_questions:
            item = {
                "pair_id": pair_id,
                "method": base.get("method"),
                "routing": base.get("routing"),
                "subject": base.get("subject"),
                "concept": base.get("concept"),
                "question": question,
                "section": question.split(".", 1)[0],
            }
            for label in labels:
                answer = per_model[label][question]["answer"]
                reasoning = per_model[label][question]["reasoning"]
                text_id = hashlib.sha1(reasoning.encode("utf-8")).hexdigest()
                all_texts[text_id] = reasoning
                item[f"{label}_answer"] = answer
                item[f"{label}_reasoning"] = reasoning
                item[f"{label}_text_id"] = text_id
            items.append(item)

    text_ids = sorted(all_texts)
    texts = [all_texts[text_id] for text_id in text_ids]
    backend_used, embeddings = get_embeddings(texts, args.backend, args.sbert_model, args.clip_model, args.batch_size)
    embedding_map = {text_id: embeddings[idx] for idx, text_id in enumerate(text_ids)}

    pair_rows: list[dict[str, Any]] = []
    by_pair = defaultdict(list)
    by_question = defaultdict(list)
    by_section = defaultdict(list)
    by_method = defaultdict(list)
    by_routing = defaultdict(list)
    by_label_delta = defaultdict(list)
    item_mean_sims: list[float] = []
    example_candidates: list[dict[str, Any]] = []

    for item in items:
        sims = []
        for left, right in combinations(labels, 2):
            sim = cosine(embedding_map[item[f"{left}_text_id"]], embedding_map[item[f"{right}_text_id"]])
            delta = abs(ANSWER_VALUES[item[f"{left}_answer"]] - ANSWER_VALUES[item[f"{right}_answer"]])
            delta_name = "same_label" if delta == 0 else "adjacent_label" if delta == 0.5 else "opposite_label"
            row = {
                "pair_id": item["pair_id"],
                "method": item["method"],
                "routing": item["routing"],
                "subject": item["subject"],
                "concept": item["concept"],
                "question": item["question"],
                "section": item["section"],
                "model_pair": f"{left}__{right}",
                "similarity": sim,
                "label_delta": delta,
                "label_delta_name": delta_name,
                f"{left}_answer": item[f"{left}_answer"],
                f"{right}_answer": item[f"{right}_answer"],
                f"{left}_reasoning": item[f"{left}_reasoning"],
                f"{right}_reasoning": item[f"{right}_reasoning"],
            }
            pair_rows.append(row)
            by_pair[f"{left}__{right}"].append(sim)
            by_question[(item["question"], f"{left}__{right}")].append(sim)
            by_section[(item["section"], f"{left}__{right}")].append(sim)
            by_method[(item["method"], f"{left}__{right}")].append(sim)
            by_routing[(item["routing"], f"{left}__{right}")].append(sim)
            by_label_delta[(delta_name, f"{left}__{right}")].append(sim)
            sims.append(sim)
        item_mean = mean(sims)
        item_mean_sims.append(item_mean)
        values = [ANSWER_VALUES[item[f"{label}_answer"]] for label in labels]
        if max(values) - min(values) == 1.0 or item_mean < 0.55:
            example = {key: item[key] for key in ("pair_id", "method", "routing", "subject", "concept", "question")}
            example["mean_reasoning_similarity"] = item_mean
            for label in labels:
                example[f"{label}_answer"] = item[f"{label}_answer"]
                example[f"{label}_reasoning"] = item[f"{label}_reasoning"]
            example_candidates.append(example)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "runs": {label: str(path) for label, path in runs.items()},
        "backend": backend_used,
        "sbert_model": args.sbert_model if backend_used == "sbert" else None,
        "clip_model": args.clip_model if backend_used == "clip" else None,
        "common_pair_count": len(common_pair_ids),
        "question_item_count": len(items),
        "unique_reasoning_count": len(texts),
        "overall_item_mean_pairwise_similarity": summarize(item_mean_sims),
        "pairwise": {pair: summarize(values) for pair, values in sorted(by_pair.items())},
        "by_section": {f"{section}:{pair}": summarize(values) for (section, pair), values in sorted(by_section.items())},
        "by_method": {f"{method}:{pair}": summarize(values) for (method, pair), values in sorted(by_method.items())},
        "by_routing": {f"{routing}:{pair}": summarize(values) for (routing, pair), values in sorted(by_routing.items())},
        "by_label_delta": {f"{delta}:{pair}": summarize(values) for (delta, pair), values in sorted(by_label_delta.items())},
    }
    (output_dir / "reasoning_semantic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    question_rows: list[dict[str, Any]] = []
    for question in sorted({item["question"] for item in items}):
        row: dict[str, Any] = {"question": question}
        all_values: list[float] = []
        for left, right in combinations(labels, 2):
            pair = f"{left}__{right}"
            values = by_question[(question, pair)]
            stats = summarize(values)
            row[f"{pair}_n"] = stats["n"]
            row[f"{pair}_mean"] = stats["mean"]
            row[f"{pair}_median"] = stats["median"]
            all_values.extend(values)
        row["all_pair_mean"] = mean(all_values) if all_values else None
        question_rows.append(row)
    question_rows.sort(key=lambda row: row["all_pair_mean"] if row["all_pair_mean"] is not None else -1)
    write_csv(
        output_dir / "reasoning_question_similarity.csv",
        question_rows,
        [
            "question",
            "all_pair_mean",
            *(f"{left}__{right}_n" for left, right in combinations(labels, 2)),
            *(f"{left}__{right}_mean" for left, right in combinations(labels, 2)),
            *(f"{left}__{right}_median" for left, right in combinations(labels, 2)),
        ],
    )

    label_delta_rows: list[dict[str, Any]] = []
    for (delta, pair), values in sorted(by_label_delta.items()):
        stats = summarize(values)
        label_delta_rows.append({"label_relation": delta, "model_pair": pair, **stats})
    write_csv(
        output_dir / "reasoning_by_label_relation.csv",
        label_delta_rows,
        ["label_relation", "model_pair", "n", "mean", "median", "p10", "p25", "p50", "p75", "p90"],
    )

    example_candidates.sort(key=lambda row: row["mean_reasoning_similarity"])
    write_csv(
        output_dir / "reasoning_disagreement_examples.csv",
        example_candidates[: args.max_examples],
        [
            "pair_id",
            "method",
            "routing",
            "subject",
            "concept",
            "question",
            "mean_reasoning_similarity",
            *(f"{label}_answer" for label in labels),
            *(f"{label}_reasoning" for label in labels),
        ],
    )

    print(json.dumps({
        "output_dir": str(output_dir),
        "backend": backend_used,
        "common_pair_count": len(common_pair_ids),
        "question_item_count": len(items),
        "unique_reasoning_count": len(texts),
        "overall_item_mean_pairwise_similarity": summary["overall_item_mean_pairwise_similarity"],
        "pairwise": summary["pairwise"],
    }, indent=2))


if __name__ == "__main__":
    main()
