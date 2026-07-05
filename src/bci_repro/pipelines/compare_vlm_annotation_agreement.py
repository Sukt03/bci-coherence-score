#!/usr/bin/env python3
"""Compare per-question annotation agreement across VLM evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


ANSWER_VALUES = {"no": 0.0, "somewhat": 0.5, "yes": 1.0}
ANSWER_ORDER = ["no", "somewhat", "yes"]


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
        default=Path("vlm_eval_runs/agreement_internvl3_sail_ola"),
    )
    return parser.parse_args()


def configured_runs(raw_runs: list[str]) -> dict[str, Path]:
    if not raw_runs:
        return DEFAULT_RUNS
    runs: dict[str, Path] = {}
    for raw in raw_runs:
        if "=" not in raw:
            raise ValueError(f"--run must be label=path, got: {raw}")
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty run label in: {raw}")
        runs[label] = Path(path)
    if len(runs) < 2:
        raise ValueError("Need at least two runs to compare.")
    return runs


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = record.get("pair_id")
            if not pair_id:
                raise ValueError(f"Missing pair_id in {path}:{line_no}")
            if pair_id in rows:
                raise ValueError(f"Duplicate pair_id {pair_id!r} in {path}")
            rows[pair_id] = record
    return rows


def normalize_answer(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("answer")
    if value is None:
        return None
    answer = str(value).strip().lower()
    return answer if answer in ANSWER_VALUES else None


def iter_answers(record: dict[str, Any], include_prescreen: bool) -> dict[str, str]:
    normalized = record.get("normalized_response") or {}
    answers: dict[str, str] = {}
    if include_prescreen:
        ps1 = normalize_answer(normalized.get("PS1_has_semantic_content"))
        if ps1:
            answers["PS1_has_semantic_content"] = ps1
    for section in ("perceptual", "semantic"):
        section_values = normalized.get(section) or {}
        if not isinstance(section_values, dict):
            continue
        for question_key, value in section_values.items():
            answer = normalize_answer(value)
            if answer:
                answers[f"{section}.{question_key}"] = answer
    return answers


def score(record: dict[str, Any], key: str) -> float | None:
    scores = record.get("scores") or {}
    value = scores.get(key)
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except TypeError:
        return None
    return float(value)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(rank_values(xs), rank_values(ys))


def cohen_kappa(left: list[str], right: list[str], weighted: bool = False) -> float | None:
    if len(left) != len(right) or not left:
        return None
    n = len(left)
    observed = 0.0
    for a, b in zip(left, right):
        if weighted:
            observed += 1.0 - abs(ANSWER_VALUES[a] - ANSWER_VALUES[b])
        else:
            observed += 1.0 if a == b else 0.0
    observed /= n

    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = 0.0
    for a in ANSWER_ORDER:
        for b in ANSWER_ORDER:
            agreement = 1.0 - abs(ANSWER_VALUES[a] - ANSWER_VALUES[b]) if weighted else float(a == b)
            expected += agreement * (left_counts[a] / n) * (right_counts[b] / n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(items: list[dict[str, str]], labels: list[str]) -> float | None:
    if not items or len(labels) < 2:
        return None
    n_raters = len(labels)
    p_i_values: list[float] = []
    category_totals = Counter()
    for item in items:
        counts = Counter(item[label] for label in labels)
        category_totals.update(counts)
        p_i = (sum(count * count for count in counts.values()) - n_raters) / (n_raters * (n_raters - 1))
        p_i_values.append(p_i)
    p_bar = mean(p_i_values)
    total_ratings = len(items) * n_raters
    p_e = sum((category_totals[answer] / total_ratings) ** 2 for answer in ANSWER_ORDER)
    if p_e == 1.0:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def summarize_items(items: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if not items:
        return {"n": 0}
    unanimous = sum(1 for item in items if len({item[label] for label in labels}) == 1)
    all_different = sum(1 for item in items if len({item[label] for label in labels}) == len(labels))
    values_by_label = {
        label: [ANSWER_VALUES[item[label]] for item in items]
        for label in labels
    }
    answer_dist = {
        label: dict(Counter(item[label] for item in items))
        for label in labels
    }
    pairwise: dict[str, dict[str, Any]] = {}
    for left, right in combinations(labels, 2):
        left_answers = [item[left] for item in items]
        right_answers = [item[right] for item in items]
        diffs = [abs(ANSWER_VALUES[a] - ANSWER_VALUES[b]) for a, b in zip(left_answers, right_answers)]
        pairwise[f"{left}__{right}"] = {
            "n": len(items),
            "exact_agreement": sum(a == b for a, b in zip(left_answers, right_answers)) / len(items),
            "adjacent_disagreement": sum(diff == 0.5 for diff in diffs) / len(items),
            "opposite_disagreement": sum(diff == 1.0 for diff in diffs) / len(items),
            "mean_abs_score_diff": mean(diffs),
            "cohen_kappa": cohen_kappa(left_answers, right_answers, weighted=False),
            "weighted_cohen_kappa": cohen_kappa(left_answers, right_answers, weighted=True),
        }
    return {
        "n": len(items),
        "unanimous_rate": unanimous / len(items),
        "two_vs_one_rate": (len(items) - unanimous - all_different) / len(items),
        "all_different_rate": all_different / len(items),
        "fleiss_kappa": fleiss_kappa(items, labels),
        "mean_answer_score": {label: mean(values) for label, values in values_by_label.items()},
        "answer_distribution": answer_dist,
        "pairwise": pairwise,
    }


def group_key(item: dict[str, Any], group: str) -> str:
    if group == "routing":
        return item["routing"]
    if group == "method":
        return item["method"]
    if group == "section":
        return item["question"].split(".", 1)[0]
    if group == "question":
        return item["question"]
    raise ValueError(group)


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
        raise ValueError("No common pair_id values across runs.")

    items_by_scope: dict[str, list[dict[str, Any]]] = {"scored_only": [], "with_prescreen": []}
    missing = defaultdict(int)
    for include_prescreen, scope in [(False, "scored_only"), (True, "with_prescreen")]:
        for pair_id in common_pair_ids:
            answers_by_label = {
                label: iter_answers(loaded[label][pair_id], include_prescreen=include_prescreen)
                for label in labels
            }
            questions = set.intersection(*(set(answers) for answers in answers_by_label.values()))
            all_questions = set.union(*(set(answers) for answers in answers_by_label.values()))
            for question in all_questions - questions:
                missing[(scope, question)] += 1
            base = loaded[labels[0]][pair_id]
            for question in sorted(questions):
                item = {
                    "pair_id": pair_id,
                    "question": question,
                    "routing": base.get("routing"),
                    "method": base.get("method"),
                    "subject": base.get("subject"),
                    "concept": base.get("concept"),
                }
                item.update({label: answers_by_label[label][question] for label in labels})
                items_by_scope[scope].append(item)

    score_rows: list[dict[str, Any]] = []
    for score_key in ("T_PAS", "T_SAS"):
        for left, right in combinations(labels, 2):
            xs: list[float] = []
            ys: list[float] = []
            for pair_id in common_pair_ids:
                x = score(loaded[left][pair_id], score_key)
                y = score(loaded[right][pair_id], score_key)
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if xs:
                diffs = [abs(x - y) for x, y in zip(xs, ys)]
                score_rows.append(
                    {
                        "score": score_key,
                        "pair": f"{left}__{right}",
                        "n": len(xs),
                        "pearson": pearson(xs, ys),
                        "spearman": spearman(xs, ys),
                        "mean_abs_diff": mean(diffs),
                        f"mean_{left}": mean(xs),
                        f"mean_{right}": mean(ys),
                    }
                )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "runs": {label: str(path) for label, path in runs.items()},
        "row_counts": {label: len(rows) for label, rows in loaded.items()},
        "common_pair_count": len(common_pair_ids),
        "scope_summary": {
            scope: summarize_items(items, labels)
            for scope, items in items_by_scope.items()
        },
        "missing_common_question_counts": {f"{scope}:{question}": count for (scope, question), count in missing.items()},
        "score_correlations": score_rows,
    }
    (output_dir / "agreement_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    question_rows: list[dict[str, Any]] = []
    for question in sorted({item["question"] for item in items_by_scope["scored_only"]}):
        question_items = [item for item in items_by_scope["scored_only"] if item["question"] == question]
        stats = summarize_items(question_items, labels)
        row = {
            "question": question,
            "n": stats["n"],
            "unanimous_rate": stats["unanimous_rate"],
            "two_vs_one_rate": stats["two_vs_one_rate"],
            "all_different_rate": stats["all_different_rate"],
            "fleiss_kappa": stats["fleiss_kappa"],
        }
        for label, value in stats["mean_answer_score"].items():
            row[f"mean_{label}"] = value
        for pair, pair_stats in stats["pairwise"].items():
            row[f"exact_{pair}"] = pair_stats["exact_agreement"]
            row[f"weighted_kappa_{pair}"] = pair_stats["weighted_cohen_kappa"]
        question_rows.append(row)
    question_rows.sort(key=lambda row: (row["unanimous_rate"], row["fleiss_kappa"] if row["fleiss_kappa"] is not None else -9))
    write_csv(
        output_dir / "question_agreement.csv",
        question_rows,
        [
            "question",
            "n",
            "unanimous_rate",
            "two_vs_one_rate",
            "all_different_rate",
            "fleiss_kappa",
            *(f"mean_{label}" for label in labels),
            *(f"exact_{left}__{right}" for left, right in combinations(labels, 2)),
            *(f"weighted_kappa_{left}__{right}" for left, right in combinations(labels, 2)),
        ],
    )

    group_rows: list[dict[str, Any]] = []
    for group in ("routing", "method", "section"):
        values = sorted({group_key(item, group) for item in items_by_scope["scored_only"]})
        for value in values:
            group_items = [item for item in items_by_scope["scored_only"] if group_key(item, group) == value]
            stats = summarize_items(group_items, labels)
            row = {
                "group": group,
                "value": value,
                "n": stats["n"],
                "unanimous_rate": stats["unanimous_rate"],
                "two_vs_one_rate": stats["two_vs_one_rate"],
                "all_different_rate": stats["all_different_rate"],
                "fleiss_kappa": stats["fleiss_kappa"],
            }
            for label, score_value in stats["mean_answer_score"].items():
                row[f"mean_{label}"] = score_value
            group_rows.append(row)
    write_csv(
        output_dir / "group_agreement.csv",
        group_rows,
        [
            "group",
            "value",
            "n",
            "unanimous_rate",
            "two_vs_one_rate",
            "all_different_rate",
            "fleiss_kappa",
            *(f"mean_{label}" for label in labels),
        ],
    )

    disagreement_rows: list[dict[str, Any]] = []
    for item in items_by_scope["scored_only"]:
        values = [ANSWER_VALUES[item[label]] for label in labels]
        if max(values) - min(values) < 1.0:
            continue
        row = {key: item[key] for key in ("pair_id", "method", "routing", "subject", "concept", "question")}
        row.update({label: item[label] for label in labels})
        disagreement_rows.append(row)
    write_csv(
        output_dir / "strong_disagreements.csv",
        disagreement_rows,
        ["pair_id", "method", "routing", "subject", "concept", "question", *labels],
    )

    write_csv(
        output_dir / "score_correlations.csv",
        score_rows,
        sorted({field for row in score_rows for field in row}),
    )

    print(json.dumps({
        "output_dir": str(output_dir),
        "common_pair_count": len(common_pair_ids),
        "scored_items": len(items_by_scope["scored_only"]),
        "with_prescreen_items": len(items_by_scope["with_prescreen"]),
        "scored_summary": report["scope_summary"]["scored_only"],
    }, indent=2))


if __name__ == "__main__":
    main()
