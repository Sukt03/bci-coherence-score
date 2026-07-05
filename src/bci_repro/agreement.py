from __future__ import annotations

from collections import Counter
from statistics import mean

ANSWER_VALUES = {"no": 0.0, "somewhat": 0.5, "yes": 1.0}
ANSWER_ORDER = ["no", "somewhat", "yes"]


def cohen_kappa(left: list[str], right: list[str], weighted: bool = False) -> float | None:
    if len(left) != len(right) or not left:
        return None
    n = len(left)
    observed = 0.0
    for a, b in zip(left, right):
        observed += 1.0 - abs(ANSWER_VALUES[a] - ANSWER_VALUES[b]) if weighted else float(a == b)
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

