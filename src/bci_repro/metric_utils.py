from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "mse",
    "psnr",
    "ssim",
    "dreamsim_score",
    "openclip_cosine",
    "dinov2_cosine",
    "lpips_alex",
    "dists",
    "topiq_fr_score",
    "pieapp_score",
    "imagereward_score",
    "blip_caption_sbert_cosine",
]

LOWER_IS_BETTER = {"mse", "dreamsim_score", "lpips_alex", "dists", "pieapp_score"}


def oriented_quality(series: pd.Series, metric: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return -values if metric in LOWER_IS_BETTER else values


def corr_value(left: pd.Series, right: pd.Series, method: str = "spearman") -> float | None:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3:
        return None
    if method == "spearman":
        left_values = frame.iloc[:, 0].rank(method="average")
        right_values = frame.iloc[:, 1].rank(method="average")
        value = left_values.corr(right_values, method="pearson")
    else:
        value = frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method)
    return None if pd.isna(value) else float(value)


def quartile_failure_rates(
    metric_values: pd.Series,
    semantic_values: pd.Series,
    metric: str,
    semantic_threshold: float = 0.5,
) -> dict[str, Any]:
    q = oriented_quality(metric_values, metric)
    frame = pd.concat([q.rename("quality"), semantic_values.rename("semantic")], axis=1).dropna()
    if frame.empty:
        return {"n": 0, "harshness_rate": None, "blind_spot_rate": None}
    q25 = float(frame["quality"].quantile(0.25))
    q75 = float(frame["quality"].quantile(0.75))
    recoverable = frame["semantic"] >= semantic_threshold
    harsh = recoverable & (frame["quality"] < q25)
    high_metric = frame["quality"] >= q75
    blind = high_metric & (frame["semantic"] < semantic_threshold)
    return {
        "n": int(len(frame)),
        "quality_q25": q25,
        "quality_q75": q75,
        "harshness_rate": float(harsh.sum() / recoverable.sum()) if recoverable.sum() else None,
        "blind_spot_rate": float(blind.sum() / high_metric.sum()) if high_metric.sum() else None,
    }


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(max_value / math.sqrt(value)))

