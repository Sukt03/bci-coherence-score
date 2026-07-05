#!/usr/bin/env python3
"""Generate paper figures from saved metric/VLM/distiller artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("paper_analysis/figures")
METRIC_DIR = Path("paper_analysis/metric_failure_20260601")
AGREEMENT_DIR = Path("vlm_eval_runs/agreement_internvl3_sail_ola_ovis")
REASONING_DIR = Path("vlm_eval_runs/reasoning_semantics_internvl3_sail_ola_ovis")
BCS_NPZ = Path("distill_runs/v4teacher_fusion_siglip_clip_dinov3_mlp_ensemble5_20260601/predictions_all.npz")

METRIC_ORDER = [
    "MSE",
    "SSIM",
    "LPIPS",
    "DISTS",
    "DreamSim",
    "OpenCLIP",
    "DINOv2",
    "ImageReward",
]

MODEL_LABELS = ["InternVL3", "SAIL", "OLA", "Ovis2"]
MODEL_KEYS = ["internvl3", "sail", "ola", "ovis"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e8ec",
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def figure_metric_audit() -> None:
    caption = pd.read_csv(METRIC_DIR / "caption_proxy_metric_audit.csv")
    caption = caption.set_index("metric_display").loc[METRIC_ORDER].reset_index()

    colors = {
        "corr": "#386cb0",
        "hr": "#fdb462",
        "bsr": "#fb8072",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.25]})

    y = np.arange(len(caption))
    axes[0].barh(y, caption["spearman_caption"], color=colors["corr"], height=0.62)
    axes[0].axvline(0, color="#2f3437", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(caption["metric_display"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"Spearman $\rho$ with caption consistency")
    axes[0].set_title("(a) Semantic-proxy correlation")
    axes[0].set_xlim(-0.10, 0.58)

    height = 0.34
    axes[1].barh(
        y - height / 2,
        caption["caption_harshness_rate"],
        color=colors["hr"],
        height=height,
        label="Caption-HR",
    )
    axes[1].barh(
        y + height / 2,
        caption["caption_blind_spot_rate"],
        color=colors["bsr"],
        height=height,
        label="Caption-BSR",
    )
    axes[1].axvline(0.25, color="#555", linestyle="--", linewidth=0.9, label="quartile chance")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Caption-proxy failure rate")
    axes[1].set_title("(b) Non-circular failure audit")
    axes[1].set_xlim(0, 0.32)
    axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle("Current metrics weakly track caption-level semantic consistency", y=1.04, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig_metric_audit_caption_proxy")


def matrix_from_pairwise(pairwise: dict[str, dict[str, float]], field: str) -> np.ndarray:
    matrix = np.full((len(MODEL_KEYS), len(MODEL_KEYS)), np.nan, dtype=np.float32)
    for i, left in enumerate(MODEL_KEYS):
        for j, right in enumerate(MODEL_KEYS):
            if i == j:
                matrix[i, j] = 1.0
                continue
            key = f"{left}__{right}" if f"{left}__{right}" in pairwise else f"{right}__{left}"
            matrix[i, j] = float(pairwise[key][field])
    return matrix


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "1.00" if i == j else f"{value:.2f}"
            color = "white" if value >= 0.62 else "#202428"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)


def figure_vlm_agreement() -> None:
    agreement = json.loads((AGREEMENT_DIR / "agreement_summary.json").read_text(encoding="utf-8"))
    reasoning = json.loads((REASONING_DIR / "reasoning_semantic_summary.json").read_text(encoding="utf-8"))
    pairwise_agreement = agreement["scope_summary"]["scored_only"]["pairwise"]
    pairwise_reasoning = reasoning["pairwise"]

    kappa = matrix_from_pairwise(pairwise_agreement, "weighted_cohen_kappa")
    reason = matrix_from_pairwise(pairwise_reasoning, "mean")

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15))
    for ax, matrix, title, vmin, vmax in [
        (axes[0], kappa, "(a) Answer agreement\nweighted $\\kappa$", 0.25, 1.0),
        (axes[1], reason, "(b) Reasoning similarity\nSBERT cosine", 0.50, 1.0),
    ]:
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(MODEL_LABELS)))
        ax.set_yticks(np.arange(len(MODEL_LABELS)))
        ax.set_xticklabels(MODEL_LABELS, rotation=35, ha="right")
        ax.set_yticklabels(MODEL_LABELS)
        ax.set_title(title)
        ax.grid(False)
        annotate_heatmap(ax, matrix)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Four-VLM scoring is consistent but not single-model dependent", y=1.03, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig_vlm_agreement_reasoning")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg
        start = end
    return ranks


def corr(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask].astype(np.float64)
    right = right[mask].astype(np.float64)
    if len(left) < 3:
        return float("nan")
    if spearman:
        left = rankdata(left)
        right = rankdata(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def aggregate_scores(values: np.ndarray, weights: np.ndarray, question_mask: Iterable[bool]) -> np.ndarray:
    question_mask = np.array(list(question_mask), dtype=bool)
    active = (weights[:, question_mask] > 0) & np.isfinite(values[:, question_mask])
    masked = np.where(active, values[:, question_mask], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(masked, axis=1)


def figure_bcs_scatter() -> None:
    data = np.load(BCS_NPZ, allow_pickle=True)
    predictions = data["predictions"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    test_indices = data["test_indices"].astype(np.int64)
    questions = [str(item) for item in data["questions"].tolist()]

    pred_test = predictions[test_indices]
    target_test = targets[test_indices]
    weight_test = weights[test_indices]

    perceptual_mask = [question.startswith("perceptual.") for question in questions]
    semantic_mask = [question.startswith("semantic.") for question in questions]
    panels = [
        ("T-PAS", aggregate_scores(target_test, weight_test, perceptual_mask), aggregate_scores(pred_test, weight_test, perceptual_mask)),
        ("T-SAS", aggregate_scores(target_test, weight_test, semantic_mask), aggregate_scores(pred_test, weight_test, semantic_mask)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.05), sharex=True, sharey=True)
    for ax, (label, target, pred) in zip(axes, panels):
        mask = np.isfinite(target) & np.isfinite(pred)
        target = target[mask]
        pred = pred[mask]
        mae = float(np.mean(np.abs(pred - target)))
        pearson = corr(pred, target)
        spearman = corr(pred, target, spearman=True)

        ax.scatter(target, pred, s=12, alpha=0.38, color="#386cb0", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="#202428", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Four-VLM consensus target")
        ax.text(
            0.04,
            0.96,
            f"MAE={mae:.3f}\n$r$={pearson:.3f}\n$\\rho$={spearman:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d8dde6", "boxstyle": "round,pad=0.25"},
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("BCI-Coherence prediction")
    fig.suptitle("BCI-Coherence Score predicts four-VLM perceptual and semantic consensus", y=1.04, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig_bcs_predicted_vs_target")


def main() -> None:
    setup_style()
    figure_metric_audit()
    figure_vlm_agreement()
    figure_bcs_scatter()
    print(f"Wrote figures to {OUT_DIR}")
    for path in sorted(OUT_DIR.glob("fig_*")):
        print(path)


if __name__ == "__main__":
    main()
