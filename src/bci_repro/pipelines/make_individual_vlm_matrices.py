#!/usr/bin/env python3
"""Export individual VLM agreement matrices for paper stitching."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


AGREEMENT_JSON = Path("vlm_eval_runs/agreement_internvl3_sail_ola_ovis/agreement_summary.json")
REASONING_JSON = Path("vlm_eval_runs/reasoning_semantics_internvl3_sail_ola_ovis/reasoning_semantic_summary.json")
OUT_DIR = Path("figures/individual")

MODEL_KEYS = ["internvl3", "sail", "ola", "ovis"]
MODEL_LABELS = ["InternVL3", "SAIL", "OLA", "Ovis2"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 350,
        }
    )


def matrix_from_pairwise(pairwise: dict[str, dict[str, float]], field: str) -> np.ndarray:
    matrix = np.full((len(MODEL_KEYS), len(MODEL_KEYS)), np.nan, dtype=np.float32)
    for row, left in enumerate(MODEL_KEYS):
        for col, right in enumerate(MODEL_KEYS):
            if left == right:
                matrix[row, col] = 1.0
                continue
            key = f"{left}__{right}"
            if key not in pairwise:
                key = f"{right}__{left}"
            matrix[row, col] = float(pairwise[key][field])
    return matrix


def annotate(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 0.62 else "#202428"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=10, color=color)


def render_matrix(matrix: np.ndarray, title: str, colorbar_label: str, stem: str, vmin: float, vmax: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.35, 3.25))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(MODEL_LABELS)))
    ax.set_yticks(np.arange(len(MODEL_LABELS)))
    ax.set_xticklabels(MODEL_LABELS, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(MODEL_LABELS)
    ax.set_title(title, pad=10, weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(MODEL_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate(ax, matrix)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    agreement = json.loads(AGREEMENT_JSON.read_text(encoding="utf-8"))
    reasoning = json.loads(REASONING_JSON.read_text(encoding="utf-8"))

    answer_matrix = matrix_from_pairwise(
        agreement["scope_summary"]["scored_only"]["pairwise"],
        "weighted_cohen_kappa",
    )
    reasoning_matrix = matrix_from_pairwise(reasoning["pairwise"], "mean")

    render_matrix(
        answer_matrix,
        "Answer Agreement",
        r"Weighted $\kappa$",
        "vlm_answer_agreement_matrix",
        0.25,
        1.0,
    )
    render_matrix(
        reasoning_matrix,
        "Reasoning Similarity",
        "SBERT cosine",
        "vlm_reasoning_similarity_matrix",
        0.50,
        1.0,
    )
    print(f"Wrote {OUT_DIR / 'vlm_answer_agreement_matrix.png'}")
    print(f"Wrote {OUT_DIR / 'vlm_answer_agreement_matrix.pdf'}")
    print(f"Wrote {OUT_DIR / 'vlm_reasoning_similarity_matrix.png'}")
    print(f"Wrote {OUT_DIR / 'vlm_reasoning_similarity_matrix.pdf'}")


if __name__ == "__main__":
    main()
