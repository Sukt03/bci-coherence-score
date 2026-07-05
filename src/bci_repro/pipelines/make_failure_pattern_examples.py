#!/usr/bin/env python3
"""Render real image-pair examples for metric failure patterns.

The figure uses real ground-truth/generated images plus the already computed
metric and VLM-consensus scores. It does not synthesize or edit images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


DEFAULT_PAIR_JSONL = Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl")
DEFAULT_SCORES_CSV = Path("paper_analysis/metric_failure_20260601/merged_metric_vlm_scores.csv")
DEFAULT_OUTPUT = Path("paper_analysis/figures/fig_failure_pattern_examples.pdf")

DEFAULT_EXAMPLES = [
    {
        "label": "Harshness",
        "subtitle": "Low pixel similarity, high semantic recoverability",
        "pair_id": "brainvis__extra__chinaware__rank1__cand2__row36",
    },
    {
        "label": "Semantic blindness",
        "subtitle": "High low-level similarity, wrong semantic content",
        "pair_id": "ATM__sub-05__eagle__rank1__cand1",
    },
]

SCORE_COLUMNS = [
    ("MSE", "mse"),
    ("PSNR", "psnr"),
    ("SSIM", "ssim"),
    ("DreamSim", "dreamsim_score"),
    ("OpenCLIP", "openclip_cosine"),
    ("BLIP-SBERT", "blip_caption_sbert_cosine"),
    ("T-PAS", "T_PAS_consensus"),
    ("T-SAS", "T_SAS_consensus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        metavar="LABEL|SUBTITLE|PAIR_ID",
        help="Override examples. Can be passed twice. Example: 'Harshness|Low SSIM, high T-SAS|PAIR_ID'",
    )
    return parser.parse_args()


def load_paths(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id")
            reference_path = row.get("reference_path")
            generated_path = row.get("generated_path")
            if not pair_id or not reference_path or not generated_path:
                raise ValueError(f"Missing path fields in {path}:{line_no}")
            rows[str(pair_id)] = {
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
            }
    return rows


def load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path)
    if "pair_id_x" in scores.columns:
        scores = scores.rename(columns={"pair_id_x": "pair_id"})
    required = {"pair_id", "method", "concept", *[column for _, column in SCORE_COLUMNS]}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return scores


def parse_examples(raw_examples: list[str]) -> list[dict[str, str]]:
    if not raw_examples:
        return DEFAULT_EXAMPLES
    examples: list[dict[str, str]] = []
    for raw in raw_examples:
        pieces = raw.split("|")
        if len(pieces) != 3:
            raise ValueError("--example must have format LABEL|SUBTITLE|PAIR_ID")
        label, subtitle, pair_id = [piece.strip() for piece in pieces]
        if not label or not subtitle or not pair_id:
            raise ValueError("--example fields cannot be empty")
        examples.append({"label": label, "subtitle": subtitle, "pair_id": pair_id})
    return examples


def image_for_panel(path: Path, size: int = 420) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def score_text(row: pd.Series) -> str:
    lines = [
        f"Method: {row['method']}",
        f"Concept: {row['concept']}",
        "",
        "Conventional / learned metrics",
    ]
    for label, column in SCORE_COLUMNS[:6]:
        lines.append(f"{label}: {number(row[column])}")
    lines.extend(["", "BCI-aware VLM consensus"])
    for label, column in SCORE_COLUMNS[6:]:
        lines.append(f"{label}: {number(row[column])}")
    return "\n".join(lines)


def validate_examples(examples: list[dict[str, str]], paths: dict[str, dict[str, str]], scores: pd.DataFrame) -> None:
    score_ids = set(scores["pair_id"].astype(str))
    for example in examples:
        pair_id = example["pair_id"]
        if pair_id not in paths:
            raise KeyError(f"{pair_id!r} not found in pair JSONL")
        if pair_id not in score_ids:
            raise KeyError(f"{pair_id!r} not found in score CSV")
        for image_key in ("reference_path", "generated_path"):
            image_path = Path(paths[pair_id][image_key])
            if not image_path.exists():
                raise FileNotFoundError(image_path)


def render(examples: list[dict[str, str]], paths: dict[str, dict[str, str]], scores: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(
        nrows=len(examples),
        ncols=3,
        figsize=(10.8, 6.7),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.18], "wspace": 0.06, "hspace": 0.32},
    )
    if len(examples) == 1:
        axes = axes[None, :]

    column_titles = ["Ground truth", "Generated", "Scores"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13, weight="bold", pad=10)

    for row_idx, example in enumerate(examples):
        pair_id = example["pair_id"]
        score_row = scores[scores["pair_id"].astype(str).eq(pair_id)].iloc[0]
        image_paths = paths[pair_id]

        gt_image = image_for_panel(Path(image_paths["reference_path"]))
        gen_image = image_for_panel(Path(image_paths["generated_path"]))

        for col_idx, image in enumerate((gt_image, gen_image)):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#cccccc")

        score_ax = axes[row_idx, 2]
        score_ax.axis("off")
        score_ax.text(
            0.0,
            0.98,
            example["label"],
            transform=score_ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            weight="bold",
        )
        score_ax.text(
            0.0,
            0.88,
            example["subtitle"],
            transform=score_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color="#444444",
            wrap=True,
        )
        score_ax.text(
            0.0,
            0.73,
            score_text(score_row),
            transform=score_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            family="monospace",
            linespacing=1.28,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    png_output = output.with_suffix(".png")
    fig.savefig(png_output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {png_output}")


def main() -> None:
    args = parse_args()
    examples = parse_examples(args.example)
    paths = load_paths(args.pair_jsonl)
    scores = load_scores(args.scores_csv)
    validate_examples(examples, paths, scores)
    render(examples, paths, scores, args.output, args.dpi)


if __name__ == "__main__":
    main()
