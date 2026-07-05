#!/usr/bin/env python3
"""Analyze conventional metric behavior against BCI-aware VLM scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUNS = {
    "internvl3": Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl"),
    "sail": Path("vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl"),
    "ola": Path("vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl"),
    "ovis": Path("vlm_eval_runs/ovis_full_both_reasoning_20260530_083349/pair_scores.jsonl"),
}

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
HIGHER_IS_BETTER = set(METRIC_COLUMNS) - LOWER_IS_BETTER

BASE_KEY = ["method", "subject", "concept", "rank_num", "candidate_index"]
EXTRA_KEY = ["method", "concept", "rank_num", "candidate_index"]
EXTRA_METHODS = {
    "brainvis",
    "dreamdiffusion",
    "thingseeg_brainvis",
    "thingseeg_dreamdiffusion",
    "cvpr40_brainvis",
    "cvpr40_dreamdiffusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-csv", type=Path, default=Path("expanded_metric_scores_clean_with_sbert.csv"))
    parser.add_argument("--extra-metric-csv", type=Path, default=Path("expanded_extra_model_885_clean_with_sbert(1).csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_analysis/metric_failure_20260601"))
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument("--perceptual-threshold", type=float, default=0.5)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional VLM run as label=pair_scores.jsonl. Defaults to InternVL3, SAIL, OLA, Ovis.",
    )
    return parser.parse_args()


def parse_numeric_suffix(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def configured_runs(raw_runs: list[str]) -> dict[str, Path]:
    if not raw_runs:
        return DEFAULT_RUNS
    runs: dict[str, Path] = {}
    for raw in raw_runs:
        if "=" not in raw:
            raise ValueError(f"--run must be label=path, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty run label in {raw!r}")
        runs[label] = Path(path)
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


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(output) else output


def load_consensus_annotations(runs: dict[str, Path]) -> pd.DataFrame:
    loaded = {label: read_jsonl(path) for label, path in runs.items()}
    labels = list(loaded)
    common_pair_ids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if not common_pair_ids:
        raise ValueError("No common VLM pair IDs.")

    rows: list[dict[str, Any]] = []
    for pair_id in common_pair_ids:
        base = loaded[labels[0]][pair_id]
        t_pas_values: list[float] = []
        t_sas_values: list[float] = []
        row = {
            "pair_id": pair_id,
            "method": str(base.get("method") or ""),
            "subject": str(base.get("subject") or ""),
            "concept": str(base.get("concept") or ""),
            "routing": str(base.get("routing") or ""),
            "rank": str(base.get("rank") or ""),
            "candidate": str(base.get("candidate") or ""),
            "rank_num": parse_numeric_suffix(base.get("rank")),
            "candidate_index": parse_numeric_suffix(base.get("candidate")),
        }
        for label in labels:
            scores = loaded[label][pair_id].get("scores") or {}
            t_pas = numeric_or_none(scores.get("T_PAS"))
            t_sas = numeric_or_none(scores.get("T_SAS"))
            row[f"{label}_T_PAS"] = t_pas
            row[f"{label}_T_SAS"] = t_sas
            if t_pas is not None:
                t_pas_values.append(t_pas)
            if t_sas is not None:
                t_sas_values.append(t_sas)
        row["T_PAS_consensus"] = float(median(t_pas_values)) if t_pas_values else np.nan
        row["T_SAS_consensus"] = float(median(t_sas_values)) if t_sas_values else np.nan
        row["T_PAS_mean"] = float(mean(t_pas_values)) if t_pas_values else np.nan
        row["T_SAS_mean"] = float(mean(t_sas_values)) if t_sas_values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_base_metrics(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    metrics = raw.copy()
    metrics["method"] = metrics["model"].astype(str)
    metrics["subject"] = metrics["subject"].astype(str)
    metrics["concept"] = metrics["class_name"].astype(str)
    metrics["rank_num"] = pd.to_numeric(metrics["selection_rank"], errors="coerce").astype("Int64")
    metrics["candidate_index"] = pd.to_numeric(metrics["candidate_index"], errors="coerce").astype("Int64")
    metrics["metric_source"] = "atm_enigma_metric_csv"
    keep = BASE_KEY + ["dataset_group", "pair_id", "metric_source"] + [column for column in METRIC_COLUMNS if column in metrics]
    return metrics[keep]


def load_extra_metrics(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    metrics = raw.copy()
    if "topiq_fr_score" not in metrics and "import_topiq_fr_score" in metrics:
        metrics["topiq_fr_score"] = metrics["import_topiq_fr_score"]
    if "pieapp_score" not in metrics and "import_pieapp_score" in metrics:
        metrics["pieapp_score"] = metrics["import_pieapp_score"]
    metrics["method"] = metrics["detected_model"].astype(str)
    metrics["subject"] = "extra"
    metrics["concept"] = metrics["class_name_or_stem"].astype(str)
    metrics["rank_num"] = pd.to_numeric(metrics["consensus_rank"], errors="coerce").astype("Int64")
    metrics["candidate_index"] = pd.to_numeric(metrics["candidate_index"], errors="coerce").astype("Int64")
    metrics["metric_source"] = "extra_metric_csv"
    metrics["pair_id"] = ""
    keep = BASE_KEY + ["dataset_group", "pair_id", "metric_source"] + [column for column in METRIC_COLUMNS if column in metrics]
    return metrics[keep]


def merge_annotations_and_metrics(annotations: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotations = annotations.copy()
    annotations["rank_num"] = pd.to_numeric(annotations["rank_num"], errors="coerce").astype("Int64")
    annotations["candidate_index"] = pd.to_numeric(annotations["candidate_index"], errors="coerce").astype("Int64")

    base_annotations = annotations[~annotations["method"].isin(EXTRA_METHODS)].copy()
    extra_annotations = annotations[annotations["method"].isin(EXTRA_METHODS)].copy()
    base_metrics = metrics[~metrics["method"].isin(EXTRA_METHODS)].copy()
    extra_metrics = metrics[metrics["method"].isin(EXTRA_METHODS)].copy()

    base_merged = base_annotations.merge(base_metrics, on=BASE_KEY, how="left", indicator="_metric_merge")
    extra_merged = extra_annotations.merge(
        extra_metrics.drop(columns=["subject"]),
        on=EXTRA_KEY,
        how="left",
        indicator="_metric_merge",
    )
    # Restore the annotation subject column name after the extra merge.
    if "subject_x" in extra_merged.columns:
        extra_merged = extra_merged.rename(columns={"subject_x": "subject"})
    if "subject_y" in extra_merged.columns:
        extra_merged = extra_merged.drop(columns=["subject_y"])

    merged = pd.concat([base_merged, extra_merged], ignore_index=True)
    merged["metric_matched"] = merged["_metric_merge"].eq("both")
    stats = {
        "annotation_rows": int(len(annotations)),
        "metric_rows": int(len(metrics)),
        "merged_rows": int(len(merged)),
        "matched_rows": int(merged["metric_matched"].sum()),
        "unmatched_rows": int((~merged["metric_matched"]).sum()),
        "base_annotation_rows": int(len(base_annotations)),
        "extra_annotation_rows": int(len(extra_annotations)),
        "base_metric_rows": int(len(base_metrics)),
        "extra_metric_rows": int(len(extra_metrics)),
        "base_metric_duplicate_keys": int(base_metrics.duplicated(BASE_KEY).sum()),
        "extra_metric_duplicate_keys": int(extra_metrics.duplicated(EXTRA_KEY).sum()),
    }
    return merged, stats


def quality_score(series: pd.Series, metric: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return -values if metric in LOWER_IS_BETTER else values


def corr_value(left: pd.Series, right: pd.Series, method: str) -> float | None:
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


def correlation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        if metric not in df:
            continue
        q = quality_score(df[metric], metric)
        for target in ("T_PAS_consensus", "T_SAS_consensus"):
            frame = pd.concat([q.rename("metric_quality"), df[target]], axis=1).dropna()
            rows.append(
                {
                    "metric": metric,
                    "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                    "target": target,
                    "n": int(len(frame)),
                    "pearson": corr_value(q, df[target], "pearson"),
                    "spearman": corr_value(q, df[target], "spearman"),
                }
            )
    return rows


def failure_rows(df: pd.DataFrame, semantic_threshold: float) -> list[dict[str, Any]]:
    object_df = df[(df["routing"] == "object") & df["T_SAS_consensus"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        if metric not in object_df:
            continue
        q = quality_score(object_df[metric], metric)
        frame = object_df.assign(_q=q).dropna(subset=["_q", "T_SAS_consensus"])
        if frame.empty:
            continue
        q25 = float(frame["_q"].quantile(0.25))
        q75 = float(frame["_q"].quantile(0.75))
        recoverable = frame["T_SAS_consensus"] >= semantic_threshold
        harsh = recoverable & (frame["_q"] < q25)
        high_metric = frame["_q"] >= q75
        blind = high_metric & (frame["T_SAS_consensus"] < semantic_threshold)
        rows.append(
            {
                "metric": metric,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "n": int(len(frame)),
                "semantic_threshold": semantic_threshold,
                "semantically_recoverable_n": int(recoverable.sum()),
                "harsh_count": int(harsh.sum()),
                "harshness_rate": float(harsh.sum() / recoverable.sum()) if recoverable.sum() else None,
                "high_metric_n": int(high_metric.sum()),
                "blind_spot_count": int(blind.sum()),
                "blind_spot_rate": float(blind.sum() / high_metric.sum()) if high_metric.sum() else None,
                "quality_q25": q25,
                "quality_q75": q75,
            }
        )
    return rows


def bin_label(t_pas: float, t_sas: float, p_threshold: float, s_threshold: float) -> str:
    if pd.isna(t_sas):
        return "abstract_or_semantic_null"
    p_high = t_pas >= p_threshold
    s_high = t_sas >= s_threshold
    if p_high and s_high:
        return "high_perceptual_high_semantic"
    if p_high and not s_high:
        return "high_perceptual_low_semantic"
    if not p_high and s_high:
        return "low_perceptual_high_semantic"
    return "low_perceptual_low_semantic"


def disagreement_bin_rows(df: pd.DataFrame, p_threshold: float, s_threshold: float) -> list[dict[str, Any]]:
    scored = df.copy()
    scored["disagreement_regime"] = [
        bin_label(t_pas, t_sas, p_threshold, s_threshold)
        for t_pas, t_sas in zip(scored["T_PAS_consensus"], scored["T_SAS_consensus"])
    ]
    rows: list[dict[str, Any]] = []
    for group_name, group_df in [("overall", scored), *[(str(method), part) for method, part in scored.groupby("method")]]:
        counts = Counter(group_df["disagreement_regime"])
        total = len(group_df)
        row = {"group": group_name, "n": int(total)}
        for label in [
            "high_perceptual_high_semantic",
            "high_perceptual_low_semantic",
            "low_perceptual_high_semantic",
            "low_perceptual_low_semantic",
            "abstract_or_semantic_null",
        ]:
            row[f"{label}_n"] = int(counts[label])
            row[f"{label}_rate"] = float(counts[label] / total) if total else None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_display(metric: str) -> str:
    names = {
        "mse": "MSE",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "dreamsim_score": "DreamSim",
        "openclip_cosine": "OpenCLIP",
        "dinov2_cosine": "DINOv2",
        "lpips_alex": "LPIPS",
        "dists": "DISTS",
        "topiq_fr_score": "TOPIQ-FR",
        "pieapp_score": "PieAPP",
        "imagereward_score": "ImageReward",
        "blip_caption_sbert_cosine": "BLIP-SBERT",
    }
    return names.get(metric, metric)


def write_latex_tables(output_dir: Path, correlations: list[dict[str, Any]], failures: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    corr_df = pd.DataFrame(correlations)
    rows = []
    for metric in METRIC_COLUMNS:
        metric_rows = corr_df[corr_df["metric"] == metric]
        tpas = metric_rows[metric_rows["target"] == "T_PAS_consensus"]
        tsas = metric_rows[metric_rows["target"] == "T_SAS_consensus"]
        if tpas.empty or tsas.empty:
            continue
        rows.append(
            [
                metric_display(metric),
                fmt_float(tpas.iloc[0]["spearman"]),
                fmt_float(tpas.iloc[0]["pearson"]),
                fmt_float(tsas.iloc[0]["spearman"]),
                fmt_float(tsas.iloc[0]["pearson"]),
            ]
        )
    table = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Metric & $\\rho$(T-PAS) & $r$(T-PAS) & $\\rho$(T-SAS) & $r$(T-SAS) \\\\",
        "\\midrule",
    ]
    table += [f"{latex_escape(row[0])} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\" for row in rows]
    table += ["\\bottomrule", "\\end{tabular}", ""]
    (output_dir / "table_metric_correlations.tex").write_text("\n".join(table), encoding="utf-8")

    failure_df = pd.DataFrame(failures).sort_values("blind_spot_rate", ascending=False)
    table = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Metric & HR $\\uparrow$ & Harsh / Recoverable & BSR $\\uparrow$ & Blind / High \\\\",
        "\\midrule",
    ]
    for _, row in failure_df.iterrows():
        table.append(
            f"{latex_escape(metric_display(row['metric']))} & "
            f"{fmt_float(row['harshness_rate'])} & "
            f"{int(row['harsh_count'])}/{int(row['semantically_recoverable_n'])} & "
            f"{fmt_float(row['blind_spot_rate'])} & "
            f"{int(row['blind_spot_count'])}/{int(row['high_metric_n'])} \\\\"
        )
    table += ["\\bottomrule", "\\end{tabular}", ""]
    (output_dir / "table_failure_rates.tex").write_text("\n".join(table), encoding="utf-8")

    bin_df = pd.DataFrame(bins)
    wanted = ["overall", "ATM", "ENIGMA", "brainvis", "dreamdiffusion", "thingseeg_brainvis", "thingseeg_dreamdiffusion"]
    bin_df = bin_df[bin_df["group"].isin(wanted)]
    table = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Group & N & High P / Low S & Low P / High S & Low P / Low S \\\\",
        "\\midrule",
    ]
    for _, row in bin_df.iterrows():
        table.append(
            f"{latex_escape(row['group'])} & {int(row['n'])} & "
            f"{int(row['high_perceptual_low_semantic_n'])} ({fmt_float(row['high_perceptual_low_semantic_rate'])}) & "
            f"{int(row['low_perceptual_high_semantic_n'])} ({fmt_float(row['low_perceptual_high_semantic_rate'])}) & "
            f"{int(row['low_perceptual_low_semantic_n'])} ({fmt_float(row['low_perceptual_low_semantic_rate'])}) \\\\"
        )
    table += ["\\bottomrule", "\\end{tabular}", ""]
    (output_dir / "table_perceptual_semantic_bins.tex").write_text("\n".join(table), encoding="utf-8")


def write_examples(df: pd.DataFrame, output_dir: Path, semantic_threshold: float) -> None:
    object_df = df[(df["routing"] == "object") & df["T_SAS_consensus"].notna()].copy()
    example_rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        if metric not in object_df:
            continue
        frame = object_df.copy()
        frame["_q"] = quality_score(frame[metric], metric)
        frame = frame.dropna(subset=["_q", "T_SAS_consensus"])
        if frame.empty:
            continue
        q25 = float(frame["_q"].quantile(0.25))
        q75 = float(frame["_q"].quantile(0.75))
        harsh = frame[(frame["T_SAS_consensus"] >= semantic_threshold) & (frame["_q"] < q25)].copy()
        blind = frame[(frame["T_SAS_consensus"] < semantic_threshold) & (frame["_q"] >= q75)].copy()
        harsh = harsh.sort_values(["T_SAS_consensus", "_q"], ascending=[False, True]).head(20)
        blind = blind.sort_values(["_q", "T_SAS_consensus"], ascending=[False, True]).head(20)
        for failure_type, rows in [("harshness", harsh), ("blind_spot", blind)]:
            for _, row in rows.iterrows():
                example_rows.append(
                    {
                        "failure_type": failure_type,
                        "metric": metric,
                        "pair_id": row["pair_id_x"] if "pair_id_x" in row else row["pair_id"],
                        "method": row["method"],
                        "subject": row["subject"],
                        "concept": row["concept"],
                        "T_PAS_consensus": row["T_PAS_consensus"],
                        "T_SAS_consensus": row["T_SAS_consensus"],
                        "metric_value": row.get(metric),
                        "metric_quality": row["_q"],
                    }
                )
    write_csv(output_dir / "failure_examples.csv", example_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_consensus_annotations(configured_runs(args.run))
    metrics = pd.concat([load_base_metrics(args.metric_csv), load_extra_metrics(args.extra_metric_csv)], ignore_index=True)
    merged, merge_stats = merge_annotations_and_metrics(annotations, metrics)

    correlations = correlation_rows(merged[merged["metric_matched"]].copy())
    failures = failure_rows(merged[merged["metric_matched"]].copy(), args.semantic_threshold)
    bins = disagreement_bin_rows(merged, args.perceptual_threshold, args.semantic_threshold)

    merged.to_csv(args.output_dir / "merged_metric_vlm_scores.csv", index=False)
    write_csv(args.output_dir / "metric_correlations.csv", correlations)
    write_csv(args.output_dir / "metric_failure_rates.csv", failures)
    write_csv(args.output_dir / "perceptual_semantic_bins.csv", bins)
    write_examples(merged[merged["metric_matched"]].copy(), args.output_dir, args.semantic_threshold)
    write_latex_tables(args.output_dir, correlations, failures, bins)

    summary = {
        "runs": {label: str(path) for label, path in configured_runs(args.run).items()},
        "metric_csv": str(args.metric_csv),
        "extra_metric_csv": str(args.extra_metric_csv),
        "semantic_threshold": args.semantic_threshold,
        "perceptual_threshold": args.perceptual_threshold,
        "merge_stats": merge_stats,
        "metric_columns": METRIC_COLUMNS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "higher_is_better": sorted(HIGHER_IS_BETTER),
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
