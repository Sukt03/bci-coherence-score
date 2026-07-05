from __future__ import annotations

import argparse

from ._legacy import run_pipeline
from ._paths import resolve_data_root


FOUR_RUN_ARGS = [
    "--run", "internvl3=internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl",
    "--run", "sail=vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl",
    "--run", "ola=vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl",
    "--run", "ovis=vlm_eval_runs/ovis_full_both_reasoning_20260530_083349/pair_scores.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate analysis artifacts from cached VLM and metric outputs.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "agreement", "reasoning", "metric-failure", "degradation"],
        default="all",
    )
    parser.add_argument("--reasoning-backend", default="auto")
    parser.add_argument("--metric-csv", default="expanded_metric_scores_clean_with_sbert.csv")
    parser.add_argument("--extra-metric-csv", default="expanded_extra_model_885_clean_with_sbert(1).csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    stages = ["agreement", "reasoning", "metric-failure", "degradation"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "agreement":
            run_pipeline(
                "compare_vlm_annotation_agreement.py",
                [*FOUR_RUN_ARGS, "--output-dir", "vlm_eval_runs/agreement_internvl3_sail_ola_ovis"],
                data_root,
            )
        elif stage == "reasoning":
            run_pipeline(
                "compare_vlm_reasoning_semantics.py",
                [
                    *FOUR_RUN_ARGS,
                    "--output-dir",
                    "vlm_eval_runs/reasoning_semantics_internvl3_sail_ola_ovis",
                    "--backend",
                    args.reasoning_backend,
                ],
                data_root,
            )
        elif stage == "metric-failure":
            run_pipeline(
                "analyze_reconstruction_metrics.py",
                [
                    *FOUR_RUN_ARGS,
                    "--metric-csv",
                    args.metric_csv,
                    "--extra-metric-csv",
                    args.extra_metric_csv,
                ],
                data_root,
            )
        elif stage == "degradation":
            run_pipeline("analyze_perceptual_degradation_probe.py", [], data_root)


if __name__ == "__main__":
    main()
