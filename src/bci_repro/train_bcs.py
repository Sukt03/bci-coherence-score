from __future__ import annotations

import argparse

from ._legacy import run_pipeline
from ._paths import resolve_data_root


SCRIPT_BY_MODE = {
    "train-v2": "train_vlm_score_distiller_v2.py",
    "ensemble": "ensemble_distiller_predictions.py",
    "ordinal-calibrate": "ordinal_calibrate_predictions.py",
    "stack": "stack_distiller_predictions.py",
    "select-ordinal-expert": "select_ordinal_expert_predictions.py",
    "summary": "summarize_distiller_results.py",
}


FOUR_TEACHER_ARGS = [
    "--run", "internvl3=internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl",
    "--run", "sail=vlm_eval_runs/sail_full_both_reasoning_20260530_030620/pair_scores.jsonl",
    "--run", "ola=vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired/pair_scores.jsonl",
    "--run", "ovis=vlm_eval_runs/ovis_full_both_reasoning_20260530_083349/pair_scores.jsonl",
    "--encoder-model", "google/siglip-base-patch16-224",
    "--encoder-model", "openai/clip-vit-large-patch14",
    "--encoder-model", "timm/vit_base_patch16_dinov3.lvd1689m",
    "--embedding-cache-dir", "distill_runs/embedding_cache",
    "--split-seed", "42",
    "--split-by", "concept",
    "--train-frac", "0.70",
    "--val-frac", "0.15",
    "--encoder-batch-size", "128",
    "--batch-size", "512",
    "--epochs", "120",
    "--patience", "16",
    "--hidden-dim", "512",
    "--blocks", "2",
    "--dropout", "0.2",
    "--lr", "0.0005",
    "--weight-decay", "0.0005",
    "--ce-weight", "0.35",
    "--mse-weight", "1.0",
    "--architecture", "mlp",
    "--device", "cuda:0",
    "--dtype", "bf16",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run BCI-Coherence Score distiller pipelines.", add_help=False)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--mode", choices=sorted(SCRIPT_BY_MODE), default="train-v2")
    parser.add_argument("--preset", choices=["none", "four-teacher"], default="none")
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.help:
        print("Usage: python -m bci_repro.train_bcs --mode train-v2 [pipeline args]")
        print("Use --preset four-teacher for the final paper teacher/encoder defaults.")
        raise SystemExit(0)
    return args, remaining


def main() -> None:
    args, remaining = parse_args()
    pipeline_args = [*FOUR_TEACHER_ARGS, *remaining] if args.preset == "four-teacher" else remaining
    run_pipeline(SCRIPT_BY_MODE[args.mode], pipeline_args, resolve_data_root(args.data_root))


if __name__ == "__main__":
    main()

