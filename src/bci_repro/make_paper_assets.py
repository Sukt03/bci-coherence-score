from __future__ import annotations

import argparse

from ._legacy import run_pipeline
from ._paths import resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper computational figures/assets.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "figures", "failure-examples", "individual-matrices"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    stages = ["figures", "failure-examples", "individual-matrices"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "figures":
            run_pipeline("make_paper_figures.py", [], data_root)
        elif stage == "failure-examples":
            run_pipeline("make_failure_pattern_examples.py", [], data_root)
        elif stage == "individual-matrices":
            run_pipeline("make_individual_vlm_matrices.py", [], data_root)


if __name__ == "__main__":
    main()

