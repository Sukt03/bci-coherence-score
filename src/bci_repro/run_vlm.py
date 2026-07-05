from __future__ import annotations

import argparse

from ._legacy import run_pipeline
from ._paths import resolve_data_root


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run a packaged VLM annotation pipeline.", add_help=False)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-key", choices=["internvl3", "sail", "ola", "ovis"], default=None)
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.help:
        print("Usage: python -m bci_repro.run_vlm --model-key internvl3|sail|ola|ovis [pipeline args]")
        print("Pass model-specific arguments after the wrapper options.")
        raise SystemExit(0)
    return args, remaining


def main() -> None:
    args, remaining = parse_args()
    if not args.model_key:
        raise SystemExit("--model-key is required")
    data_root = resolve_data_root(args.data_root)
    if args.model_key == "internvl3":
        run_pipeline("internvl3_eval_pairs.py", remaining, data_root)
    else:
        run_pipeline("vlm_eval_pairs.py", ["--model-key", args.model_key, *remaining], data_root)


if __name__ == "__main__":
    main()

