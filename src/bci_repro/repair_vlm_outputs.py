from __future__ import annotations

import argparse

from ._legacy import run_pipeline
from ._paths import resolve_data_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair saved VLM outputs.", add_help=False)
    parser.add_argument("--data-root", default=None)
    args, remaining = parser.parse_known_args()
    run_pipeline("repair_vlm_outputs.py", remaining, resolve_data_root(args.data_root))


if __name__ == "__main__":
    main()

