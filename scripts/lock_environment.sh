#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade pip==24.2 pip-tools==7.4.1
python -m piptools compile \
  --resolver=backtracking \
  --output-file requirements-lock.txt \
  requirements.in
python -m pip install -r requirements-lock.txt
python -m pip install -e .
PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" python -m bci_repro.check_environment --strict-full
