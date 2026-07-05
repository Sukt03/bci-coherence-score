#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_vlm_full.sh internvl3|sail|ola|ovis [wrapper/pipeline args]" >&2
  exit 2
fi

MODEL_KEY="$1"
shift
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m bci_repro.run_vlm --model-key "$MODEL_KEY" "$@"

