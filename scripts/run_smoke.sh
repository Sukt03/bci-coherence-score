#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/..}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m compileall src scripts
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pytest") else 1)
PY
then
  python -m pytest -q
else
  python tests/run_unit_tests.py
fi
python -m bci_repro.prepare_inputs --data-root "$DATA_ROOT" --verify-only || true

SMOKE_MANIFEST="internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955/selected_pairs.json"
SMOKE_OUT="${SMOKE_OUT:-${ROOT_DIR}/outputs/smoke_metrics.csv}"
if [[ -f "$DATA_ROOT/$SMOKE_MANIFEST" ]]; then
  python -m bci_repro.compute_metrics \
    --data-root "$DATA_ROOT" \
    --manifest "$SMOKE_MANIFEST" \
    --out "$SMOKE_OUT" \
    --metric-set fast \
    --limit 2 \
    --local-files-only
  SMOKE_OUT="$SMOKE_OUT" python - <<'PY'
import csv
import os
from pathlib import Path

required = {"pair_id", "mse", "psnr", "ssim", "openclip_cosine", "metric_errors", "expanded_metric_errors"}
path = Path(os.environ["SMOKE_OUT"])
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"smoke metric output missing columns: {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise SystemExit("smoke metric output is empty")
PY
fi

echo "Smoke checks completed."
