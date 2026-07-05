#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/..}"
RECOMPUTE_METRICS=0
METRIC_SET="${METRIC_SET:-all}"
METRIC_MANIFEST="${METRIC_MANIFEST:-internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955/selected_pairs.json}"
METRIC_OUT="${METRIC_OUT:-${ROOT_DIR}/outputs/recomputed_metric_scores.csv}"
EXTRA_METRIC_MANIFEST="${EXTRA_METRIC_MANIFEST:-consensus_rank1_gt_generated/manifest.csv}"
EXTRA_METRIC_OUT="${EXTRA_METRIC_OUT:-${ROOT_DIR}/outputs/recomputed_extra_metric_scores.csv}"

ARGS=()
resolve_data_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$DATA_ROOT/$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recompute-metrics)
      RECOMPUTE_METRICS=1
      shift
      ;;
    --metric-set)
      METRIC_SET="$2"
      shift 2
      ;;
    --metric-manifest)
      METRIC_MANIFEST="$2"
      shift 2
      ;;
    --metric-out)
      METRIC_OUT="$2"
      shift 2
      ;;
    --extra-metric-manifest)
      EXTRA_METRIC_MANIFEST="$2"
      shift 2
      ;;
    --extra-metric-out)
      EXTRA_METRIC_OUT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

if [[ "$RECOMPUTE_METRICS" -eq 1 ]]; then
  python -m bci_repro.compute_metrics \
    --data-root "$DATA_ROOT" \
    --manifest "$METRIC_MANIFEST" \
    --out "$METRIC_OUT" \
    --metric-set "$METRIC_SET" \
    --resume
  if [[ -f "$(resolve_data_path "$EXTRA_METRIC_MANIFEST")" ]]; then
    python -m bci_repro.compute_metrics \
      --data-root "$DATA_ROOT" \
      --manifest "$EXTRA_METRIC_MANIFEST" \
      --out "$EXTRA_METRIC_OUT" \
      --metric-set "$METRIC_SET" \
      --resume
  fi
  ARGS+=("--metric-csv" "$METRIC_OUT")
  if [[ -f "$EXTRA_METRIC_OUT" ]]; then
    ARGS+=("--extra-metric-csv" "$EXTRA_METRIC_OUT")
  fi
fi

python -m bci_repro.analyze_results --stage all "${ARGS[@]}"
python -m bci_repro.make_paper_assets --stage all "${ARGS[@]}"
