from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ._paths import BUNDLE_ROOT, relative_to_data_root, resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify regenerated outputs against expected paper values.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--expected", default="configs/expected_values.json")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(name: str, actual: float, expected: float, tolerance: float, failures: list[str]) -> None:
    if math.isnan(actual) or abs(actual - expected) > tolerance:
        failures.append(f"{name}: actual={actual} expected={expected}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    expected_path = Path(args.expected)
    if not expected_path.is_absolute():
        cwd_candidate = Path.cwd() / expected_path
        bundle_candidate = BUNDLE_ROOT / expected_path
        expected_path = cwd_candidate if cwd_candidate.exists() else bundle_candidate
    expected = read_json(expected_path)
    failures: list[str] = []

    for label, info in expected["vlm_runs"].items():
        summary_path = relative_to_data_root(info["summary"], data_root)
        if not summary_path.exists():
            failures.append(f"missing {summary_path}")
            continue
        summary = read_json(summary_path)
        overall = summary["overall"]
        if int(overall["n"]) != expected["pair_count"]:
            failures.append(f"{label}.n: actual={overall['n']} expected={expected['pair_count']}")
        if int(overall["valid"]) != expected["pair_count"]:
            failures.append(f"{label}.valid: actual={overall['valid']} expected={expected['pair_count']}")
        check_close(f"{label}.mean_T_PAS", float(overall["mean_T_PAS"]), info["mean_T_PAS"], args.tolerance, failures)
        check_close(f"{label}.mean_T_SAS", float(overall["mean_T_SAS"]), info["mean_T_SAS"], args.tolerance, failures)

    table1_path = relative_to_data_root("paper_analysis/metric_failure_20260601/table_caption_metric_audit_full.tex", data_root)
    if not table1_path.exists():
        failures.append(f"missing {table1_path}")

    agreement_path = relative_to_data_root("vlm_eval_runs/agreement_internvl3_sail_ola_ovis/agreement_summary.json", data_root)
    if agreement_path.exists():
        agreement = read_json(agreement_path)["scope_summary"]["scored_only"]
        check_close("agreement.unanimous_rate", float(agreement["unanimous_rate"]), expected["agreement"]["unanimous_rate"], args.tolerance, failures)
        check_close("agreement.fleiss_kappa", float(agreement["fleiss_kappa"]), expected["agreement"]["fleiss_kappa"], args.tolerance, failures)
    else:
        failures.append(f"missing {agreement_path}")

    bcs_path = relative_to_data_root("distill_runs/v4teacher_paper_ready_summary_20260601.json", data_root)
    if bcs_path.exists():
        bcs = read_json(bcs_path)["4teacher_raw_ensemble"]
        check_close("bcs.raw.per_question.mae", float(bcs["per_question"]["mae"]), expected["bcs_table6"]["4teacher_continuous"]["q_mae"], args.tolerance, failures)
        check_close("bcs.raw.T_PAS.mae", float(bcs["T_PAS"]["mae"]), expected["bcs_table6"]["4teacher_continuous"]["pas_mae"], args.tolerance, failures)
        check_close("bcs.raw.T_SAS.mae", float(bcs["T_SAS"]["mae"]), expected["bcs_table6"]["4teacher_continuous"]["sas_mae"], args.tolerance, failures)
    else:
        failures.append(f"missing {bcs_path}")

    report = {"ok": not failures, "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
