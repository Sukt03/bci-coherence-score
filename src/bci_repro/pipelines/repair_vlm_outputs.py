#!/usr/bin/env python3
"""Repair VLM result JSONL files by reparsing saved raw responses."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from internvl3_eval_pairs import (
    ImagePair,
    QUESTION_PROMPTS_BY_ROUTING,
    build_per_question_record,
    parse_label_reason_response,
    summarize_records,
    write_summary_csv,
)


DEFAULT_REPAIRS = [
    (
        Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955"),
        "internvl3_pair_scores.jsonl",
        Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired"),
    ),
    (
        Path("vlm_eval_runs/ola_full_both_reasoning_20260530_030620"),
        "pair_scores.jsonl",
        Path("vlm_eval_runs/ola_full_both_reasoning_20260530_030620_repaired"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "internvl", "ola"),
        default="all",
        help="Repair all known runs or one known run.",
    )
    parser.add_argument(
        "--require-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ola-retry-jsonl",
        type=Path,
        default=None,
        help="Optional one-row OLA retry JSONL to merge into the repaired OLA run.",
    )
    parser.add_argument(
        "--ola-retry-pair-id",
        default="ATM__sub-10__tape_recorder__rank1__cand4",
    )
    parser.add_argument(
        "--ola-retry-question-key",
        default="P6_holistic_visual_recoverability",
    )
    return parser.parse_args()


def pair_from_record(record: dict[str, Any]) -> ImagePair:
    return ImagePair(
        pair_id=record["pair_id"],
        method=record["method"],
        subject=record["subject"],
        concept=record["concept"],
        routing=record["routing"],
        rank=record["rank"],
        candidate=record["candidate"],
        reference_path=Path(record["reference_path"]),
        generated_path=Path(record["generated_path"]),
    )


def repair_record(record: dict[str, Any], require_reasoning: bool) -> dict[str, Any]:
    pair = pair_from_record(record)
    raw_responses = record.get("raw_response") or {}
    answers: dict[str, str] = {}
    reasoning: dict[str, str] = {}
    errors: list[str] = []

    for key, _question in QUESTION_PROMPTS_BY_ROUTING[pair.routing]:
        answer, reason = parse_label_reason_response(str(raw_responses.get(key, "") or ""))
        if answer is None:
            errors.append(f"missing_or_invalid_{key}")
            continue
        answers[key] = answer
        if reason:
            reasoning[key] = reason

    repaired = build_per_question_record(
        pair,
        answers,
        reasoning,
        raw_responses,
        errors,
        int(record.get("batch_size") or 1),
        float(record.get("batch_latency_s") or 0.0),
        require_reasoning,
    )
    return repaired


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_metadata(source_dir: Path, output_dir: Path) -> None:
    for name in (
        "selected_pairs.json",
        "run_config.json",
        "category_routing.json",
        "batch_probe.json",
        "chosen_batch_size.txt",
        "sanity_check.json",
    ):
        source_path = source_dir / name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / name)


def repair_run(
    source_dir: Path,
    input_name: str,
    output_dir: Path,
    require_reasoning: bool,
) -> dict[str, Any]:
    input_path = source_dir / input_name
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata(source_dir, output_dir)

    original_records = read_jsonl(input_path)
    repaired_records = [repair_record(record, require_reasoning) for record in original_records]

    output_jsonl = output_dir / input_name
    write_jsonl(repaired_records, output_jsonl)
    if input_name != "pair_scores.jsonl":
        write_jsonl(repaired_records, output_dir / "pair_scores.jsonl")

    invalid_records = [record for record in repaired_records if not record["valid"]]
    write_jsonl(invalid_records, output_dir / "invalid_pairs.jsonl")

    old_summary_path = source_dir / "summary.json"
    old_summary = (
        json.loads(old_summary_path.read_text(encoding="utf-8"))
        if old_summary_path.exists()
        else {}
    )
    summary = summarize_records(repaired_records, old_summary.get("batch_probe", []))
    for key in ("model", "output_schema"):
        if key in old_summary:
            summary[key] = old_summary[key]
    summary["repair"] = {
        "source_dir": str(source_dir),
        "source_jsonl": input_name,
        "original_rows": len(original_records),
        "original_invalid": sum(1 for record in original_records if not record.get("valid")),
        "repaired_invalid": len(invalid_records),
        "require_reasoning": require_reasoning,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_csv(summary, output_dir / "summary.csv")

    report = {
        **summary["repair"],
        "output_dir": str(output_dir),
        "output_jsonl": str(output_jsonl),
        "valid": summary["overall"]["valid"],
        "n": summary["overall"]["n"],
    }
    (output_dir / "repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def merge_ola_retry(
    retry_jsonl: Path,
    pair_id: str,
    question_key: str,
    require_reasoning: bool,
) -> dict[str, Any]:
    output_dir = DEFAULT_REPAIRS[1][2]
    repaired_jsonl = output_dir / "pair_scores.jsonl"
    if not repaired_jsonl.exists():
        raise FileNotFoundError(f"Run OLA repair before merging retry: {repaired_jsonl}")
    retry_records = read_jsonl(retry_jsonl)
    retry_record = next((record for record in retry_records if record["pair_id"] == pair_id), None)
    if retry_record is None:
        raise ValueError(f"Retry JSONL does not contain pair_id={pair_id}")
    retry_raw = retry_record.get("raw_response") or {}
    if question_key not in retry_raw:
        raise ValueError(f"Retry record does not contain raw response for {question_key}")

    records = read_jsonl(repaired_jsonl)
    merged_records = []
    replaced = False
    for record in records:
        if record["pair_id"] != pair_id:
            merged_records.append(record)
            continue
        raw = dict(record.get("raw_response") or {})
        raw[question_key] = retry_raw[question_key]
        merged_source = {**record, "raw_response": raw}
        merged_records.append(repair_record(merged_source, require_reasoning))
        replaced = True
    if not replaced:
        raise ValueError(f"Repaired OLA JSONL does not contain pair_id={pair_id}")

    write_jsonl(merged_records, repaired_jsonl)
    invalid_records = [record for record in merged_records if not record["valid"]]
    write_jsonl(invalid_records, output_dir / "invalid_pairs.jsonl")

    old_summary_path = output_dir / "summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    summary = summarize_records(merged_records, old_summary.get("batch_probe", []))
    for key in ("model", "output_schema"):
        if key in old_summary:
            summary[key] = old_summary[key]
    summary["repair"] = {
        **old_summary.get("repair", {}),
        "merged_retry_jsonl": str(retry_jsonl),
        "merged_retry_pair_id": pair_id,
        "merged_retry_question_key": question_key,
        "repaired_invalid": len(invalid_records),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_csv(summary, output_dir / "summary.csv")
    report = {
        "output_dir": str(output_dir),
        "output_jsonl": str(repaired_jsonl),
        "merged_retry_jsonl": str(retry_jsonl),
        "valid": summary["overall"]["valid"],
        "n": summary["overall"]["n"],
        "repaired_invalid": len(invalid_records),
    }
    (output_dir / "repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    selected = DEFAULT_REPAIRS
    if args.only == "internvl":
        selected = [DEFAULT_REPAIRS[0]]
    elif args.only == "ola":
        selected = [DEFAULT_REPAIRS[1]]

    reports = []
    for source_dir, input_name, output_dir in selected:
        report = repair_run(source_dir, input_name, output_dir, args.require_reasoning)
        reports.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if args.ola_retry_jsonl is not None:
        report = merge_ola_retry(
            args.ola_retry_jsonl,
            args.ola_retry_pair_id,
            args.ola_retry_question_key,
            args.require_reasoning,
        )
        reports.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if args.only == "all":
        print(json.dumps({"reports": reports}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
