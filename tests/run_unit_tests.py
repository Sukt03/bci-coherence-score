from __future__ import annotations

from test_agreement import test_cohen_kappa_perfect, test_fleiss_kappa_runs
from test_compute_metrics import (
    test_caption_cleaning_is_deterministic,
    test_manifest_loading_accepts_csv_json_jsonl,
    test_metric_direction_metadata_matches_runtime,
    test_metric_output_schema_contains_paper_columns,
    test_metric_runner_low_level_smoke,
    test_resume_completed_pair_ids,
)
from test_metric_utils import (
    test_corr_value_spearman,
    test_oriented_quality_flips_lower_is_better,
    test_quartile_failure_rates_shape,
)
from test_scoring import (
    test_aggregate_abstract_has_no_semantic_score,
    test_aggregate_object_scores,
    test_parse_label_reason_response_from_json,
)
from test_splits import test_split_indices_is_deterministic


def main() -> None:
    tests = [
        test_parse_label_reason_response_from_json,
        test_aggregate_object_scores,
        test_aggregate_abstract_has_no_semantic_score,
        test_cohen_kappa_perfect,
        test_fleiss_kappa_runs,
        test_split_indices_is_deterministic,
        test_oriented_quality_flips_lower_is_better,
        test_corr_value_spearman,
        test_quartile_failure_rates_shape,
        test_manifest_loading_accepts_csv_json_jsonl,
        test_caption_cleaning_is_deterministic,
        test_metric_output_schema_contains_paper_columns,
        test_metric_direction_metadata_matches_runtime,
        test_resume_completed_pair_ids,
        test_metric_runner_low_level_smoke,
    ]
    for test in tests:
        test()
    print(f"Ran {len(tests)} unit tests.")


if __name__ == "__main__":
    main()
