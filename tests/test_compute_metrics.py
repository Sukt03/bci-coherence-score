from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from PIL import Image

from bci_repro.compute_metrics import (
    OUTPUT_COLUMNS,
    MetricRunner,
    clean_caption,
    completed_pair_ids,
    load_metric_config,
    read_pairs,
)
from bci_repro.metric_utils import LOWER_IS_BETTER, METRIC_COLUMNS


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), color).save(path)


def test_manifest_loading_accepts_csv_json_jsonl() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        csv_path = root / "pairs.csv"
        csv_path.write_text(
            "reference_path,generated_path,model,subject,class_name,rank,candidate\n"
            "gt.png,gen.png,ATM,sub-01,airboat,rank1,cand2\n",
            encoding="utf-8",
        )
        json_path = root / "pairs.json"
        payload = [
            {
                "reference_path": "gt.png",
                "generated_path": "gen.png",
                "method": "ATM",
                "subject": "sub-01",
                "concept": "airboat",
            }
        ]
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        jsonl_path = root / "pairs.jsonl"
        jsonl_path.write_text(json.dumps(payload[0]) + "\n", encoding="utf-8")

        for path in (csv_path, json_path, jsonl_path):
            pairs = read_pairs(path, root)
            assert len(pairs) == 1
            assert pairs[0].reference_path == root / "gt.png"
            assert pairs[0].generated_path == root / "gen.png"
            assert pairs[0].metadata["model"] == "ATM"
            assert pairs[0].metadata["class_name"] == "airboat"


def test_caption_cleaning_is_deterministic() -> None:
    assert clean_caption("  A Red-Blue, Boat!!! ") == "a red blue boat"
    assert clean_caption(clean_caption("A Red-Blue, Boat!!!")) == "a red blue boat"


def test_metric_output_schema_contains_paper_columns() -> None:
    for column in METRIC_COLUMNS:
        assert column in OUTPUT_COLUMNS
    assert "metric_errors" in OUTPUT_COLUMNS
    assert "expanded_metric_errors" in OUTPUT_COLUMNS


def test_metric_direction_metadata_matches_runtime() -> None:
    config = load_metric_config(Path(__file__).resolve().parents[1] / "configs" / "metrics.json")
    metrics = config["metrics"]
    for metric in METRIC_COLUMNS:
        expected = "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"
        assert metrics[metric]["direction"] == expected


def test_resume_completed_pair_ids() -> None:
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "metrics.csv"
        out.write_text("pair_id,mse\nfirst,0.1\nsecond,0.2\n", encoding="utf-8")
        assert completed_pair_ids(out) == {"first", "second"}


def test_metric_runner_low_level_smoke() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _make_image(root / "gt.png", (0, 0, 0))
        _make_image(root / "gen.png", (255, 255, 255))
        manifest = root / "pairs.csv"
        manifest.write_text(
            "reference_path,generated_path,pair_id,model,subject,class_name\n"
            "gt.png,gen.png,pair-1,ATM,sub-01,airboat\n",
            encoding="utf-8",
        )
        pair = read_pairs(manifest, root)[0]
        runner = MetricRunner(
            metric_set="fast",
            image_size=16,
            cache_dir=root / "cache",
            device="cpu",
            local_files_only=True,
            model_revisions={"models": {}},
        )
        runner.requested = {"mse", "psnr", "ssim"}
        row = runner.compute_pair(pair)
        assert row["pair_id"] == "pair-1"
        assert float(row["mse"]) == 1.0
        assert float(row["psnr"]) == 0.0
        assert "ssim" in row
