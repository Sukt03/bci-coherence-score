# Lost in Visual Translation: A VLM-Assisted Perceptual-Semantic Coherence Framework for EEG-to-Image Reconstruction [IJCAI-ECAI 2026]



This paper is from authors at Mahindra University and was accepted at the
IJCAI-ECAI 2026 HBAI Workshop.

This repository provides the official reproducibility code for evaluating
EEG-to-image reconstructions with BCI-aware perceptual and semantic criteria.
It includes the VLM annotation protocol, metric audit utilities,
BCI-Coherence Score training pipeline, and verification scripts used to
recompute the reported computational results from supplied image artifacts.

## Contents

- `src/bci_repro/`: public Python package and command-line entrypoints.
- `src/bci_repro/pipelines/`: packaged implementations of the experiment
  pipelines used for VLM scoring, analysis, and BCS training.
- `prompts/questions.md`: final BCI-aware perceptual-semantic VLM annotation
  protocol and JSON response requirements.
- `configs/`: dataset, VLM, distiller, expected-value, and paper-output
  configuration files.
- `configs/metrics.json`: metric metadata, directions, output columns, and
  backend dependencies for full metric recomputation.
- `configs/model_revisions.json`: pinned model identifiers and resolved
  revisions for VLMs, encoders, captioning, SBERT, OpenCLIP, and DINOv2.
- `scripts/`: shell wrappers for smoke tests, VLM recomputation, BCS training,
  and analysis reproduction.
- `tests/`: focused unit tests for scoring, parsing, agreement, splitting, and
  metric utility logic.
- `environment.lock.yml`, `requirements.in`, `requirements-lock.txt`: locked
  environment files for repeatable metric, VLM, and analysis runs.

## Scope

This is a code-first release. Large image datasets, cached model outputs, and
trained checkpoints are treated as external artifacts so the project stays
portable. To reproduce paper-scale results, place the required artifacts next to
this folder or pass their location with `--data-root`.

## Expected Data Layout

When commands are launched from `final_code/`, the default data root is `..`.
The data root should contain:

- `metric_selected_images_only/`
- `consensus_rank1_gt_generated/manifest.csv`

For full metric recomputation from images, provide a manifest with
`reference_path`/`generated_path` columns, a selected-pairs JSON/JSONL file, or
the supplied `consensus_rank1_gt_generated/manifest.csv` layout.

## Environment

Create the recommended conda environment:

```bash
conda env create -f environment.lock.yml
conda activate hbai
python -m bci_repro.check_environment
```

For a lightweight parser and unit-test check, CPU-only Python is enough:

```bash
python -m pip install -e . pytest numpy pandas pillow matplotlib
scripts/run_smoke.sh
```

To regenerate the pip lock file after an intentional dependency change:

```bash
scripts/lock_environment.sh
```

## Reproduction Workflows

### 1. Smoke Test

```bash
scripts/run_smoke.sh
```

This compiles the package, runs unit tests, and checks that expected input
paths are available when present.

### 2. Analysis-Only Reproduction

Use this when cached VLM outputs and BCS prediction artifacts are available:

```bash
scripts/reproduce_analysis_outputs.sh --data-root ..
```

This regenerates the VLM agreement summaries, reasoning-similarity summaries,
metric-failure tables, controlled degradation probe outputs, and paper figures.

### 3. Full Metric Recompute

Use this when GT/generated image pairs are available and model weights can be
downloaded or loaded from the local Hugging Face/package caches:

```bash
python -m bci_repro.compute_metrics \
  --data-root .. \
  --manifest consensus_rank1_gt_generated/manifest.csv \
  --out final_code/outputs/recomputed_metrics.csv \
  --metric-set all \
  --resume
```

For a faster schema and preprocessing check:

```bash
python -m bci_repro.compute_metrics \
  --data-root .. \
  --manifest internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955/selected_pairs.json \
  --out final_code/outputs/smoke_metrics.csv \
  --metric-set fast \
  --limit 4 \
  --local-files-only
```

The output CSV includes all paper metric columns and per-row
`metric_errors`/`expanded_metric_errors` fields, so one unavailable backend does
not invalidate the entire run.

To recompute metrics before the standard analysis workflow:

```bash
scripts/reproduce_analysis_outputs.sh \
  --data-root .. \
  --recompute-metrics \
  --metric-manifest internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955/selected_pairs.json \
  --metric-out final_code/outputs/recomputed_metric_scores.csv \
  --extra-metric-manifest consensus_rank1_gt_generated/manifest.csv \
  --extra-metric-out final_code/outputs/recomputed_extra_metric_scores.csv
```

### 4. Full VLM Recompute

Use a CUDA machine with enough memory for the selected open-weight VLM:

```bash
scripts/run_vlm_full.sh internvl3 --data-root .. --output-dir internvl3_eval_runs/recomputed_internvl3
scripts/run_vlm_full.sh sail --data-root .. --output-dir vlm_eval_runs/recomputed_sail
scripts/run_vlm_full.sh ola --data-root .. --output-dir vlm_eval_runs/recomputed_ola
scripts/run_vlm_full.sh ovis --data-root .. --output-dir vlm_eval_runs/recomputed_ovis
```

These scripts use `prompts/questions.md` and the final VLM configuration in
`configs/vlm_models.json`.

### 5. BCS Distiller

After four VLM JSONL files are available:

```bash
scripts/run_bcs_distiller.sh --data-root ..
```

This runs the four-teacher BCI-Coherence Score distiller preset described in
`configs/bcs_distiller.json`.

## Verification

To compare regenerated outputs against expected paper values:

```bash
python -m bci_repro.verify_outputs --data-root .. --expected configs/expected_values.json
```

The verifier checks the reproducible computational values used by the project:
final pair counts, valid VLM summaries, metric-audit values, degradation-probe
values, VLM agreement/reasoning summaries, and BCS performance metrics.


