#!/usr/bin/env python3
"""Evaluate GT/generated image pairs with InternVL3 using questions.md scoring."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoTokenizer

from internvl3_speed_benchmark import (
    build_transform,
    dtype_from_name,
    dynamic_preprocess,
    first_model_device,
    load_model,
    synchronize,
)


ANSWER_MAP = {"yes": 1.0, "somewhat": 0.5, "no": 0.0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OBJECT_PERCEPTUAL_KEYS = [
    "P1_global_spatial_structure",
    "P2_object_shape_silhouette",
    "P3_surface_texture_material",
    "P4_color_chromatic_consistency",
    "P5_artifact_absence",
    "P6_holistic_visual_recoverability",
]
OBJECT_SEMANTIC_KEYS = [
    "S1_basic_category_identity",
    "S2_subordinate_identity",
    "S3_functional_role_purpose",
    "S4_quantity_cardinality",
    "S5_scene_context_environment",
    "S6_semantic_recoverability",
]
ABSTRACT_PERCEPTUAL_KEYS = [
    "P1_global_spatial_structure",
    "P3_surface_texture_pattern",
    "P4_color_chromatic_consistency",
    "P5_artifact_absence",
    "P6_holistic_visual_recoverability",
]
DEFAULT_ABSTRACT_CATEGORIES = {"wallpaper"}


FALLBACK_SYSTEM_PROMPT = """You are an expert evaluator for EEG-to-image reconstruction.

You will be shown two images:
  Image 1: the ground-truth reference image.
  Image 2: the EEG-generated reconstruction.

EEG-to-image outputs are often blurry, distorted, low-detail, noisy,
low-contrast, or stylistically different. Do not evaluate them as standard
photorealistic generation. Evaluate whether Image 2 preserves Image 1 under
these EEG limitations.

For every question, use exactly one answer: "yes", "somewhat", or "no".
Return only valid JSON with no explanation outside the JSON object."""

PER_QUESTION_SYSTEM_PROMPT = """You are an expert evaluator for EEG-to-image reconstruction.

You will be shown a reference image and an EEG-generated reconstruction.
EEG outputs can be blurry, low-detail, distorted, noisy, or low contrast.
Do not judge them as normal photorealistic image generation. Judge whether the
specific criterion is preserved under EEG reconstruction limitations.

For each prompt, answer with exactly one of these labels: yes, somewhat, no.
Also provide one concise reason that explains the selected label.
Return only minified JSON with exactly these fields:
{"answer":"yes|somewhat|no","reasoning":"one short sentence, max 25 words"}
No markdown, no code fences, no text outside the JSON object."""


EVAL_PROMPT = """Image 1: <image>
Image 2: <image>

Score Image 2 against Image 1 using the object-image questions from
questions.md. Use EEG tolerance: blur, low resolution, missing fine detail, mild
distortion, and approximate matches are acceptable. Penalize only the specific
failure described by each field.

Answer every field with one object containing:
  - answer: exactly one value: "yes", "somewhat", or "no"
  - reasoning: one short sentence, max 25 words

Fields:
PS1_has_semantic_content: Image 1 contains identifiable semantic content.
P1_global_spatial_structure: coarse position, size, and arrangement only.
P2_object_shape_silhouette: dominant object outline and form only.
P3_surface_texture_material: surface or material quality only.
P4_color_chromatic_consistency: dominant color palette only.
P5_artifact_absence: severe artifacts only; blur and low detail are not artifacts.
P6_holistic_visual_recoverability: overall recoverable visual content.
S1_basic_category_identity: same broad category or scene type.
S2_subordinate_identity: same specific object or type where visible.
S3_functional_role_purpose: same function or purpose.
S4_quantity_cardinality: same number of primary objects.
S5_scene_context_environment: same background or setting context.
S6_semantic_recoverability: overall recoverable intended semantic content.

Return only one minified JSON object with exactly those 13 object fields.
Do not use markdown, code fences, explanations outside JSON, or trailing commas."""


OBJECT_QUESTION_PROMPTS = [
    (
        "P1_global_spatial_structure",
        "Does Image 2 preserve the coarse spatial organization of Image 1, including approximate position, size, and arrangement of dominant regions? Judge only where things are and how space is divided. Ignore shape, color, texture, and meaning.",
    ),
    (
        "P2_object_shape_silhouette",
        "Does the dominant object or figure in Image 2 have a shape or silhouette similar to Image 1? Judge only outline, contour, and overall geometric form. Ignore color, texture, position, and meaning.",
    ),
    (
        "P3_surface_texture_material",
        "Does the surface texture and material appearance of the dominant object in Image 2 resemble Image 1? Judge only smooth vs rough, matte vs shiny, organic vs manufactured, fine-grained vs coarse. Ignore color, shape, position, and meaning.",
    ),
    (
        "P4_color_chromatic_consistency",
        "Does the dominant color palette of Image 2 reasonably match Image 1? Judge only dominant hues, approximate saturation, and broad chromatic character. Ignore texture, shape, position, and meaning.",
    ),
    (
        "P5_artifact_absence",
        "Is Image 2 free from severe visual artifacts that dominate or overwhelm the content? Judge only structured noise, grid patterns, color bleeding, repetitive hallucinated patterns, or generation failures. Do not penalize blur or low detail.",
    ),
    (
        "P6_holistic_visual_recoverability",
        "Taking Image 2 as a whole, can a human observer still recover the primary visual content of Image 1 despite EEG reconstruction degradation? Judge the combined evidence from structure, shape, texture, and color.",
    ),
    (
        "S1_basic_category_identity",
        "Does Image 2 depict an object or scene belonging to the same broad basic-level category as Image 1, such as animal, vehicle, food, furniture, tool, clothing, building, plant, person, household object, electronic device, natural scene, or indoor scene?",
    ),
    (
        "S2_subordinate_identity",
        "Does Image 2 depict the same specific type of object shown in Image 1, beyond just the broad category, such as dog vs cat, car vs truck, apple vs banana, or similar subordinate identity?",
    ),
    (
        "S3_functional_role_purpose",
        "Does the dominant object in Image 2 serve the same functional role or purpose as in Image 1? Judge what the object is for or does, not its appearance, count, or scene context.",
    ),
    (
        "S4_quantity_cardinality",
        "Does Image 2 show approximately the same number of primary objects as Image 1? Judge only quantity of the main object or objects. Ignore background and secondary decorative elements.",
    ),
    (
        "S5_scene_context_environment",
        "Does Image 2 imply the same scene context or environment as Image 1, independent of the main object? Judge setting such as indoor vs outdoor, natural vs urban, water vs land, domestic vs wild, or aerial vs ground level.",
    ),
    (
        "S6_semantic_recoverability",
        "Taking Image 2 as a whole, can the intended semantic content of Image 1 be inferred despite EEG reconstruction noise? Judge the combined evidence from category, identity, function, quantity, and scene.",
    ),
]
ABSTRACT_QUESTION_PROMPTS = [
    (
        "P1_global_spatial_structure",
        "Does Image 2 preserve the coarse spatial organization of Image 1, including the approximate arrangement, distribution, and balance of dominant visual regions? For abstract images, judge only spatial distribution of pattern or visual mass. Ignore texture, color, and meaning.",
    ),
    (
        "P3_surface_texture_pattern",
        "Does the dominant texture, pattern, or surface appearance of Image 2 resemble Image 1? For abstract images, texture or pattern is the primary content. Judge organic vs geometric, fine vs coarse, regular vs irregular, smooth vs rough, dense vs sparse, directional vs random.",
    ),
    (
        "P4_color_chromatic_consistency",
        "Does the dominant color palette of Image 2 reasonably match Image 1? Judge only dominant hues, approximate saturation, and broad chromatic character. Ignore texture, pattern structure, and spatial layout.",
    ),
    (
        "P5_artifact_absence",
        "Is Image 2 free from severe visual artifacts that dominate or overwhelm the content? Judge only structured noise, grid patterns, color bleeding, repetitive hallucinated patterns unrelated to Image 1, or generation failures. Do not penalize blur or low detail.",
    ),
    (
        "P6_holistic_visual_recoverability",
        "Taking Image 2 as a whole, can a human observer still recover the primary visual pattern or structure of Image 1 despite EEG reconstruction degradation? Judge the combined evidence from spatial structure, texture character, and color.",
    ),
]
QUESTION_PROMPTS_BY_ROUTING = {
    "object": OBJECT_QUESTION_PROMPTS,
    "abstract": ABSTRACT_QUESTION_PROMPTS,
}


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    method: str
    subject: str
    concept: str
    routing: str
    rank: str
    candidate: str
    reference_path: Path
    generated_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score sampled GT/generated pairs with InternVL3."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("metric_selected_images_only"))
    parser.add_argument(
        "--dataset-source",
        choices=("base", "extra", "both"),
        default="base",
        help="base reads metric_selected_images_only; extra reads consensus_rank1_gt_generated/manifest.csv; both combines them.",
    )
    parser.add_argument(
        "--extra-manifest",
        type=Path,
        default=Path("consensus_rank1_gt_generated/manifest.csv"),
        help="Manifest produced from extra_consensus_rank1_gt_generated.zip.",
    )
    parser.add_argument(
        "--extra-methods",
        nargs="+",
        default=["all"],
        help='Extra detected_model values to include, or "all".',
    )
    parser.add_argument(
        "--extra-subject",
        default="extra",
        help="Subject label assigned to extra-model rows.",
    )
    parser.add_argument(
        "--extra-routing",
        choices=("object", "abstract", "category"),
        default="object",
        help="Routing for extra-model rows. category uses --abstract-categories; object treats all extra rows as object images.",
    )
    parser.add_argument("--questions", type=Path, default=Path("questions.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("internvl3_eval_runs/latest"))
    parser.add_argument("--model", default="OpenGVLab/InternVL3-8B")
    parser.add_argument("--methods", nargs="+", default=["ATM", "ENIGMA"])
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["all"],
        help='Concept folder names to include, or "all". Useful for smoke tests.',
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-01"],
        help='Subject IDs to include, or "all". Default keeps the 100-pair run diverse.',
    )
    parser.add_argument(
        "--rank",
        default="rank1",
        help='Generated rank to include, e.g. rank1, rank2, or "all".',
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=29)
    parser.add_argument(
        "--sample-strategy",
        choices=("balanced", "first"),
        default="balanced",
        help="balanced alternates methods after shuffling each method group.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("per-question", "combined-json"),
        default="per-question",
        help="per-question is more reliable for InternVL3; combined-json is faster but brittle.",
    )
    parser.add_argument(
        "--abstract-categories",
        nargs="*",
        default=sorted(DEFAULT_ABSTRACT_CATEGORIES),
        help="Concept folder names routed to the abstract pipeline. All other categories are object.",
    )
    parser.add_argument(
        "--strict-pairing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a generated file cannot be mapped exactly to a shared GT category.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Candidate batch sizes to probe. Ignored when --batch-size is set.",
    )
    parser.add_argument(
        "--batch-selection",
        choices=("largest-valid", "throughput"),
        default="largest-valid",
        help="largest-valid uses the biggest successful probe to fill the GPU; throughput uses fastest observed pairs/sec.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--require-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a non-empty short reasoning string for every per-question answer.",
    )
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-batch-probe",
        action="store_true",
        help="Use --batch-size or 1 directly without probing candidates.",
    )
    return parser.parse_args()


def load_system_prompt(questions_path: Path) -> str:
    if not questions_path.exists():
        return FALLBACK_SYSTEM_PROMPT
    text = questions_path.read_text(encoding="utf-8")
    marker = "## System Prompt"
    marker_index = text.find(marker)
    if marker_index < 0:
        return FALLBACK_SYSTEM_PROMPT
    fenced = re.search(r"```(?:\w+)?\n(.*?)\n```", text[marker_index:], re.DOTALL)
    if not fenced:
        return FALLBACK_SYSTEM_PROMPT
    return fenced.group(1).strip()


def parse_rank_and_candidate(path: Path) -> tuple[str, str]:
    rank_match = re.search(r"__rank(\d+)__", path.name)
    cand_match = re.search(r"__cand(\d+)", path.stem)
    rank = f"rank{rank_match.group(1)}" if rank_match else "rank_unknown"
    candidate = f"cand{cand_match.group(1)}" if cand_match else "cand_unknown"
    return rank, candidate


def routing_for_concept(concept: str, abstract_categories: set[str]) -> str:
    return "abstract" if concept in abstract_categories else "object"


def parse_extra_candidate(filename: str) -> str:
    stem = Path(filename).stem
    match = re.search(r"(?:_|-)(\d+)$", stem)
    return f"cand{match.group(1)}" if match else "cand_unknown"


def resolve_manifest_image_path(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    manifest_parent_candidate = manifest_path.parent / path.name
    if manifest_parent_candidate.exists():
        return manifest_parent_candidate
    repo_relative_candidate = Path(path_text)
    if repo_relative_candidate.exists():
        return repo_relative_candidate
    return manifest_parent_candidate


def collect_base_pairs(args: argparse.Namespace) -> list[ImagePair]:
    return collect_pairs(args)


def collect_extra_pairs(args: argparse.Namespace) -> list[ImagePair]:
    if not args.extra_manifest.exists():
        raise FileNotFoundError(f"Missing extra manifest: {args.extra_manifest}")

    concept_filter = None if args.concepts == ["all"] else set(args.concepts)
    method_filter = None
    if args.extra_methods != ["all"]:
        method_filter = {method.lower() for method in args.extra_methods}
    rank_filter = None if args.rank == "all" else args.rank
    abstract_categories = set(args.abstract_categories)

    pairs: list[ImagePair] = []
    missing_images: list[str] = []
    seen: set[str] = set()
    with args.extra_manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "zip_row_index",
            "detected_model",
            "class_name_or_stem",
            "consensus_rank",
            "filename",
            "gt_zip_path",
            "generated_zip_path",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Extra manifest missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            method = row["detected_model"].strip()
            concept = row["class_name_or_stem"].strip()
            if method_filter is not None and method.lower() not in method_filter:
                continue
            if concept_filter is not None and concept not in concept_filter:
                continue
            rank = f"rank{row['consensus_rank'].strip() or 'unknown'}"
            if rank_filter is not None and rank != rank_filter:
                continue
            candidate = parse_extra_candidate(row["filename"])
            if args.extra_routing == "category":
                routing = routing_for_concept(concept, abstract_categories)
            else:
                routing = args.extra_routing

            reference_path = resolve_manifest_image_path(args.extra_manifest, row["gt_zip_path"])
            generated_path = resolve_manifest_image_path(args.extra_manifest, row["generated_zip_path"])
            if args.strict_pairing:
                if not reference_path.exists():
                    missing_images.append(str(reference_path))
                    continue
                if not generated_path.exists():
                    missing_images.append(str(generated_path))
                    continue

            row_index = row["zip_row_index"].strip() or "unknown"
            pair_id = f"{method}__{args.extra_subject}__{concept}__{rank}__{candidate}__row{row_index}"
            if pair_id in seen:
                raise ValueError(f"Duplicate extra pair_id: {pair_id}")
            seen.add(pair_id)
            pairs.append(
                ImagePair(
                    pair_id=pair_id,
                    method=method,
                    subject=args.extra_subject,
                    concept=concept,
                    routing=routing,
                    rank=rank,
                    candidate=candidate,
                    reference_path=reference_path,
                    generated_path=generated_path,
                )
            )

    if args.strict_pairing and missing_images:
        raise FileNotFoundError(
            "Missing extra GT/generated image(s): "
            + ", ".join(sorted(set(missing_images))[:20])
        )
    return pairs


def collect_requested_pairs(args: argparse.Namespace) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    if args.dataset_source in {"base", "both"}:
        pairs.extend(collect_base_pairs(args))
    if args.dataset_source in {"extra", "both"}:
        pairs.extend(collect_extra_pairs(args))

    seen: set[str] = set()
    duplicates: list[str] = []
    for pair in pairs:
        if pair.pair_id in seen:
            duplicates.append(pair.pair_id)
        seen.add(pair.pair_id)
    if args.strict_pairing and duplicates:
        raise ValueError("Duplicate pair_id(s): " + ", ".join(sorted(set(duplicates))[:20]))
    return pairs


def collect_pairs(args: argparse.Namespace) -> list[ImagePair]:
    gt_dir = args.data_dir / "shared_gt"
    generated_root = args.data_dir / "metric_selected_generated"
    methods = {method.upper() for method in args.methods}
    concept_filter = None if args.concepts == ["all"] else set(args.concepts)
    subject_filter = None if args.subjects == ["all"] else set(args.subjects)
    rank_filter = None if args.rank == "all" else args.rank
    abstract_categories = set(args.abstract_categories)

    pairs: list[ImagePair] = []
    missing_references: list[str] = []
    malformed_files: list[str] = []
    for method_dir in sorted(generated_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name.upper() not in methods:
            continue
        method = method_dir.name.upper()
        for subject_dir in sorted(method_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            if subject_filter is not None and subject_dir.name not in subject_filter:
                continue
            for concept_dir in sorted(subject_dir.iterdir()):
                if not concept_dir.is_dir():
                    continue
                concept = concept_dir.name
                if concept_filter is not None and concept not in concept_filter:
                    continue
                reference_path = gt_dir / f"{concept}__reference.jpg"
                if not reference_path.exists():
                    missing_references.append(str(reference_path))
                    continue
                for generated_path in sorted(concept_dir.iterdir()):
                    if generated_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if args.strict_pairing and not generated_path.name.startswith(f"{concept}__"):
                        malformed_files.append(str(generated_path))
                        continue
                    rank, candidate = parse_rank_and_candidate(generated_path)
                    if rank_filter is not None and rank != rank_filter:
                        continue
                    pair_id = f"{method}__{subject_dir.name}__{concept}__{rank}__{candidate}"
                    pairs.append(
                        ImagePair(
                            pair_id=pair_id,
                            method=method,
                            subject=subject_dir.name,
                            concept=concept,
                            routing=routing_for_concept(concept, abstract_categories),
                            rank=rank,
                            candidate=candidate,
                            reference_path=reference_path,
                            generated_path=generated_path,
                        )
                    )
    if args.strict_pairing and missing_references:
        raise FileNotFoundError(
            "Missing shared GT reference(s): " + ", ".join(sorted(set(missing_references))[:20])
        )
    if args.strict_pairing and malformed_files:
        raise ValueError(
            "Generated file(s) do not match their concept folder: "
            + ", ".join(sorted(set(malformed_files))[:20])
        )
    seen: set[str] = set()
    duplicates: list[str] = []
    for pair in pairs:
        if pair.pair_id in seen:
            duplicates.append(pair.pair_id)
        seen.add(pair.pair_id)
    if args.strict_pairing and duplicates:
        raise ValueError("Duplicate pair_id(s): " + ", ".join(sorted(set(duplicates))[:20]))
    return pairs


def apply_limit(
    pairs: list[ImagePair],
    limit: int | None,
    seed: int,
    strategy: str,
) -> list[ImagePair]:
    if limit is None or limit <= 0 or limit >= len(pairs):
        return pairs
    if strategy == "first":
        return pairs[:limit]

    rng = random.Random(seed)
    grouped: dict[str, list[ImagePair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.method, []).append(pair)
    for group in grouped.values():
        rng.shuffle(group)

    selected: list[ImagePair] = []
    method_names = sorted(grouped)
    while len(selected) < limit and any(grouped.values()):
        for method in method_names:
            if grouped[method]:
                selected.append(grouped[method].pop())
                if len(selected) >= limit:
                    break
    return selected


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = []
    for item in value.split(","):
        item = item.strip()
        if item:
            batch_sizes.append(int(item))
    return sorted({size for size in batch_sizes if size > 0})


def patch_transformers_quantizer_compat() -> None:
    """Handle older InternVL remote code with newer bitsandbytes quantizer logic."""
    try:
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

            def all_tied_weights_keys(self: torch.nn.Module) -> dict[str, str]:
                stored = getattr(self, "_compat_all_tied_weights_keys", None)
                if stored is not None:
                    return stored
                tied = getattr(self, "_tied_weights_keys", None) or {}
                if isinstance(tied, dict):
                    return tied
                return {key: key for key in tied}

            def set_all_tied_weights_keys(
                self: torch.nn.Module, value: dict[str, str] | list[str] | tuple[str, ...]
            ) -> None:
                if isinstance(value, dict):
                    self._compat_all_tied_weights_keys = value
                else:
                    self._compat_all_tied_weights_keys = {key: key for key in value}

            PreTrainedModel.all_tied_weights_keys = property(
                all_tied_weights_keys, set_all_tied_weights_keys
            )
    except Exception:
        pass

    try:
        import transformers.quantizers.base as quantizer_base
    except Exception:
        return

    original = quantizer_base.get_keys_to_not_convert
    if getattr(original, "_internvl_compat_patched", False):
        return

    def compat_get_keys_to_not_convert(model: torch.nn.Module) -> list[str]:
        if not hasattr(model, "all_tied_weights_keys"):
            setattr(model, "all_tied_weights_keys", {})
        return original(model)

    setattr(compat_get_keys_to_not_convert, "_internvl_compat_patched", True)
    quantizer_base.get_keys_to_not_convert = compat_get_keys_to_not_convert


def load_image_tiles(path: Path, max_tiles: int, transform: Any) -> torch.Tensor:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
    tiles = dynamic_preprocess(
        rgb_image,
        image_size=448,
        use_thumbnail=True,
        max_num=max_tiles,
    )
    return torch.stack([transform(tile) for tile in tiles])


def image_tokens(model: torch.nn.Module, num_patches: int) -> str:
    img_context_token = "<IMG_CONTEXT>"
    return "<img>" + img_context_token * model.num_image_token * num_patches + "</img>"


def build_query(
    model: torch.nn.Module,
    prompt: str,
    system_prompt: str,
    num_patches_list: list[int],
) -> str:
    if prompt.count("<image>") != len(num_patches_list):
        raise ValueError(
            f"Prompt has {prompt.count('<image>')} image placeholders, "
            f"but got {len(num_patches_list)} images."
        )

    template = copy.deepcopy(model.conv_template)
    template.system_message = system_prompt
    template.append_message(template.roles[0], prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for num_patches in num_patches_list:
        query = query.replace("<image>", image_tokens(model, num_patches), 1)
    return query


def build_batch_inputs(
    pairs: list[ImagePair],
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    max_tiles: int,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id

    queries = []
    pixel_batches = []
    for pair in pairs:
        reference_pixels = load_image_tiles(pair.reference_path, max_tiles, transform)
        generated_pixels = load_image_tiles(pair.generated_path, max_tiles, transform)
        num_patches_list = [reference_pixels.shape[0], generated_pixels.shape[0]]
        queries.append(build_query(model, prompt, system_prompt, num_patches_list))
        pixel_batches.append(torch.cat([reference_pixels, generated_pixels], dim=0))

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    pixel_values = torch.cat(pixel_batches, dim=0).to(dtype=dtype, device=device)
    return pixel_values, input_ids, attention_mask


def clean_response(text: str, sep: str) -> str:
    text = text.strip()
    for marker in (sep, "<|im_end|>", "<|endoftext|>"):
        if marker and marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def generate_batch(
    pairs: list[ImagePair],
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    generation_config: dict[str, Any],
    max_tiles: int,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
) -> list[str]:
    pixel_values, input_ids, attention_mask = build_batch_inputs(
        pairs,
        model,
        tokenizer,
        prompt,
        system_prompt,
        max_tiles,
        dtype,
        device,
        transform,
    )
    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
    if output_ids.shape[1] > input_ids.shape[1]:
        output_ids = output_ids[:, input_ids.shape[1] :]
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    sep = model.conv_template.sep.strip()
    return [clean_response(response, sep) for response in responses]


def is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
        return parsed, None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "no JSON object found"
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed, None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def normalize_answer(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("answer")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ANSWER_MAP:
        return normalized
    for answer in ANSWER_MAP:
        if re.fullmatch(rf".*\b{answer}\b.*", normalized):
            return answer
    return None


def clean_reasoning(value: Any) -> str | None:
    if isinstance(value, dict):
        direct_value = value.get("reasoning") or value.get("reason") or value.get("rationale")
        if direct_value is not None:
            value = direct_value
        else:
            for key, candidate in value.items():
                normalized_key = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized_key in {"reasoning", "reason", "rationale"}:
                    value = candidate
                    break
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = cleaned.rstrip(")")
    if not cleaned:
        return None
    return cleaned[:300]


def normalize_question_item(value: Any) -> dict[str, str | None] | None:
    answer = normalize_answer(value)
    if answer is None:
        return None
    item: dict[str, str | None] = {"answer": answer}
    reasoning = clean_reasoning(value)
    if reasoning:
        item["reasoning"] = reasoning
    return item


def normalize_annotation(parsed: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    if parsed is None:
        return None, ["parse_failed"]

    if "perceptual" not in parsed and any(key in parsed for key in OBJECT_PERCEPTUAL_KEYS):
        errors: list[str] = []
        perceptual = {}
        semantic = {}
        for key in OBJECT_PERCEPTUAL_KEYS:
            item = normalize_question_item(parsed.get(key))
            if item is None:
                errors.append(f"missing_or_invalid_{key}")
            else:
                perceptual[key] = item
        for key in OBJECT_SEMANTIC_KEYS:
            item = normalize_question_item(parsed.get(key))
            if item is None:
                errors.append(f"missing_or_invalid_{key}")
            else:
                semantic[key] = item
        normalized = {
            "routing": "naturalistic",
            "PS1_has_semantic_content": normalize_answer(
                parsed.get("PS1_has_semantic_content")
            )
            or "yes",
            "perceptual": perceptual,
            "semantic": semantic,
        }
        return normalized, errors

    errors: list[str] = []
    ps1 = normalize_answer(parsed.get("PS1_has_semantic_content"))
    routing = parsed.get("routing")
    if isinstance(routing, str):
        routing = routing.strip().lower()
    if routing in {"object", "naturalistic", "semantic"}:
        routing = "naturalistic"
    elif routing in {"abstract", "texture"}:
        routing = "abstract"
    elif ps1 == "yes":
        routing = "naturalistic"
    elif ps1 == "no":
        routing = "abstract"
    else:
        routing = "naturalistic"
        errors.append("missing_or_unknown_routing")

    expected_perceptual = (
        ABSTRACT_PERCEPTUAL_KEYS if routing == "abstract" else OBJECT_PERCEPTUAL_KEYS
    )
    expected_semantic = [] if routing == "abstract" else OBJECT_SEMANTIC_KEYS

    perceptual_raw = parsed.get("perceptual")
    if not isinstance(perceptual_raw, dict):
        perceptual_raw = {}
        errors.append("missing_perceptual")
    semantic_raw = parsed.get("semantic")
    if routing == "naturalistic" and not isinstance(semantic_raw, dict):
        semantic_raw = {}
        errors.append("missing_semantic")

    perceptual = {}
    for key in expected_perceptual:
        item = normalize_question_item(perceptual_raw.get(key))
        if item is None:
            errors.append(f"missing_or_invalid_{key}")
        else:
            perceptual[key] = item

    semantic = None
    if routing == "naturalistic":
        semantic = {}
        for key in expected_semantic:
            item = normalize_question_item(semantic_raw.get(key))
            if item is None:
                errors.append(f"missing_or_invalid_{key}")
            else:
                semantic[key] = item

    normalized = {
        "routing": routing,
        "PS1_has_semantic_content": "no" if routing == "abstract" else "yes",
        "perceptual": perceptual,
        "semantic": semantic,
    }
    return normalized, errors


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compute_scores(annotation: dict[str, Any] | None) -> dict[str, float | None]:
    if annotation is None:
        return {"T_PAS": None, "T_SAS": None}
    perceptual_values = [
        ANSWER_MAP[v["answer"]]
        for v in annotation["perceptual"].values()
        if isinstance(v, dict) and v.get("answer") in ANSWER_MAP
    ]
    semantic_values = []
    if isinstance(annotation.get("semantic"), dict):
        semantic_values = [
            ANSWER_MAP[v["answer"]]
            for v in annotation["semantic"].values()
            if isinstance(v, dict) and v.get("answer") in ANSWER_MAP
        ]
    return {
        "T_PAS": mean_or_none(perceptual_values),
        "T_SAS": mean_or_none(semantic_values),
    }


def make_record(
    pair: ImagePair,
    response: str,
    batch_size: int,
    batch_latency_s: float,
) -> dict[str, Any]:
    parsed, parse_error = extract_json(response)
    normalized, normalization_errors = normalize_annotation(parsed)
    scores = compute_scores(normalized)
    expected_count = (
        len(ABSTRACT_PERCEPTUAL_KEYS)
        if normalized and normalized["routing"] == "abstract"
        else len(OBJECT_PERCEPTUAL_KEYS) + len(OBJECT_SEMANTIC_KEYS)
    )
    valid_answer_count = 0
    if normalized:
        valid_answer_count += len(normalized["perceptual"])
        if isinstance(normalized.get("semantic"), dict):
            valid_answer_count += len(normalized["semantic"])

    return {
        **asdict(pair),
        "reference_path": str(pair.reference_path),
        "generated_path": str(pair.generated_path),
        "raw_response": response,
        "parsed_response": parsed,
        "parse_error": parse_error,
        "normalization_errors": normalization_errors,
        "valid": parsed is not None and not normalization_errors and valid_answer_count == expected_count,
        "normalized_response": normalized,
        "scores": scores,
        "batch_size": batch_size,
        "batch_latency_s": batch_latency_s,
        "estimated_item_latency_s": batch_latency_s / max(batch_size, 1),
    }


def question_prompt(question: str) -> str:
    return (
        "Image 1: <image>\n"
        "Image 2: <image>\n\n"
        f"{question}\n\n"
        'Return only this JSON schema: {"answer":"yes|somewhat|no","reasoning":"one short sentence, max 25 words"}'
    )


def parse_label_response(response: str) -> str | None:
    cleaned = response.strip().lower()
    cleaned = cleaned.replace("```", " ").replace('"', " ").replace("'", " ")
    match = re.search(r"\b(yes|somewhat|no)\b", cleaned)
    if not match:
        return None
    return match.group(1)


def parse_label_reason_response(response: str) -> tuple[str | None, str | None]:
    parsed, _ = extract_json(response)
    if isinstance(parsed, dict):
        answer = normalize_answer(parsed.get("answer"))
        reasoning = clean_reasoning(parsed)
        return answer, reasoning

    answer = parse_label_response(response)
    reasoning = None
    reasoning_match = re.search(
        r'"[^"A-Za-z]*(?:reasoning|reason|rationale)[^"A-Za-z]*"\s*:\s*"([^"}]+)',
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = clean_reasoning(reasoning_match.group(1))
    because_match = re.search(r"\b(?:because|reason(?:ing)?[:\s-]+)(.+)$", response, re.IGNORECASE | re.DOTALL)
    if reasoning is None and because_match:
        reasoning = clean_reasoning(because_match.group(1))
    return answer, reasoning


def build_per_question_record(
    pair: ImagePair,
    answers: dict[str, str],
    reasoning: dict[str, str],
    raw_responses: dict[str, str],
    errors: list[str],
    batch_size: int,
    batch_latency_s: float,
    require_reasoning: bool,
) -> dict[str, Any]:
    if pair.routing == "abstract":
        expected_perceptual = ABSTRACT_PERCEPTUAL_KEYS
        expected_semantic: list[str] = []
        ps1 = "no"
    else:
        expected_perceptual = OBJECT_PERCEPTUAL_KEYS
        expected_semantic = OBJECT_SEMANTIC_KEYS
        ps1 = "yes"

    perceptual = {
        key: {
            "answer": answers[key],
            **({"reasoning": reasoning[key]} if key in reasoning else {}),
        }
        for key in expected_perceptual
        if key in answers
    }
    semantic = (
        {
            key: {
                "answer": answers[key],
                **({"reasoning": reasoning[key]} if key in reasoning else {}),
            }
            for key in expected_semantic
            if key in answers
        }
        if expected_semantic
        else None
    )
    normalized = {
        "routing": pair.routing,
        "PS1_has_semantic_content": ps1,
        "perceptual": perceptual,
        "semantic": semantic,
    }
    expected_keys = expected_perceptual + expected_semantic
    expected_count = len(expected_keys)
    valid_answer_count = len(perceptual) + (len(semantic) if isinstance(semantic, dict) else 0)
    if valid_answer_count != expected_count:
        missing = sorted(set(expected_keys) - set(answers))
        errors.extend(f"missing_or_invalid_{key}" for key in missing)
    if require_reasoning:
        missing_reasoning = sorted(set(expected_keys) - set(reasoning))
        errors.extend(f"missing_reasoning_{key}" for key in missing_reasoning if key in answers)

    return {
        **asdict(pair),
        "reference_path": str(pair.reference_path),
        "generated_path": str(pair.generated_path),
        "raw_response": raw_responses,
        "parsed_response": {
            key: {
                "answer": answer,
                **({"reasoning": reasoning[key]} if key in reasoning else {}),
            }
            for key, answer in answers.items()
        },
        "parse_error": None if not errors else ";".join(errors),
        "normalization_errors": errors,
        "valid": not errors and valid_answer_count == expected_count,
        "normalized_response": normalized,
        "scores": compute_scores(normalized),
        "batch_size": batch_size,
        "batch_latency_s": batch_latency_s,
        "estimated_item_latency_s": batch_latency_s / max(batch_size, 1),
    }


def run_per_question_batch(
    pairs: list[ImagePair],
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    generation_config: dict[str, Any],
    max_tiles: int,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
    require_reasoning: bool,
) -> tuple[list[dict[str, Any]], float]:
    answers_by_pair = {pair.pair_id: {} for pair in pairs}
    reasoning_by_pair = {pair.pair_id: {} for pair in pairs}
    raw_by_pair = {pair.pair_id: {} for pair in pairs}
    errors_by_pair = {pair.pair_id: [] for pair in pairs}
    total_elapsed = 0.0

    for routing, prompts in QUESTION_PROMPTS_BY_ROUTING.items():
        routed_pairs = [pair for pair in pairs if pair.routing == routing]
        if not routed_pairs:
            continue
        for key, question in prompts:
            started = time.perf_counter()
            responses = generate_batch(
                routed_pairs,
                model,
                tokenizer,
                question_prompt(question),
                system_prompt,
                generation_config,
                max_tiles,
                dtype,
                device,
                transform,
            )
            synchronize()
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed

            for pair, response in zip(routed_pairs, responses):
                raw_by_pair[pair.pair_id][key] = response
                answer, reasoning = parse_label_reason_response(response)
                if answer is None:
                    errors_by_pair[pair.pair_id].append(f"missing_or_invalid_{key}")
                else:
                    answers_by_pair[pair.pair_id][key] = answer
                    if reasoning:
                        reasoning_by_pair[pair.pair_id][key] = reasoning

    records = [
        build_per_question_record(
            pair,
            answers_by_pair[pair.pair_id],
            reasoning_by_pair[pair.pair_id],
            raw_by_pair[pair.pair_id],
            errors_by_pair[pair.pair_id],
            len(pairs),
            total_elapsed,
            require_reasoning,
        )
        for pair in pairs
    ]
    return records, total_elapsed


def numeric_mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return statistics.mean(valid)


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    t_pas_values = [record["scores"]["T_PAS"] for record in records]
    t_sas_values = [record["scores"]["T_SAS"] for record in records]
    valid_count = sum(1 for record in records if record["valid"])
    return {
        "n": len(records),
        "valid": valid_count,
        "valid_rate": valid_count / len(records) if records else None,
        "mean_T_PAS": numeric_mean(t_pas_values),
        "mean_T_SAS": numeric_mean(t_sas_values),
    }


def summarize_records(records: list[dict[str, Any]], batch_probe: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    for method in sorted({record["method"] for record in records}):
        by_method[method] = summarize_group(
            [record for record in records if record["method"] == method]
        )

    by_routing = {}
    for routing in sorted({record["routing"] for record in records}):
        by_routing[routing] = summarize_group(
            [record for record in records if record["routing"] == routing]
        )

    by_method_routing = {}
    for method, routing in sorted({(record["method"], record["routing"]) for record in records}):
        key = f"{method}/{routing}"
        by_method_routing[key] = summarize_group(
            [
                record
                for record in records
                if record["method"] == method and record["routing"] == routing
            ]
        )

    by_method_subject = {}
    for method, subject in sorted({(record["method"], record["subject"]) for record in records}):
        key = f"{method}/{subject}"
        by_method_subject[key] = summarize_group(
            [
                record
                for record in records
                if record["method"] == method and record["subject"] == subject
            ]
        )

    per_question = {}
    for key in OBJECT_PERCEPTUAL_KEYS + ABSTRACT_PERCEPTUAL_KEYS + OBJECT_SEMANTIC_KEYS:
        values = []
        counts: Counter[str] = Counter()
        for record in records:
            normalized = record.get("normalized_response") or {}
            perceptual = normalized.get("perceptual") or {}
            semantic = normalized.get("semantic") or {}
            item = perceptual.get(key) or semantic.get(key)
            if isinstance(item, dict) and item.get("answer") in ANSWER_MAP:
                answer = item["answer"]
                counts[answer] += 1
                values.append(ANSWER_MAP[answer])
        if values:
            per_question[key] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "counts": dict(counts),
            }

    category_routing = {
        concept: routing
        for concept, routing in sorted({(record["concept"], record["routing"]) for record in records})
    }

    return {
        "overall": summarize_group(records),
        "by_method": by_method,
        "by_routing": by_routing,
        "by_method_routing": by_method_routing,
        "by_method_subject": by_method_subject,
        "per_question": per_question,
        "category_routing": category_routing,
        "batch_probe": batch_probe,
    }


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [{"group": "overall", **summary["overall"]}]
    for method, stats in summary["by_method"].items():
        rows.append({"group": f"method:{method}", **stats})
    for routing, stats in summary["by_routing"].items():
        rows.append({"group": f"routing:{routing}", **stats})
    for key, stats in summary["by_method_routing"].items():
        rows.append({"group": f"method_routing:{key}", **stats})
    for key, stats in summary["by_method_subject"].items():
        rows.append({"group": f"method_subject:{key}", **stats})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "n", "valid", "valid_rate", "mean_T_PAS", "mean_T_SAS"],
        )
        writer.writeheader()
        writer.writerows(rows)


def probe_batch_sizes(
    batch_sizes: list[int],
    pairs: list[ImagePair],
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
) -> tuple[int, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size > len(pairs):
            continue
        probe_pairs = pairs[:batch_size]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            responses = generate_batch(
                probe_pairs,
                model,
                tokenizer,
                EVAL_PROMPT,
                system_prompt,
                generation_config,
                args.max_tiles,
                dtype,
                device,
                transform,
                args.require_reasoning,
            )
            synchronize()
            elapsed = time.perf_counter() - started
            parsed_ok = 0
            for response in responses:
                parsed, _ = extract_json(response)
                normalized, errors = normalize_annotation(parsed)
                if normalized is not None and not errors:
                    parsed_ok += 1
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            result = {
                "batch_size": batch_size,
                "ok": True,
                "elapsed_s": elapsed,
                "items_per_s": batch_size / elapsed if elapsed > 0 else None,
                "peak_cuda_allocated_gb": peak_gb,
                "parsed_ok": parsed_ok,
            }
            print(
                f"batch_probe bs={batch_size} ok elapsed={elapsed:.2f}s "
                f"items/s={result['items_per_s']:.3f} peak={peak_gb:.2f}GB "
                f"parsed={parsed_ok}/{batch_size}",
                flush=True,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            torch.cuda.empty_cache()
            result = {
                "batch_size": batch_size,
                "ok": False,
                "error": "cuda_out_of_memory",
            }
            print(f"batch_probe bs={batch_size} OOM; skipping larger unsafe use", flush=True)
        results.append(result)
        if not result.get("ok") and result.get("error") == "cuda_out_of_memory":
            break

    successful = [result for result in results if result.get("ok")]
    if not successful:
        raise RuntimeError("All batch-size probes failed.")
    if not any(result.get("parsed_ok", 0) > 0 for result in successful):
        raise RuntimeError("Batch probes ran, but none returned valid score JSON.")
    if args.batch_selection == "largest-valid":
        chosen = max(successful, key=lambda item: item["batch_size"])["batch_size"]
    else:
        chosen = max(successful, key=lambda item: item.get("items_per_s") or 0.0)["batch_size"]
    return int(chosen), results


def probe_batch_sizes_per_question(
    batch_sizes: list[int],
    pairs: list[ImagePair],
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
) -> tuple[int, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if batch_size > len(pairs):
            continue
        probe_pairs = pairs[:batch_size]
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            records, elapsed = run_per_question_batch(
                probe_pairs,
                model,
                tokenizer,
                system_prompt,
                generation_config,
                args.max_tiles,
                dtype,
                device,
                transform,
                args.require_reasoning,
            )
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            valid = sum(1 for record in records if record["valid"])
            result = {
                "batch_size": batch_size,
                "ok": True,
                "elapsed_s": elapsed,
                "items_per_s": batch_size / elapsed if elapsed > 0 else None,
                "peak_cuda_allocated_gb": peak_gb,
                "parsed_ok": valid,
            }
            print(
                f"batch_probe bs={batch_size} ok elapsed={elapsed:.2f}s "
                f"items/s={result['items_per_s']:.3f} peak={peak_gb:.2f}GB "
                f"valid={valid}/{batch_size}",
                flush=True,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            torch.cuda.empty_cache()
            result = {
                "batch_size": batch_size,
                "ok": False,
                "error": "cuda_out_of_memory",
            }
            print(f"batch_probe bs={batch_size} OOM; skipping larger unsafe use", flush=True)
        results.append(result)
        if not result.get("ok") and result.get("error") == "cuda_out_of_memory":
            break

    successful = [result for result in results if result.get("ok")]
    if not successful:
        raise RuntimeError("All batch-size probes failed.")
    if not any(result.get("parsed_ok", 0) > 0 for result in successful):
        raise RuntimeError("Batch probes ran, but none returned valid per-question labels.")
    if args.batch_selection == "largest-valid":
        chosen = max(successful, key=lambda item: item["batch_size"])["batch_size"]
    else:
        chosen = max(successful, key=lambda item: item.get("items_per_s") or 0.0)["batch_size"]
    return int(chosen), results


def evaluate_pairs(
    pairs: list[ImagePair],
    batch_size: int,
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        index = 0
        current_batch_size = batch_size
        started_all = time.perf_counter()
        while index < len(pairs):
            batch_pairs = pairs[index : index + current_batch_size]
            try:
                torch.cuda.empty_cache()
                started = time.perf_counter()
                responses = generate_batch(
                    batch_pairs,
                    model,
                    tokenizer,
                    EVAL_PROMPT,
                    system_prompt,
                    generation_config,
                    args.max_tiles,
                    dtype,
                    device,
                    transform,
                    args.require_reasoning,
                )
                synchronize()
                elapsed = time.perf_counter() - started
            except RuntimeError as exc:
                if not is_oom_error(exc) or current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"eval OOM; reducing runtime batch size to {current_batch_size}",
                    flush=True,
                )
                continue

            batch_records = [
                make_record(pair, response, len(batch_pairs), elapsed)
                for pair, response in zip(batch_pairs, responses)
            ]
            for record in batch_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.extend(batch_records)

            valid = sum(1 for record in batch_records if record["valid"])
            mean_t_pas = numeric_mean([record["scores"]["T_PAS"] for record in batch_records])
            mean_t_sas = numeric_mean([record["scores"]["T_SAS"] for record in batch_records])
            completed = index + len(batch_pairs)
            elapsed_all = time.perf_counter() - started_all
            rate = completed / elapsed_all if elapsed_all > 0 else 0.0
            remaining = (len(pairs) - completed) / rate if rate > 0 else float("nan")
            print(
                f"eval {completed:03d}/{len(pairs):03d} "
                f"bs={len(batch_pairs)} elapsed={elapsed:.2f}s valid={valid}/{len(batch_pairs)} "
                f"items/s={rate:.3f} eta_min={remaining / 60:.1f} "
                f"T-PAS={mean_t_pas} T-SAS={mean_t_sas}",
                flush=True,
            )
            index += len(batch_pairs)
    return records


def evaluate_pairs_per_question(
    pairs: list[ImagePair],
    batch_size: int,
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    transform: Any,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        index = 0
        current_batch_size = batch_size
        started_all = time.perf_counter()
        while index < len(pairs):
            batch_pairs = pairs[index : index + current_batch_size]
            try:
                torch.cuda.empty_cache()
                batch_records, elapsed = run_per_question_batch(
                    batch_pairs,
                    model,
                    tokenizer,
                    system_prompt,
                    generation_config,
                    args.max_tiles,
                    dtype,
                    device,
                    transform,
                    args.require_reasoning,
                )
            except RuntimeError as exc:
                if not is_oom_error(exc) or current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"eval OOM; reducing runtime batch size to {current_batch_size}",
                    flush=True,
                )
                continue

            for record in batch_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.extend(batch_records)

            valid = sum(1 for record in batch_records if record["valid"])
            mean_t_pas = numeric_mean([record["scores"]["T_PAS"] for record in batch_records])
            mean_t_sas = numeric_mean([record["scores"]["T_SAS"] for record in batch_records])
            completed = index + len(batch_pairs)
            elapsed_all = time.perf_counter() - started_all
            rate = completed / elapsed_all if elapsed_all > 0 else 0.0
            remaining = (len(pairs) - completed) / rate if rate > 0 else float("nan")
            print(
                f"eval {completed:03d}/{len(pairs):03d} "
                f"bs={len(batch_pairs)} elapsed={elapsed:.2f}s valid={valid}/{len(batch_pairs)} "
                f"items/s={rate:.3f} eta_min={remaining / 60:.1f} "
                f"T-PAS={mean_t_pas} T-SAS={mean_t_sas}",
                flush=True,
            )
            index += len(batch_pairs)
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run inside the tmux Slurm GPU shell.")

    pairs = collect_requested_pairs(args)
    if not pairs:
        raise SystemExit("No image pairs found for the requested filters.")
    pairs = apply_limit(pairs, args.limit, args.sample_seed, args.sample_strategy)
    pair_counts = Counter((pair.method, pair.routing) for pair in pairs)
    routing_counts = Counter(pair.routing for pair in pairs)
    category_routing = {
        concept: routing
        for concept, routing in sorted({(pair.concept, pair.routing) for pair in pairs})
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_pairs.json"
    selected_path.write_text(
        json.dumps(
            [{**asdict(pair), "reference_path": str(pair.reference_path), "generated_path": str(pair.generated_path)} for pair in pairs],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "category_routing.json").write_text(
        json.dumps(category_routing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "dataset_source": args.dataset_source,
                "extra_manifest": str(args.extra_manifest),
                "extra_methods": args.extra_methods,
                "extra_subject": args.extra_subject,
                "extra_routing": args.extra_routing,
                "methods": args.methods,
                "concepts": args.concepts,
                "subjects": args.subjects,
                "rank": args.rank,
                "limit": args.limit,
                "sample_strategy": args.sample_strategy,
                "eval_mode": args.eval_mode,
                "abstract_categories": sorted(args.abstract_categories),
                "batch_sizes": args.batch_sizes,
                "batch_selection": args.batch_selection,
                "max_new_tokens": args.max_new_tokens,
                "max_tiles": args.max_tiles,
                "quantization": args.quantization,
                "dtype": args.dtype,
                "require_reasoning": args.require_reasoning,
                "pair_counts": {f"{method}/{routing}": count for (method, routing), count in pair_counts.items()},
                "routing_counts": dict(routing_counts),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dtype = dtype_from_name(args.dtype)
    system_prompt = (
        PER_QUESTION_SYSTEM_PROMPT
        if args.eval_mode == "per-question"
        else load_system_prompt(args.questions)
    )
    transform = build_transform(input_size=448)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(
        f"Pairs: {len(pairs)} methods={args.methods} subjects={args.subjects} rank={args.rank}",
        flush=True,
    )
    print(f"Routing counts: {dict(routing_counts)}", flush=True)
    print(
        "Pair counts: "
        + ", ".join(
            f"{method}/{routing}={count}"
            for (method, routing), count in sorted(pair_counts.items())
        ),
        flush=True,
    )
    print(
        f"Quantization={args.quantization} dtype={args.dtype} max_tiles={args.max_tiles}",
        flush=True,
    )
    print(f"Eval mode: {args.eval_mode}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_transformers_quantizer_compat()
    model = load_model(args, dtype)
    model.eval()
    device = first_model_device(model)
    synchronize()

    eos_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    if eos_id is None or eos_id < 0:
        eos_id = tokenizer.eos_token_id
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "eos_token_id": eos_id,
        "pad_token_id": tokenizer.pad_token_id or eos_id,
    }

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"CUDA memory after load: free={free_bytes / (1024**3):.2f}GB "
        f"total={total_bytes / (1024**3):.2f}GB",
        flush=True,
    )

    batch_probe: list[dict[str, Any]] = []
    if args.batch_size is not None:
        chosen_batch_size = args.batch_size
    elif args.skip_batch_probe:
        chosen_batch_size = 1
    else:
        probe_fn = (
            probe_batch_sizes_per_question
            if args.eval_mode == "per-question"
            else probe_batch_sizes
        )
        chosen_batch_size, batch_probe = probe_fn(
            parse_batch_sizes(args.batch_sizes),
            pairs,
            model,
            tokenizer,
            system_prompt,
            generation_config,
            args,
            dtype,
            device,
            transform,
        )
    print(f"Chosen batch size: {chosen_batch_size}", flush=True)

    jsonl_path = args.output_dir / "internvl3_pair_scores.jsonl"
    evaluate_fn = (
        evaluate_pairs_per_question
        if args.eval_mode == "per-question"
        else evaluate_pairs
    )
    records = evaluate_fn(
        pairs,
        chosen_batch_size,
        model,
        tokenizer,
        system_prompt,
        generation_config,
        args,
        dtype,
        device,
        transform,
        jsonl_path,
    )
    summary = summarize_records(records, batch_probe)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(summary, args.output_dir / "summary.csv")

    print("Summary", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {jsonl_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
