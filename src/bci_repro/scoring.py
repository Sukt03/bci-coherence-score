from __future__ import annotations

import json
import math
import re
from typing import Any

ANSWER_VALUES = {"no": 0.0, "somewhat": 0.5, "yes": 1.0}

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


def normalize_answer(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("answer")
    if value is None:
        return None
    answer = str(value).strip().lower()
    return answer if answer in ANSWER_VALUES else None


def clean_reasoning(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("reasoning") or value.get("reason") or value.get("rationale")
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def parse_label_reason_response(text: str) -> tuple[str | None, str | None]:
    """Parse a VLM response into the canonical label/reason pair."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return normalize_answer(parsed), clean_reasoning(parsed)

    answer = None
    match = re.search(r"\b(yes|somewhat|no)\b", raw, flags=re.IGNORECASE)
    if match:
        answer = match.group(1).lower()
    reason = None
    reason_match = re.search(
        r'"(?:reasoning|reason|rationale)"\s*:\s*"([^"]+)"',
        raw,
        flags=re.IGNORECASE,
    )
    if reason_match:
        reason = clean_reasoning(reason_match.group(1))
    return answer, reason


def aggregate_scores(normalized_response: dict[str, Any]) -> dict[str, float | None]:
    routing = str(normalized_response.get("routing") or "object")
    perceptual = normalized_response.get("perceptual") or {}
    semantic = normalized_response.get("semantic") or {}
    if routing == "abstract":
        perceptual_keys = ABSTRACT_PERCEPTUAL_KEYS
        semantic_keys: list[str] = []
    else:
        perceptual_keys = OBJECT_PERCEPTUAL_KEYS
        semantic_keys = OBJECT_SEMANTIC_KEYS

    def mean_for(section: dict[str, Any], keys: list[str]) -> float | None:
        values = [
            ANSWER_VALUES[answer]
            for key in keys
            if (answer := normalize_answer(section.get(key))) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "T_PAS": mean_for(perceptual, perceptual_keys),
        "T_SAS": mean_for(semantic, semantic_keys) if semantic_keys else None,
    }


def is_close(left: float | None, right: float | None, tolerance: float = 1e-6) -> bool:
    if left is None or right is None:
        return left is right
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return abs(left - right) <= tolerance

