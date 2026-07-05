#!/usr/bin/env python3
"""Evaluate GT/generated image pairs with multiple VLM backends.

This runner intentionally reuses the pair manifest and scoring helpers from the
InternVL evaluator so follow-on VLM runs are directly comparable.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import sys
import time
import types
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from internvl3_eval_pairs import (
    ANSWER_MAP,
    PER_QUESTION_SYSTEM_PROMPT,
    QUESTION_PROMPTS_BY_ROUTING,
    ImagePair,
    build_per_question_record,
    is_oom_error,
    numeric_mean,
    parse_label_reason_response,
    summarize_records,
    write_summary_csv,
)
from internvl3_speed_benchmark import (
    build_transform,
    dtype_from_name,
    dynamic_preprocess,
    first_model_device,
    synchronize,
)


MODEL_DEFAULTS = {
    "r4b": {
        "model_id": "YannQi/R-4B",
        "slug": "r4b",
        "display": "YannQi/R-4B",
    },
    "sail": {
        "model_id": "BytedanceDouyinContent/SAIL-VL-1d6-8B",
        "slug": "sail-vl-1d6-8b",
        "display": "BytedanceDouyinContent/SAIL-VL-1d6-8B",
    },
    "wethink": {
        "model_id": "yangjie-cv/WeThink-Qwen2.5VL-7B",
        "slug": "wethink-qwen2.5vl-7b",
        "display": "yangjie-cv/WeThink-Qwen2.5VL-7B",
    },
    "ola": {
        "model_id": "THUdyh/Ola-7b",
        "slug": "ola-7b",
        "display": "THUdyh/Ola-7b",
    },
    "ovis": {
        "model_id": "AIDC-AI/Ovis2-8B",
        "slug": "ovis2-8b",
        "display": "AIDC-AI/Ovis2-8B",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image pairs with non-InternVL VLMs.")
    parser.add_argument("--model-key", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--selected-pairs",
        type=Path,
        default=Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955/selected_pairs.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=6885)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=0,
        help="Use a deterministic mixed smoke subset of this size before any limit.",
    )
    parser.add_argument("--batch-sizes", default="4,8,16,24,32,48,64,80")
    parser.add_argument(
        "--batch-selection",
        choices=("largest-valid", "throughput"),
        default="largest-valid",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-batch-probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--qwen-max-pixels", type=int, default=448 * 448)
    parser.add_argument("--r4b-thinking-mode", default="auto")
    parser.add_argument("--invalid-retries", type=int, default=3)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--sanity-check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sanity-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def vlm_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    return dtype_from_name(name)


def parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if item:
            sizes.append(int(item))
    return sorted({size for size in sizes if size > 0})


def load_selected_pairs(path: Path, expected_count: int | None) -> list[ImagePair]:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected pair manifest: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if expected_count and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, found {len(rows)} in {path}")

    pairs: list[ImagePair] = []
    missing_paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        reference_path = Path(row["reference_path"])
        generated_path = Path(row["generated_path"])
        if not reference_path.exists():
            missing_paths.append(str(reference_path))
        if not generated_path.exists():
            missing_paths.append(str(generated_path))
        pair_id = str(row["pair_id"])
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id in selected manifest: {pair_id}")
        seen.add(pair_id)
        pairs.append(
            ImagePair(
                pair_id=pair_id,
                method=str(row["method"]),
                subject=str(row["subject"]),
                concept=str(row["concept"]),
                routing=str(row["routing"]),
                rank=str(row["rank"]),
                candidate=str(row["candidate"]),
                reference_path=reference_path,
                generated_path=generated_path,
            )
        )
    if missing_paths:
        raise FileNotFoundError("Missing image path(s): " + ", ".join(missing_paths[:20]))
    return pairs


def choose_smoke_pairs(pairs: list[ImagePair], count: int) -> list[ImagePair]:
    if count <= 0:
        return pairs

    selected: list[ImagePair] = []
    used: set[str] = set()

    def add_first(predicate) -> None:
        for pair in pairs:
            if pair.pair_id not in used and predicate(pair):
                selected.append(pair)
                used.add(pair.pair_id)
                return

    for routing in ("abstract", "object"):
        add_first(lambda pair, routing=routing: pair.routing == routing)
    for method in (
        "ATM",
        "ENIGMA",
        "brainvis",
        "dreamdiffusion",
        "thingseeg_brainvis",
        "thingseeg_dreamdiffusion",
        "cvpr40_brainvis",
        "cvpr40_dreamdiffusion",
    ):
        add_first(lambda pair, method=method: pair.method == method)

    for pair in pairs:
        if len(selected) >= count:
            break
        if pair.pair_id not in used:
            selected.append(pair)
            used.add(pair.pair_id)
    return selected[:count]


def plain_question_prompt(question: str) -> str:
    return (
        "The first image is Image 1, the ground-truth reference. "
        "The second image is Image 2, the EEG-generated reconstruction.\n\n"
        f"{question}\n\n"
        'Return only this JSON schema: {"answer":"yes|somewhat|no","reasoning":"one short sentence, max 25 words"}'
    )


def image_placeholder_question_prompt(question: str) -> str:
    return (
        "Image 1: <image>\n"
        "Image 2: <image>\n\n"
        f"{question}\n\n"
        'Return only this JSON schema: {"answer":"yes|somewhat|no","reasoning":"one short sentence, max 25 words"}'
    )


def strip_response(text: str) -> str:
    text = text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def tensor_inputs_to_device(inputs: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    data = dict(inputs)
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            if key in {"pixel_values", "image_grid_thw"}:
                data[key] = value.to(device)
            elif value.dtype.is_floating_point:
                data[key] = value.to(device=device, dtype=dtype)
            else:
                data[key] = value.to(device)
    if "pixel_values" in data and torch.is_tensor(data["pixel_values"]):
        data["pixel_values"] = data["pixel_values"].to(dtype=dtype)
    return data


def patch_transformers_tied_weights_compat() -> None:
    """Support remote-code models that still define `_tied_weights_keys` as a list."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

        def all_tied_weights_keys(self: torch.nn.Module) -> dict[str, str]:
            stored = getattr(self, "_compat_all_tied_weights_keys", None)
            if stored is not None:
                return stored
            tied = getattr(self, "_tied_weights_keys", None) or {}
            if isinstance(tied, dict):
                return tied
            return {str(key): str(key) for key in tied}

        def set_all_tied_weights_keys(
            self: torch.nn.Module,
            value: dict[str, str] | list[str] | tuple[str, ...],
        ) -> None:
            if isinstance(value, dict):
                self._compat_all_tied_weights_keys = value
            else:
                self._compat_all_tied_weights_keys = {str(key): str(key) for key in value}

        PreTrainedModel.all_tied_weights_keys = property(
            all_tied_weights_keys, set_all_tied_weights_keys
        )

    original = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original is None:
        return
    if getattr(original, "_hbai_list_compat", False):
        return

    def compat_get_expanded_tied_weights_keys(
        self: torch.nn.Module, all_submodels: bool = False
    ) -> dict[str, str]:
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, (list, tuple, set)):
            if all_submodels:
                expanded: dict[str, str] = {}
                for prefix, submodule in self.named_modules(remove_duplicate=False):
                    if isinstance(submodule, PreTrainedModel):
                        sub_tied = compat_get_expanded_tied_weights_keys(
                            submodule, all_submodels=False
                        )
                        if prefix:
                            sub_tied = {
                                f"{prefix}.{key}": f"{prefix}.{value}"
                                for key, value in sub_tied.items()
                            }
                        expanded.update(sub_tied)
                return expanded
            return {str(key): str(key) for key in tied}
        return original(self, all_submodels=all_submodels)

    setattr(compat_get_expanded_tied_weights_keys, "_hbai_list_compat", True)
    PreTrainedModel.get_expanded_tied_weights_keys = compat_get_expanded_tied_weights_keys


def patch_transformers_config_diff_compat() -> None:
    """Let remote config classes without safe default constructors still log/serialize."""
    try:
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return

    original = PretrainedConfig.to_diff_dict
    if getattr(original, "_hbai_default_ctor_compat", False):
        return

    def sanitize_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): sanitize_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize_jsonable(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def compat_to_diff_dict(self: Any) -> dict[str, Any]:
        try:
            data = original(self)
        except Exception:
            data = self.to_dict()
        return sanitize_jsonable(data)

    setattr(compat_to_diff_dict, "_hbai_default_ctor_compat", True)
    PretrainedConfig.to_diff_dict = compat_to_diff_dict


def patch_transformers_default_rope_compat() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return
    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def default_rope_parameters(
        config: Any,
        device: torch.device | None = None,
        seq_len: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, float]:
        del seq_len, kwargs
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size") // getattr(config, "num_attention_heads")
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters


def patch_transformers_dynamic_cache_legacy_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "to_legacy_cache"):
        return

    def to_legacy_cache(self: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        legacy = []
        for layer in getattr(self, "layers", []):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            legacy.append((keys, values))
        return tuple(legacy)

    DynamicCache.to_legacy_cache = to_legacy_cache


def patch_transformers_multimodal_data_compat() -> None:
    """Backport the processor-only MultiModalData helper expected by R-4B."""
    try:
        import transformers.processing_utils as processing_utils
    except Exception:
        return
    if hasattr(processing_utils, "MultiModalData"):
        return

    @dataclass
    class MultiModalData:
        num_image_tokens: list[int] | None = None
        num_video_tokens: list[int] | None = None
        num_audio_tokens: list[int] | None = None
        num_image_patches: list[int] | None = None

        def __contains__(self, key: str) -> bool:
            return hasattr(self, key) and getattr(self, key) is not None

        def __getitem__(self, key: str) -> Any:
            if hasattr(self, key):
                return getattr(self, key)
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    processing_utils.MultiModalData = MultiModalData


def patch_transformers_image_processing_r4b_compat() -> None:
    """Backport image-processing helpers required by R-4B remote code."""
    try:
        import transformers.image_processing_utils as image_processing_utils
        import transformers.image_utils as image_utils
        from transformers.image_utils import get_image_size
    except Exception:
        return

    if not hasattr(image_processing_utils, "get_patch_output_size"):

        def get_patch_output_size(
            image: Any,
            target_resolution: tuple[int, int],
            input_data_format: Any,
        ) -> tuple[int, int]:
            original_height, original_width = get_image_size(
                image, channel_dim=input_data_format
            )
            target_height, target_width = target_resolution
            scale_w = target_width / original_width
            scale_h = target_height / original_height
            if scale_w < scale_h:
                new_width = target_width
                new_height = min(math.ceil(original_height * scale_w), target_height)
            else:
                new_height = target_height
                new_width = min(math.ceil(original_width * scale_h), target_width)
            return new_height, new_width

        image_processing_utils.get_patch_output_size = get_patch_output_size

    if not hasattr(image_utils, "make_flat_list_of_images"):

        def is_valid_list_of_images(images: Any) -> bool:
            return isinstance(images, (list, tuple)) and bool(images) and all(
                image_utils.is_valid_image(image) for image in images
            )

        def make_flat_list_of_images(
            images: Any,
            expected_ndims: int = 3,
        ) -> list[Any]:
            if (
                isinstance(images, (list, tuple))
                and all(isinstance(item, (list, tuple)) for item in images)
                and all(is_valid_list_of_images(item) or not item for item in images)
            ):
                return [image for image_list in images for image in image_list]

            if isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
                first = images[0]
                if image_utils.is_pil_image(first) or first.ndim == expected_ndims:
                    return list(images)
                if first.ndim == expected_ndims + 1:
                    return [image for image_list in images for image in image_list]

            if image_utils.is_valid_image(images):
                if image_utils.is_pil_image(images) or images.ndim == expected_ndims:
                    return [images]
                if images.ndim == expected_ndims + 1:
                    return list(images)

            raise ValueError(f"Could not make a flat list of images from {images}")

        image_utils.make_flat_list_of_images = make_flat_list_of_images


class VLMAdapter(ABC):
    def __init__(self, args: argparse.Namespace, model_id: str, dtype: torch.dtype) -> None:
        self.args = args
        self.model_id = model_id
        self.dtype = dtype
        self.system_prompt = PER_QUESTION_SYSTEM_PROMPT
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.device = torch.device("cuda")

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        raise NotImplementedError

    def generate_sanity(
        self,
        pair: ImagePair,
        generation_config: dict[str, Any],
    ) -> str:
        del generation_config
        response = self.generate_question(
            [pair],
            "Briefly describe the reference image in one short sentence.",
            {"max_new_tokens": 64, "do_sample": False, "num_beams": 1},
        )[0]
        return strip_response(response)

    def cleanup(self) -> None:
        synchronize()
        torch.cuda.empty_cache()


class R4BAdapter(VLMAdapter):
    def load(self) -> None:
        patch_transformers_multimodal_data_compat()
        patch_transformers_image_processing_r4b_compat()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        ).eval()
        self.model.to(device="cuda", dtype=self.dtype)
        self.device = first_model_device(self.model)
        self.patch_prepare_inputs_for_generation()

    def patch_prepare_inputs_for_generation(self) -> None:
        original_prepare = self.model.prepare_inputs_for_generation

        def compat_prepare(
            model_self: torch.nn.Module,
            input_ids: torch.Tensor,
            past_key_values: Any = None,
            inputs_embeds: torch.Tensor | None = None,
            pixel_values: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            cache_position: torch.Tensor | None = None,
            logits_to_keep: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if cache_position is None:
                if past_key_values is None:
                    cache_position = torch.arange(
                        input_ids.shape[1],
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                else:
                    cache_position = torch.ones(
                        input_ids.shape[1],
                        device=input_ids.device,
                        dtype=torch.long,
                    )
            elif past_key_values is not None and cache_position.numel() and cache_position[0].item() == 0:
                cache_position = torch.ones_like(cache_position)
            return original_prepare(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(compat_prepare, self.model)

    def build_messages(self, pair: ImagePair, question: str) -> list[dict[str, Any]]:
        text = self.system_prompt + "\n\n" + plain_question_prompt(question)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(pair.reference_path)},
                    {"type": "image", "image": str(pair.generated_path)},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def render_text(self, messages: list[dict[str, Any]]) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=self.args.r4b_thinking_mode,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        texts = [self.render_text(self.build_messages(pair, question)) for pair in pairs]
        image_groups = [
            [
                Image.open(pair.reference_path).convert("RGB"),
                Image.open(pair.generated_path).convert("RGB"),
            ]
            for pair in pairs
        ]
        inputs = self.processor(
            images=image_groups,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        inputs = tensor_inputs_to_device(inputs, self.device, self.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_config)
        input_ids = inputs.get("input_ids")
        if input_ids is not None and output_ids.shape[1] > input_ids.shape[1]:
            output_ids = output_ids[:, input_ids.shape[1] :]
        tokenizer = self.tokenizer or self.processor.tokenizer
        return [strip_response(text) for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

    def generate_sanity(
        self,
        pair: ImagePair,
        generation_config: dict[str, Any],
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(pair.reference_path)},
                    {"type": "text", "text": "Describe this image in one short sentence."},
                ],
            }
        ]
        text = self.render_text(messages)
        with Image.open(pair.reference_path) as image:
            rgb_image = image.convert("RGB")
        inputs = self.processor(images=rgb_image, text=text, return_tensors="pt")
        inputs = tensor_inputs_to_device(inputs, self.device, self.dtype)
        config = dict(generation_config)
        config["max_new_tokens"] = min(
            self.args.sanity_max_new_tokens,
            max(config.get("max_new_tokens", self.args.sanity_max_new_tokens), 64),
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **config)
        input_ids = inputs.get("input_ids")
        if input_ids is not None and output_ids.shape[1] > input_ids.shape[1]:
            output_ids = output_ids[:, input_ids.shape[1] :]
        tokenizer = self.tokenizer or self.processor.tokenizer
        return strip_response(tokenizer.decode(output_ids[0], skip_special_tokens=True))


class OvisAdapter(VLMAdapter):
    def load(self) -> None:
        config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        )
        if getattr(config, "llm_attn_implementation", None) == "flash_attention_2":
            config.llm_attn_implementation = None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=self.dtype,
            multimodal_max_length=32768,
            low_cpu_mem_usage=True,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        ).eval()
        self.model.cuda()
        self.device = first_model_device(self.model)
        self.tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def preprocess_sample(
        self,
        query: str,
        images: list[Image.Image],
        max_partition: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        _, input_ids, pixel_values = self.model.preprocess_inputs(
            query,
            images,
            max_partition=max_partition,
        )
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype,
                device=self.visual_tokenizer.device,
            )
        return input_ids, attention_mask, pixel_values

    def pad_inputs(
        self,
        input_ids: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_input_ids = pad_sequence(
            [ids.flip(dims=[0]) for ids in input_ids],
            batch_first=True,
            padding_value=0,
        ).flip(dims=[1])
        batch_attention_mask = pad_sequence(
            [mask.flip(dims=[0]) for mask in attention_masks],
            batch_first=True,
            padding_value=False,
        ).flip(dims=[1])
        max_length = getattr(self.model.config, "multimodal_max_length", 32768)
        return batch_input_ids[:, -max_length:], batch_attention_mask[:, -max_length:]

    def generation_kwargs(self, generation_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(generation_config)
        config["eos_token_id"] = getattr(self.model.generation_config, "eos_token_id", None)
        config["pad_token_id"] = self.tokenizer.pad_token_id
        config["use_cache"] = True
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            config.setdefault(key, None)
        return config

    def decode_generated(
        self,
        output_ids: torch.Tensor,
        prompt_length: int,
    ) -> list[str]:
        decoded: list[str] = []
        for row in output_ids:
            candidate_ids = row[prompt_length:] if row.shape[0] > prompt_length else row
            candidate = strip_response(
                self.tokenizer.decode(candidate_ids, skip_special_tokens=True)
            )
            if not candidate and row.shape[0] <= prompt_length:
                candidate = strip_response(
                    self.tokenizer.decode(row, skip_special_tokens=True)
                )
            decoded.append(candidate)
        return decoded

    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        prompt = self.system_prompt + "\n\n" + image_placeholder_question_prompt(question)
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        pixel_values: list[torch.Tensor | None] = []
        for pair in pairs:
            with Image.open(pair.reference_path) as reference_image, Image.open(pair.generated_path) as generated_image:
                images = [
                    reference_image.convert("RGB"),
                    generated_image.convert("RGB"),
                ]
                sample_input_ids, sample_attention_mask, sample_pixel_values = self.preprocess_sample(
                    prompt,
                    images,
                    max_partition=4,
                )
            input_ids.append(sample_input_ids)
            attention_masks.append(sample_attention_mask)
            pixel_values.append(sample_pixel_values)

        batch_input_ids, batch_attention_mask = self.pad_inputs(input_ids, attention_masks)
        config = self.generation_kwargs(generation_config)
        with torch.inference_mode():
            output_ids = self.model.generate(
                batch_input_ids,
                pixel_values=pixel_values,
                attention_mask=batch_attention_mask,
                **config,
            )
        return self.decode_generated(output_ids, batch_input_ids.shape[1])

    def generate_sanity(
        self,
        pair: ImagePair,
        generation_config: dict[str, Any],
    ) -> str:
        with Image.open(pair.reference_path) as image:
            images = [image.convert("RGB")]
            input_ids, attention_mask, pixel_values = self.preprocess_sample(
                "<image>\nDescribe this image in one short sentence.",
                images,
                max_partition=9,
            )
        batch_input_ids = input_ids.unsqueeze(0)
        batch_attention_mask = attention_mask.unsqueeze(0)
        config = self.generation_kwargs(generation_config)
        config["max_new_tokens"] = min(
            self.args.sanity_max_new_tokens,
            max(config.get("max_new_tokens", self.args.sanity_max_new_tokens), 64),
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                batch_input_ids,
                pixel_values=[pixel_values],
                attention_mask=batch_attention_mask,
                **config,
            )
        return self.decode_generated(output_ids, batch_input_ids.shape[1])[0]


class SailAdapter(VLMAdapter):
    def force_eager_attention(self, config: Any) -> Any:
        for candidate in (config, getattr(config, "llm_config", None)):
            if candidate is None:
                continue
            setattr(candidate, "_attn_implementation", "eager")
            setattr(candidate, "_attn_implementation_internal", "eager")
            setattr(candidate, "attn_implementation", "eager")
        return config

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=False,
            token=self.args.hf_token,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        )
        config = self.force_eager_attention(config)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        ).eval()
        self.model.to("cuda")
        self.device = first_model_device(self.model)
        self.transform = build_transform(input_size=self.args.image_size)
        self.get_conv_template = self.model.__class__.chat.__globals__.get("get_conv_template")
        if self.get_conv_template is None:
            raise RuntimeError("SAIL-VL remote code does not expose get_conv_template")
        if int(transformers.__version__.split(".", 1)[0]) >= 5:
            self.patch_language_model_prepare_inputs()
            self.patch_rotary_length_compat()

    def patch_language_model_prepare_inputs(self) -> None:
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            return
        original_prepare = language_model.prepare_inputs_for_generation
        if getattr(original_prepare, "_hbai_cache_position_compat", False):
            return

        def compat_prepare(
            lm_self: torch.nn.Module,
            input_ids: torch.Tensor | None = None,
            past_key_values: Any = None,
            attention_mask: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            cache_position: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            use_cache: bool = True,
            num_logits_to_keep: int | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if inputs_embeds is not None:
                embed_len = int(inputs_embeds.shape[1])
                if attention_mask is not None and int(attention_mask.shape[1]) != embed_len:
                    attention_mask = attention_mask[:, -embed_len:]
                if input_ids is not None and int(input_ids.shape[1]) != embed_len:
                    input_ids = input_ids[:, -embed_len:]
            if cache_position is None:
                source = inputs_embeds if inputs_embeds is not None else input_ids
                if source is None and attention_mask is not None:
                    source = attention_mask
                if source is None:
                    raise RuntimeError("SAIL-VL generation could not infer cache_position source.")
                seq_len = int(source.shape[1])
                device = source.device
                if past_key_values is None or inputs_embeds is not None:
                    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
                    if inputs_embeds is not None:
                        past_key_values = None
                else:
                    cache_position = torch.arange(
                        max(seq_len - 1, 0),
                        seq_len,
                        device=device,
                        dtype=torch.long,
                    )
            model_inputs = original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )
            return model_inputs

        setattr(compat_prepare, "_hbai_cache_position_compat", True)
        language_model.prepare_inputs_for_generation = types.MethodType(
            compat_prepare, language_model
        )

    def patch_rotary_length_compat(self) -> None:
        try:
            attn_module_name = (
                self.model.language_model.model.layers[0].self_attn.__class__.__module__
            )
            module = sys.modules[attn_module_name]
            original_apply = module.apply_rotary_pos_emb
        except Exception:
            return
        if getattr(original_apply, "_hbai_rotary_length_compat", False):
            return

        def compat_apply_rotary_pos_emb(
            q: torch.Tensor,
            k: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            unsqueeze_dim: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            target_len = int(q.shape[2 if unsqueeze_dim == 1 else 1])
            cos_len = int(cos.shape[-2])
            if cos_len != target_len:
                cos = cos[..., -target_len:, :]
                sin = sin[..., -target_len:, :]
                if position_ids is not None and position_ids.shape[-1] != target_len:
                    position_ids = position_ids[..., -target_len:]
            return original_apply(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)

        setattr(compat_apply_rotary_pos_emb, "_hbai_rotary_length_compat", True)
        module.apply_rotary_pos_emb = compat_apply_rotary_pos_emb

    def load_tiles(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
        tiles = dynamic_preprocess(
            rgb_image,
            image_size=self.args.image_size,
            use_thumbnail=True,
            max_num=self.args.max_tiles,
        )
        return torch.stack([self.transform(tile) for tile in tiles])

    def build_query(self, question: str, num_patches_list: list[int]) -> str:
        template = self.get_conv_template(self.model.template)
        if hasattr(template, "system_message"):
            template.system_message = self.system_prompt
        template.append_message(template.roles[0], image_placeholder_question_prompt(question))
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches + "</img>"
            query = query.replace("<image>", image_tokens, 1)
        return query

    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        queries = []
        pixel_batches = []
        template = self.get_conv_template(self.model.template)
        for pair in pairs:
            reference_pixels = self.load_tiles(pair.reference_path)
            generated_pixels = self.load_tiles(pair.generated_path)
            num_patches_list = [reference_pixels.shape[0], generated_pixels.shape[0]]
            queries.append(self.build_query(question, num_patches_list))
            pixel_batches.append(torch.cat([reference_pixels, generated_pixels], dim=0))

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        pixel_values = torch.cat(pixel_batches, dim=0).to(dtype=self.dtype, device=self.device)
        eos_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        config = dict(generation_config)
        if eos_id is not None and eos_id >= 0:
            config["eos_token_id"] = eos_id
        config["pad_token_id"] = self.tokenizer.pad_token_id or config.get("eos_token_id")
        with torch.inference_mode():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **config,
            )
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        sep = template.sep if isinstance(template.sep, str) else ""
        cleaned = []
        for response in responses:
            if sep and sep in response:
                response = response.split(sep, 1)[0]
            cleaned.append(strip_response(response))
        return cleaned

    def generate_sanity(
        self,
        pair: ImagePair,
        generation_config: dict[str, Any],
    ) -> str:
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        pixel_values = self.load_tiles(pair.reference_path).to(dtype=self.dtype, device=self.device)
        config = dict(generation_config)
        config["max_new_tokens"] = min(
            self.args.sanity_max_new_tokens,
            max(config.get("max_new_tokens", self.args.sanity_max_new_tokens), 64),
        )
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            "<image>\nDescribe this image in one short sentence.",
            config,
        )
        return strip_response(response)


def set_ola_env_defaults() -> None:
    defaults = {
        "LOWRES_RESIZE": "384x32",
        "HIGHRES_BASE": "0x32",
        "VIDEO_RESIZE": "0x64",
        "VIDEO_MAXRES": "480",
        "VIDEO_MINRES": "288",
        "MAXRES": "1536",
        "MINRES": "0",
        "REGIONAL_POOL": "2x",
        "FORCE_NO_DOWNSAMPLE": "1",
        "LOAD_VISION_EARLY": "1",
        "SKIP_LOAD_VIT": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def install_ola_import_shims() -> None:
    if "flash_attn" not in sys.modules:
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

        def flash_attn_func(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dropout_p: float = 0.0,
            softmax_scale: float | None = None,
            causal: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor:
            del kwargs
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                dropout_p=dropout_p,
                scale=softmax_scale,
                is_causal=causal,
            )
            return out.transpose(1, 2)

        def flash_attn_varlen_func(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int | None = None,
            max_seqlen_k: int | None = None,
            dropout_p: float = 0.0,
            softmax_scale: float | None = None,
            causal: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor:
            del max_seqlen_q, max_seqlen_k, kwargs
            chunks: list[torch.Tensor] = []
            for idx in range(cu_seqlens_q.numel() - 1):
                qs, qe = int(cu_seqlens_q[idx]), int(cu_seqlens_q[idx + 1])
                ks, ke = int(cu_seqlens_k[idx]), int(cu_seqlens_k[idx + 1])
                q_i = q[qs:qe].unsqueeze(0)
                k_i = k[ks:ke].unsqueeze(0)
                v_i = v[ks:ke].unsqueeze(0)
                chunks.append(
                    flash_attn_func(
                        q_i,
                        k_i,
                        v_i,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    ).squeeze(0)
                )
            return torch.cat(chunks, dim=0) if chunks else q.new_empty(q.shape)

        flash_attn.flash_attn_func = flash_attn_func
        flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
        sys.modules["flash_attn"] = flash_attn

    if "deepspeed" not in sys.modules:
        deepspeed = types.ModuleType("deepspeed")
        deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
        zero = types.ModuleType("deepspeed.zero")
        zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", loader=None)

        class GatheredParameters:
            def __init__(self, params: Any) -> None:
                self.params = params

            def __enter__(self) -> Any:
                return self.params

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        def register_external_parameter(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        zero.GatheredParameters = GatheredParameters
        zero.register_external_parameter = register_external_parameter
        deepspeed.zero = zero
        runtime = types.ModuleType("deepspeed.runtime")
        runtime.__spec__ = importlib.machinery.ModuleSpec("deepspeed.runtime", loader=None)
        runtime_zero = types.ModuleType("deepspeed.runtime.zero")
        runtime_zero.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed.runtime.zero", loader=None
        )
        partition_parameters = types.ModuleType(
            "deepspeed.runtime.zero.partition_parameters"
        )
        partition_parameters.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed.runtime.zero.partition_parameters", loader=None
        )

        class ZeroParamStatus:
            NOT_AVAILABLE = "NOT_AVAILABLE"

        partition_parameters.ZeroParamStatus = ZeroParamStatus
        sys.modules["deepspeed"] = deepspeed
        sys.modules["deepspeed.zero"] = zero
        sys.modules["deepspeed.runtime"] = runtime
        sys.modules["deepspeed.runtime.zero"] = runtime_zero
        sys.modules[
            "deepspeed.runtime.zero.partition_parameters"
        ] = partition_parameters

    if "whisper" not in sys.modules:
        whisper = types.ModuleType("whisper")
        whisper.__spec__ = importlib.machinery.ModuleSpec("whisper", loader=None)

        def unavailable(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("Whisper audio loading is disabled for image-only OLA eval.")

        whisper.load_model = unavailable
        whisper.pad_or_trim = lambda value, *args, **kwargs: value
        sys.modules["whisper"] = whisper


class OlaAdapter(VLMAdapter):
    def load(self) -> None:
        set_ola_env_defaults()
        install_ola_import_shims()
        try:
            from ola.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from ola.conversation import conv_templates
            from ola.datasets.preprocess import tokenizer_image_token
            from ola.mm_utils import process_anyres_highres_image
            from ola.model.language_model.ola_qwen import OlaConfigQwen, OlaQwenForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "OLA package is not importable. Install the official Ola-Omni/Ola "
                "package in the hbai environment before running OLA."
            ) from exc

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self.process_anyres_highres_image = process_anyres_highres_image

        config = OlaConfigQwen.from_pretrained(
            self.model_id,
            token=self.args.hf_token,
            trust_remote_code=self.args.trust_remote_code,
        )
        for attr in ("speech_encoder", "speech_encoder_type", "speech_projector_type", "music_encoder"):
            if hasattr(config, attr):
                delattr(config, attr)
        if getattr(config, "rope_parameters", None) is None:
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 1000000.0),
            }
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        config.attn_implementation = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=False,
            token=self.args.hf_token,
            trust_remote_code=self.args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = OlaQwenForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            token=self.args.hf_token,
        ).eval()
        self.model.to(device="cuda", dtype=self.dtype)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = first_model_device(self.model)

        self.model.get_model().speech_encoder = torch.nn.Identity()

        def fake_encode_speech(
            model_self: torch.nn.Module,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor | None = None,
            speech_wav: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del speech_lengths, speech_wav
            return torch.zeros(
                (0, model_self.config.hidden_size),
                device=speech.device,
                dtype=self.dtype,
            )

        self.model.encode_speech = types.MethodType(fake_encode_speech, self.model)

        vision_tower = self.model.get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("OLA model did not expose a vision tower.")
        if not getattr(vision_tower, "is_loaded", False):
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.dtype)
        vision_tower.eval()
        self.image_processor = vision_tower.image_processor

    def build_prompt(self, question: str, image_count: int = 2) -> str:
        conv = self.conv_templates["qwen_1_5"].copy()
        image_tokens = "\n".join([self.DEFAULT_IMAGE_TOKEN] * image_count)
        text = f"{image_tokens}\n{self.system_prompt}\n\n{plain_question_prompt(question)}"
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        tensors = [
            self.tokenizer_image_token(
                prompt,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            for prompt in prompts
        ]
        input_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        attention_mask = input_ids.ne(pad_id).long()
        return input_ids.to(self.device), attention_mask.to(self.device)

    def process_image(self, path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            image_size = rgb_image.size
            lowres, highres = self.process_anyres_highres_image(rgb_image, self.image_processor)
        return (
            lowres.to(device=self.device, dtype=self.dtype),
            highres.to(device=self.device, dtype=self.dtype),
            image_size,
        )

    def dummy_speech(
        self, batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        speech = [
            torch.zeros((1, 3000, 128), device=self.device, dtype=self.dtype)
            for _ in range(batch_size)
        ]
        speech_lengths = [
            torch.LongTensor([3000]).to(self.device)
            for _ in range(batch_size)
        ]
        speech_chunks = [
            torch.LongTensor([1]).to(self.device)
            for _ in range(batch_size)
        ]
        speech_wav = [
            torch.zeros((1, 480000), device=self.device, dtype=torch.float32)
            for _ in range(batch_size)
        ]
        return speech, speech_lengths, speech_chunks, speech_wav

    def generate_with_images(
        self,
        prompts: list[str],
        image_paths_by_prompt: list[list[Path]],
        generation_config: dict[str, Any],
    ) -> list[str]:
        input_ids, attention_mask = self.encode_prompts(prompts)
        images: list[torch.Tensor] = []
        images_highres: list[torch.Tensor] = []
        image_sizes: list[tuple[int, int]] = []
        modalities: list[str] = []
        for image_paths in image_paths_by_prompt:
            for image_path in image_paths:
                lowres, highres, image_size = self.process_image(image_path)
                images.append(lowres)
                images_highres.append(highres)
                image_sizes.append(image_size)
                modalities.append("image")
        speech, speech_lengths, speech_chunks, speech_wav = self.dummy_speech(len(prompts))
        config = dict(generation_config)
        config.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        config.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=images,
                images_highres=images_highres,
                image_sizes=image_sizes,
                modalities=modalities,
                speech=speech,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wav,
                attention_mask=attention_mask,
                use_cache=True,
                **config,
            )
        if output_ids.ndim == 1:
            output_ids = output_ids.unsqueeze(0)
        if output_ids.shape[1] > input_ids.shape[1]:
            decoded_ids = output_ids[:, input_ids.shape[1] :]
        else:
            decoded_ids = output_ids
        return [strip_response(text) for text in self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)]

    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        prompts = [self.build_prompt(question, image_count=2) for _ in pairs]
        images = [[pair.reference_path, pair.generated_path] for pair in pairs]
        return self.generate_with_images(prompts, images, generation_config)

    def generate_sanity(
        self,
        pair: ImagePair,
        generation_config: dict[str, Any],
    ) -> str:
        conv = self.conv_templates["qwen_1_5"].copy()
        conv.append_message(
            conv.roles[0],
            f"{self.DEFAULT_IMAGE_TOKEN}\nDescribe this image in one short sentence.",
        )
        conv.append_message(conv.roles[1], None)
        config = dict(generation_config)
        config["max_new_tokens"] = min(
            self.args.sanity_max_new_tokens,
            max(config.get("max_new_tokens", self.args.sanity_max_new_tokens), 64),
        )
        return self.generate_with_images(
            [conv.get_prompt()],
            [[pair.reference_path]],
            config,
        )[0]


class WeThinkAdapter(VLMAdapter):
    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.args.trust_remote_code,
            max_pixels=self.args.qwen_max_pixels,
            token=self.args.hf_token,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.args.trust_remote_code,
            token=self.args.hf_token,
        ).eval()
        self.model.to("cuda")
        self.device = first_model_device(self.model)

    def build_messages(self, pair: ImagePair, question: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(pair.reference_path)},
                    {"type": "image", "image": str(pair.generated_path)},
                    {"type": "text", "text": plain_question_prompt(question)},
                ],
            },
        ]

    def make_inputs(self, pairs: list[ImagePair], question: str) -> dict[str, Any]:
        conversations = [self.build_messages(pair, question) for pair in pairs]
        try:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                add_vision_id=True,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        return tensor_inputs_to_device(inputs, self.device, self.dtype)

    def generate_question(
        self,
        pairs: list[ImagePair],
        question: str,
        generation_config: dict[str, Any],
    ) -> list[str]:
        inputs = self.make_inputs(pairs, question)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_config)
        input_ids = inputs["input_ids"]
        generated_ids = [
            output_ids[idx, input_ids.shape[1] :]
            for idx in range(output_ids.shape[0])
        ]
        tokenizer = self.tokenizer or self.processor.tokenizer
        return [strip_response(text) for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def build_adapter(args: argparse.Namespace, dtype: torch.dtype) -> VLMAdapter:
    patch_transformers_tied_weights_compat()
    patch_transformers_config_diff_compat()
    patch_transformers_default_rope_compat()
    patch_transformers_dynamic_cache_legacy_compat()
    model_id = args.model_id or MODEL_DEFAULTS[args.model_key]["model_id"]
    if args.model_key == "r4b":
        adapter: VLMAdapter = R4BAdapter(args, model_id, dtype)
    elif args.model_key == "sail":
        adapter = SailAdapter(args, model_id, dtype)
    elif args.model_key == "wethink":
        adapter = WeThinkAdapter(args, model_id, dtype)
    elif args.model_key == "ola":
        adapter = OlaAdapter(args, model_id, dtype)
    elif args.model_key == "ovis":
        adapter = OvisAdapter(args, model_id, dtype)
    else:
        raise ValueError(f"Unsupported model key: {args.model_key}")
    adapter.load()
    return adapter


def run_per_question_batch(
    pairs: list[ImagePair],
    adapter: VLMAdapter,
    generation_config: dict[str, Any],
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
            responses = adapter.generate_question(routed_pairs, question, generation_config)
            synchronize()
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            for pair, response in zip(routed_pairs, responses):
                raw_by_pair[pair.pair_id][key] = response
                answer, reason = parse_label_reason_response(response)
                if answer is None:
                    errors_by_pair[pair.pair_id].append(f"missing_or_invalid_{key}")
                else:
                    answers_by_pair[pair.pair_id][key] = answer
                    if reason:
                        reasoning_by_pair[pair.pair_id][key] = reason

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


def probe_batch_sizes(
    batch_sizes: list[int],
    pairs: list[ImagePair],
    adapter: VLMAdapter,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
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
                probe_pairs, adapter, generation_config, args.require_reasoning
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
            result = {"batch_size": batch_size, "ok": False, "error": "cuda_out_of_memory"}
            print(f"batch_probe bs={batch_size} OOM; stopping probe", flush=True)
        results.append(result)
        if not result.get("ok") and result.get("error") == "cuda_out_of_memory":
            break

    successful = [result for result in results if result.get("ok")]
    fully_valid = [result for result in successful if result.get("parsed_ok") == result["batch_size"]]
    candidates = fully_valid or [result for result in successful if result.get("parsed_ok", 0) > 0]
    if not candidates:
        raise RuntimeError("Batch probes ran, but none produced valid rows.")
    if args.batch_selection == "largest-valid":
        chosen = max(candidates, key=lambda item: item["batch_size"])["batch_size"]
    else:
        chosen = max(candidates, key=lambda item: item.get("items_per_s") or 0.0)["batch_size"]
    return int(chosen), results


def write_records(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_pairs(
    pairs: list[ImagePair],
    adapter: VLMAdapter,
    batch_size: int,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
    jsonl_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_batch_size = batch_size
    started_all = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        index = 0
        while index < len(pairs):
            batch_pairs = pairs[index : index + current_batch_size]
            try:
                torch.cuda.empty_cache()
                batch_records, elapsed = run_per_question_batch(
                    batch_pairs, adapter, generation_config, args.require_reasoning
                )
            except RuntimeError as exc:
                if not is_oom_error(exc) or current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(f"eval OOM; reducing runtime batch size to {current_batch_size}", flush=True)
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
                f"eval {completed:04d}/{len(pairs):04d} "
                f"bs={len(batch_pairs)} elapsed={elapsed:.2f}s valid={valid}/{len(batch_pairs)} "
                f"items/s={rate:.3f} eta_min={remaining / 60:.1f} "
                f"T-PAS={mean_t_pas} T-SAS={mean_t_sas}",
                flush=True,
            )
            index += len(batch_pairs)
    return records


def retry_invalid_records(
    records: list[dict[str, Any]],
    pairs_by_id: dict[str, ImagePair],
    adapter: VLMAdapter,
    generation_config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    current = {record["pair_id"]: record for record in records}
    for retry_idx in range(1, args.invalid_retries + 1):
        invalid_ids = [pair_id for pair_id, record in current.items() if not record["valid"]]
        if not invalid_ids:
            break
        print(f"invalid_retry {retry_idx}/{args.invalid_retries} rows={len(invalid_ids)}", flush=True)
        for pair_id in invalid_ids:
            pair = pairs_by_id[pair_id]
            try:
                replacement, _ = run_per_question_batch(
                    [pair], adapter, generation_config, args.require_reasoning
                )
            except RuntimeError as exc:
                if is_oom_error(exc):
                    torch.cuda.empty_cache()
                    continue
                raise
            if replacement and replacement[0]["valid"]:
                current[pair_id] = replacement[0]
    return [current[record["pair_id"]] for record in records]


def write_invalid_records(records: list[dict[str, Any]], path: Path) -> None:
    invalid = [record for record in records if not record["valid"]]
    write_records(invalid, path)


def run_sanity_check(
    adapter: VLMAdapter,
    pair: ImagePair,
    generation_config: dict[str, Any],
    output_dir: Path,
) -> None:
    started = time.perf_counter()
    response = adapter.generate_sanity(pair, generation_config)
    synchronize()
    elapsed = time.perf_counter() - started
    record = {
        "pair_id": pair.pair_id,
        "reference_path": str(pair.reference_path),
        "response": response,
        "elapsed_s": elapsed,
        "ok": bool(response.strip()),
    }
    (output_dir / "sanity_check.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"sanity_check ok={record['ok']} elapsed={elapsed:.2f}s response={response[:160]!r}", flush=True)
    if not record["ok"]:
        raise RuntimeError("Sanity check produced an empty response.")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run inside an allocated GPU shell.")

    model_id = args.model_id or MODEL_DEFAULTS[args.model_key]["model_id"]
    slug = MODEL_DEFAULTS[args.model_key]["slug"]
    dtype = vlm_dtype_from_name(args.dtype)

    all_pairs = load_selected_pairs(args.selected_pairs, args.expected_count)
    pairs = choose_smoke_pairs(all_pairs, args.smoke_samples)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No image pairs selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_pairs.json"
    selected_path.write_text(
        json.dumps(
            [
                {
                    **asdict(pair),
                    "reference_path": str(pair.reference_path),
                    "generated_path": str(pair.generated_path),
                }
                for pair in pairs
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pair_counts = Counter((pair.method, pair.routing) for pair in pairs)
    run_config = {
        "model_key": args.model_key,
        "model_id": model_id,
        "model_slug": slug,
        "selected_pairs": str(args.selected_pairs),
        "expected_count": args.expected_count,
        "evaluated_count": len(pairs),
        "smoke_samples": args.smoke_samples,
        "limit": args.limit,
        "batch_sizes": args.batch_sizes,
        "batch_size": args.batch_size,
        "skip_batch_probe": args.skip_batch_probe,
        "probe_only": args.probe_only,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "require_reasoning": args.require_reasoning,
        "max_tiles": args.max_tiles,
        "image_size": args.image_size,
        "qwen_max_pixels": args.qwen_max_pixels,
        "sanity_check": args.sanity_check,
        "sanity_max_new_tokens": args.sanity_max_new_tokens,
        "pair_counts": {f"{method}/{routing}": count for (method, routing), count in sorted(pair_counts.items())},
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"Pairs: {len(pairs)} selected from {len(all_pairs)}", flush=True)
    print(f"Pair counts: {run_config['pair_counts']}", flush=True)

    adapter = build_adapter(args, dtype)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    if args.model_key == "sail":
        generation_config.update(
            {
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            }
        )
    if adapter.tokenizer is not None:
        pad_id = getattr(adapter.tokenizer, "pad_token_id", None)
        eos_id = getattr(adapter.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            generation_config["eos_token_id"] = eos_id
        if pad_id is not None:
            generation_config["pad_token_id"] = pad_id

    if args.sanity_check:
        run_sanity_check(adapter, pairs[0], generation_config, args.output_dir)

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
        chosen_batch_size, batch_probe = probe_batch_sizes(
            parse_batch_sizes(args.batch_sizes),
            pairs,
            adapter,
            generation_config,
            args,
        )
    print(f"Chosen batch size: {chosen_batch_size}", flush=True)
    (args.output_dir / "batch_probe.json").write_text(
        json.dumps(batch_probe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "chosen_batch_size.txt").write_text(
        f"{chosen_batch_size}\n",
        encoding="utf-8",
    )

    if args.probe_only:
        print("Probe-only run complete.", flush=True)
        return

    jsonl_path = args.output_dir / "pair_scores.jsonl"
    records = evaluate_pairs(
        pairs,
        adapter,
        chosen_batch_size,
        generation_config,
        args,
        jsonl_path,
    )
    if args.invalid_retries > 0:
        write_invalid_records(records, args.output_dir / "invalid_pairs.pre_retry.jsonl")
        records = retry_invalid_records(
            records,
            {pair.pair_id: pair for pair in pairs},
            adapter,
            generation_config,
            args,
        )
        write_records(records, jsonl_path)

    write_invalid_records(records, args.output_dir / "invalid_pairs.jsonl")
    summary = summarize_records(records, batch_probe)
    summary["model"] = {
        "model_key": args.model_key,
        "model_id": model_id,
        "model_slug": slug,
    }
    summary["output_schema"] = "internvl3-compatible-per-question-reasoning"
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(summary, args.output_dir / "summary.csv")

    print("Summary", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {jsonl_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
