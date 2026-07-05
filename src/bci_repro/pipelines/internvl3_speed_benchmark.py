#!/usr/bin/env python3
"""Benchmark OpenGVLab/InternVL3-8B image-chat inference latency.

Default settings target a 20 GB A100 MIG slice: 4-bit weights, bf16 compute,
one 448px image tile, batch size 1.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure InternVL3-8B VLM inference speed on one GPU."
    )
    parser.add_argument("--model", default="OpenGVLab/InternVL3-8B")
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        help="Optional local image path(s). Pass multiple paths for multi-image prompts.",
    )
    parser.add_argument(
        "--prompt",
        default="<image>\nDescribe this image in one concise sentence.",
        help="Prompt sent to model.chat(). Include <image> for image input.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of identical requests to run in one generate() call.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
        help="Use 4bit on a 20 GB MIG slice unless you know unquantized fits.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Compute dtype for activations and non-quantized weights.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable FlashAttention 2 if it is installed in the environment.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for OpenGVLab/InternVL3-8B custom model code.",
    )
    return parser.parse_args()


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Iterable[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 1,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def make_synthetic_image() -> Image.Image:
    image = Image.new("RGB", (896, 448), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 64, 360, 384), fill=(38, 92, 130), outline=(15, 41, 61), width=6)
    draw.ellipse((520, 84, 824, 388), fill=(231, 174, 72), outline=(115, 80, 26), width=6)
    draw.text((64, 32), "InternVL3 speed test", fill=(20, 24, 31))
    draw.text((558, 202), "A100 MIG", fill=(20, 24, 31))
    return image


def load_image_tiles(image_path: Path | None, max_tiles: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB") if image_path else make_synthetic_image()
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_tiles)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def load_images(image_paths: list[Path] | None, max_tiles: int) -> tuple[torch.Tensor, list[int]]:
    image_paths = image_paths or [None]
    all_pixel_values = []
    num_patches_list = []
    for image_path in image_paths:
        image_pixel_values = load_image_tiles(image_path, max_tiles)
        all_pixel_values.append(image_pixel_values)
        num_patches_list.append(image_pixel_values.shape[0])
    return torch.cat(all_pixel_values, dim=0), num_patches_list


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def first_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters") from exc


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    common_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "use_flash_attn": args.flash_attn,
    }

    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map="auto",
            **common_kwargs,
        )
    elif args.quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map="auto",
            **common_kwargs,
        )
    else:
        model = AutoModel.from_pretrained(args.model, **common_kwargs).eval().cuda()

    return model.eval()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_batched_inputs(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    prompt: str,
    num_patches_list: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_context_token = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
    model.img_context_token_id = img_context_token_id

    queries = []
    for _ in range(batch_size):
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = "<img>" + img_context_token * model.num_image_token * num_patches + "</img>"
            query = query.replace("<image>", image_tokens, 1)
        queries.append(query)

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)
    batched_pixel_values = torch.cat([pixel_values] * batch_size, dim=0)
    return batched_pixel_values, input_ids, attention_mask


def run_once(
    model,
    tokenizer,
    pixel_values,
    prompt: str,
    generation_config: dict,
    num_patches_list: list[int],
    batch_size: int,
) -> list[str]:
    with torch.inference_mode():
        if batch_size == 1:
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                dict(generation_config),
                num_patches_list=num_patches_list,
            )
            return [response]

        batched_pixel_values, input_ids, attention_mask = build_batched_inputs(
            model, tokenizer, pixel_values, prompt, num_patches_list, batch_size
        )
        batched_generation_config = dict(generation_config)
        batched_generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids(
            model.conv_template.sep.strip()
        )
        generation_output = model.generate(
            pixel_values=batched_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **batched_generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return [response.split(model.conv_template.sep.strip())[0].strip() for response in responses]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this inside the Slurm GPU allocation.")

    dtype = dtype_from_name(args.dtype)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(
        f"Quantization: {args.quantization}, dtype: {args.dtype}, "
        f"max_tiles: {args.max_tiles}, batch_size: {args.batch_size}"
    )

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, use_fast=False
    )
    model = load_model(args, dtype)
    synchronize()
    load_seconds = time.perf_counter() - load_started

    device = first_model_device(model)
    pixel_values, num_patches_list = load_images(args.image, args.max_tiles)
    pixel_values = pixel_values.to(dtype=dtype, device=device)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
    }
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    print(f"Load time: {load_seconds:.2f}s")
    print(f"Batch size: {args.batch_size}")
    print(f"Input images: {len(num_patches_list)}")
    print(f"Input image tiles: {pixel_values.shape[0]} ({num_patches_list})")
    print(f"Prompt: {args.prompt!r}")

    for _ in range(args.warmup):
        run_once(
            model,
            tokenizer,
            pixel_values,
            args.prompt,
            generation_config,
            num_patches_list,
            args.batch_size,
        )
    synchronize()

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    token_counts = []
    last_responses = []

    for run_idx in range(1, args.runs + 1):
        started = time.perf_counter()
        last_responses = run_once(
            model,
            tokenizer,
            pixel_values,
            args.prompt,
            generation_config,
            num_patches_list,
            args.batch_size,
        )
        synchronize()
        elapsed = time.perf_counter() - started

        output_tokens = sum(
            len(tokenizer(response, add_special_tokens=False).input_ids)
            for response in last_responses
        )
        latencies.append(elapsed)
        token_counts.append(output_tokens)
        rate = output_tokens / elapsed if elapsed > 0 else float("nan")
        print(
            f"run={run_idx} latency={elapsed:.3f}s "
            f"batch_output_tokens={output_tokens} aggregate_output_tok_per_s={rate:.2f}"
        )

    total_tokens = sum(token_counts)
    total_seconds = sum(latencies)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print("\nSummary")
    print(f"avg_latency_s: {statistics.mean(latencies):.3f}")
    print(f"median_latency_s: {statistics.median(latencies):.3f}")
    print(f"avg_batch_output_tokens: {statistics.mean(token_counts):.1f}")
    print(f"avg_output_tokens_per_item: {statistics.mean(token_counts) / args.batch_size:.1f}")
    print(f"aggregate_output_tok_per_s: {total_tokens / total_seconds:.2f}")
    print(f"peak_cuda_allocated_gb: {peak_gb:.2f}")
    print(f"last_response_0: {last_responses[0] if last_responses else ''}")


if __name__ == "__main__":
    main()
