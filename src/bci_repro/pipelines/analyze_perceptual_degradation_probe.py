#!/usr/bin/env python3
"""Probe metric behavior under controlled perceptual degradations.

This analysis uses real ground-truth images from the VLM evaluation manifest,
creates deterministic degraded variants, and measures whether image metrics
track the known degradation severity. It is independent of T-PAS/T-SAS.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_PAIR_JSONL = Path("internvl3_eval_runs/full_both_reasoning_bs64_20260529_225955_repaired/pair_scores.jsonl")
DEFAULT_OUTPUT_DIR = Path("paper_analysis/perceptual_degradation_probe_20260601")
IMAGE_SIZE = 256


@dataclass(frozen=True)
class Degradation:
    name: str
    levels: tuple[float, ...]
    apply: Callable[[Image.Image, float], Image.Image]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-jsonl", type=Path, default=DEFAULT_PAIR_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all unique references.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--embedding-max-images", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_reference_paths(path: Path, max_images: int, seed: int) -> list[Path]:
    refs: dict[str, Path] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            reference_path = row.get("reference_path")
            if not reference_path:
                raise ValueError(f"Missing reference_path in {path}:{line_no}")
            ref = Path(reference_path)
            refs[str(ref)] = ref

    paths = sorted(refs.values(), key=lambda item: item.as_posix())
    if max_images and len(paths) > max_images:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, max_images), key=lambda item: item.as_posix())
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference images: {missing[:5]}")
    return paths


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def degrade_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def degrade_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    seed = int(float(sigma) * 100000) + int(arr[0, 0, 0] * 1000)
    rng = np.random.default_rng(seed)
    noisy = np.clip(arr + rng.normal(0.0, float(sigma), arr.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def degrade_downsample(image: Image.Image, scale: float) -> Image.Image:
    side = max(8, int(round(IMAGE_SIZE * float(scale))))
    small = image.resize((side, side), Image.Resampling.BICUBIC)
    return small.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)


def degrade_jpeg(image: Image.Image, quality: float) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade_color(image: Image.Image, saturation: float) -> Image.Image:
    return ImageEnhance.Color(image).enhance(float(saturation))


def degrade_shift(image: Image.Image, pixels: float) -> Image.Image:
    shift = int(round(float(pixels)))
    arr = np.asarray(image)
    padded = cv2.copyMakeBorder(arr, shift, shift, shift, shift, cv2.BORDER_REFLECT_101)
    shifted = padded[shift * 2 : shift * 2 + IMAGE_SIZE, shift * 2 : shift * 2 + IMAGE_SIZE]
    return Image.fromarray(shifted, mode="RGB")


DEGRADATIONS = [
    Degradation("Blur", (1.0, 2.0, 4.0, 8.0), degrade_blur),
    Degradation("Noise", (0.03, 0.06, 0.12, 0.24), degrade_noise),
    Degradation("Downsample", (0.75, 0.50, 0.25, 0.125), degrade_downsample),
    Degradation("JPEG", (60.0, 40.0, 25.0, 10.0), degrade_jpeg),
    Degradation("Color", (0.75, 0.50, 0.25, 0.0), degrade_color),
    Degradation("Shift", (4.0, 8.0, 16.0, 32.0), degrade_shift),
]


def as_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = a[:, :, channel]
        y = b[:, :, channel]
        mux = cv2.GaussianBlur(x, (11, 11), 1.5)
        muy = cv2.GaussianBlur(y, (11, 11), 1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mux2
        sigy2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - muy2
        sigxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - muxy
        numerator = (2 * muxy + c1) * (2 * sigxy + c2)
        denominator = (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        scores.append(np.mean(numerator / np.maximum(denominator, 1e-12)))
    return float(np.mean(scores))


def edge_cosine(a: np.ndarray, b: np.ndarray) -> float:
    gray_a = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_b = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_a_x = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
    grad_a_y = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
    grad_b_x = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
    grad_b_y = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
    vec_a = np.concatenate([grad_a_x.ravel(), grad_a_y.ravel()])
    vec_b = np.concatenate([grad_b_x.ravel(), grad_b_y.ravel()])
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return float(np.dot(vec_a, vec_b) / denom) if denom > 1e-12 else 0.0


def color_hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = (a * 255).astype(np.uint8)
    arr_b = (b * 255).astype(np.uint8)
    hist_a = cv2.calcHist([arr_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).astype(np.float32)
    hist_b = cv2.calcHist([arr_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).astype(np.float32)
    hist_a /= max(float(hist_a.sum()), 1e-12)
    hist_b /= max(float(hist_b.sum()), 1e-12)
    return float(np.minimum(hist_a, hist_b).sum())


LOW_LEVEL_METRICS = {
    "MSE": lambda a, b: mse(a, b),
    "PSNR": lambda a, b: psnr(a, b),
    "SSIM": lambda a, b: ssim(a, b),
    "Edge cosine": lambda a, b: edge_cosine(a, b),
    "Color hist.": lambda a, b: color_hist_intersection(a, b),
}

LOWER_IS_BETTER = {"MSE"}


def compute_low_level(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_index, path in enumerate(paths):
        base_image = load_image(path)
        base = as_float_array(base_image)
        for degradation in DEGRADATIONS:
            for severity_index, level in enumerate(degradation.levels, start=1):
                degraded_image = degradation.apply(base_image, level)
                degraded = as_float_array(degraded_image)
                for metric, fn in LOW_LEVEL_METRICS.items():
                    value = fn(base, degraded)
                    quality = -value if metric in LOWER_IS_BETTER else value
                    rows.append(
                        {
                            "image_index": image_index,
                            "image_path": str(path),
                            "degradation": degradation.name,
                            "severity_index": severity_index,
                            "level": level,
                            "metric": metric,
                            "value": value,
                            "quality": quality,
                        }
                    )
    return rows


def embed_images(model_name: str, images: list[Image.Image], batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import CLIPImageProcessor, CLIPModel, SiglipImageProcessor, SiglipModel

    def tensor_from_output(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        raise TypeError(f"Cannot extract tensor from {type(output)!r}")

    if model_name == "CLIP-L/14":
        model_id = "openai/clip-vit-large-patch14"
        processor = CLIPImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()

        def forward(batch: list[Image.Image]) -> torch.Tensor:
            inputs = processor(images=batch, return_tensors="pt").to(device)
            return tensor_from_output(model.get_image_features(**inputs))

    elif model_name == "SigLIP":
        model_id = "google/siglip-base-patch16-224"
        processor = SiglipImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = SiglipModel.from_pretrained(model_id, local_files_only=True).to(device).eval()

        def forward(batch: list[Image.Image]) -> torch.Tensor:
            inputs = processor(images=batch, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):
                return tensor_from_output(model.get_image_features(**inputs))
            return tensor_from_output(model.vision_model(**inputs))

    else:
        raise ValueError(model_name)

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            features = forward(images[start : start + batch_size]).float()
            features = torch.nn.functional.normalize(features, dim=-1)
            outputs.append(features.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_embedding_rows(paths: list[Path], max_images: int, batch_size: int, device: str) -> list[dict[str, Any]]:
    selected = paths[:max_images] if max_images else paths
    base_images = [load_image(path) for path in selected]
    variants: list[tuple[int, str, int, float, Image.Image]] = []
    for image_index, base_image in enumerate(base_images):
        for degradation in DEGRADATIONS:
            for severity_index, level in enumerate(degradation.levels, start=1):
                variants.append((image_index, degradation.name, severity_index, level, degradation.apply(base_image, level)))

    rows: list[dict[str, Any]] = []
    for metric in ("CLIP-L/14", "SigLIP"):
        base_embeddings = embed_images(metric, base_images, batch_size, device)
        variant_embeddings = embed_images(metric, [item[-1] for item in variants], batch_size, device)
        for variant_index, (image_index, degradation, severity_index, level, _image) in enumerate(variants):
            value = float(np.dot(base_embeddings[image_index], variant_embeddings[variant_index]))
            rows.append(
                {
                    "image_index": image_index,
                    "image_path": str(selected[image_index]),
                    "degradation": degradation,
                    "severity_index": severity_index,
                    "level": level,
                    "metric": metric,
                    "value": value,
                    "quality": value,
                }
            )
    return rows


def spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3:
        return float("nan")
    return float(frame.iloc[:, 0].rank(method="average").corr(frame.iloc[:, 1].rank(method="average"), method="pearson"))


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (metric, degradation), group in df.groupby(["metric", "degradation"], sort=False):
        rho = spearman(group["quality"], -group["severity_index"])
        first = group[group["severity_index"] == 1]["quality"].mean()
        last = group[group["severity_index"] == group["severity_index"].max()]["quality"].mean()
        summary_rows.append(
            {
                "metric": metric,
                "degradation": degradation,
                "spearman_severity": rho,
                "mean_quality_level1": float(first),
                "mean_quality_level4": float(last),
                "n_pairs": int(len(group)),
                "n_images": int(group["image_path"].nunique()),
            }
        )
    return pd.DataFrame(summary_rows)


def latex_escape(value: str) -> str:
    return value.replace("&", "\\&").replace("_", "\\_")


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["MSE", "PSNR", "SSIM", "Edge cosine", "Color hist.", "CLIP-L/14", "SigLIP"]
    degradations = [item.name for item in DEGRADATIONS]
    pivot = summary.pivot(index="metric", columns="degradation", values="spearman_severity")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{\\footnotesize Controlled perceptual degradation probe. Entries are Spearman correlations between oriented metric quality and negative degradation severity; higher is better.}",
        "\\label{tab:perceptual_degradation_probe}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Metric} & Blur & Noise & Downsample & JPEG & Color & Shift \\\\",
        "\\midrule",
    ]
    for metric in metrics:
        if metric not in pivot.index:
            continue
        values = []
        for degradation in degradations:
            value = pivot.loc[metric, degradation] if degradation in pivot.columns else np.nan
            values.append("--" if pd.isna(value) else f"{value:.3f}")
        lines.append(f"{latex_escape(metric)} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / "table_perceptual_degradation_probe.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = read_reference_paths(args.pair_jsonl, args.max_images, args.seed)
    rows = compute_low_level(paths)
    embedding_status: dict[str, Any] = {"requested": bool(args.include_embeddings), "status": "skipped"}
    if args.include_embeddings:
        try:
            rows.extend(compute_embedding_rows(paths, args.embedding_max_images, args.embedding_batch_size, args.device))
            embedding_status = {
                "requested": True,
                "status": "ok",
                "embedding_max_images": args.embedding_max_images,
                "embedding_batch_size": args.embedding_batch_size,
                "device": args.device,
            }
        except Exception as exc:
            embedding_status = {"requested": True, "status": "failed", "error": repr(exc)}

    raw = pd.DataFrame(rows)
    summary = summarize(rows)
    raw.to_csv(args.output_dir / "perceptual_degradation_raw.csv", index=False)
    summary.to_csv(args.output_dir / "perceptual_degradation_summary.csv", index=False)
    write_latex_table(summary, args.output_dir)
    metadata = {
        "pair_jsonl": str(args.pair_jsonl),
        "unique_reference_images_used": len(paths),
        "max_images": args.max_images,
        "seed": args.seed,
        "image_size": IMAGE_SIZE,
        "degradations": {item.name: item.levels for item in DEGRADATIONS},
        "embedding_status": embedding_status,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(summary.pivot(index="metric", columns="degradation", values="spearman_severity").round(3).to_string())


if __name__ == "__main__":
    main()
