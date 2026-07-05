from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from ._paths import relative_to_data_root, resolve_data_root
from .metric_utils import METRIC_COLUMNS

REF_PATH_COLUMNS = (
    "reference_path",
    "gt_path_resolved",
    "gt_zip_path",
    "original_gt_path",
    "gt_path",
)
GEN_PATH_COLUMNS = (
    "generated_path",
    "gen_path_resolved",
    "generated_zip_path",
    "original_generated_path",
    "gen_path",
)

BASE_COLUMNS = [
    "dataset_group",
    "model",
    "subject",
    "class_name",
    "pair_id",
    "selection_rank",
    "top2_rank",
    "candidate_index",
    "consensus_rank",
    "dreamsim_rank",
    "pieapp_rank",
    "topiq_fr_score",
    "pieapp_score",
    "gt_path_resolved",
    "gen_path_resolved",
    "mse",
    "psnr",
    "ssim",
    "dreamsim_score",
    "imagereward_score",
    "openclip_cosine",
    "dinov2_cosine",
    "lpips_alex",
    "dists",
    "blip_caption_gt",
    "blip_caption_gen",
    "blip_caption_gt_clean",
    "blip_caption_gen_clean",
    "blip_caption_sbert_model",
    "blip_caption_sbert_cosine",
    "warning",
    "expanded_metric_errors",
]

EXTRA_COLUMNS = [
    "detected_model",
    "duplicate_source_model",
    "class_name_or_stem",
    "stem",
    "source_zip",
    "pairing_status",
    "selected_by_consensus",
    "consensus_score",
    "import_dreamsim_score",
    "import_topiq_fr_score",
    "import_pieapp_score",
    "metric_errors",
]

OUTPUT_COLUMNS = BASE_COLUMNS[:]
for column in EXTRA_COLUMNS:
    if column not in OUTPUT_COLUMNS:
        OUTPUT_COLUMNS.append(column)
for column in METRIC_COLUMNS:
    if column not in OUTPUT_COLUMNS:
        OUTPUT_COLUMNS.append(column)

FAST_METRICS = {"mse", "psnr", "ssim", "openclip_cosine"}
ALL_METRICS = set(METRIC_COLUMNS)
CAPTION_COLUMNS = {
    "blip_caption_gt",
    "blip_caption_gen",
    "blip_caption_gt_clean",
    "blip_caption_gen_clean",
    "blip_caption_sbert_model",
    "blip_caption_sbert_cosine",
}


class BackendUnavailable(RuntimeError):
    """Raised when an optional metric backend is not installed or not cached."""


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    reference_path: Path
    generated_path: Path
    metadata: dict[str, Any]
    row_index: int


@dataclass
class Backend:
    name: str
    value: Any
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paper metric CSVs from GT/generated image pairs."
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric-set", choices=("fast", "all"), default="all")
    parser.add_argument("--metrics-config", default="configs/metrics.json")
    parser.add_argument("--model-revisions", default="configs/model_revisions.json")
    parser.add_argument("--cache-dir", default="outputs/metric_cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested metric backend is unavailable or a row errors.",
    )
    return parser.parse_args()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _first_present(row: dict[str, Any], columns: tuple[str, ...] | list[str]) -> str:
    for column in columns:
        value = _text(row.get(column)).strip()
        if value:
            return value
    return ""


def _parse_index(value: Any, prefix: str) -> str:
    text = _text(value).strip()
    if not text:
        return ""
    match = re.search(rf"{re.escape(prefix)}[-_ ]*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return ""


def _candidate_from_filename(filename: str) -> str:
    match = re.search(r"(?:cand|candidate)[-_]?(\d+)", filename, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _relative_path(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_image(value: str, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path


def _stable_id(parts: list[str]) -> str:
    clean = [part for part in parts if part]
    if clean:
        return "__".join(clean)
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"pair__{digest}"


def _normalize_metadata(row: dict[str, Any], row_index: int, data_root: Path) -> dict[str, Any]:
    model = _first_present(row, ["model", "method", "detected_model"])
    detected_model = _text(row.get("detected_model")).strip()
    subject = _first_present(row, ["subject", "sub"])
    if detected_model and not subject:
        subject = "extra"
    class_name = _first_present(row, ["class_name", "concept", "class_name_or_stem", "stem"])
    selection_rank = _first_present(row, ["selection_rank", "rank_num"])
    if not selection_rank:
        selection_rank = _parse_index(row.get("rank"), "rank")
    consensus_rank = _text(row.get("consensus_rank")).strip()
    candidate_index = _text(row.get("candidate_index")).strip()
    if not candidate_index:
        candidate_index = _parse_index(row.get("candidate"), "cand")
    if not candidate_index:
        candidate_index = _candidate_from_filename(_text(row.get("filename")))

    pair_id = _text(row.get("pair_id")).strip()
    if not pair_id:
        rank_token = f"rank{selection_rank or consensus_rank}" if (selection_rank or consensus_rank) else ""
        candidate_token = f"cand{candidate_index}" if candidate_index else ""
        row_token = f"row{_text(row.get('zip_row_index')).strip()}" if row.get("zip_row_index") is not None else ""
        pair_id = _stable_id([model or detected_model, subject, class_name, rank_token, candidate_token, row_token])

    stem = _first_present(row, ["stem"])
    filename = _text(row.get("filename")).strip()
    if not stem and filename:
        stem = Path(filename).stem

    return {
        "dataset_group": _first_present(row, ["dataset_group"]) or model or detected_model,
        "model": model,
        "subject": subject,
        "class_name": class_name,
        "pair_id": pair_id,
        "selection_rank": selection_rank,
        "top2_rank": _text(row.get("top2_rank")).strip(),
        "candidate_index": candidate_index,
        "consensus_rank": consensus_rank,
        "dreamsim_rank": _text(row.get("dreamsim_rank")).strip(),
        "pieapp_rank": _text(row.get("pieapp_rank")).strip(),
        "warning": _text(row.get("warning")).strip(),
        "detected_model": detected_model,
        "duplicate_source_model": _text(row.get("duplicate_source_model")).strip(),
        "class_name_or_stem": _first_present(row, ["class_name_or_stem"]) or class_name,
        "stem": stem,
        "source_zip": _text(row.get("source_zip")).strip(),
        "pairing_status": _text(row.get("pairing_status")).strip(),
        "selected_by_consensus": _text(row.get("selected_by_consensus")).strip(),
        "consensus_score": _text(row.get("consensus_score")).strip(),
        "import_dreamsim_score": _text(row.get("import_dreamsim_score") or row.get("dreamsim_score")).strip(),
        "import_topiq_fr_score": _text(row.get("import_topiq_fr_score") or row.get("topiq_fr_score")).strip(),
        "import_pieapp_score": _text(row.get("import_pieapp_score") or row.get("pieapp_score")).strip(),
        "source_manifest_index": row_index,
        "_data_root": str(data_root),
    }


def _read_json_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "pairs" in payload and isinstance(payload["pairs"], list):
            return payload["pairs"]
        if "rows" in payload and isinstance(payload["rows"], list):
            return payload["rows"]
    if not isinstance(payload, list):
        raise ValueError(f"JSON manifest must be a list or contain a pairs/rows list: {path}")
    return payload


def _read_jsonl_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object: {path}")
                rows.append(value)
    return rows


def _read_csv_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_pair_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json_manifest(path)
    if suffix == ".jsonl":
        return _read_jsonl_manifest(path)
    if suffix == ".csv":
        return _read_csv_manifest(path)
    raise ValueError(f"Unsupported manifest extension: {path.suffix}")


def read_pairs(path: Path, data_root: Path) -> list[ImagePair]:
    rows = read_pair_rows(path)
    pairs: list[ImagePair] = []
    for row_index, row in enumerate(rows):
        reference = _first_present(row, REF_PATH_COLUMNS)
        generated = _first_present(row, GEN_PATH_COLUMNS)
        if not reference or not generated:
            raise ValueError(
                f"Manifest row {row_index} lacks reference/generated paths. "
                f"Accepted reference columns: {', '.join(REF_PATH_COLUMNS)}; "
                f"generated columns: {', '.join(GEN_PATH_COLUMNS)}"
            )
        metadata = _normalize_metadata(row, row_index=row_index, data_root=data_root)
        pairs.append(
            ImagePair(
                pair_id=metadata["pair_id"],
                reference_path=_resolve_image(reference, data_root),
                generated_path=_resolve_image(generated, data_root),
                metadata=metadata,
                row_index=row_index,
            )
        )
    return pairs


def clean_caption(caption: str) -> str:
    caption = caption.lower().strip()
    caption = caption.replace("-", " ")
    caption = caption.translate(str.maketrans({char: " " for char in string.punctuation}))
    caption = re.sub(r"\s+", " ", caption)
    return caption.strip()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_array(path: Path, size: int) -> np.ndarray:
    image = load_rgb(path)
    image = ImageOps.fit(
        image,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def mse_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr_score(a: np.ndarray, b: np.ndarray) -> float:
    value = mse_score(a, b)
    if value <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))
    except Exception:
        # Deterministic fallback used for smoke tests when scikit-image is absent.
        c1 = 0.01**2
        c2 = 0.03**2
        values = []
        for channel in range(3):
            x = a[:, :, channel]
            y = b[:, :, channel]
            mu_x = float(x.mean())
            mu_y = float(y.mean())
            sigma_x = float(x.var())
            sigma_y = float(y.var())
            sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())
            numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            values.append(numerator / denominator if denominator else 0.0)
        return float(np.mean(values))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cache_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _tensor_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _json_error(metric: str, exc: BaseException) -> str:
    return f"{metric}: {type(exc).__name__}: {exc}"


class MetricRunner:
    def __init__(
        self,
        metric_set: str,
        image_size: int,
        cache_dir: Path,
        device: str | None,
        local_files_only: bool,
        model_revisions: dict[str, Any],
    ) -> None:
        self.metric_set = metric_set
        self.requested = FAST_METRICS if metric_set == "fast" else ALL_METRICS
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or self._default_device()
        self.local_files_only = local_files_only
        self.model_revisions = model_revisions
        self.backends: dict[str, Backend] = {}

    def _default_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def wants(self, metric: str) -> bool:
        return metric in self.requested

    def _hf_model(self, key: str, default: str) -> tuple[str, str | None]:
        models = self.model_revisions.get("models", {})
        entry = models.get(key, {})
        return entry.get("model_id", default), entry.get("revision")

    def _backend(self, name: str, loader: Callable[[], Any]) -> Backend:
        if name in self.backends:
            return self.backends[name]
        try:
            backend = Backend(name=name, value=loader())
        except Exception as exc:
            backend = Backend(name=name, value=None, error=f"{type(exc).__name__}: {exc}")
        self.backends[name] = backend
        return backend

    def _embedding_cache_path(self, model_key: str, image_path: Path) -> Path:
        revision = self.model_revisions.get("models", {}).get(model_key, {}).get("revision", "")
        key = cache_key(f"{model_key}|{revision}|{image_path.resolve()}")
        return self.cache_dir / "embeddings" / model_key / f"{key}.npy"

    def _caption_cache_path(self, image_path: Path) -> Path:
        key = cache_key(str(image_path.resolve()))
        return self.cache_dir / "captions" / f"{key}.json"

    def _text_cache_path(self, text: str) -> Path:
        key = cache_key(text)
        return self.cache_dir / "text_embeddings" / f"{key}.npy"

    def _pair_cache_path(self, metric: str, reference_path: Path, generated_path: Path) -> Path:
        key = cache_key(f"{metric}|{reference_path.resolve()}|{generated_path.resolve()}")
        return self.cache_dir / "pair_metrics" / metric / f"{key}.json"

    def _torch_image_tensor(self, path: Path, size: int | None = None, minus_one_to_one: bool = True) -> Any:
        import torch

        image_size = size or self.image_size
        array = resize_array(path, image_size)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        if minus_one_to_one:
            tensor = tensor * 2.0 - 1.0
        return tensor.to(self.device)

    def _low_level(self, row: dict[str, Any], pair: ImagePair, errors: list[str]) -> None:
        try:
            reference = resize_array(pair.reference_path, self.image_size)
            generated = resize_array(pair.generated_path, self.image_size)
            row["mse"] = mse_score(reference, generated)
            row["psnr"] = psnr_score(reference, generated)
            row["ssim"] = ssim_score(reference, generated)
        except Exception as exc:
            errors.append(_json_error("low_level", exc))

    def _openclip_backend(self) -> Any:
        import torch

        def transformers_backend(model_key: str) -> Any:
            from transformers import CLIPImageProcessor, CLIPModel

            model_id, revision = self._hf_model(model_key, "laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            kwargs = {"local_files_only": self.local_files_only}
            if revision:
                kwargs["revision"] = revision
            processor = CLIPImageProcessor.from_pretrained(model_id, **kwargs)
            model = CLIPModel.from_pretrained(model_id, **kwargs).to(self.device)
            model.eval()
            return ("transformers_clip", model, processor, torch)

        if self.local_files_only:
            return transformers_backend("openclip")

        try:
            import open_clip

            model_name = self.model_revisions.get("models", {}).get("openclip", {}).get("open_clip_model", "ViT-L-14")
            pretrained = self.model_revisions.get("models", {}).get("openclip", {}).get(
                "open_clip_pretrained", "laion2b_s32b_b82k"
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
            model.eval()
            return ("open_clip", model, preprocess, torch)
        except Exception:
            try:
                return transformers_backend("openclip")
            except Exception:
                return transformers_backend("openclip_fallback")

    def _openclip_embedding(self, image_path: Path) -> np.ndarray:
        cache_path = self._embedding_cache_path("openclip", image_path)
        if cache_path.exists():
            return np.load(cache_path)
        backend = self._backend("openclip", self._openclip_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "OpenCLIP unavailable")
        kind, model, preprocess, torch = backend.value
        image = load_rgb(image_path)
        with torch.inference_mode():
            if kind == "open_clip":
                tensor = preprocess(image).unsqueeze(0).to(self.device)
                features = model.encode_image(tensor)
            else:
                inputs = preprocess(images=image, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        array = features[0].detach().cpu().float().numpy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, array)
        return array

    def _dinov2_backend(self) -> Any:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model_id, revision = self._hf_model("dinov2", "facebook/dinov2-large")
        kwargs = {"local_files_only": self.local_files_only}
        if revision:
            kwargs["revision"] = revision
        processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device)
        model.eval()
        return model, processor, torch

    def _dinov2_embedding(self, image_path: Path) -> np.ndarray:
        cache_path = self._embedding_cache_path("dinov2", image_path)
        if cache_path.exists():
            return np.load(cache_path)
        backend = self._backend("dinov2", self._dinov2_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "DINOv2 unavailable")
        model, processor, torch = backend.value
        image = load_rgb(image_path)
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
            features = features / features.norm(dim=-1, keepdim=True)
        array = features[0].detach().cpu().float().numpy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, array)
        return array

    def _dreamsim_backend(self) -> Any:
        import torch
        from dreamsim import dreamsim

        model, preprocess = dreamsim(pretrained=True, device=self.device)
        model.eval()
        return model, preprocess, torch

    def _dreamsim_distance(self, reference_path: Path, generated_path: Path) -> float:
        cache_path = self._pair_cache_path("dreamsim", reference_path, generated_path)
        if cache_path.exists():
            return float(json.loads(cache_path.read_text(encoding="utf-8"))["value"])
        backend = self._backend("dreamsim", self._dreamsim_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "DreamSim unavailable")
        model, preprocess, torch = backend.value
        with torch.inference_mode():
            reference = preprocess(load_rgb(reference_path)).to(self.device)
            generated = preprocess(load_rgb(generated_path)).to(self.device)
            value = _tensor_to_float(model(reference, generated))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"value": value}), encoding="utf-8")
        return value

    def _lpips_backend(self) -> Any:
        import lpips

        model = lpips.LPIPS(net="alex").to(self.device)
        model.eval()
        return model

    def _lpips_distance(self, reference_path: Path, generated_path: Path) -> float:
        backend = self._backend("lpips_alex", self._lpips_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "LPIPS unavailable")
        import torch

        with torch.inference_mode():
            reference = self._torch_image_tensor(reference_path, minus_one_to_one=False)
            generated = self._torch_image_tensor(generated_path, minus_one_to_one=False)
            return _tensor_to_float(backend.value(reference, generated))

    def _dists_backend(self) -> Any:
        try:
            from DISTS_pytorch import DISTS
        except Exception:
            from dists_pytorch import DISTS

        model = DISTS().to(self.device)
        model.eval()
        return model

    def _dists_distance(self, reference_path: Path, generated_path: Path) -> float:
        backend = self._backend("dists", self._dists_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "DISTS unavailable")
        import torch

        with torch.inference_mode():
            reference = self._torch_image_tensor(reference_path)
            generated = self._torch_image_tensor(generated_path)
            return _tensor_to_float(backend.value(reference, generated))

    def _pyiqa_backend(self, metric_name: str) -> Any:
        import pyiqa

        metric = pyiqa.create_metric(metric_name, device=self.device)
        return metric

    def _pyiqa_score(self, metric_name: str, reference_path: Path, generated_path: Path) -> float:
        cache_path = self._pair_cache_path(metric_name, reference_path, generated_path)
        if cache_path.exists():
            return float(json.loads(cache_path.read_text(encoding="utf-8"))["value"])
        backend = self._backend(metric_name, lambda: self._pyiqa_backend(metric_name))
        if not backend.available:
            raise BackendUnavailable(backend.error or f"{metric_name} unavailable")
        value = _tensor_to_float(backend.value(str(generated_path), str(reference_path)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"value": value}), encoding="utf-8")
        return value

    def _blip_backend(self) -> Any:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_id, revision = self._hf_model("blip_captioner", "Salesforce/blip-image-captioning-base")
        kwargs = {"local_files_only": self.local_files_only}
        if revision:
            kwargs["revision"] = revision
        processor = BlipProcessor.from_pretrained(model_id, **kwargs)
        model = BlipForConditionalGeneration.from_pretrained(model_id, **kwargs).to(self.device)
        model.eval()
        return model, processor, torch

    def caption(self, image_path: Path) -> str:
        cache_path = self._caption_cache_path(image_path)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["caption"]
        backend = self._backend("blip_captioner", self._blip_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "BLIP captioner unavailable")
        model, processor, torch = backend.value
        image = load_rgb(image_path)
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=32)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"caption": caption}), encoding="utf-8")
        return caption

    def _sbert_backend(self) -> Any:
        from sentence_transformers import SentenceTransformer

        model_id, revision = self._hf_model("sbert", "sentence-transformers/all-MiniLM-L6-v2")
        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if self.local_files_only:
            kwargs["local_files_only"] = True
        model = SentenceTransformer(model_id, device=self.device, **kwargs)
        return model

    def _text_embedding(self, text: str) -> np.ndarray:
        cache_path = self._text_cache_path(text)
        if cache_path.exists():
            return np.load(cache_path)
        backend = self._backend("sbert", self._sbert_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "SBERT unavailable")
        embedding = backend.value.encode(text, normalize_embeddings=True)
        array = np.asarray(embedding, dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, array)
        return array

    def _imagereward_backend(self) -> Any:
        import ImageReward as RM

        model = RM.load("ImageReward-v1.0")
        return model

    def _imagereward_score(self, prompt: str, generated_path: Path) -> float:
        backend = self._backend("imagereward", self._imagereward_backend)
        if not backend.available:
            raise BackendUnavailable(backend.error or "ImageReward unavailable")
        return float(backend.value.score(prompt, str(generated_path)))

    def compute_pair(self, pair: ImagePair) -> dict[str, Any]:
        row = {column: "" for column in OUTPUT_COLUMNS}
        for key, value in pair.metadata.items():
            if key in row:
                row[key] = value
        row["pair_id"] = pair.pair_id
        row["gt_path_resolved"] = _relative_path(pair.reference_path, Path(pair.metadata["_data_root"]))
        row["gen_path_resolved"] = _relative_path(pair.generated_path, Path(pair.metadata["_data_root"]))

        errors: list[str] = []
        if any(self.wants(metric) for metric in ("mse", "psnr", "ssim")):
            self._low_level(row, pair, errors)

        metric_jobs: list[tuple[str, Callable[[], float]]] = [
            (
                "openclip_cosine",
                lambda: cosine(self._openclip_embedding(pair.reference_path), self._openclip_embedding(pair.generated_path)),
            ),
            (
                "dinov2_cosine",
                lambda: cosine(self._dinov2_embedding(pair.reference_path), self._dinov2_embedding(pair.generated_path)),
            ),
            ("dreamsim_score", lambda: self._dreamsim_distance(pair.reference_path, pair.generated_path)),
            ("lpips_alex", lambda: self._lpips_distance(pair.reference_path, pair.generated_path)),
            ("dists", lambda: self._dists_distance(pair.reference_path, pair.generated_path)),
            ("topiq_fr_score", lambda: self._pyiqa_score("topiq_fr", pair.reference_path, pair.generated_path)),
            ("pieapp_score", lambda: self._pyiqa_score("pieapp", pair.reference_path, pair.generated_path)),
        ]
        for metric, func in metric_jobs:
            if self.wants(metric):
                try:
                    row[metric] = func()
                except Exception as exc:
                    errors.append(_json_error(metric, exc))

        need_captions = self.wants("blip_caption_sbert_cosine") or self.wants("imagereward_score")
        if need_captions:
            try:
                gt_caption = self.caption(pair.reference_path)
                gen_caption = self.caption(pair.generated_path)
                row["blip_caption_gt"] = gt_caption
                row["blip_caption_gen"] = gen_caption
                row["blip_caption_gt_clean"] = clean_caption(gt_caption)
                row["blip_caption_gen_clean"] = clean_caption(gen_caption)
            except Exception as exc:
                errors.append(_json_error("blip_captioner", exc))

        if self.wants("blip_caption_sbert_cosine") and row["blip_caption_gt_clean"] and row["blip_caption_gen_clean"]:
            try:
                row["blip_caption_sbert_model"] = self.model_revisions.get("models", {}).get("sbert", {}).get(
                    "model_id", "sentence-transformers/all-MiniLM-L6-v2"
                )
                row["blip_caption_sbert_cosine"] = cosine(
                    self._text_embedding(row["blip_caption_gt_clean"]),
                    self._text_embedding(row["blip_caption_gen_clean"]),
                )
            except Exception as exc:
                errors.append(_json_error("blip_caption_sbert_cosine", exc))

        if self.wants("imagereward_score"):
            try:
                prompt = row["blip_caption_gt_clean"] or row["class_name"] or row["class_name_or_stem"]
                if not prompt:
                    raise ValueError("no prompt available from GT caption or class name")
                row["imagereward_score"] = self._imagereward_score(prompt, pair.generated_path)
            except Exception as exc:
                errors.append(_json_error("imagereward_score", exc))

        if errors:
            row["metric_errors"] = json.dumps(errors, sort_keys=True)
            row["expanded_metric_errors"] = json.dumps(errors, sort_keys=True)
        return row


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_metric_config(path: Path) -> dict[str, Any]:
    return load_json(path, default={})


def completed_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {_text(row.get("pair_id")).strip() for row in reader if _text(row.get("pair_id")).strip()}


def write_rows(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    manifest = relative_to_data_root(args.manifest, data_root)
    output = relative_to_data_root(args.out, data_root)
    metrics_config = relative_to_data_root(args.metrics_config, Path.cwd())
    revisions_config = relative_to_data_root(args.model_revisions, Path.cwd())
    cache_dir = relative_to_data_root(args.cache_dir, Path.cwd())

    # Load now so invalid JSON fails early. The metric metadata is also used by tests
    # and downstream documentation, while the code keeps backend defaults for safety.
    load_metric_config(metrics_config)
    model_revisions = load_json(revisions_config, default={})

    pairs = read_pairs(manifest, data_root)
    if args.limit:
        pairs = pairs[: args.limit]

    skip_ids = completed_pair_ids(output) if args.resume else set()
    runner = MetricRunner(
        metric_set=args.metric_set,
        image_size=args.image_size,
        cache_dir=cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        model_revisions=model_revisions,
    )
    rows: list[dict[str, Any]] = []
    failed_rows = 0
    for pair in pairs:
        if pair.pair_id in skip_ids:
            continue
        row = runner.compute_pair(pair)
        if row.get("metric_errors") or row.get("expanded_metric_errors"):
            failed_rows += 1
        rows.append(row)

    write_rows(output, rows, append=args.resume)
    summary = {
        "out": str(output),
        "metric_set": args.metric_set,
        "rows_written": len(rows),
        "rows_skipped_resume": len(skip_ids),
        "rows_with_errors": failed_rows,
        "device": runner.device,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and failed_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
