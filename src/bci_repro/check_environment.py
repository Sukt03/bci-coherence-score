from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any

from ._paths import BUNDLE_ROOT, relative_to_data_root

CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("Pillow", "PIL"),
    ("opencv-python-headless", "cv2"),
    ("scikit-image", "skimage"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("sentence-transformers", "sentence_transformers"),
    ("timm", "timm"),
    ("huggingface-hub", "huggingface_hub"),
]

OPTIONAL_BACKENDS = {
    "openclip_cosine": [("open-clip-torch", "open_clip")],
    "dreamsim_score": [("dreamsim", "dreamsim")],
    "lpips_alex": [("lpips", "lpips")],
    "dists": [("DISTS-pytorch", "DISTS_pytorch"), ("DISTS-pytorch", "dists_pytorch")],
    "topiq_fr_score": [("pyiqa", "pyiqa")],
    "pieapp_score": [("pyiqa", "pyiqa")],
    "imagereward_score": [("ImageReward", "ImageReward")],
    "blip_caption_sbert_cosine": [
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report package, CUDA, and model-lock status.")
    parser.add_argument("--model-revisions", default="configs/model_revisions.json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument(
        "--strict-full",
        action="store_true",
        help="Exit nonzero if any full-metric optional backend is unavailable.",
    )
    return parser.parse_args()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_package(package: str, module: str) -> dict[str, Any]:
    return {
        "package": package,
        "module": module,
        "version": _package_version(package),
        "importable": _has_module(module),
    }


def _check_any(candidates: list[tuple[str, str]]) -> dict[str, Any]:
    checks = [_check_package(package, module) for package, module in candidates]
    return {
        "available": any(check["importable"] for check in checks),
        "candidates": checks,
    }


def _cuda_info() -> dict[str, Any]:
    if not _has_module("torch"):
        return {"torch_importable": False}
    import torch

    info: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return info


def _load_model_revisions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "models": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {}) if isinstance(payload, dict) else {}
    return {
        "path": str(path),
        "exists": True,
        "model_count": len(models),
        "models": {
            key: {
                "model_id": value.get("model_id"),
                "revision": value.get("revision"),
                "source": value.get("source"),
            }
            for key, value in models.items()
            if isinstance(value, dict)
        },
    }


def build_report(model_revisions_path: Path) -> dict[str, Any]:
    core = [_check_package(package, module) for package, module in CORE_PACKAGES]
    optional = {metric: _check_any(candidates) for metric, candidates in OPTIONAL_BACKENDS.items()}
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "core_packages": core,
        "optional_metric_backends": optional,
        "cuda": _cuda_info(),
        "model_revisions": _load_model_revisions(model_revisions_path),
    }


def _print_human(report: dict[str, Any]) -> None:
    print("Environment")
    print(f"  Python: {report['python']['implementation']} {report['python']['version']}")
    print(f"  Platform: {report['python']['platform']}")
    print("Core packages")
    for item in report["core_packages"]:
        status = "ok" if item["importable"] else "missing"
        version = item["version"] or "unknown"
        print(f"  {item['package']}: {status} ({version})")
    cuda = report["cuda"]
    print("CUDA")
    print(f"  torch importable: {cuda.get('torch_importable')}")
    print(f"  available: {cuda.get('cuda_available')}")
    print(f"  cuda version: {cuda.get('cuda_version')}")
    print(f"  devices: {cuda.get('devices', [])}")
    print("Optional metric backends")
    for metric, item in report["optional_metric_backends"].items():
        status = "ok" if item["available"] else "missing"
        names = ", ".join(candidate["module"] for candidate in item["candidates"])
        print(f"  {metric}: {status} ({names})")
    revisions = report["model_revisions"]
    print("Model revisions")
    print(f"  config: {revisions['path']} exists={revisions['exists']}")
    print(f"  models: {revisions.get('model_count', 0)}")


def main() -> None:
    args = parse_args()
    revisions_path = relative_to_data_root(args.model_revisions, BUNDLE_ROOT)
    report = build_report(revisions_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    missing_core = [item["package"] for item in report["core_packages"] if not item["importable"]]
    missing_optional = [
        metric for metric, item in report["optional_metric_backends"].items() if not item["available"]
    ]
    if missing_core or (args.strict_full and missing_optional):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
