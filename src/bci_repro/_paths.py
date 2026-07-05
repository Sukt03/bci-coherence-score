from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PACKAGE_ROOT.parents[1]
PIPELINE_DIR = PACKAGE_ROOT / "pipelines"


def resolve_data_root(value: str | None) -> Path:
    """Resolve the external data root used by legacy pipeline scripts."""
    if value:
        return Path(value).expanduser().resolve()
    return BUNDLE_ROOT.parent.resolve()


def relative_to_data_root(path: str | Path, data_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return data_root / path
