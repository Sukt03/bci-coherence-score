from __future__ import annotations

import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._paths import PIPELINE_DIR


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_pipeline(script_name: str, argv: list[str], data_root: Path) -> None:
    """Execute a packaged pipeline script as if it were run from data_root."""
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(script_path), *argv]
        sys.path.insert(0, str(PIPELINE_DIR))
        sys.path.insert(0, str(data_root))
        with pushd(data_root):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path

