"""Small I/O helpers for safe data artifact writes."""

from __future__ import annotations

import os
import tempfile
import json
import shutil
import time
from pathlib import Path

import pandas as pd

_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_SECONDS = 0.5


def _replace_with_retry(tmp_path: Path, target: Path) -> None:
    # On Windows, antivirus/indexer scans briefly hold freshly written files,
    # making os.replace fail with a transient PermissionError.
    delay = _REPLACE_BACKOFF_SECONDS
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp_path, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def write_csv_atomically(
    df: pd.DataFrame,
    path: Path | str,
    *,
    refuse_empty: bool = False,
) -> Path:
    """Write a CSV through a temp file and rename it into place.

    When ``refuse_empty`` is true, an empty dataframe is rejected before any
    existing artifact is replaced.
    """
    target = Path(path)
    if refuse_empty and df.empty:
        raise ValueError(f"Refusing to overwrite {target} with an empty dataframe")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def write_json_atomically(
    payload: object,
    path: Path | str,
    *,
    indent: int = 2,
) -> Path:
    """Write JSON through a temp file and rename it into place."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=indent) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def copy_file_atomically(src: Path | str, dst: Path | str) -> Path:
    """Copy a file through a temp file and rename it into place."""
    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp_path)
        _replace_with_retry(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target
