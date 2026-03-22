"""Small I/O helpers for safe data artifact writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


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
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target
