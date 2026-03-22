"""Sync the official UFC active roster into a canonical local CSV artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster


def _summary(df: pd.DataFrame, *, output_path: Path) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "resolved_ufcstats_urls": int(df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "resolved_ufcstats_names": int(df.get("ufcstats_name", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "with_profile_details": int(df.get("profile_status", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "output_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OFFICIAL_ACTIVE_ROSTER_PATH)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--skip-profile-details", action="store_true")
    parser.add_argument("--skip-ufcstats-resolution", action="store_true")
    args = parser.parse_args()

    df = sync_official_active_roster(
        output_path=args.output,
        fetch_profile_details=not args.skip_profile_details,
        max_pages=args.max_pages,
        resolve_ufcstats=not args.skip_ufcstats_resolution,
    )
    print(json.dumps(_summary(df, output_path=args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
