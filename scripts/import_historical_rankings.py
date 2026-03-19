"""Download and normalize historical UFC rankings into local raw artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DATA_DIR  # noqa: E402


DEFAULT_URL = "https://andrewlor.me/demos/historical-ufc-rankings/data/rankings_history.json"


def _normalize_payload_rows(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_text, divisions in payload.items():
        if not isinstance(divisions, dict):
            continue
        for division, ranked_rows in divisions.items():
            if not isinstance(ranked_rows, list):
                continue
            for ranked in ranked_rows:
                if not isinstance(ranked, dict):
                    continue
                fighter = str(ranked.get("fighter", "") or "").strip()
                rank = ranked.get("rank")
                if not fighter or rank in (None, ""):
                    continue
                rows.append(
                    {
                        "date": date_text,
                        "weightclass": division,
                        "fighter": fighter,
                        "rank": rank,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.date
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame.dropna(subset=["date", "fighter", "weightclass", "rank"])
    frame = frame.sort_values(["date", "weightclass", "rank", "fighter"]).reset_index(drop=True)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=RAW_DATA_DIR / "rankings" / "rankings_history.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=RAW_DATA_DIR / "kaggle_rankings" / "rankings_history.csv",
    )
    args = parser.parse_args()

    response = requests.get(args.url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise SystemExit("Historical rankings payload was not a JSON object")

    normalized = _normalize_payload_rows(payload)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    normalized.to_csv(args.csv_out, index=False)

    print(
        {
            "source_url": args.url,
            "json_out": str(args.json_out),
            "csv_out": str(args.csv_out),
            "rows": int(len(normalized)),
            "min_date": normalized["date"].min().isoformat() if not normalized.empty else None,
            "max_date": normalized["date"].max().isoformat() if not normalized.empty else None,
            "divisions": int(normalized["weightclass"].nunique()) if not normalized.empty else 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
