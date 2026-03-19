"""Backfill historical method-odds gaps from BestFightOdds into a durable CSV."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DATA_DIR  # noqa: E402
from src.data.method_odds import (  # noqa: E402
    HEADERS,
    _parse_bfo_method_odds,
)
from src.data.ufc_refresh import _fight_key_series  # noqa: E402


OUTPUT_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds.csv"
_BFO_SEARCH_CACHE: dict[str, str] = {}


def _load_missing_fights(path: Path) -> pd.DataFrame:
    fights = pd.read_csv(path, parse_dates=["event_date"])
    required = {"event_date", "fighter_a", "fighter_b"}
    missing_cols = required - set(fights.columns)
    if missing_cols:
        raise SystemExit(f"Missing required columns in {path}: {sorted(missing_cols)}")
    optional_cols = [col for col in ["event", "event_name", "event_title"] if col in fights.columns]
    keep_cols = ["event_date", "fighter_a", "fighter_b", *optional_cols]
    fights = fights[keep_cols].dropna(subset=["event_date", "fighter_a", "fighter_b"]).drop_duplicates()
    fights = fights.sort_values(["event_date", "fighter_a", "fighter_b"]).reset_index(drop=True)
    return fights


def _event_title(row: pd.Series) -> str:
    for column in ("event", "event_name", "event_title"):
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    existing = pd.read_csv(path, parse_dates=["event_date"])
    if existing.empty:
        return existing
    existing["fight_key"] = _fight_key_series(existing)
    return existing


def _candidate_bfo_urls(fighter_a: str, fighter_b: str, event_title: str) -> list[str]:
    queries = [
        event_title,
        f"{event_title} {fighter_a} {fighter_b}" if event_title else "",
        " ".join(part for part in [fighter_a, fighter_b, event_title] if part),
        f"{fighter_a} {fighter_b}",
        fighter_a,
        fighter_b,
    ]

    urls: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = str(query or "").strip()
        if not query:
            continue
        html = _BFO_SEARCH_CACHE.get(query)
        if html is None:
            response = requests.get(
                "https://www.bestfightodds.com/search",
                params={"query": query},
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            html = response.text
            _BFO_SEARCH_CACHE[query] = html
        soup = BeautifulSoup(html, "lxml")
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            if "/events/" not in href and "/fighters/" not in href:
                continue
            full_url = href if href.startswith("http") else f"https://www.bestfightodds.com{href}"
            if full_url in seen:
                continue
            seen.add(full_url)
            urls.append(full_url)
    return urls


def _save(existing: pd.DataFrame, new_rows: list[dict], path: Path) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    frames = [frame for frame in (existing, new_df) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["fight_key"] = _fight_key_series(combined)
    combined = combined.dropna(subset=["fight_key"]).drop_duplicates(subset=["fight_key"], keep="last")
    combined = combined.drop(columns=["fight_key"], errors="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "coverage_audits" / "v4_144_test_after_rankings" / "method_odds_missing_fights.csv",
    )
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    fights = _load_missing_fights(args.inventory_csv)
    if args.from_date:
        fights = fights[fights["event_date"] >= pd.Timestamp(args.from_date)]
    if args.to_date:
        fights = fights[fights["event_date"] <= pd.Timestamp(args.to_date)]
    if args.limit is not None:
        fights = fights.head(args.limit)
    fights = fights.reset_index(drop=True)

    existing = _load_existing(args.output_path)
    existing_keys = set(existing["fight_key"].dropna()) if "fight_key" in existing.columns else set()

    pending = fights.copy()
    pending["fight_key"] = _fight_key_series(pending)
    pending = pending[~pending["fight_key"].isin(existing_keys)].reset_index(drop=True)

    print(
        {
            "inventory_csv": str(args.inventory_csv),
            "candidate_fights": int(len(fights)),
            "pending_fights": int(len(pending)),
            "output_path": str(args.output_path),
        }
    )
    if pending.empty:
        return 0

    new_rows: list[dict] = []
    matched = 0
    attempted = 0
    soup_cache: dict[str, BeautifulSoup] = {}
    for _, row in pending.iterrows():
        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]
        event_title = _event_title(row)
        attempted += 1

        source_url = None
        parsed = None
        try:
            candidate_urls = _candidate_bfo_urls(fighter_a, fighter_b, event_title)
        except Exception:
            candidate_urls = []
        for candidate_url in candidate_urls:
            try:
                soup = soup_cache.get(candidate_url)
                if soup is None:
                    response = requests.get(candidate_url, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "lxml")
                    soup_cache[candidate_url] = soup
                parsed = _parse_bfo_method_odds(soup, fighter_a, fighter_b)
            except Exception:
                parsed = None
            if parsed is not None:
                source_url = candidate_url
                break

        if parsed is None or source_url is None:
            continue

        matched += 1
        new_rows.append(
            {
                "event_date": pd.Timestamp(row["event_date"]).date().isoformat(),
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "event_title": event_title,
                "source": "bestfightodds",
                "source_url": source_url,
                "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                **parsed,
            }
        )

        if len(new_rows) % max(int(args.save_every), 1) == 0:
            existing = _save(existing, new_rows, args.output_path)
            existing_keys = set(_fight_key_series(existing).dropna())
            print(
                {
                    "attempted": attempted,
                    "matched": matched,
                    "saved_rows": int(len(existing)),
                }
            )
            new_rows = []

    final_df = _save(existing, new_rows, args.output_path)
    print(
        {
            "attempted": attempted,
            "matched": matched,
            "saved_rows_total": int(len(final_df)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
