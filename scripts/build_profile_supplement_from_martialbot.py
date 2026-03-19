"""
Build or extend the local supplemental fighter-profile artifact from MartialBot.

This is a URL-map-driven helper for known MartialBot pages.
For general gap recovery, prefer the broader external-profile supplement builder,
which now uses MartialBot's public search endpoint directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.name_utils import normalize_person_name  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_profile_supplement.csv"
DEFAULT_URL_MAP = REPO_ROOT / "data" / "raw" / "martialbot_fighter_urls.csv"
TARGET_FIELDS = ("reach", "dob")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "N/A", "nan", "??"}


def _load_existing_rows(output_path: Path) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    existing_df = pd.read_csv(output_path)
    if existing_df.empty:
        return []
    return existing_df.to_dict(orient="records")


def _build_effective_profile_state(
    scraped_df: pd.DataFrame,
    existing_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    state: dict[str, dict[str, object]] = {}

    def _merge_row(row: dict[str, object] | pd.Series) -> None:
        name_key = normalize_person_name(row.get("name"))
        if not name_key:
            return
        profile = state.setdefault(name_key, {field: "" for field in TARGET_FIELDS})
        for field in TARGET_FIELDS:
            if _blank(profile.get(field)) and not _blank(row.get(field)):
                profile[field] = row.get(field)

    for _, row in scraped_df.iterrows():
        _merge_row(row)
    for row in existing_rows:
        _merge_row(row)
    return state


def _normalize_existing_source_keys(existing_rows: list[dict[str, object]]) -> set[tuple[str, str]]:
    return {
        (normalize_person_name(row.get("name")), str(row.get("source", "")).strip().lower())
        for row in existing_rows
        if not _blank(row.get("name")) and not _blank(row.get("source"))
    }


def _parse_martialbot_page(fighter_url: str) -> dict[str, str]:
    response = requests.get(fighter_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    name = ""
    heading = soup.find("h1")
    if heading:
        name = " ".join(heading.stripped_strings)

    details: dict[str, str] = {}
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        key = " ".join(dt.stripped_strings)
        value = " ".join(dd.stripped_strings)
        details[key] = value

    return {
        "name": name,
        "reach": details.get("Reach", ""),
        "dob": details.get("Born", ""),
    }


def _build_row(
    mapping_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(mapping_row.get("name", "") or "").strip()
    fighter_url = str(mapping_row.get("fighter_url", "") or "").strip()
    if not fighter_name or not fighter_url:
        return None

    current_profile = current_state.setdefault(
        normalize_person_name(fighter_name),
        {field: "" for field in TARGET_FIELDS},
    )
    profile = _parse_martialbot_page(fighter_url)
    row = {
        "name": fighter_name,
        "source": "martialbot",
        "source_name": profile.get("name", ""),
        "search_name": fighter_name,
        "fighter_url": fighter_url,
        "height": "",
        "reach": "",
        "weight": "",
        "stance": "",
        "dob": "",
    }

    recovered_any = False
    if _blank(current_profile.get("reach")) and not _blank(profile.get("reach")):
        row["reach"] = profile["reach"]
        recovered_any = True
    if _blank(current_profile.get("dob")) and not _blank(profile.get("dob")):
        row["dob"] = profile["dob"]
        recovered_any = True
    return row if recovered_any else None


def _update_state_from_row(state: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    fighter_key = normalize_person_name(row.get("name"))
    if not fighter_key:
        return
    profile = state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    for field in TARGET_FIELDS:
        if _blank(profile.get(field)) and not _blank(row.get(field)):
            profile[field] = row.get(field)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--url-map-csv", type=Path, default=DEFAULT_URL_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scraped_df = pd.read_csv(args.scraped_fighters_path)
    url_map_df = pd.read_csv(args.url_map_csv)
    if args.limit is not None:
        url_map_df = url_map_df.head(args.limit).copy()

    existing_rows = _load_existing_rows(args.output)
    existing_source_keys = _normalize_existing_source_keys(existing_rows)
    current_state = _build_effective_profile_state(scraped_df, existing_rows)

    results: list[dict[str, object]] = []
    attempted = 0
    for _, row in url_map_df.iterrows():
        fighter_key = normalize_person_name(row.get("name"))
        if (fighter_key, "martialbot") in existing_source_keys:
            continue
        attempted += 1
        supplement_row = _build_row(row, current_state)
        if supplement_row is None:
            continue
        results.append(supplement_row)
        existing_source_keys.add((fighter_key, "martialbot"))
        _update_state_from_row(current_state, supplement_row)

    combined_rows = existing_rows + results
    output_df = pd.DataFrame(combined_rows)
    if not output_df.empty:
        output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    summary = {
        "candidate_rows": int(len(url_map_df)),
        "attempted_rows": attempted,
        "recovered_rows": int(len(results)),
        "output_path": str(args.output),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
