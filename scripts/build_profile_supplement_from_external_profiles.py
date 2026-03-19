"""
Build or extend the local supplemental fighter-profile artifact from public sources.

This focuses on the remaining recoverable V4 profile gaps without fabricating data.
It currently uses:
- MartialBot for direct fighter-page reach and exact DOB when available
- Tapology for direct fighter-page reach and exact DOB when available
- Wikipedia infoboxes for reach and exact DOB when available
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fallback_scrapers import (  # noqa: E402
    clear_fallback_cache,
    scrape_fightdx_profile,
    scrape_martialbot_profile,
    scrape_tapology_profile,
    search_fightdx,
    search_martialbot,
    search_tapology,
)
from src.data.name_utils import normalize_person_name, same_person_name  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_profile_supplement.csv"
TARGET_FIELDS = ("reach", "dob")
TARGET_GAP_FIELDS = {"reach", "age"}
MARTIALBOT_SEARCH_NAME_OVERRIDES = {
    "tsuyoshi kohsaka": "Tsuyoshi Kosaka",
}
TAPOLOGY_SEARCH_NAME_OVERRIDES = {
    "felix lee mitchell": "Felix Mitchell",
}
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {
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
    return str(value).strip() in {"", "-", "--", "N/A", "nan", "??"}


def _load_candidates(scraped_fighters_path: Path, gap_audit_csv: Path | None = None) -> pd.DataFrame:
    scraped_df = pd.read_csv(scraped_fighters_path)
    if scraped_df.empty or "name" not in scraped_df.columns:
        raise SystemExit(f"Scraped fighter profile artifact is empty or missing 'name': {scraped_fighters_path}")

    if gap_audit_csv is not None:
        gap_df = pd.read_csv(gap_audit_csv)
        names = sorted(
            {
                row.get("fighter_name")
                for _, row in gap_df.iterrows()
                if row.get("field") in TARGET_GAP_FIELDS and not _blank(row.get("fighter_name"))
            }
        )
        return scraped_df[scraped_df["name"].isin(names)].copy()

    mask = pd.Series(False, index=scraped_df.index)
    for field in TARGET_FIELDS:
        if field not in scraped_df.columns:
            continue
        mask = mask | scraped_df[field].apply(_blank)
    return scraped_df[mask].copy()


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


def _build_base_row(
    fighter_name: str,
    *,
    source: str,
    source_name: str,
    search_name: str,
    fighter_url: str,
) -> dict[str, object]:
    return {
        "name": fighter_name,
        "source": source,
        "source_name": source_name,
        "search_name": search_name,
        "fighter_url": fighter_url,
        "height": "",
        "reach": "",
        "weight": "",
        "stance": "",
        "dob": "",
    }


def _update_state_from_row(state: dict[str, dict[str, object]], row: dict[str, object]) -> None:
    fighter_key = normalize_person_name(row.get("name"))
    if not fighter_key:
        return
    profile = state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    for field in TARGET_FIELDS:
        if _blank(profile.get(field)) and not _blank(row.get(field)):
            profile[field] = row.get(field)


def _build_tapology_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name = TAPOLOGY_SEARCH_NAME_OVERRIDES.get(fighter_key, fighter_name)
    fighter_url = search_tapology(search_name)
    if not fighter_url:
        return None

    profile = scrape_tapology_profile(fighter_url)
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="tapology",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = False
    if _blank(current_profile.get("reach")) and not _blank(profile.get("reach_raw")):
        supplement["reach"] = profile.get("reach_raw", "")
        recovered_any = True
    if _blank(current_profile.get("dob")) and not _blank(profile.get("dob")):
        supplement["dob"] = profile.get("dob", "")
        recovered_any = True
    return supplement if recovered_any else None


def _build_fightdx_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    fighter_url = search_fightdx(fighter_name)
    if not fighter_url:
        return None

    profile = scrape_fightdx_profile(fighter_url)
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="fightdx",
        source_name=str(profile.get("name", "") or ""),
        search_name=fighter_name,
        fighter_url=fighter_url,
    )

    recovered_any = False
    if _blank(current_profile.get("reach")) and not _blank(profile.get("reach_raw")):
        supplement["reach"] = profile.get("reach_raw", "")
        recovered_any = True
    if _blank(current_profile.get("dob")) and not _blank(profile.get("dob")):
        supplement["dob"] = profile.get("dob", "")
        recovered_any = True
    return supplement if recovered_any else None


def _build_martialbot_row(
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name = MARTIALBOT_SEARCH_NAME_OVERRIDES.get(fighter_key, fighter_name)
    fighter_url = search_martialbot(search_name)
    if not fighter_url:
        return None

    profile = scrape_martialbot_profile(fighter_url)
    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="martialbot",
        source_name=str(profile.get("name", "") or ""),
        search_name=search_name,
        fighter_url=fighter_url,
    )

    recovered_any = False
    if _blank(current_profile.get("reach")) and not _blank(profile.get("reach_raw")):
        supplement["reach"] = profile.get("reach_raw", "")
        recovered_any = True
    if _blank(current_profile.get("dob")) and not _blank(profile.get("dob")):
        supplement["dob"] = profile.get("dob", "")
        recovered_any = True
    return supplement if recovered_any else None


def _wiki_api(session: requests.Session, **params) -> dict:
    response = session.get(WIKIPEDIA_API_URL, params=params, headers=WIKIPEDIA_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def _wikipedia_find_title(session: requests.Session, fighter_name: str) -> str | None:
    search_name = fighter_name

    exact_page = _wiki_api(
        session,
        action="query",
        titles=search_name,
        prop="info",
        format="json",
        formatversion=2,
    )["query"]["pages"][0]
    if not exact_page.get("missing") and same_person_name(fighter_name, exact_page.get("title", "")):
        return exact_page.get("title")

    search_results = _wiki_api(
        session,
        action="query",
        list="search",
        srsearch=search_name,
        format="json",
        formatversion=2,
        srlimit=5,
    ).get("query", {}).get("search", [])

    normalized_search_name = normalize_person_name(search_name)
    for result in search_results:
        title = result.get("title", "")
        if same_person_name(search_name, title):
            return title
        candidate_key = normalize_person_name(title)
        if normalized_search_name and (
            normalized_search_name in candidate_key or candidate_key in normalized_search_name
        ):
            return title
    return None


def _extract_infobox_field(wikitext: str, field: str) -> str:
    if not wikitext:
        return ""

    in_infobox = False
    brace_depth = 0
    for raw_line in wikitext.splitlines():
        line = raw_line.rstrip()
        if not in_infobox and line.startswith("{{Infobox"):
            in_infobox = True
        if not in_infobox:
            continue

        brace_depth += line.count("{{") - line.count("}}")
        if re.match(rf"^\|\s*{re.escape(field)}\s*=", line):
            return line.split("=", 1)[1].strip()
        if brace_depth <= 0:
            break
    return ""


def _parse_wikipedia_template_number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    fraction_match = re.fullmatch(r"(\d+)\s*\+\s*(\d+)\s*/\s*(\d+)", text)
    if fraction_match:
        whole = float(fraction_match.group(1))
        numerator = float(fraction_match.group(2))
        denominator = float(fraction_match.group(3))
        if denominator:
            return whole + (numerator / denominator)
    try:
        return float(text)
    except ValueError:
        return None


def _parse_wikipedia_reach(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if _blank(text):
        return ""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{efn\|.*?\}\}", "", text, flags=re.IGNORECASE)
    text = text.replace("[[", "").replace("]]", "").replace("{{nbsp}}", " ")
    text = text.strip()

    convert_match = re.search(r"\{\{\s*convert\|([^}]*)\}\}", text, flags=re.IGNORECASE)
    if convert_match:
        parts = [part.strip() for part in convert_match.group(1).split("|") if part.strip()]
        if len(parts) >= 2:
            numeric = _parse_wikipedia_template_number(parts[0])
            unit = parts[1].lower()
            if numeric is not None and unit in {"in", "inch", "inches"}:
                return f"{numeric:g} in"
            if numeric is not None and unit == "cm":
                return f"{numeric:g} cm"

    metric_match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", text, flags=re.IGNORECASE)
    if metric_match:
        return f"{metric_match.group(1)} cm"
    imperial_match = re.search(r"(\d+(?:\.\d+)?)\s*in\b", text, flags=re.IGNORECASE)
    if imperial_match:
        return f"{imperial_match.group(1)} in"
    return ""


def _parse_wikipedia_birth_date(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if _blank(text):
        return ""

    template_match = re.search(r"\{\{\s*birth[^|}]*\|(.+?)\}\}", text, flags=re.IGNORECASE)
    if template_match:
        parts = [part.strip() for part in template_match.group(1).split("|") if part.strip()]
        named: dict[str, int] = {}
        numeric_parts: list[int] = []
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in {"year", "month", "day"} and value.isdigit():
                    named[key] = int(value)
                continue
            if part.isdigit():
                numeric_parts.append(int(part))

        year = named.get("year")
        month = named.get("month")
        day = named.get("day")
        if year and month and day:
            return f"{year:04d}-{month:02d}-{day:02d}"
        if len(numeric_parts) >= 3:
            return f"{numeric_parts[0]:04d}-{numeric_parts[1]:02d}-{numeric_parts[2]:02d}"
        return ""

    plain_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if plain_match:
        return plain_match.group(0)
    return ""


def _build_wikipedia_row(
    session: requests.Session,
    scraped_row: pd.Series,
    current_state: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    fighter_key = normalize_person_name(fighter_name)
    search_name = fighter_name
    title = _wikipedia_find_title(session, fighter_name)
    if not title:
        return None

    page = _wiki_api(
        session,
        action="query",
        titles=title,
        prop="revisions",
        rvprop="content",
        format="json",
        formatversion=2,
    )["query"]["pages"][0]
    wikitext = page.get("revisions", [{}])[0].get("content", "")
    reach = _parse_wikipedia_reach(_extract_infobox_field(wikitext, "reach"))
    dob = _parse_wikipedia_birth_date(_extract_infobox_field(wikitext, "birth_date"))

    current_profile = current_state.setdefault(fighter_key, {field: "" for field in TARGET_FIELDS})
    supplement = _build_base_row(
        fighter_name,
        source="wikipedia",
        source_name=title,
        search_name=search_name,
        fighter_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    )

    recovered_any = False
    if _blank(current_profile.get("reach")) and not _blank(reach):
        supplement["reach"] = reach
        recovered_any = True
    if _blank(current_profile.get("dob")) and not _blank(dob):
        supplement["dob"] = dob
        recovered_any = True
    return supplement if recovered_any else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gap-audit-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    clear_fallback_cache()
    candidates = _load_candidates(args.scraped_fighters_path, gap_audit_csv=args.gap_audit_csv)
    candidates = candidates.sort_values("name").reset_index(drop=True)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    existing_rows = _load_existing_rows(args.output)
    existing_source_keys = _normalize_existing_source_keys(existing_rows)
    current_state = _build_effective_profile_state(
        pd.read_csv(args.scraped_fighters_path),
        existing_rows,
    )

    session = requests.Session()
    results: list[dict[str, object]] = []
    source_recoveries = {"martialbot": 0, "fightdx": 0, "tapology": 0, "wikipedia": 0}
    attempted = 0

    for _, row in candidates.iterrows():
        attempted += 1
        fighter_key = normalize_person_name(row.get("name"))

        if (fighter_key, "martialbot") not in existing_source_keys:
            martialbot_row = _build_martialbot_row(row, current_state)
            if martialbot_row is not None:
                results.append(martialbot_row)
                existing_source_keys.add((fighter_key, "martialbot"))
                _update_state_from_row(current_state, martialbot_row)
                source_recoveries["martialbot"] += 1

        if (fighter_key, "fightdx") not in existing_source_keys:
            fightdx_row = _build_fightdx_row(row, current_state)
            if fightdx_row is not None:
                results.append(fightdx_row)
                existing_source_keys.add((fighter_key, "fightdx"))
                _update_state_from_row(current_state, fightdx_row)
                source_recoveries["fightdx"] += 1

        if (fighter_key, "tapology") not in existing_source_keys:
            tapology_row = _build_tapology_row(row, current_state)
            if tapology_row is not None:
                results.append(tapology_row)
                existing_source_keys.add((fighter_key, "tapology"))
                _update_state_from_row(current_state, tapology_row)
                source_recoveries["tapology"] += 1

        if (fighter_key, "wikipedia") not in existing_source_keys:
            wikipedia_row = _build_wikipedia_row(session, row, current_state)
            if wikipedia_row is not None:
                results.append(wikipedia_row)
                existing_source_keys.add((fighter_key, "wikipedia"))
                _update_state_from_row(current_state, wikipedia_row)
                source_recoveries["wikipedia"] += 1

    combined_rows = existing_rows + results
    output_df = pd.DataFrame(combined_rows)
    if not output_df.empty:
        output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    summary = {
        "candidate_rows": int(len(candidates)),
        "attempted_rows": attempted,
        "recovered_rows": int(len(results)),
        "recovered_by_source": source_recoveries,
        "output_path": str(args.output),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Recovered by source: {summary['recovered_by_source']}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
