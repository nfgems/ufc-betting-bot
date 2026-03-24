"""Backfill missing local UFCStats raw rows for fighters on the official active roster."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically
from src.data.scraper import scrape_fighter
from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_DELAY_SECONDS = 0.35
RESULTS_PATH = RAW_DATA_DIR / "ufc-fight-results.csv"
STATS_PATH = RAW_DATA_DIR / "ufc-fight-stats.csv"
FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
PROFILE_REFRESH_FIELDS = ("height", "reach", "weight", "stance", "dob")

RESULTS_COLUMNS = [
    "EVENT",
    "BOUT",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
    "REFEREE",
    "DETAILS",
    "URL",
]
STATS_COLUMNS = [
    "EVENT",
    "BOUT",
    "ROUND",
    "FIGHTER",
    "KD",
    "SIG.STR.",
    "SIG.STR. %",
    "TOTAL STR.",
    "TD",
    "TD %",
    "SUB.ATT",
    "REV.",
    "CTRL",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
]


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _get_soup(url: str, *, session: requests.Session) -> BeautifulSoup:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(response.text, "lxml")


def _load_official_roster(*, refresh: bool) -> pd.DataFrame:
    if refresh or not OFFICIAL_ACTIVE_ROSTER_PATH.exists():
        return sync_official_active_roster()
    return pd.read_csv(OFFICIAL_ACTIVE_ROSTER_PATH)


def _extract_completed_fight_urls(fighter_url: str, *, session: requests.Session) -> list[str]:
    soup = _get_soup(fighter_url, session=session)
    urls: list[str] = []
    seen: set[str] = set()
    for row in soup.select("tr.b-fight-details__table-row"):
        detail_url = str(row.get("data-link") or "").strip()
        if not detail_url or "fight-details" not in detail_url or detail_url in seen:
            continue
        seen.add(detail_url)
        urls.append(detail_url)
    return urls


def _parse_outcome(soup: BeautifulSoup) -> str:
    statuses: list[str] = []
    for status in soup.select("i.b-fight-details__person-status")[:2]:
        text = _clean_text(status.get_text(" ", strip=True)).upper()
        if text.startswith("W"):
            statuses.append("W")
        elif text.startswith("L"):
            statuses.append("L")
        elif text.startswith("D"):
            statuses.append("D")
        elif "NC" in text:
            statuses.append("NC")
        else:
            statuses.append(text or "?")
    return "/".join(statuses)


def _parse_text_value_map(soup: BeautifulSoup) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in soup.select("i.b-fight-details__text-item, i.b-fight-details__text-item_first"):
        label_el = item.select_one("i.b-fight-details__label")
        if label_el is None:
            continue
        label = _clean_text(label_el.get_text(" ", strip=True)).rstrip(":").upper()
        full_text = _clean_text(item.get_text(" ", strip=True))
        value = full_text.replace(_clean_text(label_el.get_text(" ", strip=True)), "", 1).strip()
        parsed[label] = value

    for paragraph in soup.select("p.b-fight-details__text"):
        text = _clean_text(paragraph.get_text(" ", strip=True))
        if text.startswith("Details:"):
            parsed["DETAILS"] = text.replace("Details:", "", 1).strip()
            break
    return parsed


def _parse_round_pair_table(
    table,
    *,
    value_columns: list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    parsed: dict[tuple[str, str], dict[str, str]] = {}
    body = table.select_one("tbody")
    if body is None:
        return parsed

    current_round = ""
    for child in body.children:
        if getattr(child, "name", None) == "thead":
            current_round = _clean_text(child.get_text(" ", strip=True))
            continue
        if getattr(child, "name", None) != "tr":
            continue
        cells = child.select("td.b-fight-details__table-col, td")
        if len(cells) < len(value_columns) + 1:
            continue
        fighter_tags = cells[0].select("p")
        fighter_names = [_clean_text(tag.get_text(" ", strip=True)) for tag in fighter_tags[:2]]
        if len(fighter_names) < 2:
            continue
        for idx, fighter_name in enumerate(fighter_names):
            row = {"ROUND": current_round, "FIGHTER": fighter_name}
            for cell_index, column_name in enumerate(value_columns, start=1):
                value_tags = cells[cell_index].select("p")
                value = _clean_text(value_tags[idx].get_text(" ", strip=True)) if len(value_tags) > idx else ""
                row[column_name] = value
            parsed[(current_round, fighter_name)] = row
    return parsed


def _parse_fight_detail(detail_url: str, *, session: requests.Session) -> tuple[dict[str, str], list[dict[str, str]]]:
    soup = _get_soup(detail_url, session=session)
    event_link = soup.select_one("h2.b-content__title a.b-link")
    event_title = _clean_text(event_link.get_text(" ", strip=True)) if event_link else ""

    fighters = [
        _clean_text(name.get_text(" ", strip=True))
        for name in soup.select("h3.b-fight-details__person-name a")[:2]
    ]
    if len(fighters) < 2:
        raise ValueError(f"Could not parse fighter names from {detail_url}")
    bout = f"{fighters[0]} vs. {fighters[1]}"

    fight_title_el = soup.select_one("i.b-fight-details__fight-title")
    weight_class = _clean_text(fight_title_el.get_text(" ", strip=True)) if fight_title_el else ""
    text_map = _parse_text_value_map(soup)
    result_row = {
        "EVENT": event_title,
        "BOUT": bout,
        "OUTCOME": _parse_outcome(soup),
        "WEIGHTCLASS": weight_class,
        "METHOD": text_map.get("METHOD", ""),
        "ROUND": text_map.get("ROUND", ""),
        "TIME": text_map.get("TIME", ""),
        "TIME FORMAT": text_map.get("TIME FORMAT", ""),
        "REFEREE": text_map.get("REFEREE", ""),
        "DETAILS": text_map.get("DETAILS", ""),
        "URL": detail_url,
    }

    round_tables = soup.select("table.b-fight-details__table.js-fight-table")
    totals_rounds = {}
    sig_rounds = {}
    if round_tables:
        totals_rounds = _parse_round_pair_table(
            round_tables[0],
            value_columns=[
                "KD",
                "SIG.STR.",
                "SIG.STR. %",
                "TOTAL STR.",
                "TD",
                "TD %",
                "SUB.ATT",
                "REV.",
                "CTRL",
            ],
        )
    if len(round_tables) > 1:
        sig_rounds = _parse_round_pair_table(
            round_tables[1],
            value_columns=[
                "SIG.STR.",
                "SIG.STR. %",
                "HEAD",
                "BODY",
                "LEG",
                "DISTANCE",
                "CLINCH",
                "GROUND",
            ],
        )

    stats_rows: list[dict[str, str]] = []
    for key, total_row in totals_rounds.items():
        merged = {
            "EVENT": event_title,
            "BOUT": bout,
            "ROUND": total_row.get("ROUND", ""),
            "FIGHTER": total_row.get("FIGHTER", ""),
            "KD": total_row.get("KD", ""),
            "SIG.STR.": total_row.get("SIG.STR.", ""),
            "SIG.STR. %": total_row.get("SIG.STR. %", ""),
            "TOTAL STR.": total_row.get("TOTAL STR.", ""),
            "TD": total_row.get("TD", ""),
            "TD %": total_row.get("TD %", ""),
            "SUB.ATT": total_row.get("SUB.ATT", ""),
            "REV.": total_row.get("REV.", ""),
            "CTRL": total_row.get("CTRL", ""),
            "HEAD": "",
            "BODY": "",
            "LEG": "",
            "DISTANCE": "",
            "CLINCH": "",
            "GROUND": "",
        }
        sig_row = sig_rounds.get(key, {})
        for column in ("HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"):
            merged[column] = sig_row.get(column, "")
        stats_rows.append(merged)

    return result_row, stats_rows


def _profile_row_from_url(fighter_url: str) -> dict | None:
    try:
        return scrape_fighter(fighter_url)
    except Exception:
        return None


def _blank_profile_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "nan", "NaN"}


def _profile_row_needs_refresh(row: pd.Series | dict | None) -> bool:
    if row is None:
        return True
    return any(_blank_profile_value(row.get(field)) for field in PROFILE_REFRESH_FIELDS)


def _merge_profile_rows(existing_row: dict, refreshed_row: dict) -> tuple[dict, bool]:
    merged = dict(existing_row)
    changed = False
    for field, value in refreshed_row.items():
        if field in merged and not _blank_profile_value(merged.get(field)):
            continue
        if _blank_profile_value(value):
            continue
        merged[field] = value
        changed = True
    return merged, changed


def _append_missing_profiles(roster_df: pd.DataFrame) -> tuple[int, int]:
    if not FIGHTERS_PATH.exists():
        logger.warning("Scraped fighters file does not exist: %s", FIGHTERS_PATH)
        existing_df = pd.DataFrame(
            columns=[
                "name",
                "record",
                "fighter_url",
                "height",
                "weight",
                "reach",
                "stance",
                "dob",
                "slpm",
                "str_acc",
                "sapm",
                "str_def",
                "td_avg",
                "td_acc",
                "td_def",
                "sub_avg",
            ]
        )
    else:
        existing_df = pd.read_csv(FIGHTERS_PATH)

    # Diagnostic: log pre-backfill state
    def _stance_count(df: pd.DataFrame) -> int:
        if "stance" not in df.columns:
            return 0
        return int(df["stance"].fillna("").astype(str).str.strip().ne("").sum())

    logger.info(
        "Profile backfill starting: FIGHTERS_PATH=%s rows=%d stance=%d",
        FIGHTERS_PATH, len(existing_df), _stance_count(existing_df),
    )

    for column in (
        "name",
        "record",
        "fighter_url",
        "height",
        "weight",
        "reach",
        "stance",
        "dob",
        "slpm",
        "str_acc",
        "sapm",
        "str_def",
        "td_avg",
        "td_acc",
        "td_def",
        "sub_avg",
    ):
        if column not in existing_df.columns:
            existing_df[column] = pd.NA

    existing_index_by_url = {
        str(url).strip(): index
        for index, url in existing_df.get("fighter_url", pd.Series(dtype="object")).items()
        if str(url).strip()
    }
    new_rows: list[dict] = []
    updated_rows = 0
    needed_refresh = 0
    failed_scrapes = 0
    for fighter_url in roster_df.get("ufcstats_url", pd.Series(dtype="object")).dropna().astype(str):
        fighter_url = fighter_url.strip()
        if not fighter_url:
            continue
        existing_index = existing_index_by_url.get(fighter_url)
        existing_row = (
            existing_df.loc[existing_index].to_dict()
            if existing_index is not None
            else None
        )
        if not _profile_row_needs_refresh(existing_row):
            continue
        needed_refresh += 1
        profile_row = _profile_row_from_url(fighter_url)
        if profile_row is None:
            failed_scrapes += 1
            continue
        profile_row = dict(profile_row)
        if existing_index is None:
            new_rows.append(profile_row)
            continue
        merged_row, changed = _merge_profile_rows(existing_row, profile_row)
        if changed:
            for field, value in merged_row.items():
                existing_df.at[existing_index, field] = value
            updated_rows += 1

    if new_rows:
        appended_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_df, appended_df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["fighter_url"], keep="first")
    else:
        combined = existing_df

    if new_rows or updated_rows:
        write_csv_atomically(combined, FIGHTERS_PATH, refuse_empty=True)

    logger.info(
        "Profile backfill complete: needed_refresh=%d new=%d updated=%d failed_scrapes=%d "
        "final_rows=%d final_stance=%d",
        needed_refresh, len(new_rows), updated_rows, failed_scrapes,
        len(combined), _stance_count(combined),
    )
    return len(new_rows), updated_rows


def _merge_results(existing_df: pd.DataFrame, new_rows: list[dict[str, str]]) -> pd.DataFrame:
    if not new_rows:
        return existing_df
    new_df = pd.DataFrame(new_rows, columns=RESULTS_COLUMNS)
    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined["__url_key"] = combined["URL"].fillna("").astype(str).str.strip()
    combined = combined.drop_duplicates(subset=["__url_key"], keep="first").drop(columns="__url_key")
    return combined


def _merge_stats(existing_df: pd.DataFrame, new_rows: list[dict[str, str]]) -> pd.DataFrame:
    if not new_rows:
        return existing_df
    new_df = pd.DataFrame(new_rows, columns=STATS_COLUMNS)
    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined["__stat_key"] = (
        combined["EVENT"].fillna("").astype(str).str.strip()
        + "||"
        + combined["BOUT"].fillna("").astype(str).str.strip()
        + "||"
        + combined["ROUND"].fillna("").astype(str).str.strip()
        + "||"
        + combined["FIGHTER"].fillna("").astype(str).str.strip()
    )
    combined = combined.drop_duplicates(subset=["__stat_key"], keep="first").drop(columns="__stat_key")
    return combined


def run_backfill(
    *,
    refresh_roster: bool = False,
    limit_fighters: int | None = None,
    roster_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    if roster_df is None:
        roster_df = _load_official_roster(refresh=refresh_roster)
    roster_df = roster_df.copy()
    roster_df["ufcstats_url"] = roster_df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    roster_df = roster_df[roster_df["ufcstats_url"] != ""].reset_index(drop=True)
    if limit_fighters is not None:
        roster_df = roster_df.head(limit_fighters).copy()

    results_df = pd.read_csv(RESULTS_PATH)
    stats_df = pd.read_csv(STATS_PATH)
    existing_urls = {
        str(url).strip()
        for url in results_df.get("URL", pd.Series(dtype="object")).dropna().astype(str)
        if str(url).strip()
    }

    scraped_profiles_added, scraped_profiles_updated = _append_missing_profiles(roster_df)

    session = requests.Session()
    missing_fight_urls: list[str] = []
    seen_missing_urls: set[str] = set()
    fighters_checked = 0
    fighters_with_missing = 0

    for _, row in roster_df.iterrows():
        fighter_url = row["ufcstats_url"]
        try:
            fight_urls = _extract_completed_fight_urls(fighter_url, session=session)
        except Exception:
            continue
        fighters_checked += 1
        fighter_missing = 0
        for fight_url in fight_urls:
            if fight_url in existing_urls or fight_url in seen_missing_urls:
                continue
            seen_missing_urls.add(fight_url)
            missing_fight_urls.append(fight_url)
            fighter_missing += 1
        if fighter_missing:
            fighters_with_missing += 1

    new_result_rows: list[dict[str, str]] = []
    new_stat_rows: list[dict[str, str]] = []
    failed_urls: list[str] = []
    for fight_url in missing_fight_urls:
        try:
            result_row, stat_rows = _parse_fight_detail(fight_url, session=session)
        except Exception:
            failed_urls.append(fight_url)
            continue
        new_result_rows.append(result_row)
        new_stat_rows.extend(stat_rows)

    merged_results = _merge_results(results_df, new_result_rows)
    merged_stats = _merge_stats(stats_df, new_stat_rows)
    write_csv_atomically(merged_results, RESULTS_PATH, refuse_empty=True)
    write_csv_atomically(merged_stats, STATS_PATH, refuse_empty=True)

    return {
        "roster_rows_with_ufcstats_url": int(len(roster_df)),
        "fighters_checked": int(fighters_checked),
        "fighters_with_missing_fights": int(fighters_with_missing),
        "missing_fight_urls_found": int(len(missing_fight_urls)),
        "new_result_rows": int(len(new_result_rows)),
        "new_stat_rows": int(len(new_stat_rows)),
        "scraped_profiles_added": int(scraped_profiles_added),
        "scraped_profiles_updated": int(scraped_profiles_updated),
        "failed_fight_urls": failed_urls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-roster", action="store_true")
    parser.add_argument("--limit-fighters", type=int, default=None)
    args = parser.parse_args()

    summary = run_backfill(
        refresh_roster=args.refresh_roster,
        limit_fighters=args.limit_fighters,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
