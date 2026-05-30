"""Backfill missing local UFCStats raw rows for fighters on the official active roster."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
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
from src.data.ufcstats_http import (
    DEFAULT_UFCSTATS_HEADERS,
    UFCStatsChallengeError,
    looks_like_ufcstats_challenge,
    request_ufcstats,
)
from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster

logger = logging.getLogger(__name__)

HEADERS = dict(DEFAULT_UFCSTATS_HEADERS)
REQUEST_DELAY_SECONDS = 0.35
MAX_FAILURE_DETAILS = 50
RESULTS_PATH = RAW_DATA_DIR / "ufc-fight-results.csv"
STATS_PATH = RAW_DATA_DIR / "ufc-fight-stats.csv"
FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
PROFILE_SCRAPE_FAILURES_PATH = RAW_DATA_DIR / "ufcstats_profile_scrape_failures.csv"
PROFILE_REFRESH_FIELDS = ("height", "reach", "weight", "stance", "dob")
PROFILE_RATE_FIELDS = ("slpm", "sapm", "td_avg", "sub_avg", "str_acc", "str_def", "td_acc", "td_def")
PROFILE_SCRAPE_FAILURE_COLUMNS = [
    "fighter",
    "ufcstats_url",
    "failure_reason",
    "cached_profile_present",
    "missing_fields",
]

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
    response = request_ufcstats(url, session=session, headers=HEADERS, timeout=30)
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(response.text, "lxml")


def _looks_like_ufcstats_challenge_page(soup: BeautifulSoup) -> bool:
    return looks_like_ufcstats_challenge(str(soup))


def _load_official_roster(*, refresh: bool) -> pd.DataFrame:
    if refresh or not OFFICIAL_ACTIVE_ROSTER_PATH.exists():
        return sync_official_active_roster()
    return pd.read_csv(OFFICIAL_ACTIVE_ROSTER_PATH)


def _extract_completed_fight_urls(fighter_url: str, *, session: requests.Session) -> list[str]:
    soup = _get_soup(fighter_url, session=session)
    title_el = soup.select_one("h2.b-content__title span, span.b-content__title-highlight")
    if _looks_like_ufcstats_challenge_page(soup) or title_el is None:
        raise ValueError("UFCStats fighter page did not expose profile markup")

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
    if _looks_like_ufcstats_challenge_page(soup):
        raise ValueError("UFCStats fight page did not expose fight markup")

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
    return scrape_fighter(fighter_url)


def _profile_failure_reason_from_exception(exc: Exception) -> str:
    if isinstance(exc, UFCStatsChallengeError):
        return "ufcstats_challenge_page"
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return f"http_{status_code}" if status_code is not None else "http_error"
    if isinstance(exc, requests.RequestException):
        return "request_error"
    return "scraper_exception"


def _profile_row_from_url_with_reason(fighter_url: str) -> tuple[dict | None, str]:
    try:
        profile_row = _profile_row_from_url(fighter_url)
    except Exception as exc:
        return None, _profile_failure_reason_from_exception(exc)
    if profile_row is None:
        return None, "profile_markup_missing"
    return profile_row, ""


def _write_profile_scrape_failure_report(details: list[dict[str, object]]) -> None:
    failure_df = pd.DataFrame(details, columns=PROFILE_SCRAPE_FAILURE_COLUMNS)
    write_csv_atomically(failure_df, PROFILE_SCRAPE_FAILURES_PATH)


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


def _profile_rate_stats_nonzero(row: pd.Series | dict) -> bool:
    for field in PROFILE_RATE_FIELDS:
        raw = row.get(field)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip().replace("%", "")
        if not text or text in {"--", "nan", "NaN"}:
            continue
        try:
            if float(text) > 0:
                return True
        except ValueError:
            continue
    return False


def _nonblank_count(df: pd.DataFrame, fields: tuple[str, ...]) -> int:
    total = 0
    for field in fields:
        if field not in df.columns:
            continue
        total += int(
            df[field]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"--": "", "nan": "", "NaN": ""})
            .ne("")
            .sum()
        )
    return total


def _raise_if_profile_merge_loses_observed_data(existing_df: pd.DataFrame, combined: pd.DataFrame) -> None:
    if len(combined) < len(existing_df):
        raise ValueError(
            f"Refusing to overwrite {FIGHTERS_PATH}: profile merge would reduce rows "
            f"from {len(existing_df)} to {len(combined)}"
        )

    existing_profile_fields = _nonblank_count(existing_df, PROFILE_REFRESH_FIELDS)
    combined_profile_fields = _nonblank_count(combined, PROFILE_REFRESH_FIELDS)
    if combined_profile_fields < existing_profile_fields:
        raise ValueError(
            f"Refusing to overwrite {FIGHTERS_PATH}: profile merge would reduce observed "
            f"profile fields from {existing_profile_fields} to {combined_profile_fields}"
        )

    existing_rate_fields = _nonblank_count(existing_df, PROFILE_RATE_FIELDS)
    combined_rate_fields = _nonblank_count(combined, PROFILE_RATE_FIELDS)
    if combined_rate_fields < existing_rate_fields:
        raise ValueError(
            f"Refusing to overwrite {FIGHTERS_PATH}: profile merge would reduce observed "
            f"rate fields from {existing_rate_fields} to {combined_rate_fields}"
        )


def _raise_if_raw_merge_loses_rows(existing_df: pd.DataFrame, merged_df: pd.DataFrame, path: Path) -> None:
    if len(merged_df) < len(existing_df):
        raise ValueError(
            f"Refusing to overwrite {path}: merge would reduce rows from "
            f"{len(existing_df)} to {len(merged_df)}"
        )


def _count_roster_profiles_with_nonzero_rate_stats(roster_df: pd.DataFrame) -> int:
    if not FIGHTERS_PATH.exists() or roster_df.empty:
        return 0
    try:
        fighters_df = pd.read_csv(FIGHTERS_PATH)
    except Exception:
        return 0
    if fighters_df.empty or "fighter_url" not in fighters_df.columns:
        return 0
    roster_urls = {
        str(url).strip()
        for url in roster_df.get("ufcstats_url", pd.Series(dtype="object")).dropna().astype(str)
        if str(url).strip()
    }
    if not roster_urls:
        return 0
    matched = fighters_df[
        fighters_df["fighter_url"].fillna("").astype(str).str.strip().isin(roster_urls)
    ]
    return int(sum(_profile_rate_stats_nonzero(row) for _, row in matched.iterrows()))


def _append_missing_profiles_with_summary(roster_df: pd.DataFrame) -> dict[str, object]:
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
        existing_df[column] = existing_df[column].astype("object")

    existing_index_by_url = {
        str(url).strip(): index
        for index, url in existing_df.get("fighter_url", pd.Series(dtype="object")).items()
        if str(url).strip()
    }
    new_rows: list[dict] = []
    updated_rows = 0
    needed_refresh = 0
    failed_scrapes = 0
    failed_profile_urls: list[str] = []
    failed_profile_details: list[dict[str, object]] = []
    failure_reason_counts: Counter[str] = Counter()
    for _, roster_row in roster_df.iterrows():
        fighter_url = str(roster_row.get("ufcstats_url") or "").strip()
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
        profile_row, failure_reason = _profile_row_from_url_with_reason(fighter_url)
        if profile_row is None:
            failed_scrapes += 1
            failed_profile_urls.append(fighter_url)
            reason = failure_reason or "unknown"
            failure_reason_counts[reason] += 1
            missing_fields = [
                field
                for field in PROFILE_REFRESH_FIELDS
                if existing_row is None or _blank_profile_value(existing_row.get(field))
            ]
            failed_profile_details.append(
                {
                    "fighter": _clean_text(
                        roster_row.get("official_name")
                        or roster_row.get("ufcstats_name")
                        or ""
                    ),
                    "ufcstats_url": fighter_url,
                    "failure_reason": reason,
                    "cached_profile_present": bool(existing_index is not None),
                    "missing_fields": ",".join(missing_fields),
                }
            )
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
        _raise_if_profile_merge_loses_observed_data(existing_df, combined)
        write_csv_atomically(combined, FIGHTERS_PATH, refuse_empty=True)

    _write_profile_scrape_failure_report(failed_profile_details)
    if failed_scrapes:
        logger.warning(
            "Profile scrape failures by reason: %s (details_path=%s truncated=%d)",
            dict(failure_reason_counts),
            PROFILE_SCRAPE_FAILURES_PATH,
            max(0, failed_scrapes - MAX_FAILURE_DETAILS),
        )

    logger.info(
        "Profile backfill complete: needed_refresh=%d new=%d updated=%d failed_scrapes=%d "
        "final_rows=%d final_stance=%d",
        needed_refresh, len(new_rows), updated_rows, failed_scrapes,
        len(combined), _stance_count(combined),
    )
    return {
        "scraped_profiles_added": int(len(new_rows)),
        "scraped_profiles_updated": int(updated_rows),
        "scraped_profiles_needed_refresh": int(needed_refresh),
        "scraped_profile_scrape_failures": int(failed_scrapes),
        "scraped_profile_scrape_failure_reasons": dict(failure_reason_counts),
        "scraped_profile_scrape_failure_details": failed_profile_details[:MAX_FAILURE_DETAILS],
        "scraped_profile_scrape_failure_details_truncated": max(
            0,
            int(failed_scrapes - MAX_FAILURE_DETAILS),
        ),
        "scraped_profile_scrape_failure_details_path": str(PROFILE_SCRAPE_FAILURES_PATH),
        "failed_profile_urls": failed_profile_urls,
    }


def _append_missing_profiles(roster_df: pd.DataFrame) -> tuple[int, int]:
    summary = _append_missing_profiles_with_summary(roster_df)
    return int(summary["scraped_profiles_added"]), int(summary["scraped_profiles_updated"])


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
    active_roster_rows_total = int(len(roster_df))
    if "coverage_eligible" in roster_df.columns:
        coverage_mask = roster_df["coverage_eligible"].fillna(True).astype(str).str.lower().isin(
            {"true", "1", "yes", "on"}
        ) | roster_df["coverage_eligible"].fillna(True).eq(True)
    else:
        coverage_mask = roster_df.get("combat_sport", pd.Series("mma", index=roster_df.index)).fillna("").astype(str).str.lower().ne("power_slap")
    roster_df = roster_df[coverage_mask].copy()
    active_roster_rows_coverage_eligible = int(len(roster_df))
    roster_df["ufcstats_url"] = roster_df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    roster_without_ufcstats_url = int((roster_df["ufcstats_url"] == "").sum())
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

    profile_summary = _append_missing_profiles_with_summary(roster_df)

    session = requests.Session()
    missing_fight_urls: list[str] = []
    seen_missing_urls: set[str] = set()
    fighters_checked = 0
    fighters_with_missing = 0
    fighters_with_completed_ufcstats_bouts = 0
    fighters_without_completed_ufcstats_bouts = 0
    fighter_fight_list_failure_count = 0
    fighter_fight_list_failures: list[dict[str, str]] = []

    for _, row in roster_df.iterrows():
        fighter_url = row["ufcstats_url"]
        try:
            fight_urls = _extract_completed_fight_urls(fighter_url, session=session)
        except Exception as exc:
            fighter_fight_list_failure_count += 1
            if len(fighter_fight_list_failures) < MAX_FAILURE_DETAILS:
                fighter_fight_list_failures.append(
                    {
                        "fighter": str(row.get("official_name") or row.get("ufcstats_name") or ""),
                        "ufcstats_url": fighter_url,
                        "error": str(exc),
                    }
                )
            continue
        fighters_checked += 1
        if fight_urls:
            fighters_with_completed_ufcstats_bouts += 1
        else:
            fighters_without_completed_ufcstats_bouts += 1
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
    _raise_if_raw_merge_loses_rows(results_df, merged_results, RESULTS_PATH)
    _raise_if_raw_merge_loses_rows(stats_df, merged_stats, STATS_PATH)
    write_csv_atomically(merged_results, RESULTS_PATH, refuse_empty=True)
    write_csv_atomically(merged_stats, STATS_PATH, refuse_empty=True)

    return {
        "active_roster_rows_total": active_roster_rows_total,
        "active_roster_rows_coverage_eligible": active_roster_rows_coverage_eligible,
        "roster_rows_with_ufcstats_url": int(len(roster_df)),
        "roster_rows_without_ufcstats_url": roster_without_ufcstats_url,
        "fighters_checked": int(fighters_checked),
        "fighters_with_completed_ufcstats_bouts": int(fighters_with_completed_ufcstats_bouts),
        "fighters_without_completed_ufcstats_bouts": int(fighters_without_completed_ufcstats_bouts),
        "fighters_with_missing_fights": int(fighters_with_missing),
        "missing_fight_urls_found": int(len(missing_fight_urls)),
        "new_result_rows": int(len(new_result_rows)),
        "new_stat_rows": int(len(new_stat_rows)),
        "fighters_with_nonzero_ufcstats_rate_stats": _count_roster_profiles_with_nonzero_rate_stats(roster_df),
        **profile_summary,
        "fighter_fight_list_scrape_failures": int(fighter_fight_list_failure_count),
        "fighter_fight_list_failure_details": fighter_fight_list_failures,
        "fighter_fight_list_failure_details_truncated": max(
            0,
            int(fighter_fight_list_failure_count - len(fighter_fight_list_failures)),
        ),
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
