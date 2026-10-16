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
from src.data.io_utils import (
    csv_pair_manifest_path,
    read_csv_pair_verified,
    write_csv_atomically,
    write_csvs_atomically,
)
from src.data.fight_results import (
    RESULT_CLASS_DQ,
    RESULT_CLASS_DRAW,
    RESULT_CLASS_NO_CONTEST,
    RESULT_CLASS_NORMAL,
    RESULT_POLICY_VERSION,
    classify_official_result,
    has_result_policy_marker,
    validate_marked_result,
)
from src.data.name_utils import normalize_person_name, normalize_ufcstats_id
from src.data.scraper import scrape_fighter
from src.data.ufcstats_http import (
    DEFAULT_UFCSTATS_HEADERS,
    UFCStatsChallengeError,
    looks_like_ufcstats_challenge,
    request_ufcstats,
)
from src.data.ufc_active_roster import (
    OFFICIAL_ACTIVE_ROSTER_PATH,
    _apply_profile_status_eligibility,
    _explicit_boolean,
    sync_official_active_roster,
)

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
PROFILE_REFRESHABLE_OBSERVED_FIELDS = frozenset(
    ("record", *PROFILE_REFRESH_FIELDS, *PROFILE_RATE_FIELDS)
)
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
    "fighter_a_id",
    "fighter_b_id",
    "RESULT_POLICY_VERSION",
    "RESULT_CLASS",
    "BINARY_TARGET_ELIGIBLE",
    "RESULT_QUARANTINE_REASON",
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
_REQUIRED_RESULT_FIELDS = (
    "EVENT",
    "BOUT",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
    "URL",
    "fighter_a_id",
    "fighter_b_id",
)
_REQUIRED_STAT_FIELDS = tuple(STATS_COLUMNS)
_ROUND_STATS_STATUS_AVAILABLE = "available"
_ROUND_STATS_STATUS_EXPLICITLY_UNAVAILABLE = "explicitly_unavailable"
_ROUND_STATS_STATUS_MISSING_UNEXPLAINED = "missing_unexplained"
_ROUND_STATS_EXPLICITLY_UNAVAILABLE_TEXT = (
    "round-by-round stats not currently available."
)


def _clean_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


# UFCStats currently serves these two reviewed profile identities from each
# other's historical URL.  Frozen/accepted data retain the canonical IDs below;
# source reads may use the alternate URL only after the returned name is proved.
_REVIEWED_CANONICAL_UFCSTATS_IDENTITIES = {
    "4e30f2761b6bd3b5": {
        "names": frozenset(
            normalize_person_name(name)
            for name in ("Jose Montanha", "Jose Montanha da Silva")
        ),
        "alternate_source_id": "1f8cd7634e8286ff",
    },
    "1f8cd7634e8286ff": {
        "names": frozenset(
            {normalize_person_name("Henrique Da Silva Lopes")}
        ),
        "alternate_source_id": "4e30f2761b6bd3b5",
    },
}
_REVIEWED_UFCSTATS_PAIR_IDS = frozenset(_REVIEWED_CANONICAL_UFCSTATS_IDENTITIES)


def _fighter_url_with_id(url: str, fighter_id: str) -> str:
    current_id = normalize_ufcstats_id(url)
    if not current_id:
        raise ValueError(f"UFCStats fighter URL has no stable ID: {url}")
    return re.sub(re.escape(current_id) + r"/?$", fighter_id, str(url).rstrip("/"))


def _reviewed_identity_name_matches(canonical_id: str, name: object) -> bool:
    identity = _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES.get(canonical_id)
    if identity is None:
        return True
    return normalize_person_name(name) in identity["names"]


def _validate_reviewed_roster_identity(
    canonical_url: str,
    observed_name: object,
    *,
    source: str,
) -> None:
    canonical_id = normalize_ufcstats_id(canonical_url)
    if canonical_id not in _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES:
        return
    if not _reviewed_identity_name_matches(str(canonical_id), observed_name):
        raise ValueError(
            "reviewed UFCStats identity mismatch: "
            f"canonical_id={canonical_id} source={source} observed_name={_clean_text(observed_name)}"
        )


def _reviewed_identity_source_urls(canonical_url: str) -> list[str]:
    canonical_id = normalize_ufcstats_id(canonical_url)
    identity = _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES.get(str(canonical_id))
    if identity is None:
        return [canonical_url]
    alternate = _fighter_url_with_id(
        canonical_url,
        str(identity["alternate_source_id"]),
    )
    return list(dict.fromkeys((canonical_url, alternate)))


def _canonicalize_reviewed_detail_identity(
    fighter_name: object,
    fighter_url: object,
) -> tuple[str, dict[str, str] | None]:
    """Map only the reviewed swapped pair and reject every inconsistent pair."""
    observed_id = normalize_ufcstats_id(fighter_url) or ""
    normalized_name = normalize_person_name(fighter_name)
    matching_ids = [
        canonical_id
        for canonical_id, identity in _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES.items()
        if normalized_name in identity["names"]
    ]
    if not matching_ids and observed_id not in _REVIEWED_UFCSTATS_PAIR_IDS:
        return observed_id, None
    if len(matching_ids) != 1 or observed_id not in _REVIEWED_UFCSTATS_PAIR_IDS:
        raise ValueError(
            "reviewed UFCStats identity mismatch: "
            f"fighter={_clean_text(fighter_name)} observed_id={observed_id or 'missing'}"
        )
    canonical_id = matching_ids[0]
    if observed_id == canonical_id:
        return canonical_id, None
    correction = {
        "fighter": _clean_text(fighter_name),
        "observed_id": observed_id,
        "canonical_id": canonical_id,
    }
    return canonical_id, correction


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


_COMPLETED_FIGHT_HISTORY_HEADERS = (
    "w/l",
    "fighter",
    "kd",
    "str",
    "td",
    "sub",
    "event",
    "method",
    "round",
    "time",
)


def _fight_history_is_explicit_empty(fight_table, rows: list[object]) -> bool:
    """Recognize UFCStats' complete zero-bout table without accepting blank markup."""
    headers = tuple(
        _clean_text(header.get_text(" ", strip=True)).casefold()
        for header in fight_table.select("thead th")
    )
    if headers != _COMPLETED_FIGHT_HISTORY_HEADERS or len(rows) != 1:
        return False
    row = rows[0]
    cells = row.select("td")
    return bool(
        "b-statistics__table-row" in (row.get("class") or [])
        and not str(row.get("data-link") or "").strip()
        and len(cells) == 1
        and not _clean_text(row.get_text(" ", strip=True))
    )


def _extract_completed_fight_urls(
    fighter_url: str,
    *,
    session: requests.Session,
    expected_canonical_url: str | None = None,
) -> list[str]:
    soup = _get_soup(fighter_url, session=session)
    title_el = soup.select_one("h2.b-content__title span, span.b-content__title-highlight")
    if _looks_like_ufcstats_challenge_page(soup) or title_el is None:
        raise ValueError("UFCStats fighter page did not expose profile markup")
    if expected_canonical_url is not None:
        _validate_reviewed_roster_identity(
            expected_canonical_url,
            title_el.get_text(" ", strip=True),
            source=fighter_url,
        )

    fight_table = soup.select_one("table.b-fight-details__table")
    if fight_table is None:
        raise ValueError("UFCStats fighter page did not expose a fight-history table")
    fight_body = fight_table.select_one("tbody")
    if fight_body is None:
        raise ValueError("UFCStats fighter fight-history table did not expose a body")

    urls: list[str] = []
    seen: set[str] = set()
    explicit_upcoming_row = False
    rows = fight_body.select("tr")
    if not rows and fight_body.select_one("td") is not None:
        raise ValueError("UFCStats fighter fight-history body contained malformed rows")
    explicit_empty_history = _fight_history_is_explicit_empty(fight_table, rows)
    for row in rows:
        detail_url = str(row.get("data-link") or "").strip()
        cells = row.select("td")
        if not cells:
            # Header rows can appear inside tbody in some UFCStats responses.
            continue
        row_text = _clean_text(row.get_text(" ", strip=True))
        if not row_text:
            # UFCStats emits an empty spacer row ahead of profile history.
            continue
        if not detail_url:
            if row_text.casefold().startswith("next "):
                explicit_upcoming_row = True
                continue
            raise ValueError("UFCStats fighter fight-history row omitted its detail URL")
        if "fight-details" not in detail_url:
            raise ValueError("UFCStats fighter fight-history row exposed an invalid detail URL")
        if detail_url in seen:
            raise ValueError("UFCStats fighter fight-history contained a duplicate detail URL")
        seen.add(detail_url)
        urls.append(detail_url)
    if not urls and not explicit_upcoming_row and not explicit_empty_history:
        raise ValueError(
            "UFCStats fighter page did not positively identify zero completed fights"
        )
    return urls


def _extract_completed_fight_urls_for_roster_identity(
    canonical_url: str,
    *,
    session: requests.Session,
) -> tuple[list[str], str]:
    """Read a reviewed profile's proven current page without changing its ID."""
    canonical_id = normalize_ufcstats_id(canonical_url)
    if canonical_id not in _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES:
        return _extract_completed_fight_urls(canonical_url, session=session), canonical_url

    failures: list[str] = []
    for source_url in _reviewed_identity_source_urls(canonical_url):
        try:
            urls = _extract_completed_fight_urls(
                source_url,
                session=session,
                expected_canonical_url=canonical_url,
            )
        except Exception as exc:
            failures.append(f"{source_url}:{type(exc).__name__}:{exc}")
            continue
        return urls, source_url
    raise ValueError(
        "reviewed UFCStats identity routes failed: " + " | ".join(failures)
    )


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


def _round_statistics_source_status(
    soup: BeautifulSoup,
    *,
    round_tables: list[object],
) -> str:
    if round_tables:
        return _ROUND_STATS_STATUS_AVAILABLE
    section_texts = {
        _clean_text(section.get_text(" ", strip=True)).casefold()
        for section in soup.select(
            "section.b-fight-details__section.js-fight-section"
        )
    }
    if _ROUND_STATS_EXPLICITLY_UNAVAILABLE_TEXT in section_texts:
        return _ROUND_STATS_STATUS_EXPLICITLY_UNAVAILABLE
    return _ROUND_STATS_STATUS_MISSING_UNEXPLAINED


def _parse_fight_detail(
    detail_url: str,
    *,
    session: requests.Session,
    source_metadata: dict[str, object] | None = None,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    soup = _get_soup(detail_url, session=session)
    if _looks_like_ufcstats_challenge_page(soup):
        raise ValueError("UFCStats fight page did not expose fight markup")

    event_link = soup.select_one("h2.b-content__title a.b-link")
    event_title = _clean_text(event_link.get_text(" ", strip=True)) if event_link else ""

    fighter_links = soup.select("h3.b-fight-details__person-name a")[:2]
    fighters = [
        _clean_text(link.get_text(" ", strip=True))
        for link in fighter_links
    ]
    if len(fighters) < 2:
        raise ValueError(f"Could not parse fighter names from {detail_url}")
    bout = f"{fighters[0]} vs. {fighters[1]}"
    fighter_ids: list[str] = []
    for fighter_name, fighter_link in zip(fighters, fighter_links):
        fighter_id, correction = _canonicalize_reviewed_detail_identity(
            fighter_name,
            fighter_link.get("href"),
        )
        if correction is not None:
            logger.warning(
                "Applied reviewed UFCStats identity correction on %s: %s",
                detail_url,
                correction,
            )
        fighter_ids.append(fighter_id)

    fight_title_el = soup.select_one("i.b-fight-details__fight-title")
    weight_class = _clean_text(fight_title_el.get_text(" ", strip=True)) if fight_title_el else ""
    text_map = _parse_text_value_map(soup)
    outcome = _parse_outcome(soup)
    method = text_map.get("METHOD", "")
    classification = classify_official_result(outcome, method)
    result_row = {
        "EVENT": event_title,
        "BOUT": bout,
        "OUTCOME": outcome,
        "WEIGHTCLASS": weight_class,
        "METHOD": method,
        "ROUND": text_map.get("ROUND", ""),
        "TIME": text_map.get("TIME", ""),
        "TIME FORMAT": text_map.get("TIME FORMAT", ""),
        "REFEREE": text_map.get("REFEREE", ""),
        "DETAILS": text_map.get("DETAILS", ""),
        "URL": detail_url,
        "fighter_a_id": fighter_ids[0],
        "fighter_b_id": fighter_ids[1],
        "RESULT_POLICY_VERSION": RESULT_POLICY_VERSION,
        "RESULT_CLASS": classification.result_class,
        "BINARY_TARGET_ELIGIBLE": str(
            classification.binary_target_eligible
        ).lower(),
        "RESULT_QUARANTINE_REASON": classification.quarantine_reason,
    }

    round_tables = soup.select("table.b-fight-details__table.js-fight-table")
    round_statistics_status = _round_statistics_source_status(
        soup,
        round_tables=list(round_tables),
    )
    if source_metadata is not None:
        source_metadata.update(
            {
                "round_statistics_status": round_statistics_status,
                "round_statistics_table_count": int(len(round_tables)),
            }
        )
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


def _profile_row_for_roster_identity(
    canonical_url: str,
    *,
    expected_name: object,
) -> tuple[dict | None, str, str]:
    """Fetch a profile only from a source page proven to match its roster ID."""
    canonical_id = normalize_ufcstats_id(canonical_url)
    if canonical_id not in _REVIEWED_CANONICAL_UFCSTATS_IDENTITIES:
        profile_row, reason = _profile_row_from_url_with_reason(canonical_url)
        return profile_row, reason, canonical_url
    try:
        _validate_reviewed_roster_identity(
            canonical_url,
            expected_name,
            source="active_roster",
        )
    except ValueError:
        return None, "reviewed_source_identity_mismatch", ""

    failures: list[str] = []
    identity_mismatch = False
    for source_url in _reviewed_identity_source_urls(canonical_url):
        profile_row, reason = _profile_row_from_url_with_reason(source_url)
        if profile_row is None:
            failures.append(reason or "unknown")
            continue
        try:
            _validate_reviewed_roster_identity(
                canonical_url,
                profile_row.get("name"),
                source=source_url,
            )
        except ValueError:
            identity_mismatch = True
            continue
        profile_row = dict(profile_row)
        profile_row["fighter_url"] = canonical_url
        return profile_row, "", source_url
    if identity_mismatch:
        return None, "reviewed_source_identity_mismatch", ""
    return None, (failures[-1] if failures else "profile_markup_missing"), ""


def _write_profile_scrape_failure_report(
    details: list[dict[str, object]],
    *,
    profile_failures_path: Path | str | None = None,
) -> None:
    profile_failures_path = Path(
        profile_failures_path
        if profile_failures_path is not None
        else PROFILE_SCRAPE_FAILURES_PATH
    )
    failure_df = pd.DataFrame(details, columns=PROFILE_SCRAPE_FAILURE_COLUMNS)
    write_csv_atomically(failure_df, profile_failures_path)


def _blank_profile_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "nan", "NaN"}


def _profile_rate_observation(
    profile_row: dict[str, object],
    *,
    fighter: str,
    fighter_url: str,
) -> dict[str, object]:
    """Describe the raw scrape before cached profile values can mask gaps."""
    observed_fields = [
        field
        for field in PROFILE_RATE_FIELDS
        if not _blank_profile_value(profile_row.get(field))
    ]
    return {
        "fighter": fighter,
        "ufcstats_url": fighter_url,
        "observed_rate_fields": observed_fields,
        "observed_rate_field_count": len(observed_fields),
    }


def _classify_profile_rate_observations(
    observations: list[dict[str, object]],
    fight_history_by_url: dict[str, dict[str, object]],
) -> dict[str, object]:
    classified: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    for observation in observations:
        row = dict(observation)
        observed_count = int(row.get("observed_rate_field_count") or 0)
        history = fight_history_by_url.get(str(row.get("ufcstats_url") or ""), {})
        history_complete = history.get("scan_complete") is True
        completed_fights = history.get("completed_fight_count")
        if observed_count == len(PROFILE_RATE_FIELDS):
            status = "complete"
            reason = "complete_rate_bundle"
        elif observed_count:
            status = "suspicious_partial"
            reason = f"partial_rate_bundle_{observed_count}_of_{len(PROFILE_RATE_FIELDS)}"
        elif history_complete and int(completed_fights or 0) == 0:
            status = "source_limited"
            reason = "all_rate_fields_blank_zero_completed_fights"
        elif history_complete:
            status = "suspicious_missing"
            reason = "all_rate_fields_blank_with_completed_fights"
        else:
            status = "suspicious_missing"
            reason = "all_rate_fields_blank_fight_history_unavailable"
        row.update(
            {
                "rate_bundle_status": status,
                "rate_bundle_reason": reason,
                "fight_history_scan_complete": history_complete,
                "completed_fight_count": completed_fights,
            }
        )
        classified.append(row)
        status_counts[status] += 1

    suspicious = [
        row
        for row in classified
        if str(row.get("rate_bundle_status") or "").startswith("suspicious_")
    ]
    source_limited = [
        row
        for row in classified
        if row.get("rate_bundle_status") == "source_limited"
    ]
    return {
        "scraped_profile_rate_bundle_status_counts": dict(status_counts),
        "scraped_profile_rate_bundle_blocked": bool(suspicious),
        "scraped_profile_suspicious_rate_bundle_count": len(suspicious),
        "scraped_profile_source_limited_rate_bundle_count": len(source_limited),
        "scraped_profile_rate_bundle_details": classified[:MAX_FAILURE_DETAILS],
        "scraped_profile_rate_bundle_details_truncated": max(
            0, len(classified) - MAX_FAILURE_DETAILS
        ),
    }


def _profile_row_needs_refresh(row: pd.Series | dict | None) -> bool:
    if row is None:
        return True
    return any(_blank_profile_value(row.get(field)) for field in PROFILE_REFRESH_FIELDS)


def _merge_profile_rows(
    existing_row: dict,
    refreshed_row: dict,
    *,
    refresh_existing_values: bool = False,
    preserve_existing_profile_change_fields: bool = False,
) -> tuple[dict, bool]:
    merged = dict(existing_row)
    changed = False
    for field, value in refreshed_row.items():
        if _blank_profile_value(value):
            continue
        if preserve_existing_profile_change_fields and field in PROFILE_REFRESH_FIELDS:
            continue
        existing_value = merged.get(field)
        if field in merged and not _blank_profile_value(existing_value):
            if not refresh_existing_values or field not in PROFILE_REFRESHABLE_OBSERVED_FIELDS:
                continue
            if str(existing_value).strip() == str(value).strip():
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


def _raise_if_profile_merge_loses_observed_data(
    existing_df: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    fighters_path: Path | str | None = None,
) -> None:
    fighters_path = Path(fighters_path if fighters_path is not None else FIGHTERS_PATH)
    if len(combined) < len(existing_df):
        raise ValueError(
            f"Refusing to overwrite {fighters_path}: profile merge would reduce rows "
            f"from {len(existing_df)} to {len(combined)}"
        )

    existing_profile_fields = _nonblank_count(existing_df, PROFILE_REFRESH_FIELDS)
    combined_profile_fields = _nonblank_count(combined, PROFILE_REFRESH_FIELDS)
    if combined_profile_fields < existing_profile_fields:
        raise ValueError(
            f"Refusing to overwrite {fighters_path}: profile merge would reduce observed "
            f"profile fields from {existing_profile_fields} to {combined_profile_fields}"
        )

    existing_rate_fields = _nonblank_count(existing_df, PROFILE_RATE_FIELDS)
    combined_rate_fields = _nonblank_count(combined, PROFILE_RATE_FIELDS)
    if combined_rate_fields < existing_rate_fields:
        raise ValueError(
            f"Refusing to overwrite {fighters_path}: profile merge would reduce observed "
            f"rate fields from {existing_rate_fields} to {combined_rate_fields}"
        )


def _raise_if_raw_merge_loses_rows(existing_df: pd.DataFrame, merged_df: pd.DataFrame, path: Path) -> None:
    if len(merged_df) < len(existing_df):
        raise ValueError(
            f"Refusing to overwrite {path}: merge would reduce rows from "
            f"{len(existing_df)} to {len(merged_df)}"
        )


def _count_roster_profiles_with_nonzero_rate_stats(
    roster_df: pd.DataFrame,
    *,
    fighters_path: Path | str | None = None,
) -> int:
    fighters_path = Path(fighters_path if fighters_path is not None else FIGHTERS_PATH)
    if not fighters_path.exists() or roster_df.empty:
        return 0
    try:
        fighters_df = pd.read_csv(fighters_path)
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


def _append_missing_profiles_with_summary(
    roster_df: pd.DataFrame,
    *,
    refresh_existing_profiles: bool = False,
    preserve_existing_profile_change_fields: bool = False,
    fighters_path: Path | str | None = None,
    profile_failures_path: Path | str | None = None,
) -> dict[str, object]:
    fighters_path = Path(fighters_path if fighters_path is not None else FIGHTERS_PATH)
    profile_failures_path = Path(
        profile_failures_path
        if profile_failures_path is not None
        else PROFILE_SCRAPE_FAILURES_PATH
    )
    if not fighters_path.exists():
        logger.warning("Scraped fighters file does not exist: %s", fighters_path)
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
        existing_df = pd.read_csv(fighters_path)

    # Diagnostic: log pre-backfill state
    def _stance_count(df: pd.DataFrame) -> int:
        if "stance" not in df.columns:
            return 0
        return int(df["stance"].fillna("").astype(str).str.strip().ne("").sum())

    logger.info(
        "Profile backfill starting: FIGHTERS_PATH=%s rows=%d stance=%d",
        fighters_path, len(existing_df), _stance_count(existing_df),
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
    targeted_profiles = 0
    failed_scrapes = 0
    failed_profile_urls: list[str] = []
    failed_profile_details: list[dict[str, object]] = []
    failure_reason_counts: Counter[str] = Counter()
    rate_observations: list[dict[str, object]] = []
    reviewed_profile_source_redirects = 0
    preserved_profile_change_observations: list[dict[str, object]] = []
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
        profile_needs_refresh = _profile_row_needs_refresh(existing_row)
        if not profile_needs_refresh and not refresh_existing_profiles:
            continue
        if profile_needs_refresh:
            needed_refresh += 1
        targeted_profiles += 1
        expected_name = _clean_text(
            roster_row.get("official_name") or roster_row.get("ufcstats_name") or ""
        )
        profile_row, failure_reason, profile_source_url = (
            _profile_row_for_roster_identity(
                fighter_url,
                expected_name=expected_name,
            )
        )
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
        if profile_source_url and profile_source_url != fighter_url:
            reviewed_profile_source_redirects += 1
        profile_row = dict(profile_row)
        rate_observations.append(
            _profile_rate_observation(
                profile_row,
                fighter=_clean_text(
                    roster_row.get("official_name")
                    or roster_row.get("ufcstats_name")
                    or profile_row.get("name")
                    or ""
                ),
                fighter_url=fighter_url,
            )
        )
        if existing_index is None:
            new_rows.append(profile_row)
            continue
        if preserve_existing_profile_change_fields:
            for field in PROFILE_REFRESH_FIELDS:
                observed_value = profile_row.get(field)
                if (
                    _blank_profile_value(existing_row.get(field))
                    and not _blank_profile_value(observed_value)
                ):
                    preserved_profile_change_observations.append(
                        {
                            "fighter": expected_name,
                            "ufcstats_url": fighter_url,
                            "field": field,
                            "accepted_value": "",
                            "observed_value": str(observed_value).strip(),
                            "reason": "existing_profile_change_requires_independent_decision",
                        }
                    )
        merged_row, changed = _merge_profile_rows(
            existing_row,
            profile_row,
            refresh_existing_values=refresh_existing_profiles,
            preserve_existing_profile_change_fields=(
                preserve_existing_profile_change_fields
            ),
        )
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
        _raise_if_profile_merge_loses_observed_data(
            existing_df,
            combined,
            fighters_path=fighters_path,
        )
        write_csv_atomically(combined, fighters_path, refuse_empty=True)

    _write_profile_scrape_failure_report(
        failed_profile_details,
        profile_failures_path=profile_failures_path,
    )
    if failed_scrapes:
        logger.warning(
            "Profile scrape failures by reason: %s (details_path=%s truncated=%d)",
            dict(failure_reason_counts),
            profile_failures_path,
            max(0, failed_scrapes - MAX_FAILURE_DETAILS),
        )

    logger.info(
        "Profile backfill complete: targeted=%d needed_refresh=%d new=%d updated=%d failed_scrapes=%d "
        "final_rows=%d final_stance=%d",
        targeted_profiles, needed_refresh, len(new_rows), updated_rows, failed_scrapes,
        len(combined), _stance_count(combined),
    )
    return {
        "refresh_existing_profiles": bool(refresh_existing_profiles),
        "preserve_existing_profile_change_fields": bool(
            preserve_existing_profile_change_fields
        ),
        "preserved_existing_profile_change_count": len(
            preserved_profile_change_observations
        ),
        "preserved_existing_profile_change_observations": (
            preserved_profile_change_observations
        ),
        "scraped_profiles_targeted": int(targeted_profiles),
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
        "scraped_profile_scrape_failure_details_path": str(profile_failures_path),
        "failed_profile_urls": failed_profile_urls,
        "reviewed_profile_source_redirects": int(reviewed_profile_source_redirects),
        "scraped_profile_rate_observations": rate_observations,
    }


def _append_missing_profiles(
    roster_df: pd.DataFrame,
    *,
    refresh_existing_profiles: bool = False,
    preserve_existing_profile_change_fields: bool = False,
    fighters_path: Path | str | None = None,
    profile_failures_path: Path | str | None = None,
) -> tuple[int, int]:
    summary = _append_missing_profiles_with_summary(
        roster_df,
        refresh_existing_profiles=refresh_existing_profiles,
        preserve_existing_profile_change_fields=(
            preserve_existing_profile_change_fields
        ),
        fighters_path=fighters_path,
        profile_failures_path=profile_failures_path,
    )
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


def _observed(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().casefold() not in {"", "nan", "none", "<na>"}


def _round_number(value: object) -> int | None:
    """Parse UFCStats result rounds ("1") and table headings ("Round 1")."""
    match = re.fullmatch(r"(?:round\s+)?([0-9]+)", str(value or "").strip(), flags=re.I)
    if match is None:
        return None
    round_number = int(match.group(1))
    return round_number if round_number >= 1 else None


def _fight_observation_completeness(
    result_row: dict[str, str],
    stat_rows: list[dict[str, str]],
) -> tuple[bool, list[str]]:
    """Require one complete result and both fighters for every observed round."""
    reasons = [
        f"missing_result_{field.casefold().replace(' ', '_')}"
        for field in _REQUIRED_RESULT_FIELDS
        if not _observed(result_row.get(field))
    ]
    participants = [
        name.strip()
        for name in re.split(r"\s+vs\.?\s+", str(result_row.get("BOUT") or ""), flags=re.I)
        if name.strip()
    ]
    participant_keys = {normalize_person_name(name) for name in participants}
    if len(participants) != 2 or len(participant_keys) != 2:
        reasons.append("invalid_result_participants")
    classification = classify_official_result(
        result_row.get("OUTCOME"),
        result_row.get("METHOD"),
    )
    if classification.quarantined:
        reasons.append(classification.quarantine_reason)
    if (
        not classification.quarantined
        and has_result_policy_marker(result_row.get("RESULT_POLICY_VERSION"))
    ):
        try:
            validate_marked_result(
                result_row.get("OUTCOME"),
                result_row.get("METHOD"),
                policy_version=result_row.get("RESULT_POLICY_VERSION"),
                persisted_result_class=result_row.get("RESULT_CLASS"),
                persisted_binary_target_eligible=result_row.get(
                    "BINARY_TARGET_ELIGIBLE"
                ),
            )
        except ValueError as exc:
            reasons.append(f"invalid_result_policy:{exc}")
    finish_round = _round_number(result_row.get("ROUND"))
    if finish_round is None:
        reasons.append("invalid_result_round")

    rows_by_round: dict[int, list[dict[str, str]]] = {}
    for row in stat_rows:
        round_number = _round_number(row.get("ROUND"))
        if round_number is None:
            reasons.append("invalid_stat_round")
            continue
        rows_by_round.setdefault(round_number, []).append(row)
        for field in _REQUIRED_STAT_FIELDS:
            if not _observed(row.get(field)):
                reasons.append(f"missing_stat_{field.casefold().replace(' ', '_')}")

    if finish_round is not None:
        expected_rounds = set(range(1, finish_round + 1))
        if set(rows_by_round) != expected_rounds:
            reasons.append("incomplete_round_set")
        for round_number in sorted(expected_rounds):
            round_rows = rows_by_round.get(round_number, [])
            fighters = {
                normalize_person_name(row.get("FIGHTER"))
                for row in round_rows
                if normalize_person_name(row.get("FIGHTER"))
            }
            if len(round_rows) != 2 or fighters != participant_keys:
                reasons.append(f"incomplete_round_{round_number}_fighters")

    result_event = normalize_person_name(result_row.get("EVENT"))
    result_bout = normalize_person_name(result_row.get("BOUT"))
    for row in stat_rows:
        if normalize_person_name(row.get("EVENT")) != result_event:
            reasons.append("stat_event_mismatch")
        if normalize_person_name(row.get("BOUT")) != result_bout:
            reasons.append("stat_bout_mismatch")
    deduped = list(dict.fromkeys(reasons))
    return not deduped, deduped


def _explicit_source_limited_fight_reason(
    result_row: dict[str, str],
    stat_rows: list[dict[str, str]],
    reasons: list[str],
    source_metadata: dict[str, object],
) -> str:
    """Recognize only UFCStats' positive no-stats marker with a complete result."""
    if (
        source_metadata.get("round_statistics_status")
        != _ROUND_STATS_STATUS_EXPLICITLY_UNAVAILABLE
        or int(source_metadata.get("round_statistics_table_count") or 0) != 0
        or stat_rows
    ):
        return ""
    finish_round = _round_number(result_row.get("ROUND"))
    if finish_round is None:
        return ""
    expected_reasons = {
        "incomplete_round_set",
        *(f"incomplete_round_{round_number}_fighters" for round_number in range(1, finish_round + 1)),
    }
    if set(reasons) != expected_reasons:
        return ""
    return "round_by_round_stats_not_currently_available"


def run_backfill(
    *,
    refresh_roster: bool = False,
    limit_fighters: int | None = None,
    roster_df: pd.DataFrame | None = None,
    refresh_existing_profiles: bool = False,
    preserve_existing_profile_change_fields: bool = False,
    fighters_path: Path | str | None = None,
    profile_failures_path: Path | str | None = None,
    results_path: Path | str | None = None,
    stats_path: Path | str | None = None,
) -> dict[str, object]:
    profile_path_kwargs: dict[str, Path | str] = {}
    if fighters_path is not None:
        profile_path_kwargs["fighters_path"] = fighters_path
    if profile_failures_path is not None:
        profile_path_kwargs["profile_failures_path"] = profile_failures_path
    explicit_fighters_path = fighters_path is not None
    fighters_path = Path(fighters_path if fighters_path is not None else FIGHTERS_PATH)
    profile_failures_path = Path(
        profile_failures_path
        if profile_failures_path is not None
        else PROFILE_SCRAPE_FAILURES_PATH
    )
    results_path = Path(results_path if results_path is not None else RESULTS_PATH)
    stats_path = Path(stats_path if stats_path is not None else STATS_PATH)
    if roster_df is None:
        roster_df = _load_official_roster(refresh=refresh_roster)
    roster_df = roster_df.copy()
    active_roster_rows_total = int(len(roster_df))
    roster_df = _apply_profile_status_eligibility(roster_df)
    coverage_values = roster_df.get(
        "coverage_eligible",
        pd.Series(False, index=roster_df.index, dtype="object"),
    )
    coverage_mask = coverage_values.map(
        lambda value: _explicit_boolean(value) is True
    )
    roster_df = roster_df[coverage_mask].copy()
    active_roster_rows_coverage_eligible = int(len(roster_df))
    roster_df["ufcstats_url"] = roster_df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    roster_without_ufcstats_url = int((roster_df["ufcstats_url"] == "").sum())
    roster_df = roster_df[roster_df["ufcstats_url"] != ""].reset_index(drop=True)
    if limit_fighters is not None:
        roster_df = roster_df.head(limit_fighters).copy()

    pair_manifest = csv_pair_manifest_path(results_path, stats_path)
    results_df, stats_df = read_csv_pair_verified(
        results_path,
        stats_path,
        manifest_path=pair_manifest,
    )
    existing_urls = {
        str(url).strip()
        for url in results_df.get("URL", pd.Series(dtype="object")).dropna().astype(str)
        if str(url).strip()
    }

    if refresh_existing_profiles:
        profile_summary = _append_missing_profiles_with_summary(
            roster_df,
            refresh_existing_profiles=True,
            **profile_path_kwargs,
        )
    elif preserve_existing_profile_change_fields:
        profile_summary = _append_missing_profiles_with_summary(
            roster_df,
            preserve_existing_profile_change_fields=True,
            **profile_path_kwargs,
        )
    else:
        profile_summary = _append_missing_profiles_with_summary(
            roster_df,
            **profile_path_kwargs,
        )

    session = requests.Session()
    missing_fight_urls: list[str] = []
    seen_missing_urls: set[str] = set()
    fighters_checked = 0
    fighters_with_missing = 0
    fighters_with_completed_ufcstats_bouts = 0
    fighter_ids_with_completed_ufcstats_bouts: list[str] = []
    fighters_without_completed_ufcstats_bouts = 0
    fighter_fight_list_failure_count = 0
    fighter_fight_list_failures: list[dict[str, str]] = []
    fight_history_by_url: dict[str, dict[str, object]] = {}
    reviewed_fight_list_source_redirects = 0

    for _, row in roster_df.iterrows():
        fighter_url = row["ufcstats_url"]
        try:
            fight_urls, fight_list_source_url = (
                _extract_completed_fight_urls_for_roster_identity(
                    fighter_url,
                    session=session,
                )
            )
        except Exception as exc:
            fight_history_by_url[fighter_url] = {
                "scan_complete": False,
                "completed_fight_count": None,
            }
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
        if fight_list_source_url != fighter_url:
            reviewed_fight_list_source_redirects += 1
        fight_history_by_url[fighter_url] = {
            "scan_complete": True,
            "completed_fight_count": len(fight_urls),
        }
        fighters_checked += 1
        if fight_urls:
            fighters_with_completed_ufcstats_bouts += 1
            fighter_id = normalize_ufcstats_id(fighter_url)
            if fighter_id:
                fighter_ids_with_completed_ufcstats_bouts.append(fighter_id)
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

    rate_observations = list(
        profile_summary.get("scraped_profile_rate_observations") or []
    )
    profile_summary.update(
        _classify_profile_rate_observations(
            rate_observations,
            fight_history_by_url,
        )
    )
    profile_summary["scraped_profile_rate_observations"] = rate_observations[
        :MAX_FAILURE_DETAILS
    ]
    profile_summary["scraped_profile_rate_observations_truncated"] = max(
        0, len(rate_observations) - MAX_FAILURE_DETAILS
    )

    new_result_rows: list[dict[str, str]] = []
    new_stat_rows: list[dict[str, str]] = []
    complete_fight_urls: list[str] = []
    source_deferred_fight_urls: list[str] = []
    source_deferred_fight_details: list[dict[str, object]] = []
    partial_fight_urls: list[str] = []
    partial_fight_details: list[dict[str, object]] = []
    quarantined_fight_urls: list[str] = []
    quarantined_fight_details: list[dict[str, object]] = []
    failed_urls: list[str] = []
    failed_fight_details: list[dict[str, str]] = []
    for fight_url in missing_fight_urls:
        source_metadata: dict[str, object] = {}
        try:
            result_row, stat_rows = _parse_fight_detail(
                fight_url,
                session=session,
                source_metadata=source_metadata,
            )
        except Exception as exc:
            failed_urls.append(fight_url)
            if len(failed_fight_details) < MAX_FAILURE_DETAILS:
                failed_fight_details.append(
                    {
                        "fight_url": fight_url,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue
        complete, reasons = _fight_observation_completeness(result_row, stat_rows)
        if not complete:
            source_deferred_reason = _explicit_source_limited_fight_reason(
                result_row,
                stat_rows,
                reasons,
                source_metadata,
            )
            if source_deferred_reason:
                source_deferred_fight_urls.append(fight_url)
                if len(source_deferred_fight_details) < MAX_FAILURE_DETAILS:
                    source_deferred_fight_details.append(
                        {
                            "fight_url": fight_url,
                            "event": result_row.get("EVENT", ""),
                            "bout": result_row.get("BOUT", ""),
                            "outcome": result_row.get("OUTCOME", ""),
                            "method": result_row.get("METHOD", ""),
                            "finish_round": result_row.get("ROUND", ""),
                            "result_class": result_row.get("RESULT_CLASS", ""),
                            "reason": source_deferred_reason,
                        }
                    )
                continue
            partial_fight_urls.append(fight_url)
            if len(partial_fight_details) < MAX_FAILURE_DETAILS:
                partial_fight_details.append({"fight_url": fight_url, "reasons": reasons})
            quarantine_reasons = [
                reason
                for reason in reasons
                if reason.startswith("unknown_result_outcome")
                or reason.startswith("contradictory_result_outcome")
                or reason.startswith("invalid_result_policy:")
            ]
            if quarantine_reasons:
                quarantined_fight_urls.append(fight_url)
                if len(quarantined_fight_details) < MAX_FAILURE_DETAILS:
                    quarantined_fight_details.append(
                        {"fight_url": fight_url, "reasons": quarantine_reasons}
                    )
            continue
        new_result_rows.append(result_row)
        new_stat_rows.extend(stat_rows)
        complete_fight_urls.append(fight_url)

    result_quarantine_blocked = bool(quarantined_fight_urls)
    candidate_class_counts = Counter(
        str(row.get("RESULT_CLASS") or "").strip()
        for row in new_result_rows
    )
    result_class_counts = {
        result_class: int(candidate_class_counts.get(result_class, 0))
        for result_class in (
            RESULT_CLASS_NORMAL,
            RESULT_CLASS_DQ,
            RESULT_CLASS_DRAW,
            RESULT_CLASS_NO_CONTEST,
        )
    }
    result_exclusion_counts = {
        result_class: result_class_counts[result_class]
        for result_class in (
            RESULT_CLASS_DQ,
            RESULT_CLASS_DRAW,
            RESULT_CLASS_NO_CONTEST,
        )
    }
    accepted_result_rows = [] if result_quarantine_blocked else new_result_rows
    accepted_stat_rows = [] if result_quarantine_blocked else new_stat_rows
    merged_results = _merge_results(results_df, accepted_result_rows)
    merged_stats = _merge_stats(stats_df, accepted_stat_rows)
    _raise_if_raw_merge_loses_rows(results_df, merged_results, results_path)
    _raise_if_raw_merge_loses_rows(stats_df, merged_stats, stats_path)
    canonical_files_written = bool(accepted_result_rows)
    if canonical_files_written:
        write_csvs_atomically(
            ((merged_results, results_path), (merged_stats, stats_path)),
            refuse_empty=True,
            manifest_path=pair_manifest,
        )

    fight_detail_total = (
        len(complete_fight_urls)
        + len(source_deferred_fight_urls)
        + len(partial_fight_urls)
        + len(failed_urls)
    )
    fight_detail_status_counts = {
        "total": int(fight_detail_total),
        "complete": int(len(complete_fight_urls)),
        "source_deferred": int(len(source_deferred_fight_urls)),
        "partial": int(len(partial_fight_urls)),
        "failed": int(len(failed_urls)),
        "complete_pct": (
            round(100.0 * len(complete_fight_urls) / fight_detail_total, 6)
            if fight_detail_total
            else 0.0
        ),
        "source_deferred_pct": (
            round(100.0 * len(source_deferred_fight_urls) / fight_detail_total, 6)
            if fight_detail_total
            else 0.0
        ),
        "partial_pct": (
            round(100.0 * len(partial_fight_urls) / fight_detail_total, 6)
            if fight_detail_total
            else 0.0
        ),
        "failed_pct": (
            round(100.0 * len(failed_urls) / fight_detail_total, 6)
            if fight_detail_total
            else 0.0
        ),
    }

    return {
        "active_roster_rows_total": active_roster_rows_total,
        "active_roster_rows_coverage_eligible": active_roster_rows_coverage_eligible,
        "roster_rows_with_ufcstats_url": int(len(roster_df)),
        "roster_rows_without_ufcstats_url": roster_without_ufcstats_url,
        "fighters_checked": int(fighters_checked),
        "fighters_with_completed_ufcstats_bouts": int(fighters_with_completed_ufcstats_bouts),
        "fighter_ids_with_completed_ufcstats_bouts": sorted(
            set(fighter_ids_with_completed_ufcstats_bouts)
        ),
        "fighters_without_completed_ufcstats_bouts": int(fighters_without_completed_ufcstats_bouts),
        "fighters_with_missing_fights": int(fighters_with_missing),
        "missing_fight_urls_found": int(len(missing_fight_urls)),
        "new_result_rows": int(len(accepted_result_rows)),
        "new_stat_rows": int(len(accepted_stat_rows)),
        "candidate_result_rows": int(len(new_result_rows)),
        "candidate_stat_rows": int(len(new_stat_rows)),
        "result_class_counts": result_class_counts,
        "result_exclusion_counts": result_exclusion_counts,
        "fight_detail_status_counts": fight_detail_status_counts,
        "complete_fight_urls": complete_fight_urls,
        "source_deferred_fight_urls": source_deferred_fight_urls,
        "source_deferred_fight_details": source_deferred_fight_details,
        "source_deferred_fight_details_truncated": max(
            0,
            len(source_deferred_fight_urls) - len(source_deferred_fight_details),
        ),
        "partial_fight_urls": partial_fight_urls,
        "partial_fight_details": partial_fight_details,
        "partial_fight_details_truncated": max(
            0, len(partial_fight_urls) - len(partial_fight_details)
        ),
        "result_quarantine_blocked": result_quarantine_blocked,
        "quarantined_fight_urls": quarantined_fight_urls,
        "quarantined_fight_details": quarantined_fight_details,
        "quarantined_fight_details_truncated": max(
            0, len(quarantined_fight_urls) - len(quarantined_fight_details)
        ),
        "canonical_files_written": canonical_files_written,
        "fighters_path": str(fighters_path),
        "results_path": str(results_path),
        "stats_path": str(stats_path),
        "pair_manifest_path": str(pair_manifest),
        "fighters_with_nonzero_ufcstats_rate_stats": (
            _count_roster_profiles_with_nonzero_rate_stats(
                roster_df,
                fighters_path=fighters_path,
            )
            if explicit_fighters_path
            else _count_roster_profiles_with_nonzero_rate_stats(roster_df)
        ),
        **profile_summary,
        "fighter_fight_list_scrape_failures": int(fighter_fight_list_failure_count),
        "fighter_fight_list_failure_details": fighter_fight_list_failures,
        "fighter_fight_list_failure_details_truncated": max(
            0,
            int(fighter_fight_list_failure_count - len(fighter_fight_list_failures)),
        ),
        "reviewed_fight_list_source_redirects": int(
            reviewed_fight_list_source_redirects
        ),
        "failed_fight_urls": failed_urls,
        "failed_fight_details": failed_fight_details,
        "failed_fight_details_truncated": max(
            0, len(failed_urls) - len(failed_fight_details)
        ),
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
