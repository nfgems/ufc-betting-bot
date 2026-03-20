"""
Snapshot-backed UFC rankings collection and lookup.

Collection still uses the same external sources, but live inference only reads
the latest successful on-disk snapshot.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DATA_DIR
from src.data.name_utils import normalize_person_name, same_person_name

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5

RANKINGS_SNAPSHOT_DIR = RAW_DATA_DIR / "rankings"
RANKINGS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_SCHEMA_VERSION = 1
RANKINGS_SNAPSHOT_MAX_AGE = timedelta(days=14)

# Default rank for unranked fighters (same as training pipeline)
UNRANKED_DEFAULT = 16

# Weight class name normalization: map common variants to canonical names
_WC_ALIASES = {
    "strawweight": "strawweight",
    "women's strawweight": "women's strawweight",
    "flyweight": "flyweight",
    "women's flyweight": "women's flyweight",
    "bantamweight": "bantamweight",
    "women's bantamweight": "women's bantamweight",
    "featherweight": "featherweight",
    "women's featherweight": "women's featherweight",
    "lightweight": "lightweight",
    "welterweight": "welterweight",
    "middleweight": "middleweight",
    "light heavyweight": "light heavyweight",
    "heavyweight": "heavyweight",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_name(name: str) -> str:
    """Normalize a fighter name for matching: lowercase, strip extra whitespace."""
    return normalize_person_name(name)


def _canonical_wc(wc: str) -> str:
    """Map a weight class string to its canonical key."""
    wc_lower = str(wc or "").lower().strip()
    for alias, canonical in _WC_ALIASES.items():
        if alias in wc_lower:
            return canonical
    return wc_lower


def _name_tokens(name: str) -> list[str]:
    return [token for token in _normalize_name(name).split() if token]


def _names_match(query_name: str, ranked_name: str) -> bool:
    """Prefer exact/full-name matches and only then allow controlled fallback."""
    return same_person_name(query_name, ranked_name)


def _last_name_token(name: str) -> str:
    tokens = _name_tokens(name)
    return tokens[-1] if tokens else ""


def _snapshot_path(snapshot_time: str) -> Path:
    timestamp = datetime.fromisoformat(snapshot_time).strftime("%Y%m%d_%H%M%S")
    return RANKINGS_SNAPSHOT_DIR / f"rankings_{timestamp}.json"


def _save_snapshot(snapshot: dict) -> Path:
    path = _snapshot_path(snapshot["snapshot_time"])
    path.write_text(json.dumps(snapshot, indent=2))
    logger.info("Saved rankings snapshot: %s", path)
    return path


def _load_snapshot(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read rankings snapshot %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("snapshot_path", str(path))
    return data


def _snapshot_age(snapshot_time: object) -> Optional[timedelta]:
    parsed = _parse_snapshot_timestamp(snapshot_time)
    if parsed is None:
        return None
    return datetime.now(timezone.utc) - parsed


def _parse_snapshot_timestamp(
    value: object,
    *,
    end_of_day_if_date_only: bool = False,
) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if end_of_day_if_date_only and "T" not in str(value) and " " not in str(value):
        parsed = parsed + timedelta(days=1) - timedelta(microseconds=1)
    return parsed


def _snapshot_is_stale(snapshot: dict, *, max_age: Optional[timedelta] = None) -> bool:
    if max_age is None:
        max_age = RANKINGS_SNAPSHOT_MAX_AGE
    if max_age is None:
        return False
    age = _snapshot_age(snapshot.get("snapshot_time"))
    if age is None:
        return False
    return age > max_age


def load_latest_rankings_snapshot(
    *,
    require_success: bool = True,
    allow_stale: bool = False,
    max_age: Optional[timedelta] = None,
    as_of_date: Optional[str] = None,
) -> Optional[dict]:
    """Load the newest rankings snapshot from disk."""
    cutoff = _parse_snapshot_timestamp(as_of_date, end_of_day_if_date_only=True)
    for path in sorted(RANKINGS_SNAPSHOT_DIR.glob("rankings_*.json"), reverse=True):
        snapshot = _load_snapshot(path)
        if snapshot is None:
            continue
        if cutoff is not None:
            snapshot_time = _parse_snapshot_timestamp(snapshot.get("snapshot_time"))
            if snapshot_time is None or snapshot_time > cutoff:
                continue
        if require_success and (
            snapshot.get("status") != "success"
            or snapshot.get("acquisition_failed")
            or snapshot.get("source") == "none"
        ):
            continue
        if cutoff is None and not allow_stale and _snapshot_is_stale(snapshot, max_age=max_age):
            continue
        return snapshot
    return None


def _fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Rankings fetch failed for %s: %s", url, exc)
        return None

    time.sleep(REQUEST_DELAY)
    return resp.text


def _normalize_rankings_payload(payload: dict, *, source: str) -> Optional[dict]:
    wc_rankings = {
        _canonical_wc(division): {
            _normalize_name(name): int(rank)
            for name, rank in rankings.items()
            if _normalize_name(name)
        }
        for division, rankings in payload.get("wc", {}).items()
        if _canonical_wc(division)
    }
    pfp_rankings = {
        _normalize_name(name): int(rank)
        for name, rank in payload.get("pfp", {}).items()
        if _normalize_name(name)
    }

    if not wc_rankings and not pfp_rankings:
        return None

    return {
        "wc": wc_rankings,
        "pfp": pfp_rankings,
        "source": source,
    }


def _parse_ufc_rankings_html(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")

    wc_rankings: dict[str, dict[str, int]] = {}
    pfp_rankings: dict[str, int] = {}

    divisions = soup.select("div.view-grouping")
    if not divisions:
        divisions = soup.select("div.rankings--athlete--container")
    if not divisions:
        logger.warning("UFC.com: could not find ranking divisions in HTML")
        return None

    for div in divisions:
        header = div.select_one("h4, h3, .view-grouping-header")
        if not header:
            continue
        division_name = _normalize_name(header.get_text())
        is_pfp = "pound" in division_name

        fighters: list[str] = []
        for selector in [
            "tr td.views-field-title a",
            "a.e-author__name",
            "td.views-field a",
            ".view-content tbody tr",
        ]:
            found = div.select(selector)
            if found:
                fighters = [_normalize_name(el.get_text()) for el in found if el.get_text().strip()]
                break

        if not fighters:
            items = div.select("li, tr")
            for item in items:
                text = item.get_text(strip=True)
                match = re.match(r"(?:\d+\.?\s*)?([\w\s'-]+)", text)
                if not match:
                    continue
                name = _normalize_name(match.group(1))
                if len(name) > 2 and name not in {"champion", "title", "division"}:
                    fighters.append(name)

        if not fighters:
            continue

        if is_pfp:
            for rank, name in enumerate(fighters[:15], 1):
                pfp_rankings[name] = rank
        else:
            wc_key = _canonical_wc(division_name)
            wc_rankings[wc_key] = {}
            for rank, name in enumerate(fighters[:15], 1):
                wc_rankings[wc_key][name] = rank

    return _normalize_rankings_payload({"wc": wc_rankings, "pfp": pfp_rankings}, source="ufc.com")


def _parse_espn_rankings_html(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")

    wc_rankings: dict[str, dict[str, int]] = {}
    pfp_rankings: dict[str, int] = {}

    tables = soup.select("div.Table__Scroller, table.Table")
    if not tables:
        logger.warning("ESPN: could not find ranking tables in HTML")
        return None

    sections = soup.select("section, div.rankings-section, div.ResponsiveTable")
    for section in sections:
        header = section.select_one("h2, h3, .Table__Title, caption")
        if not header:
            continue
        division_name = _normalize_name(header.get_text())
        is_pfp = "pound" in division_name

        fighters: list[str] = []
        for row in section.select("tr"):
            name_cell = row.select_one("td:nth-of-type(2), td a, span.AnchorLink")
            if not name_cell:
                continue
            name = _normalize_name(name_cell.get_text())
            if len(name) > 2:
                fighters.append(name)

        if not fighters:
            continue

        if is_pfp:
            for rank, name in enumerate(fighters[:15], 1):
                pfp_rankings[name] = rank
        else:
            wc_key = _canonical_wc(division_name)
            wc_rankings[wc_key] = {}
            for rank, name in enumerate(fighters[:15], 1):
                wc_rankings[wc_key][name] = rank

    return _normalize_rankings_payload({"wc": wc_rankings, "pfp": pfp_rankings}, source="espn.com")


def _parse_tapology_rankings_html(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")

    wc_rankings: dict[str, dict[str, int]] = {}
    pfp_rankings: dict[str, int] = {}

    sections = soup.select("div.rankingTable, div.ranking, section.division")
    if not sections:
        sections = soup.select("div[class*='ranking'], table[class*='ranking']")

    for section in sections:
        header = section.select_one("h2, h3, h4, .rankingTitle, caption")
        if not header:
            continue
        division_name = _normalize_name(header.get_text())
        is_pfp = "pound" in division_name

        fighters: list[str] = []
        for row in section.select("tr, li.rankingItemsItem"):
            name_el = row.select_one("a[href*='fighter'], td.name, span.name")
            if not name_el:
                continue
            name = _normalize_name(name_el.get_text())
            if len(name) > 2:
                fighters.append(name)

        if not fighters:
            continue

        if is_pfp:
            for rank, name in enumerate(fighters[:15], 1):
                pfp_rankings[name] = rank
        else:
            wc_key = _canonical_wc(division_name)
            wc_rankings[wc_key] = {}
            for rank, name in enumerate(fighters[:15], 1):
                wc_rankings[wc_key][name] = rank

    return _normalize_rankings_payload({"wc": wc_rankings, "pfp": pfp_rankings}, source="tapology.com")


def _scrape_ufc_rankings() -> Optional[dict]:
    html = _fetch_html("https://www.ufc.com/rankings")
    return _parse_ufc_rankings_html(html) if html else None


def _scrape_espn_rankings() -> Optional[dict]:
    html = _fetch_html("https://www.espn.com/mma/rankings")
    return _parse_espn_rankings_html(html) if html else None


def _scrape_tapology_rankings() -> Optional[dict]:
    html = _fetch_html("https://www.tapology.com/rankings/current-ufc-rankings")
    return _parse_tapology_rankings_html(html) if html else None


def _source_result(
    source: str,
    *,
    status: str,
    captured_at: str,
    rankings: Optional[dict] = None,
    error: str = "",
) -> dict:
    result = {
        "source": source,
        "status": status,
        "captured_at": captured_at,
    }
    if error:
        result["error"] = error
    if rankings is not None:
        result["num_divisions"] = len(rankings.get("wc", {}))
        result["num_pfp"] = len(rankings.get("pfp", {}))
        result["wc"] = rankings.get("wc", {})
        result["pfp"] = rankings.get("pfp", {})
    return result


def collect_rankings_snapshot() -> dict:
    """
    Fetch rankings from live sources and persist a timestamped normalized snapshot.

    Inference should not call this. It exists for scheduled/offline collection.
    """
    source_chain: list[tuple[str, Callable[[], Optional[dict]]]] = [
        ("ufc.com", _scrape_ufc_rankings),
        ("espn.com", _scrape_espn_rankings),
        ("tapology.com", _scrape_tapology_rankings),
    ]

    attempted_sources: list[dict] = []
    winning_payload: Optional[dict] = None

    for source, scrape_fn in source_chain:
        captured_at = _now_iso()
        try:
            rankings = scrape_fn()
        except Exception as exc:
            logger.warning("Rankings scraper failed for %s: %s", source, exc)
            attempted_sources.append(
                _source_result(source, status="failed", captured_at=captured_at, error=str(exc))
            )
            continue

        if rankings is None:
            attempted_sources.append(
                _source_result(source, status="failed", captured_at=captured_at, error="no rankings parsed")
            )
            continue

        attempted_sources.append(
            _source_result(source, status="success", captured_at=captured_at, rankings=rankings)
        )
        winning_payload = rankings
        break

    if winning_payload is not None:
        source = winning_payload.get("source", "unknown")
        status = "success"
        acquisition_failed = False
        wc_rankings = winning_payload.get("wc", {})
        pfp_rankings = winning_payload.get("pfp", {})
    else:
        source = "none"
        status = "failed"
        acquisition_failed = True
        wc_rankings = {}
        pfp_rankings = {}

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": _now_iso(),
        "status": status,
        "source": source,
        "acquisition_failed": acquisition_failed,
        "wc": wc_rankings,
        "pfp": pfp_rankings,
        "sources": attempted_sources,
    }
    path = _save_snapshot(snapshot)
    snapshot["snapshot_path"] = str(path)
    return snapshot


def get_rankings(*, as_of_date: Optional[str] = None) -> dict:
    """
    Load the latest successful rankings snapshot from disk.

    If no successful snapshot exists, callers receive an acquisition_failed flag
    and should emit NaN features rather than fabricated "unranked" values.
    """
    snapshot = load_latest_rankings_snapshot(
        require_success=True,
        allow_stale=as_of_date is not None,
        as_of_date=as_of_date,
    )
    if snapshot is not None:
        logger.debug("Using rankings snapshot from %s", snapshot.get("snapshot_time"))
        return snapshot

    if as_of_date is not None:
        logger.warning("No successful rankings snapshot is available on or before %s", as_of_date)
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": None,
            "status": "failed",
            "source": "none",
            "acquisition_failed": True,
            "wc": {},
            "pfp": {},
            "sources": [],
        }

    stale_snapshot = load_latest_rankings_snapshot(require_success=True, allow_stale=True)
    if stale_snapshot is not None and _snapshot_is_stale(stale_snapshot):
        logger.warning("Newest rankings snapshot is stale: %s", stale_snapshot.get("snapshot_time"))
        return {
            "schema_version": stale_snapshot.get("schema_version", SNAPSHOT_SCHEMA_VERSION),
            "snapshot_time": stale_snapshot.get("snapshot_time"),
            "status": "stale",
            "source": stale_snapshot.get("source", "none"),
            "acquisition_failed": True,
            "stale_snapshot": True,
            "wc": stale_snapshot.get("wc", {}),
            "pfp": stale_snapshot.get("pfp", {}),
            "sources": stale_snapshot.get("sources", []),
            "snapshot_path": stale_snapshot.get("snapshot_path"),
        }

    logger.warning("No successful rankings snapshot is available")
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": None,
        "status": "failed",
        "source": "none",
        "acquisition_failed": True,
        "wc": {},
        "pfp": {},
        "sources": [],
    }


def get_fighter_rankings(
    fighter_name: str,
    weight_class: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Get a fighter's weight class rank and P4P rank.

    Returns dict with:
      - wc_rank_feat: weight class rank (1-15, 16 for true unranked, NaN on source failure)
      - pfp_rank_feat: P4P rank (1-15, 16 for true unranked, NaN on source failure)
    """
    rankings = get_rankings(as_of_date=as_of_date)
    name_lower = _normalize_name(fighter_name)
    if rankings.get("acquisition_failed") or rankings.get("source") == "none":
        return {
            "wc_rank_feat": np.nan,
            "pfp_rank_feat": np.nan,
        }

    wc_rank = UNRANKED_DEFAULT
    if weight_class:
        wc_key = _canonical_wc(weight_class)
        wc_division = rankings.get("wc", {}).get(wc_key, {})
        for ranked_name, rank in wc_division.items():
            if _names_match(name_lower, ranked_name):
                wc_rank = rank
                break

    pfp_rank = UNRANKED_DEFAULT
    pfp_division = rankings.get("pfp", {})
    for ranked_name, rank in pfp_division.items():
        if _names_match(name_lower, ranked_name):
            pfp_rank = rank
            break

    return {
        "wc_rank_feat": wc_rank,
        "pfp_rank_feat": pfp_rank,
    }
