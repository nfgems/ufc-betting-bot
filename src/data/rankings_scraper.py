"""
Snapshot-backed UFC rankings collection and lookup.

Collection still uses the same external sources, but live inference only reads
the latest successful on-disk snapshot.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config import (
    RANKINGS_SNAPSHOT_DAILY_KEEP,
    RANKINGS_SNAPSHOT_FULL_RESOLUTION_DAYS,
    RANKINGS_SNAPSHOT_PRUNE_INTERVAL_SECONDS,
    RANKINGS_SNAPSHOT_RETENTION_DAYS,
    RAW_DATA_DIR,
)
from src.data.name_utils import normalize_person_name, same_person_name
from src.data.rankings_scopes import WC_ALIASES as _WC_ALIASES
from src.storage_retention import prune_json_snapshot_history

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}
REQUEST_DELAY = 1.5
REQUEST_TIMEOUT_SECONDS = (5, 15)
REQUEST_ATTEMPTS = 2
REQUEST_RETRY_DELAY_SECONDS = 1.0

RANKINGS_SNAPSHOT_DIR = RAW_DATA_DIR / "rankings"
RANKINGS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_SCHEMA_VERSION = 1
RANKINGS_SNAPSHOT_MAX_AGE = timedelta(days=14)
_rankings_snapshot_prune_lock = threading.Lock()
_last_rankings_snapshot_prune_monotonic = 0.0

# Default rank for unranked fighters (same as training pipeline)
UNRANKED_DEFAULT = 16

_WOMENS_WC_LEGACY_KEYS = {
    "women's strawweight": "strawweight",
    "women's flyweight": "flyweight",
    "women's bantamweight": "bantamweight",
    "women's featherweight": "featherweight",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_name(name: str) -> str:
    """Normalize a fighter name for matching: lowercase, strip extra whitespace."""
    return normalize_person_name(name)


def _canonical_wc(wc: str) -> str:
    """Map a weight class string to its canonical key."""
    wc_lower = str(wc or "").lower().strip()
    wc_normalized = _normalize_name(wc_lower)
    if wc_lower in _WC_ALIASES:
        return _WC_ALIASES[wc_lower]
    if wc_normalized in _WC_ALIASES:
        return _WC_ALIASES[wc_normalized]
    for alias, canonical in sorted(_WC_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in wc_lower or alias in wc_normalized:
            return canonical
    return wc_lower


def _wc_lookup_keys(wc: str) -> list[str]:
    wc_key = _canonical_wc(wc)
    keys = [wc_key]
    legacy_key = _WOMENS_WC_LEGACY_KEYS.get(wc_key)
    if legacy_key and legacy_key not in keys:
        keys.append(legacy_key)
    return keys


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


def _successful_rankings_snapshot(snapshot: dict) -> bool:
    return bool(
        snapshot.get("status") == "success"
        and not snapshot.get("acquisition_failed")
        and snapshot.get("source") != "none"
    )


def prune_rankings_snapshots(
    *,
    now: datetime | float | int | None = None,
    protected_paths: tuple[Path, ...] = (),
    force: bool = False,
) -> int:
    """Preserve ordinary snapshots until every consumer owner is accounted for."""
    # Direct finalized history and a selected consumer manifest do not establish
    # the global accepted/training/monthly/recovery/uncertainty inventory. Force
    # bypasses no ownership requirement. Existing data receives no deletion gate.
    return 0


def _save_snapshot(snapshot: dict, *, reviewed_context=None, capture=None, history_root=None) -> Path:
    if snapshot.get("observation_reference_schema") == 1:
        from src.data.method_observation_history import _digest, _json_bytes, _write_immutable
        from src.data.rankings_observation_history import HISTORY_SUBDIR, finalize_snapshot
        from src.data.observation_consumer_references import record_observation_consumer_artifact
        body = _json_bytes(snapshot)
        base = _snapshot_path(snapshot["snapshot_time"])
        path = base.with_name(f"{base.stem}_{_digest(body)}.json")
        _write_immutable(path, body)
        evidence = []
        if reviewed_context is not None:
            evidence.append(Path(reviewed_context["path"]))
        if snapshot.get("collection_intent"):
            evidence.append(Path(snapshot["collection_intent"]["path"]))
        for descriptor in snapshot.get("response_captures", []):
            evidence.extend(Path(descriptor[key]["path"]) for key in ("raw_response", "response_envelope"))
        try:
            if snapshot.get("status") == "success":
                finalized = finalize_snapshot(path, history_root=(Path(history_root) if history_root is not None else RAW_DATA_DIR / "kaggle_rankings") / HISTORY_SUBDIR,
                                              reviewed_context=reviewed_context, capture=capture)
                evidence.append(finalized)
        finally:
            # Even a refused finalization retains its original snapshot and all
            # completed response attempts. Processing is outside request retry.
            record_observation_consumer_artifact(path, kind="rankings_snapshot", evidence_paths=evidence)
        return path
    path = _snapshot_path(snapshot["snapshot_time"])
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info("Saved rankings snapshot: %s", path)
    try:
        prune_rankings_snapshots(protected_paths=(path,))
    except Exception as exc:
        logger.warning("Rankings snapshot retention skipped: %s", exc)
    return path


def _load_snapshot(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read rankings snapshot %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    if "observation_reference_schema" in data:
        # This names-only legacy loader cannot consume a stable-ID observation,
        # even if its final history is absent or stored in another selected root.
        # A partial save must not turn an empty compatibility payload into 16.
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


def _fetch_html(url: str, *, response_capture=None, transport=None) -> Optional[str | bytes]:
    last_exc: Exception | None = None
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            resp = (transport or requests.get)(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            if response_capture is not None:
                response_capture(resp, requested_url=url, format="html")
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.content if response_capture is not None else resp.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < REQUEST_ATTEMPTS:
                logger.info(
                    "Rankings fetch failed for %s (attempt %s/%s): %s; retrying in %.1fs",
                    url,
                    attempt,
                    REQUEST_ATTEMPTS,
                    exc,
                    REQUEST_RETRY_DELAY_SECONDS,
                )
                time.sleep(REQUEST_RETRY_DELAY_SECONDS)

    logger.warning(
        "Rankings fetch failed for %s after %s attempt(s): %s",
        url,
        REQUEST_ATTEMPTS,
        last_exc,
    )
    return None


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


def _parse_ufc_rankings_html(html: str | bytes, *, strict: bool = False) -> Optional[dict]:
    if strict:
        from src.data.rankings_observation_history import parse_official_rankings_html
        return parse_official_rankings_html(html)
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


def _parse_espn_article_rankings(html: str) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Parse ranked fighters from an ESPN article-format rankings page.

    ESPN switched from table-based rankings to article pages where division
    headers are ``<h2>`` elements and individual ranked entries are ``<h3>``
    elements with text like ``1. Fighter Name``.
    """
    soup = BeautifulSoup(html, "lxml")

    wc_rankings: dict[str, dict[str, int]] = {}
    pfp_rankings: dict[str, int] = {}

    current_division: Optional[str] = None
    is_pfp = False

    # Walk h2 (division headers) and h3 (ranked entries) in document order.
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True)
        if not text:
            continue

        if tag.name == "h2":
            lower = text.lower()
            # Detect P4P headers — must match "pound-for-pound" / "pound for
            # pound", NOT weight-limit text like "(up to 265 pounds)".
            if "pound-for-pound" in lower or "pound for pound" in lower:
                current_division = None
                is_pfp = True
                continue
            # Strip parenthesized weight limits before weight class detection.
            wc_text = re.sub(r"\(.*?\)", "", lower).strip()
            wc_key = _canonical_wc(wc_text)
            if wc_key:
                current_division = wc_key
                is_pfp = False
                wc_rankings.setdefault(current_division, {})
            continue

        # h3: expect "N. Fighter Name" or "N. Fighter Name UFC champion" etc.
        match = re.match(r"(\d+)\.\s+(.+)", text)
        if not match:
            continue
        rank = int(match.group(1))
        raw_name = match.group(2)
        # Strip trailing titles like "UFC champion", "PFL champion"
        raw_name = re.sub(r"\s+(UFC|PFL|Bellator)\s+champion.*", "", raw_name, flags=re.IGNORECASE)
        name = _normalize_name(raw_name.strip())
        if len(name) <= 2:
            continue

        if is_pfp:
            pfp_rankings[name] = rank
        elif current_division is not None:
            wc_rankings[current_division][name] = rank

    return wc_rankings, pfp_rankings


def _parse_espn_rankings_html(html: str, pfp_html: Optional[str] = None) -> Optional[dict]:
    wc_rankings, _ = _parse_espn_article_rankings(html)
    pfp_rankings: dict[str, int] = {}

    if pfp_html:
        _, pfp_rankings = _parse_espn_article_rankings(pfp_html)

    if not wc_rankings and not pfp_rankings:
        logger.warning("ESPN: could not find any rankings in article HTML")
        return None

    return _normalize_rankings_payload(
        {"wc": wc_rankings, "pfp": pfp_rankings}, source="espn.com"
    )


def _parse_tapology_rankings_html(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "lxml")
    overview = _parse_tapology_rankings_overview_html(soup)
    if overview is not None:
        return overview

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

    parsed = _normalize_rankings_payload({"wc": wc_rankings, "pfp": pfp_rankings}, source="tapology.com")
    if parsed is not None:
        return parsed
    return _parse_tapology_rankings_overview_html(soup)


def _tapology_overview_division_key(text: str) -> str:
    lower = _normalize_name(text)
    return _WC_ALIASES.get(lower, "")


def _parse_tapology_rankings_overview_html(soup: BeautifulSoup) -> Optional[dict]:
    """Parse Tapology's current /rankings/ufc overview cards."""
    wc_rankings: dict[str, dict[str, int]] = {}
    current_division = ""

    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        raw_text = link.get_text(" ", strip=True)
        text = _normalize_name(raw_text)
        if not text:
            continue

        if "/rankings/ufc/" in href:
            division_key = _tapology_overview_division_key(text)
            if division_key:
                current_division = division_key
                wc_rankings.setdefault(current_division, {})
            continue

        if current_division and "/fightcenter/fighters/" in href:
            text = _normalize_name(re.sub(r'"[^"]+"', " ", raw_text))
            division = wc_rankings.setdefault(current_division, {})
            if text in division:
                continue
            if len(division) >= 15:
                continue
            division[text] = len(division) + 1

    return _normalize_rankings_payload({"wc": wc_rankings, "pfp": {}}, source="tapology.com")


def _scrape_ufc_rankings() -> Optional[dict]:
    html = _fetch_html("https://www.ufc.com/rankings")
    return _parse_ufc_rankings_html(html) if html else None


_ESPN_DIVISIONAL_URL = (
    "https://www.espn.com/mma/story/_/id/21807736"
    "/mma-divisional-rankings-ufc-bellator-pfl-rankings-weight-class"
)
_ESPN_PFP_URL = (
    "https://www.espn.com/mma/story/_/id/24067525"
    "/mma-pound-pound-rankings"
)


def _scrape_espn_rankings() -> Optional[dict]:
    html = _fetch_html(_ESPN_DIVISIONAL_URL)
    if not html:
        return None
    pfp_html = _fetch_html(_ESPN_PFP_URL)
    return _parse_espn_rankings_html(html, pfp_html=pfp_html)


def _scrape_tapology_rankings() -> Optional[dict]:
    try:
        from src.data.fallback_scrapers import _get_tapology_soup

        for url in (
            "https://www.tapology.com/rankings/ufc",
            "https://www.tapology.com/rankings/current-ufc-rankings",
        ):
            try:
                soup = _get_tapology_soup(url)
                rankings = _parse_tapology_rankings_html(str(soup))
                if rankings is not None:
                    return rankings
            except Exception as exc:
                log_fn = (
                    logger.info
                    if "Tapology blocked from this environment" in str(exc)
                    else logger.warning
                )
                log_fn("Tapology rankings browser-aware fetch failed for %s: %s", url, exc)
    except Exception as exc:
        logger.warning("Tapology rankings browser-aware fetch unavailable: %s", exc)
    return None


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


def collect_rankings_snapshot(*, strict: bool = False, reviewed_context=None, history_root=None, transport=None) -> dict:
    """
    Fetch rankings from live sources and persist a timestamped normalized snapshot.

    Inference should not call this. It exists for scheduled/offline collection.
    """
    if strict or reviewed_context is not None:
        return _collect_verified_rankings_snapshot(reviewed_context=reviewed_context,
                                                   history_root=history_root, transport=transport)
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


def _collect_verified_rankings_snapshot(*, reviewed_context, history_root, transport):
    from src.data.method_observation_history import ObservationRefusal, _attachment, _digest, _json_bytes, _read_pin, _require, _safe_path, _write_immutable
    from src.data.rankings_observation_history import CAPTURE_SUBDIR, SOURCE_URL, load_reviewed_registry
    from src.data.source_response_capture import SourceCaptureRefusal, SourceResponseCapture
    # No source work may establish or silently substitute the identity registry.
    registry = load_reviewed_registry(reviewed_context)
    original_context = _read_pin(reviewed_context)
    _require(transport is None or callable(transport), "invalid synthetic rankings transport")
    root = _safe_path(Path(history_root or RANKINGS_SNAPSHOT_DIR))
    collection_root = _safe_path(root / CAPTURE_SUBDIR / "collections" / _digest(registry["collection_id"].encode()))
    intent_path, ready_path = collection_root / "intent.json", collection_root / "ready.json"
    _safe_path(intent_path)
    _safe_path(ready_path)
    intent = {"schema": 1, "reviewed_context": reviewed_context, "context_sha256": _digest(original_context),
              "registry_original": _attachment(reviewed_context)}
    intent_body = _json_bytes(intent)
    intent_pin = {"path": str(intent_path), "bytes": len(intent_body), "sha256": _digest(intent_body)}
    if intent_path.exists():
        _require(json.loads(intent_path.read_bytes()) == intent, "rankings collection identity binds different original input")
        _require(ready_path.exists(), "original rankings collection remains interrupted; automatic reacquisition refused")
        ready = json.loads(_safe_path(ready_path).read_bytes())
        _require(isinstance(ready, dict) and set(ready) == {"schema", "kind", "snapshot"}
                 and type(ready["schema"]) is int and ready["schema"] == 1
                 and ready["kind"] == "rankings_collection_ready_v1",
                 "invalid original rankings ready checkpoint")
        snapshot = ready["snapshot"]
        _require(isinstance(snapshot, dict) and snapshot.get("reviewed_context") == reviewed_context
                 and snapshot.get("collection_id") == registry["collection_id"]
                 and snapshot.get("collection_intent") == intent_pin,
                 "rankings ready state differs from original registry/collection pin")
        captures = snapshot["response_captures"]
        path = _save_snapshot(snapshot, reviewed_context=reviewed_context,
                              capture=captures[-1] if captures else None, history_root=history_root)
        if snapshot["status"] == "processing_refused":
            raise SourceCaptureRefusal(f"original rankings processing remains refused: {snapshot.get('error')}", captures=captures)
        return {**snapshot, "snapshot_path": str(path)}
    collection_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        collection_root.mkdir()
    except FileExistsError as exc:
        raise ObservationRefusal("original rankings claim is incomplete or concurrent; reacquisition refused") from exc
    _write_immutable(intent_path, intent_body)
    captures = []
    sink = SourceResponseCapture("ufc_rankings", history_root=root,
                                 capture_subdir=CAPTURE_SUBDIR, capture_log=captures)
    snapshot = {"schema_version": 2, "observation_reference_schema": 1,
                "status": "failed", "source": "ufc.com", "acquisition_failed": True,
                "wc": {}, "pfp": {}, "sources": [], "response_captures": captures,
                "reviewed_context": reviewed_context, "collection_id": registry["collection_id"],
                "collection_intent": intent_pin}
    capture, processing_error = None, None
    try:
        html = _fetch_html(SOURCE_URL, response_capture=sink, transport=transport)
        if html is not None:
            capture = captures[-1]
            envelope = json.loads(_read_pin(capture["response_envelope"]))
            snapshot["captured_at"] = envelope["received_at"]
            snapshot["rankings_observation"] = _parse_ufc_rankings_html(html, strict=True)
            snapshot.update(status="success", acquisition_failed=False)
            snapshot["sources"].append({"source": "ufc.com", "status": "success", "captured_at": envelope["received_at"]})
        else:
            snapshot["sources"].append({"source": "ufc.com", "status": "acquisition_failed"})
    except Exception as exc:
        processing_error = exc
        snapshot.update(status="processing_refused", acquisition_failed=False, error=str(exc))
        snapshot["sources"].append({"source": "ufc.com", "status": "processing_refused", "error": str(exc)})
    snapshot["snapshot_time"] = datetime.now(timezone.utc).isoformat()
    if captures:
        # This is original receipt, never the timestamp taken before acquisition.
        last = json.loads(_read_pin(captures[-1]["response_envelope"]))
        snapshot.setdefault("captured_at", last["received_at"])
    snapshot["sources"].extend([
        {"source": "espn.com", "status": "strict_refused", "reason": "equivalent original identity/response proof unavailable"},
        {"source": "tapology.com", "status": "strict_refused", "reason": "equivalent original identity/response proof unavailable"},
    ])
    # The ready checkpoint is retained collection evidence. Its nested snapshot
    # is published separately with a consumer sidecar after finalization.
    _write_immutable(ready_path, _json_bytes({"schema": 1, "kind": "rankings_collection_ready_v1", "snapshot": snapshot}))
    path = _save_snapshot(snapshot, reviewed_context=reviewed_context, capture=capture, history_root=history_root)
    snapshot["snapshot_path"] = str(path)
    if processing_error is not None:
        raise SourceCaptureRefusal(f"rankings processing refused; original snapshot retained at {path}: {processing_error}",
                                   captures=captures) from processing_error
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
    *, fighter_id: Optional[str] = None, strict: bool = False, history_root=None,
) -> dict:
    """
    Get a fighter's weight class rank and P4P rank.

    Returns dict with:
      - wc_rank_feat: weight class rank (1-15, 16 for true unranked, NaN on source failure)
      - pfp_rank_feat: P4P rank (1-15, 16 for true unranked, NaN on source failure)
    """
    from src.data.rankings_observation_history import HISTORY_SUBDIR, read_finalized_rankings, select_finalized_rankings
    from src.data.selected_source_inputs import selected_path
    retained_root = selected_path("rankings_observations", Path(history_root) if history_root is not None else RAW_DATA_DIR / "kaggle_rankings" / HISTORY_SUBDIR)
    if strict or fighter_id is not None or retained_root.exists():
        rows = read_finalized_rankings(retained_root)
        now = datetime.now(timezone.utc)
        if as_of_date is None:
            rows = [row for row in rows if now - _parse_snapshot_timestamp(row["observed_at"]) <= RANKINGS_SNAPSHOT_MAX_AGE]
        result = select_finalized_rankings(rows, fighter_id=fighter_id, weight_class=weight_class,
                                           as_of=as_of_date or now.isoformat())
        return {key: np.nan if value is None else value for key, value in result.items()}
    rankings = get_rankings(as_of_date=as_of_date)
    name_lower = _normalize_name(fighter_name)
    if rankings.get("acquisition_failed") or rankings.get("source") == "none":
        return {
            "wc_rank_feat": np.nan,
            "pfp_rank_feat": np.nan,
        }

    wc_rank = UNRANKED_DEFAULT
    if weight_class:
        for wc_key in _wc_lookup_keys(weight_class):
            wc_division = rankings.get("wc", {}).get(wc_key, {})
            for ranked_name, rank in wc_division.items():
                if _names_match(name_lower, ranked_name):
                    wc_rank = rank
                    break
            if wc_rank != UNRANKED_DEFAULT:
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
