"""Recover post-cutoff method odds with exact, pre-fight provenance.

The campaign deliberately reuses the project's existing BestFightOdds direct,
Playwright, and Wayback acquisition paths, but all returned HTML is parsed by
``src.data.method_odds._parse_bfo_method_odds``.  That parser accepts only one
exact matchup-scoped block.  No fuzzy or page-wide result is publishable.

This script does not train or score a model.  Its deterministic output chooses
the registered 211- or 205-feature contract solely from the fixed data-quality
gate, before any model metrics can be consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qsl, quote_plus, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MIN_FIGHTER_FIGHTS, ODDS_API_BASE_URL, ODDS_API_KEY
from src.data import method_odds
from src.data.io_utils import write_csv_atomically, write_json_atomically
from src.data.name_utils import normalize_cross_source_name, normalize_person_name
from src.model.training_spec import V4_METHOD_ODDS_FEATURE_COLS


CUTOFF_DATE = pd.Timestamp("2026-03-07")
PREFIGHT_LEAD = timedelta(hours=24)
RECENT_WINDOW_DAYS = 60
MIN_RECENT_ESTABLISHED_ROWS = 50
MIN_COMPLETE_COVERAGE = 0.70
MAX_DATA_LAG_DAYS = 35
RECOVERY_PASS_ORDER = ("revalidate", "direct", "playwright", "wayback")
_ACQUIRER_PASS_NAMES = {
    "direct_bfo": "direct",
    "playwright_bfo": "playwright",
    "wayback_bfo": "wayback",
}
_PASS_SOURCES = {
    "revalidate": "schema_v1",
    "direct": "direct_bfo",
    "playwright": "playwright_bfo",
    "wayback": "wayback_bfo",
}
_ATTEMPT_STATUSES = {
    "accepted",
    "candidate_available",
    "error",
    "no_evidence",
    "not_revalidated",
    "precondition_failed",
    "rejected",
    "validated_redundant",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "key",
    "password",
    "secret",
    "signature",
    "token",
}

# Exact source-label variants reviewed only for commence-time recovery.  They
# are intentionally local to this evidence campaign and never broaden model,
# market, or live fighter identity matching.
_COMMENCE_SOURCE_NAME_ALIASES = {
    normalize_person_name(source): normalize_person_name(target)
    for source, target in {
        "Beatriz Mesquita": "Bia Mesquita",
        "Montserrat Rendon": "Montse Rendon",
        "Su Young You": "SuYoung You",
        "Enrique Hecher Sosa": "Hecher Sosa",
        "Abdulrakhman Yakhyaev": "Abdul Rakhman Yakhyaev",
        "Landon Vannata": "Lando Vannata",
        "Paulo Henrique Costa": "Paulo Costa",
        "Lupita Godinez": "Loopy Godinez",
        "Darya Zheleznyakova": "Daria Zhelezniakova",
        "J.J. Aldrich": "JJ Aldrich",
        "Michael Aswell": "Michael Aswell Jr.",
        "Charlie Radtke": "Charles Radtke",
        "Alex Hernandez": "Alexander Hernandez",
        "Thomas Gantt": "Tommy Gantt",
        "Stephen Erceg": "Steve Erceg",
        "Baysangur Susurkaev": "Baisangur Susurkaev",
        "Marco Tulio Silva": "Marco Tulio",
        "Ming Shi": "Shi Ming",
        "Su Mudaerji": "Sumudaerji",
        "Jingnan Xiong": "Xiong Jingnan",
        "Aori Qileng": "Aoriqileng",
        "Meng Ding": "Ding Meng",
        "Jose Henrique": "Jose Souza",
        "José Henrique": "Jose Souza",
        "Yi Sak Lee": "YiSak Lee",
        "Luis Dias de Assis": "Luis Felipe Dias",
        "Kangjie Zhu": "Zhu Kangjie",
        "Sergey Pavlovich": "Sergei Pavlovich",
        "Zhu Rong": "Rongzhu",
        "Abusupyian Magomedov": "Abus Magomedov",
        "Javier Reyes Rugeles": "Javier Reyes",
        "Muhammad Said": "Muhammad Saidov",
        "Ramazonbek Temirov": "Ramazan Temirov",
        # Exact line-history labels observed in the frozen integrity-v2 source
        # inventory.  These are reviewed one-to-one fighter identities, not a
        # fuzzy suffix/spelling rule.
        "Steve Garcia Jr.": "Steve Garcia",
        "Asu Almabaev": "Asu Almabayev",
        "Kai Kamaka": "Kai Kamaka III",
        "Zachary Reese": "Zach Reese",
        "Levi Rodrigues": "Levi Rodrigues Jr.",
        "Seokhyeon Ko": "Seok Hyun Ko",
    }.items()
}

METHOD_FEATURE_COLUMNS = (
    "a_ko_odds_prob",
    "a_sub_odds_prob",
    "a_dec_odds_prob",
    "b_ko_odds_prob",
    "b_sub_odds_prob",
    "b_dec_odds_prob",
)
if set(METHOD_FEATURE_COLUMNS) != set(V4_METHOD_ODDS_FEATURE_COLS):
    raise RuntimeError("Recovery method feature list diverges from V4_METHOD_ODDS_FEATURE_COLS")

CONTRACT_211 = "full_live_contract_v6_integrity_211"
CONTRACT_205 = "full_live_contract_v6_integrity_205"
CONTRACT_211_FULLFIT = "full_live_contract_v6_integrity_211_fullfit"
CONTRACT_205_FULLFIT = "full_live_contract_v6_integrity_205_fullfit"

TERMINAL_COLUMNS = (
    "target_key",
    "event_date",
    "commence_time",
    "commence_time_source_kind",
    "commence_time_source_locator",
    "commence_time_source_observed_at",
    "commence_time_source_sha256",
    "commence_time_source_event_id",
    "commence_time_source_matchup",
    "commence_time_semantics",
    "event_id",
    "event_title",
    "fighter_a",
    "fighter_b",
    "a_num_fights",
    "b_num_fights",
    "established_fighters",
    "status",
    "terminal_reason",
    "recovery_preconditions_met",
    "target_recovery_complete",
    "attempted_sources",
    "success_source",
    "source_url",
    "original_source_url",
    "observation_captured_at",
    "validation_captured_at",
    "page_sha256",
    "parser_contract",
    "revalidated_schema_v1",
    "schema_v1_snapshot_path",
    "schema_v1_snapshot_sha256",
    "method_odds_complete",
    *METHOD_FEATURE_COLUMNS,
)


class RecoveryInputError(ValueError):
    """Raised when the target manifest cannot define a safe campaign."""


class RecoveryRunCache:
    """Small success-only cache whose lifetime is one recovery invocation."""

    def __init__(self) -> None:
        self._event_urls: dict[str, list[str]] = {}
        self._direct_pages: dict[str, dict] = {}
        self._playwright_pages: dict[str, dict] = {}
        self._wayback_cdx: dict[tuple[str, str], tuple[str, ...]] = {}
        self._wayback_pages: dict[str, dict] = {}

    @staticmethod
    def _event_key(target: dict) -> str:
        event_date = _text(target.get("event_date"))[:10]
        event_title = re.sub(
            r"\s+", " ", _text(target.get("event_title")).casefold()
        ).strip()
        if event_date and event_title:
            return f"date-title:{event_date}|{event_title}"
        event_id = _text(target.get("event_id"))
        return f"id:{event_id.casefold()}" if event_id else ""

    @staticmethod
    def _is_bfo_event_url(url: str) -> bool:
        parsed = urlparse(url)
        return bool(
            parsed.scheme in {"http", "https"}
            and (parsed.hostname or "").casefold()
            in {"bestfightodds.com", "www.bestfightodds.com"}
            and parsed.path.startswith("/events/")
        )

    def event_urls_for(self, target: dict) -> list[str]:
        key = self._event_key(target)
        return list(self._event_urls.get(key, ())) if key else []

    def remember_exact_event_url(self, target: dict, url: object) -> None:
        """Remember a URL only after the caller parsed this exact matchup."""
        key = self._event_key(target)
        normalized_url = _text(url)
        if not key or not self._is_bfo_event_url(normalized_url):
            return
        urls = self._event_urls.setdefault(key, [])
        if normalized_url not in urls:
            urls.append(normalized_url)

    def _page_store(self, acquisition: str) -> Optional[dict[str, dict]]:
        if acquisition == "direct_bfo":
            return self._direct_pages
        if acquisition == "playwright_bfo":
            return self._playwright_pages
        if acquisition == "wayback_bfo":
            return self._wayback_pages
        return None

    def raw_page(self, acquisition: str, url: str) -> Optional[dict]:
        store = self._page_store(acquisition)
        cached = store.get(url) if store is not None else None
        return dict(cached) if cached is not None else None

    def remember_raw_page(self, acquisition: str, url: str, raw: dict) -> None:
        """Cache only a complete successful response, preserving first capture."""
        store = self._page_store(acquisition)
        if store is None or url in store or _utc_datetime(raw.get("captured_at")) is None:
            return
        html = raw.get("html")
        if isinstance(html, bytes):
            has_content = bool(html)
        else:
            has_content = bool(str(html or "").strip())
        if not has_content or _text(raw.get("source_url")) != url:
            return
        store[url] = dict(raw)

    @staticmethod
    def _cutoff_key(cutoff: datetime) -> str:
        return cutoff.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")

    def wayback_timestamps(self, original_url: str, cutoff: datetime) -> list[str]:
        return list(
            self._wayback_cdx.get(
                (original_url, self._cutoff_key(cutoff)),
                (),
            )
        )

    def remember_wayback_timestamps(
        self,
        original_url: str,
        cutoff: datetime,
        timestamps: Iterable[str],
    ) -> None:
        values = tuple(timestamps)
        # An empty CDX result is a negative result and is deliberately retried.
        if values:
            self._wayback_cdx.setdefault(
                (original_url, self._cutoff_key(cutoff)),
                values,
            )


def _text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _utc_datetime(value: object) -> Optional[datetime]:
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pair_key(fighter_a: object, fighter_b: object) -> str:
    names = sorted(
        [
            normalize_cross_source_name(fighter_a),
            normalize_cross_source_name(fighter_b),
        ]
    )
    return "|".join(names) if all(names) else ""


def _target_key(row: dict) -> str:
    event_id = _text(row.get("event_id"))
    identity = event_id or _text(row.get("event_date"))[:10]
    return f"{identity}|{_pair_key(row.get('fighter_a'), row.get('fighter_b'))}"


def _safe_public_url(value: object) -> str:
    url = _text(value)
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    if any(
        key.casefold() in _SENSITIVE_QUERY_KEYS
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        return ""
    return url


def _safe_detail(value: object) -> str:
    detail = _text(value)[:500]
    detail = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|authorization)=([^\s;&]+)",
        r"\1=<redacted>",
        detail,
    )
    return re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1<redacted>@", detail)


def _url_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        text = _text(value)
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            candidates = parsed if isinstance(parsed, list) else [text]
        else:
            candidates = text.split(";")
    urls: list[str] = []
    for candidate in candidates:
        url = _safe_public_url(candidate)
        if url and url not in urls:
            urls.append(url)
    return urls


def _optional_count(value: object) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(numeric)


def _established(row: dict) -> bool:
    a_count = _optional_count(row.get("a_num_fights"))
    b_count = _optional_count(row.get("b_num_fights"))
    return bool(
        a_count is not None
        and b_count is not None
        and a_count >= MIN_FIGHTER_FIGHTS
        and b_count >= MIN_FIGHTER_FIGHTS
    )


def prepare_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, filter, deduplicate, and deterministically order target fights."""
    required = {"event_date", "fighter_a", "fighter_b"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RecoveryInputError(f"target manifest is missing columns: {missing}")

    targets = frame.copy()
    parsed_dates = pd.to_datetime(targets["event_date"], errors="coerce")
    if parsed_dates.isna().any():
        bad_rows = targets.index[parsed_dates.isna()].tolist()[:10]
        raise RecoveryInputError(f"target manifest has invalid event_date rows: {bad_rows}")
    targets["event_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    targets = targets[parsed_dates > CUTOFF_DATE].copy()
    if targets.empty:
        raise RecoveryInputError("target manifest has no fights after 2026-03-07")

    for column in (
        "commence_time",
        "event_id",
        "event_title",
        "source_url",
        "a_num_fights",
        "b_num_fights",
        "commence_time_source_kind",
        "commence_time_source_locator",
        "commence_time_source_observed_at",
        "commence_time_source_sha256",
        "commence_time_source_event_id",
        "commence_time_source_matchup",
        "commence_time_semantics",
    ):
        if column not in targets.columns:
            targets[column] = ""

    targets["fighter_a"] = targets["fighter_a"].map(_text)
    targets["fighter_b"] = targets["fighter_b"].map(_text)
    if (targets["fighter_a"] == "").any() or (targets["fighter_b"] == "").any():
        raise RecoveryInputError("every target requires two fighter names")

    records = targets.to_dict(orient="records")
    targets["target_key"] = [_target_key(record) for record in records]
    duplicate_keys = sorted(
        targets.loc[targets["target_key"].duplicated(keep=False), "target_key"].unique()
    )
    if duplicate_keys:
        raise RecoveryInputError(f"target manifest has duplicate fight keys: {duplicate_keys[:10]}")

    targets = targets.sort_values(
        ["event_date", "target_key"], kind="mergesort"
    ).reset_index(drop=True)
    return targets


def _event_url_map(path: Optional[Path]) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    fighter_a_col = "fighter_a" if "fighter_a" in frame.columns else "fighter1"
    fighter_b_col = "fighter_b" if "fighter_b" in frame.columns else "fighter2"
    url_col = "source_url" if "source_url" in frame.columns else "bfo_url"
    if not {"event_date", fighter_a_col, fighter_b_col, url_col}.issubset(frame.columns):
        return {}
    mapped: dict[str, list[str]] = {}
    for row in frame.to_dict(orient="records"):
        key = f"{_text(row.get('event_date'))[:10]}|{_pair_key(row.get(fighter_a_col), row.get(fighter_b_col))}"
        for url in _url_values(row.get(url_col)):
            mapped.setdefault(key, [])
            if url not in mapped[key]:
                mapped[key].append(url)
    return mapped


def _enrich_commence_times_from_line_history(
    targets: pd.DataFrame,
    line_history_dirs: Iterable[Path],
) -> pd.DataFrame:
    """Fill starts from the latest unambiguous verified exact-pair snapshot."""
    from src.data.historical_backfill import load_line_history_odds

    starts: dict[str, list[dict]] = {}
    source_receipts: dict[Path, tuple[str, str]] = {}
    for directory in line_history_dirs:
        directory = Path(directory)
        history = load_line_history_odds(directory)
        for row in history.to_dict(orient="records"):
            commence = _utc_datetime(row.get("commence_time"))
            observed = _utc_datetime(row.get("observed_at") or row.get("snapshot_time"))
            pair = _pair_key(row.get("fighter_a"), row.get("fighter_b"))
            event_date = _text(row.get("event_date"))[:10]
            if commence is None or observed is None or not pair or not event_date:
                continue
            source_file = Path(_text(row.get("source_file")))
            source_path = source_file if source_file.is_absolute() else directory / source_file
            resolved_source = source_path.resolve()
            receipt = source_receipts.get(resolved_source)
            if receipt is None:
                try:
                    source_bytes = resolved_source.read_bytes()
                except OSError:
                    continue
                try:
                    source_locator = resolved_source.relative_to(
                        REPO_ROOT.resolve()
                    ).as_posix()
                except ValueError:
                    source_locator = str(resolved_source)
                receipt = (source_locator, _sha256_bytes(source_bytes))
                source_receipts[resolved_source] = receipt
            source_locator, source_sha256 = receipt
            starts.setdefault(f"{event_date}|{pair}", []).append(
                {
                    "observed": observed,
                    "commence": commence.isoformat(),
                    "source_locator": source_locator,
                    "source_sha256": source_sha256,
                    "event_id": _text(row.get("event_id")),
                    "matchup": (
                        f"{_text(row.get('fighter_a'))} vs "
                        f"{_text(row.get('fighter_b'))}"
                    ),
                }
            )

    result = targets.copy()
    for index, row in result.iterrows():
        key = f"{row['event_date']}|{_pair_key(row['fighter_a'], row['fighter_b'])}"
        candidates = starts.get(key, [])
        existing = _utc_datetime(row.get("commence_time"))
        if existing is not None:
            candidates = [
                candidate
                for candidate in candidates
                if _utc_datetime(candidate["commence"]) == existing
            ]
        if candidates:
            latest_observed = max(candidate["observed"] for candidate in candidates)
            latest_starts = {
                candidate["commence"]
                for candidate in candidates
                if candidate["observed"] == latest_observed
            }
            if len(latest_starts) == 1:
                selected = sorted(
                    (
                        candidate
                        for candidate in candidates
                        if candidate["observed"] == latest_observed
                        and candidate["commence"] in latest_starts
                    ),
                    key=lambda candidate: (
                        candidate["source_locator"],
                        candidate["source_sha256"],
                    ),
                )
                selected = selected[0]
                result.at[index, "commence_time"] = selected["commence"]
                result.at[index, "commence_time_source_kind"] = "line_history"
                result.at[index, "commence_time_source_locator"] = selected[
                    "source_locator"
                ]
                result.at[index, "commence_time_source_observed_at"] = (
                    latest_observed.isoformat()
                )
                result.at[index, "commence_time_source_sha256"] = selected[
                    "source_sha256"
                ]
                result.at[index, "commence_time_source_event_id"] = selected[
                    "event_id"
                ]
                result.at[index, "commence_time_source_matchup"] = selected[
                    "matchup"
                ]
                result.at[index, "commence_time_semantics"] = "scheduled_bout_time"
    return result


def _commence_name_key(value: object) -> str:
    normalized = normalize_person_name(value)
    return _COMMENCE_SOURCE_NAME_ALIASES.get(normalized, normalized)


def _commence_event_match_rank(target: dict, event: dict) -> Optional[int]:
    """Return an exact reviewed-label rank, or None when the pair is not exact."""
    target_names = [
        normalize_person_name(target.get("fighter_a")),
        normalize_person_name(target.get("fighter_b")),
    ]
    source_names = [
        normalize_person_name(event.get("home_team")),
        normalize_person_name(event.get("away_team")),
    ]
    if not all(target_names) or not all(source_names):
        return None

    best_exact = -1
    for ordered_source in (source_names, list(reversed(source_names))):
        if [_commence_name_key(value) for value in target_names] != [
            _commence_name_key(value) for value in ordered_source
        ]:
            continue
        exact = sum(
            target_name == source_name
            for target_name, source_name in zip(target_names, ordered_source, strict=True)
        )
        best_exact = max(best_exact, exact)
    return 2 - best_exact if best_exact >= 0 else None


def _select_exact_commence_event(target: dict, events: Iterable[dict]) -> Optional[dict]:
    ranked: list[tuple[int, dict]] = []
    target_date = _text(target.get("event_date"))[:10]
    for event in events:
        if not isinstance(event, dict):
            continue
        commence = _utc_datetime(event.get("commence_time") or event.get("startDate"))
        if commence is None:
            continue
        source_date = pd.Timestamp(commence).tz_convert("America/New_York").strftime(
            "%Y-%m-%d"
        )
        if source_date != target_date:
            continue
        rank = _commence_event_match_rank(target, event)
        if rank is not None:
            ranked.append((rank, event))
    if not ranked:
        return None
    best_rank = min(rank for rank, _ in ranked)
    best = [event for rank, event in ranked if rank == best_rank]
    identities = {
        (
            _text(event.get("id") or event.get("@id")),
            _text(event.get("commence_time") or event.get("startDate")),
            normalize_person_name(event.get("home_team")),
            normalize_person_name(event.get("away_team")),
        )
        for event in best
    }
    return best[0] if len(identities) == 1 else None


def _stamp_commence_source(
    result: pd.DataFrame,
    index: object,
    *,
    event: dict,
    source_kind: str,
    source_locator: str,
    source_observed_at: object,
    source_sha256: str,
    source_semantics: str = "scheduled_bout_time",
) -> None:
    commence = _utc_datetime(event.get("commence_time") or event.get("startDate"))
    if commence is None:
        return
    result.at[index, "commence_time"] = commence.isoformat()
    result.at[index, "commence_time_source_kind"] = source_kind
    result.at[index, "commence_time_source_locator"] = source_locator
    result.at[index, "commence_time_source_observed_at"] = _text(source_observed_at)
    result.at[index, "commence_time_source_sha256"] = source_sha256
    result.at[index, "commence_time_source_event_id"] = _text(
        event.get("id") or event.get("@id")
    )
    result.at[index, "commence_time_source_matchup"] = (
        f"{_text(event.get('home_team'))} vs {_text(event.get('away_team'))}"
    )
    result.at[index, "commence_time_semantics"] = source_semantics


def enrich_commence_times_from_odds_api_history(
    targets: pd.DataFrame,
    *,
    api_key: str,
    request_get: Callable = requests.get,
) -> pd.DataFrame:
    """Fill missing exact starts from one historical event snapshot per card date."""
    result = targets.copy()
    missing = result["commence_time"].map(_utc_datetime).isna()
    if not missing.any() or not _text(api_key):
        return result

    endpoint = (
        f"{ODDS_API_BASE_URL}/historical/sports/"
        "mma_mixed_martial_arts/events"
    )
    for event_date, group in result.loc[missing].groupby("event_date", sort=True):
        date = pd.Timestamp(event_date)
        query_times = (
            (date - pd.Timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z"),
            date.strftime("%Y-%m-%dT00:00:00Z"),
        )
        for query_at in query_times:
            pending = [
                index
                for index in group.index
                if _utc_datetime(result.at[index, "commence_time"]) is None
            ]
            if not pending:
                break
            try:
                response = request_get(
                    endpoint,
                    params={"apiKey": api_key, "date": query_at},
                    timeout=30,
                )
            except requests.RequestException:
                continue
            if getattr(response, "status_code", None) != 200:
                continue
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                continue
            raw_bytes = getattr(response, "content", b"")
            if not isinstance(raw_bytes, bytes) or not raw_bytes:
                raw_bytes = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            locator = f"{endpoint}?date={query_at}"
            for index in pending:
                event = _select_exact_commence_event(
                    result.loc[index].to_dict(), payload["data"]
                )
                if event is None:
                    continue
                _stamp_commence_source(
                    result,
                    index,
                    event=event,
                    source_kind="odds_api_historical_events",
                    source_locator=locator,
                    source_observed_at=payload.get("timestamp") or query_at,
                    source_sha256=_sha256_bytes(raw_bytes),
                )
    return result


def _iter_jsonld_sports_events(value: object) -> Iterable[dict]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_jsonld_sports_events(item)
        return
    if not isinstance(value, dict):
        return
    types = value.get("@type")
    type_values = types if isinstance(types, list) else [types]
    if "SportsEvent" in type_values:
        competitors = value.get("competitor") or value.get("performer") or []
        names = [
            _text(item.get("name"))
            for item in competitors
            if isinstance(item, dict) and _text(item.get("name"))
        ]
        if len(names) == 2:
            event = dict(value)
            event["home_team"], event["away_team"] = names
            event["commence_time"] = value.get("startDate")
            yield event
    for nested in value.values():
        if isinstance(nested, (dict, list)):
            yield from _iter_jsonld_sports_events(nested)


def enrich_commence_times_from_jsonld_pages(
    targets: pd.DataFrame,
    urls: Iterable[str],
    *,
    request_get: Callable = requests.get,
) -> pd.DataFrame:
    """Fill remaining starts from explicit exact-match JSON-LD SportsEvent pages."""
    result = targets.copy()
    for url in dict.fromkeys(
        safe_url for value in urls if (safe_url := _safe_public_url(value))
    ):
        if not result["commence_time"].map(_utc_datetime).isna().any():
            break
        try:
            response = request_get(
                url,
                headers={"User-Agent": "ufc-betting-bot-integrity-recovery/1.0"},
                timeout=30,
            )
        except requests.RequestException:
            continue
        raw_bytes = getattr(response, "content", b"")
        if getattr(response, "status_code", None) != 200 or not raw_bytes:
            continue
        soup = BeautifulSoup(raw_bytes, "html.parser")
        events: list[dict] = []
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                payload = json.loads(script.get_text())
            except (TypeError, json.JSONDecodeError):
                continue
            events.extend(_iter_jsonld_sports_events(payload))
        if not events:
            continue
        observed_at = _now_iso()
        starts = [
            _text(event.get("commence_time") or event.get("startDate"))
            for event in events
        ]
        repeated_starts = {
            start for start in starts if start and starts.count(start) > 1
        }
        missing = result["commence_time"].map(_utc_datetime).isna()
        for index, target in result.loc[missing].iterrows():
            event = _select_exact_commence_event(target.to_dict(), events)
            if event is None:
                continue
            _stamp_commence_source(
                result,
                index,
                event=event,
                source_kind="jsonld_sports_event",
                source_locator=url,
                source_observed_at=observed_at,
                source_sha256=_sha256_bytes(raw_bytes),
                source_semantics=(
                    "card_start_lower_bound"
                    if _text(event.get("commence_time") or event.get("startDate"))
                    in repeated_starts
                    else "scheduled_bout_time"
                ),
            )
    return result


def load_targets(
    path: Path,
    *,
    event_url_map: Optional[Path] = None,
    line_history_dirs: Iterable[Path] = (),
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    wanted = {
        "event_date",
        "event_name",
        "fighter_a",
        "fighter_b",
        "commence_time",
        "event_id",
        "event_title",
        "source_url",
        "a_num_fights",
        "b_num_fights",
    }
    frame = pd.read_csv(path, usecols=[column for column in header.columns if column in wanted])
    if "event_name" in frame.columns:
        if "event_title" not in frame.columns:
            frame = frame.rename(columns={"event_name": "event_title"})
        else:
            missing_title = frame["event_title"].map(_text).eq("")
            frame.loc[missing_title, "event_title"] = frame.loc[
                missing_title, "event_name"
            ]
            frame = frame.drop(columns="event_name")
    targets = prepare_targets(frame)
    targets = _enrich_commence_times_from_line_history(targets, line_history_dirs)
    mapped = _event_url_map(event_url_map)
    for index, row in targets.iterrows():
        map_key = f"{row['event_date']}|{_pair_key(row['fighter_a'], row['fighter_b'])}"
        urls = _url_values(row.get("source_url"))
        for url in mapped.get(map_key, []):
            if url not in urls:
                urls.append(url)
        targets.at[index, "source_url"] = ";".join(urls)
    return targets


def _method_values(payload: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in METHOD_FEATURE_COLUMNS:
        try:
            value = float(payload.get(column))
        except (TypeError, ValueError):
            value = np.nan
        values[column] = value if math.isfinite(value) and 0.0 <= value <= 1.0 else np.nan
    return values


def _complete(values: dict[str, float]) -> bool:
    return all(math.isfinite(float(values.get(column, np.nan))) for column in METHOD_FEATURE_COLUMNS)


def _values_match(candidate: dict[str, float], evidence: dict[str, float]) -> bool:
    compared = 0
    for column in METHOD_FEATURE_COLUMNS:
        candidate_value = float(candidate.get(column, np.nan))
        if not math.isfinite(candidate_value):
            continue
        evidence_value = float(evidence.get(column, np.nan))
        if not math.isfinite(evidence_value):
            return False
        compared += 1
        if round(candidate_value, 6) != round(evidence_value, 6):
            return False
    return compared > 0


def load_schema_v1_candidates(snapshot_dirs: Iterable[Path]) -> dict[str, list[dict]]:
    """Load schema-v1 records as quarantined candidates, never inference data."""
    indexed: dict[str, list[dict]] = {}
    paths: set[Path] = set()
    for directory in snapshot_dirs:
        paths.update(Path(directory).glob("method_odds_*.json"))

    for path in sorted(paths):
        try:
            raw_bytes = path.read_bytes()
            snapshot = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            continue
        snapshot_time = _text(snapshot.get("snapshot_time"))
        for record in snapshot.get("records") or []:
            if not isinstance(record, dict):
                continue
            key = _pair_key(record.get("fighter_a"), record.get("fighter_b"))
            if not key:
                continue
            indexed.setdefault(key, []).append(
                {
                    "record": record,
                    "snapshot_time": snapshot_time,
                    "snapshot_path": str(path),
                    "snapshot_sha256": _sha256_bytes(raw_bytes),
                }
            )
    return indexed


def _v1_record_event_date(record: dict) -> str:
    explicit = _text(record.get("event_date"))
    if explicit:
        parsed = pd.to_datetime(explicit, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")
    commence = _utc_datetime(record.get("commence_time"))
    if commence is None:
        return ""
    return pd.Timestamp(commence).tz_convert("America/New_York").strftime("%Y-%m-%d")


def _enrich_commence_times_from_v1(
    targets: pd.DataFrame,
    v1_index: dict[str, list[dict]],
) -> pd.DataFrame:
    """Recover scheduling metadata only; schema-v1 method values stay quarantined."""
    result = targets.copy()
    for index, target in result.iterrows():
        existing = _utc_datetime(target.get("commence_time"))
        if existing is not None and _SHA256_RE.fullmatch(
            _text(target.get("commence_time_source_sha256"))
        ):
            continue
        pair = _pair_key(target.get("fighter_a"), target.get("fighter_b"))
        event_date = _text(target.get("event_date"))[:10]
        starts: dict[str, list[dict]] = {}
        for entry in v1_index.get(pair, []):
            record = entry.get("record") if isinstance(entry, dict) else None
            if not isinstance(record, dict):
                continue
            if _pair_key(record.get("fighter_a"), record.get("fighter_b")) != pair:
                continue
            if _v1_record_event_date(record) != event_date:
                continue
            commence = _utc_datetime(record.get("commence_time"))
            if commence is not None and (existing is None or commence == existing):
                starts.setdefault(commence.isoformat(), []).append(entry)
        if len(starts) == 1:
            commence_time, entries = next(iter(starts.items()))
            result.at[index, "commence_time"] = commence_time
            selected = sorted(
                entries,
                key=lambda item: (
                    _text(item.get("snapshot_path")),
                    _text(item.get("snapshot_sha256")),
                ),
            )[0]
            result.at[index, "commence_time_source_kind"] = "schema_v1_snapshot"
            result.at[index, "commence_time_source_locator"] = _text(
                selected.get("snapshot_path")
            )
            result.at[index, "commence_time_source_observed_at"] = _text(
                selected.get("snapshot_time")
            )
            result.at[index, "commence_time_source_sha256"] = _text(
                selected.get("snapshot_sha256")
            )
            record = selected.get("record") or {}
            result.at[index, "commence_time_source_event_id"] = _text(
                record.get("event_id")
            )
            result.at[index, "commence_time_source_matchup"] = (
                f"{_text(record.get('fighter_a'))} vs {_text(record.get('fighter_b'))}"
            )
            result.at[index, "commence_time_semantics"] = "scheduled_bout_time"
    return result


def _commence_provenance_complete(target: dict) -> bool:
    if _utc_datetime(target.get("commence_time")) is None:
        return False
    if _text(target.get("commence_time_source_kind")) not in {
        "line_history",
        "schema_v1_snapshot",
        "odds_api_historical_events",
        "jsonld_sports_event",
    }:
        return False
    if _text(target.get("commence_time_semantics")) not in {
        "scheduled_bout_time",
        "card_start_lower_bound",
    }:
        return False
    if not _text(target.get("commence_time_source_locator")):
        return False
    if _utc_datetime(target.get("commence_time_source_observed_at")) is None:
        return False
    if not _SHA256_RE.fullmatch(_text(target.get("commence_time_source_sha256"))):
        return False
    matchup = _text(target.get("commence_time_source_matchup"))
    matchup_names = re.split(r"\s+vs\.?\s+", matchup, maxsplit=1, flags=re.IGNORECASE)
    return len(matchup_names) == 2 and _commence_event_match_rank(
        target,
        {"home_team": matchup_names[0], "away_team": matchup_names[1]},
    ) is not None


def _v1_candidates_for(target: dict, index: dict[str, list[dict]]) -> list[dict]:
    commence = _utc_datetime(target.get("commence_time"))
    event_id = _text(target.get("event_id"))
    candidates: list[dict] = []
    for entry in index.get(_pair_key(target.get("fighter_a"), target.get("fighter_b")), []):
        record = entry["record"]
        matched, home_is_a = method_odds._match_fight(
            record.get("fighter_a", ""),
            record.get("fighter_b", ""),
            target.get("fighter_a", ""),
            target.get("fighter_b", ""),
        )
        if not matched or home_is_a is None:
            continue
        record_event_id = _text(record.get("event_id"))
        if event_id and record_event_id and event_id != record_event_id:
            continue
        record_commence = _utc_datetime(record.get("commence_time"))
        if commence is None or record_commence is None:
            continue
        if abs((record_commence - commence).total_seconds()) > 12 * 60 * 60:
            continue
        captured_at = _utc_datetime(record.get("captured_at") or entry.get("snapshot_time"))
        if captured_at is None:
            continue
        try:
            oriented = method_odds._orient_result_for_query(record, home_is_a)
        except (TypeError, ValueError, KeyError):
            continue
        values = _method_values(oriented)
        if not any(math.isfinite(value) for value in values.values()):
            continue
        candidates.append(
            {
                **entry,
                "captured_at": captured_at,
                "source_url": _text(record.get("source_url")),
                "values": values,
                "prefight": captured_at <= commence - PREFIGHT_LEAD,
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["captured_at"], reverse=True)


def _acquirer_payload(
    evidence: Optional[list[dict]] = None,
    *,
    urls: Optional[list[str]] = None,
    error: str = "",
) -> dict:
    return {"evidence": evidence or [], "urls": urls or [], "error": error}


def acquire_direct_bfo(
    target: dict,
    known_urls: list[str],
    *,
    cache: Optional[RecoveryRunCache] = None,
) -> dict:
    urls = list(known_urls)
    errors: list[str] = []
    if not urls:
        try:
            discovered = method_odds._search_bfo_candidate_url(
                target["fighter_a"],
                target["fighter_b"],
                event_title=_text(target.get("event_title")) or None,
            )
        except Exception as exc:
            discovered = None
            errors.append(f"discovery:{type(exc).__name__}")
        discovered = _safe_public_url(discovered)
        if discovered:
            urls.append(discovered)

    evidence: list[dict] = []
    for url in urls:
        cached = cache.raw_page("direct_bfo", url) if cache is not None else None
        if cached is not None:
            evidence.append(cached)
            continue
        try:
            response = method_odds._bfo_get(url)
        except Exception as exc:
            response = None
            errors.append(f"fetch:{type(exc).__name__}")
        if response is None:
            continue
        raw = {
            "acquisition": "direct_bfo",
            "source_url": url,
            "original_source_url": url,
            "captured_at": _now_iso(),
            "html": response.text,
        }
        evidence.append(raw)
        if cache is not None:
            cache.remember_raw_page("direct_bfo", url, raw)
    return _acquirer_payload(evidence, urls=urls, error=";".join(errors))


class PlaywrightBfoAcquirer:
    """One shared headless browser for the existing BFO browser path."""

    def __init__(self, *, cache: Optional[RecoveryRunCache] = None):
        self._manager = None
        self._browser = None
        self._page = None
        self._startup_error = ""
        self._cache = cache

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright

            self._manager = sync_playwright().start()
            self._browser = self._manager.chromium.launch(headless=True)
            self._page = self._browser.new_page()
        except Exception as exc:
            self._startup_error = type(exc).__name__
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._browser is not None:
            self._browser.close()
        if self._manager is not None:
            self._manager.stop()

    def _discover(self, target: dict) -> list[str]:
        if self._page is None:
            return []
        query = quote_plus(f"{target['fighter_a']} {target['fighter_b']}")
        self._page.goto(
            f"{method_odds.BFO_LATEST_URL.rstrip('/')}/search?query={query}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        urls: list[str] = []
        for link in self._page.query_selector_all("a[href*='/events/']"):
            href = _text(link.get_attribute("href"))
            if not href or href in {"/events", "/events/"}:
                continue
            if not href.startswith("http"):
                href = f"https://www.bestfightodds.com{href}"
            if href not in urls:
                urls.append(href)
        return urls

    def __call__(self, target: dict, known_urls: list[str]) -> dict:
        urls = list(known_urls)
        errors: list[str] = []
        if not urls:
            if self._page is None:
                return _acquirer_payload(
                    error=self._startup_error or "browser unavailable"
                )
            try:
                urls.extend(self._discover(target))
            except Exception as exc:
                errors.append(f"discovery:{type(exc).__name__}")

        evidence: list[dict] = []
        for url in urls:
            cached = (
                self._cache.raw_page("playwright_bfo", url)
                if self._cache is not None
                else None
            )
            if cached is not None:
                evidence.append(cached)
                continue
            if self._page is None:
                errors.append(self._startup_error or "browser unavailable")
                continue
            try:
                self._page.goto(url, wait_until="networkidle", timeout=45_000)
                try:
                    self._page.wait_for_selector("tr[id^='mu-']", timeout=10_000)
                except Exception:
                    pass
                raw = {
                    "acquisition": "playwright_bfo",
                    "source_url": url,
                    "original_source_url": url,
                    "captured_at": _now_iso(),
                    "html": self._page.content(),
                }
                evidence.append(raw)
                if self._cache is not None:
                    self._cache.remember_raw_page("playwright_bfo", url, raw)
            except Exception as exc:
                errors.append(f"fetch:{type(exc).__name__}")
        return _acquirer_payload(evidence, urls=urls, error=";".join(errors))


def acquire_wayback_bfo(
    target: dict,
    known_urls: list[str],
    *,
    cache: Optional[RecoveryRunCache] = None,
) -> dict:
    commence = _utc_datetime(target.get("commence_time"))
    if commence is None:
        return _acquirer_payload(error="missing commence_time")
    cutoff = commence - PREFIGHT_LEAD
    evidence: list[dict] = []
    errors: list[str] = []

    for original_url in known_urls:
        timestamps = (
            cache.wayback_timestamps(original_url, cutoff)
            if cache is not None
            else []
        )
        if not timestamps:
            try:
                response = requests.get(
                    "https://web.archive.org/cdx/search/cdx",
                    params={
                        "url": original_url,
                        "output": "json",
                        "fl": "timestamp,statuscode",
                        "filter": "statuscode:200",
                        "collapse": "digest",
                        "to": cutoff.strftime("%Y%m%d%H%M%S"),
                        "limit": 1000,
                    },
                    headers=method_odds.HEADERS,
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                errors.append(f"cdx:{type(exc).__name__}")
                continue

            timestamps = []
            for row in payload[1:] if isinstance(payload, list) else []:
                timestamp = _text(row[0] if isinstance(row, list) and row else "")
                if not re.fullmatch(r"\d{14}", timestamp):
                    continue
                captured_at = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                if captured_at <= cutoff:
                    timestamps.append(timestamp)
            timestamps = sorted(set(timestamps), reverse=True)
            if cache is not None:
                cache.remember_wayback_timestamps(original_url, cutoff, timestamps)

        for timestamp in sorted(set(timestamps), reverse=True):
            captured_at = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            if captured_at > cutoff:
                continue
            archive_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
            raw = cache.raw_page("wayback_bfo", archive_url) if cache is not None else None
            if raw is None:
                try:
                    archived = requests.get(
                        archive_url,
                        headers=method_odds.HEADERS,
                        timeout=45,
                    )
                    archived.raise_for_status()
                except Exception as exc:
                    errors.append(f"archive:{type(exc).__name__}")
                    continue
                raw = {
                    "acquisition": "wayback_bfo",
                    "source_url": archive_url,
                    "original_source_url": original_url,
                    "captured_at": captured_at.isoformat(),
                    "html": archived.text,
                }
                if cache is not None:
                    cache.remember_raw_page("wayback_bfo", archive_url, raw)
            # Avoid hauling every archived page through the campaign once an
            # exact-scoped snapshot has been found for this target.
            parsed = method_odds._parse_bfo_method_odds(
                BeautifulSoup(_text(raw.get("html")), "lxml"),
                target["fighter_a"],
                target["fighter_b"],
            )
            if parsed is None:
                continue
            evidence.append(dict(raw))
            break
    return _acquirer_payload(evidence, urls=list(known_urls), error=";".join(errors))


def _parse_evidence(target: dict, raw: dict) -> tuple[Optional[dict], str]:
    html = raw.get("html")
    if isinstance(html, bytes):
        html_bytes = html
        html_text = html.decode("utf-8", errors="replace")
    else:
        html_text = _text(html)
        html_bytes = html_text.encode("utf-8")
    if not html_text:
        return None, "empty_page"
    parsed = method_odds._parse_bfo_method_odds(
        BeautifulSoup(html_text, "lxml"),
        target["fighter_a"],
        target["fighter_b"],
    )
    if parsed is None:
        return None, "no_unique_exact_matchup_scope"
    captured_at = _utc_datetime(raw.get("captured_at"))
    if captured_at is None:
        return None, "invalid_capture_time"
    source_url = _safe_public_url(raw.get("source_url"))
    original_url = _safe_public_url(raw.get("original_source_url")) or source_url
    if not source_url:
        return None, "invalid_source_url"
    return (
        {
            "acquisition": _text(raw.get("acquisition")) or "unknown",
            "source_url": source_url,
            "original_source_url": original_url,
            "captured_at": captured_at,
            "page_sha256": _sha256_bytes(html_bytes),
            "values": _method_values(parsed),
        },
        "",
    )


def _base_terminal(target: dict) -> dict:
    a_count = _optional_count(target.get("a_num_fights"))
    b_count = _optional_count(target.get("b_num_fights"))
    row = {
        "target_key": target["target_key"],
        "event_date": _text(target.get("event_date"))[:10],
        "commence_time": _text(target.get("commence_time")),
        "commence_time_source_kind": _text(target.get("commence_time_source_kind")),
        "commence_time_source_locator": _text(target.get("commence_time_source_locator")),
        "commence_time_source_observed_at": _text(
            target.get("commence_time_source_observed_at")
        ),
        "commence_time_source_sha256": _text(target.get("commence_time_source_sha256")),
        "commence_time_source_event_id": _text(
            target.get("commence_time_source_event_id")
        ),
        "commence_time_source_matchup": _text(
            target.get("commence_time_source_matchup")
        ),
        "commence_time_semantics": _text(target.get("commence_time_semantics")),
        "event_id": _text(target.get("event_id")),
        "event_title": _text(target.get("event_title")),
        "fighter_a": _text(target.get("fighter_a")),
        "fighter_b": _text(target.get("fighter_b")),
        "a_num_fights": a_count if a_count is not None else np.nan,
        "b_num_fights": b_count if b_count is not None else np.nan,
        "established_fighters": _established(target),
        "status": "failed",
        "terminal_reason": "",
        "recovery_preconditions_met": False,
        "target_recovery_complete": False,
        "attempted_sources": "[]",
        "success_source": "",
        "source_url": "",
        "original_source_url": "",
        "observation_captured_at": "",
        "validation_captured_at": "",
        "page_sha256": "",
        "parser_contract": "",
        "revalidated_schema_v1": False,
        "schema_v1_snapshot_path": "",
        "schema_v1_snapshot_sha256": "",
        "method_odds_complete": False,
    }
    row.update({column: np.nan for column in METHOD_FEATURE_COLUMNS})
    return row


def _success_source(acquisition: str, revalidated_v1: bool) -> str:
    if revalidated_v1:
        return "bestfightodds_v1_revalidated"
    if acquisition == "wayback_bfo":
        return "wayback_bestfightodds_recovery"
    return "bestfightodds_recovery"


def _strict_true(value: object) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _attempt_pass_sequence(value: object) -> Optional[tuple[str, ...]]:
    """Return the ordered pass groups only for a structurally valid ledger."""
    try:
        attempts = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(attempts, list) or not attempts:
        return None

    sequence: list[str] = []
    seen: set[str] = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            return None
        pass_name = _text(attempt.get("pass"))
        source = _text(attempt.get("source"))
        status = _text(attempt.get("status"))
        detail = _text(attempt.get("detail"))
        if (
            not pass_name
            or source != _PASS_SOURCES.get(pass_name)
            or status not in _ATTEMPT_STATUSES
            or not detail
        ):
            return None
        if not sequence or sequence[-1] != pass_name:
            if pass_name in seen:
                return None
            sequence.append(pass_name)
            seen.add(pass_name)
    return tuple(sequence)


def _attempt_ledger_complete(
    value: object,
    expected_passes: Iterable[str],
) -> bool:
    return _attempt_pass_sequence(value) == tuple(expected_passes)


def _recovery_record(target: dict, terminal: dict) -> dict:
    values = {column: terminal[column] for column in METHOD_FEATURE_COLUMNS}
    record = method_odds._snapshot_record(
        fighter_a=terminal["fighter_a"],
        fighter_b=terminal["fighter_b"],
        method_odds=values,
        source=terminal["success_source"],
        captured_at=terminal["observation_captured_at"],
        event_id=terminal["event_id"],
        commence_time=terminal["commence_time"],
        event_title=terminal["event_title"],
        source_url=terminal["source_url"],
        page_sha256=terminal["page_sha256"],
        parser_contract=method_odds.BFO_PARSER_CONTRACT,
    )
    record.update(
        {
            "original_source_url": terminal["original_source_url"],
            "validation_captured_at": terminal["validation_captured_at"],
            "revalidated_from_schema_version": (
                1 if terminal["revalidated_schema_v1"] else None
            ),
            "schema_v1_snapshot_path": terminal["schema_v1_snapshot_path"],
            "schema_v1_snapshot_sha256": terminal["schema_v1_snapshot_sha256"],
            "target_key": target["target_key"],
        }
    )
    return record


def recover_target(
    target: dict,
    *,
    v1_index: dict[str, list[dict]],
    acquirers: Iterable[tuple[str, Callable[[dict, list[str]], dict]]],
    run_cache: Optional[RecoveryRunCache] = None,
) -> tuple[dict, Optional[dict]]:
    terminal = _base_terminal(target)
    configured_acquirers = tuple(acquirers)
    configured_passes = [
        _ACQUIRER_PASS_NAMES.get(source_name, source_name)
        for source_name, _ in configured_acquirers
    ]
    # This lookup is the quarantined schema-v1 revalidation pass. It never
    # makes a v1 record inference-eligible without later exact page evidence.
    commence = _utc_datetime(target.get("commence_time"))
    if commence is None or not _commence_provenance_complete(target):
        precondition_detail = (
            "missing_or_invalid_commence_time"
            if commence is None
            else "missing_or_invalid_commence_time_provenance"
        )
        precondition_attempts = [
            {
                "pass": pass_name,
                "source": "schema_v1" if pass_name == "revalidate" else source_name,
                "status": "precondition_failed",
                "detail": precondition_detail,
            }
            for pass_name, source_name in [
                ("revalidate", "schema_v1"),
                *zip(configured_passes, [name for name, _ in configured_acquirers], strict=True),
            ]
        ]
        terminal["terminal_reason"] = precondition_detail
        terminal["attempted_sources"] = json.dumps(
            precondition_attempts, sort_keys=True, separators=(",", ":")
        )
        return terminal, None

    terminal["recovery_preconditions_met"] = True
    v1_candidates = _v1_candidates_for(target, v1_index)
    known_urls = _url_values(target.get("source_url"))
    if run_cache is not None:
        for url in run_cache.event_urls_for(target):
            if url not in known_urls:
                known_urls.append(url)
    for candidate in v1_candidates:
        for url in _url_values(candidate.get("source_url")):
            if url not in known_urls:
                known_urls.append(url)

    attempts: list[dict] = [
        {
            "pass": "revalidate",
            "source": "schema_v1",
            "status": "candidate_available" if v1_candidates else "no_evidence",
            "detail": (
                f"{len(v1_candidates)} exact prefight candidate(s) quarantined"
                if v1_candidates
                else "no_exact_prefight_schema_v1_candidate"
            ),
        }
    ]
    recovered = False
    for source_name, acquirer in configured_acquirers:
        pass_name = _ACQUIRER_PASS_NAMES.get(source_name, source_name)
        try:
            result = acquirer(target, list(known_urls))
        except Exception as exc:
            attempts.append(
                {
                    "pass": pass_name,
                    "source": source_name,
                    "status": "error",
                    "detail": type(exc).__name__,
                }
            )
            continue
        if not isinstance(result, dict):
            result = _acquirer_payload(error="invalid acquirer result")
        for url in result.get("urls") or []:
            if url not in known_urls:
                known_urls.append(url)
        raw_evidence = result.get("evidence") or []
        if not raw_evidence:
            attempts.append(
                {
                    "pass": pass_name,
                    "source": source_name,
                    "status": "no_evidence",
                    "detail": _safe_detail(result.get("error"))
                    or "no_evidence_returned",
                }
            )
            continue

        for raw in raw_evidence:
            try:
                evidence, error = _parse_evidence(target, raw)
            except Exception as exc:
                attempts.append(
                    {
                        "pass": pass_name,
                        "source": source_name,
                        "status": "error",
                        "detail": f"parse:{type(exc).__name__}",
                    }
                )
                continue
            if evidence is None:
                attempts.append(
                    {
                        "pass": pass_name,
                        "source": source_name,
                        "status": "rejected",
                        "detail": error,
                    }
                )
                continue

            # URL reuse is deliberately stricter than raw-response caching:
            # only an event page that parsed this exact target is shared with
            # another fight on the same unambiguous event identity.
            if run_cache is not None:
                run_cache.remember_exact_event_url(
                    target,
                    evidence["original_source_url"],
                )

            values = evidence["values"]
            evidence_prefight = evidence["captured_at"] <= commence - PREFIGHT_LEAD
            matched_candidate = None
            if not evidence_prefight:
                for candidate in v1_candidates:
                    if candidate["prefight"] and _values_match(candidate["values"], values):
                        matched_candidate = candidate
                        break
                if matched_candidate is None:
                    attempts.append(
                        {
                            "pass": pass_name,
                            "source": source_name,
                            "status": "rejected",
                            "detail": "post_fight_page_without_exact_prefight_v1_value_match",
                        }
                    )
                    continue
                values = matched_candidate["values"]
                attempts[0].update(
                    {
                        "status": "accepted",
                        "detail": "exact_prefight_schema_v1_values_revalidated",
                    }
                )

            observed_at = (
                matched_candidate["captured_at"]
                if matched_candidate is not None
                else evidence["captured_at"]
            )
            success_source = _success_source(
                evidence["acquisition"], matched_candidate is not None
            )
            attempts.append(
                {
                    "pass": pass_name,
                    "source": source_name,
                    "status": "accepted" if not recovered else "validated_redundant",
                    "detail": success_source,
                }
            )
            if not recovered:
                terminal.update(
                    {
                        "status": "success",
                        "terminal_reason": "validated_exact_prefight_method_odds",
                        "success_source": success_source,
                        "source_url": evidence["source_url"],
                        "original_source_url": evidence["original_source_url"],
                        "observation_captured_at": observed_at.isoformat(),
                        "validation_captured_at": evidence["captured_at"].isoformat(),
                        "page_sha256": evidence["page_sha256"],
                        "parser_contract": method_odds.BFO_PARSER_CONTRACT,
                        "revalidated_schema_v1": matched_candidate is not None,
                        "schema_v1_snapshot_path": (
                            matched_candidate["snapshot_path"] if matched_candidate else ""
                        ),
                        "schema_v1_snapshot_sha256": (
                            matched_candidate["snapshot_sha256"] if matched_candidate else ""
                        ),
                        "method_odds_complete": _complete(values),
                        **values,
                    }
                )
                recovered = True
            # A single accepted exact page is enough evidence for this pass;
            # later configured passes still run and receive explicit outcomes.
            break

    if attempts[0]["status"] == "candidate_available":
        attempts[0].update(
            {
                "status": "not_revalidated",
                "detail": "no_acquired_exact_page_revalidated_schema_v1_values",
            }
        )
    expected_passes = ["revalidate", *configured_passes]
    terminal["target_recovery_complete"] = _attempt_ledger_complete(
        attempts, expected_passes
    )
    terminal["attempted_sources"] = json.dumps(
        attempts, sort_keys=True, separators=(",", ":")
    )
    if recovered:
        return terminal, _recovery_record(target, terminal)
    terminal["terminal_reason"] = "no_exact_prefight_or_revalidated_evidence"
    return terminal, None


def run_recovery(
    targets: pd.DataFrame,
    *,
    v1_index: Optional[dict[str, list[dict]]] = None,
    acquirers: Iterable[tuple[str, Callable[[dict, list[str]], dict]]],
    run_cache: Optional[RecoveryRunCache] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    if "target_key" not in targets.columns:
        targets = prepare_targets(targets)
    configured_acquirers = tuple(acquirers)
    terminal_rows: list[dict] = []
    recovered_records: list[dict] = []
    index = v1_index or {}
    cache = run_cache or RecoveryRunCache()
    targets = _enrich_commence_times_from_v1(targets, index)
    for target in targets.to_dict(orient="records"):
        terminal, recovered = recover_target(
            target,
            v1_index=index,
            acquirers=configured_acquirers,
            run_cache=cache,
        )
        terminal_rows.append(terminal)
        if recovered is not None:
            recovered_records.append(recovered)

    terminal = pd.DataFrame(terminal_rows, columns=TERMINAL_COLUMNS)
    if len(terminal) != len(targets) or terminal["target_key"].nunique() != len(targets):
        raise RuntimeError("terminal report must contain exactly one row per target")
    terminal.attrs["passes_completed"] = [
        "revalidate",
        *[
            _ACQUIRER_PASS_NAMES[name]
            for name, _ in configured_acquirers
            if name in _ACQUIRER_PASS_NAMES
        ],
    ]
    return terminal, recovered_records


def _decision_input_hash(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.sort_values("target_key", kind="mergesort").to_dict(orient="records"):
        payload = {
            "target_key": _text(row.get("target_key")),
            "event_date": _text(row.get("event_date"))[:10],
            "commence_time": _text(row.get("commence_time")),
            "commence_time_source_kind": _text(
                row.get("commence_time_source_kind")
            ),
            "commence_time_source_locator": _text(
                row.get("commence_time_source_locator")
            ),
            "commence_time_source_observed_at": _text(
                row.get("commence_time_source_observed_at")
            ),
            "commence_time_source_sha256": _text(
                row.get("commence_time_source_sha256")
            ),
            "commence_time_source_event_id": _text(
                row.get("commence_time_source_event_id")
            ),
            "commence_time_source_matchup": _text(
                row.get("commence_time_source_matchup")
            ),
            "commence_time_semantics": _text(row.get("commence_time_semantics")),
            "established_fighters": _strict_true(row.get("established_fighters")),
            "status": _text(row.get("status")),
            "terminal_reason": _text(row.get("terminal_reason")),
            "recovery_preconditions_met": _strict_true(
                row.get("recovery_preconditions_met")
            ),
            "target_recovery_complete": _strict_true(
                row.get("target_recovery_complete")
            ),
            "attempted_sources": _text(row.get("attempted_sources")),
            "method_odds_complete": _strict_true(row.get("method_odds_complete")),
        }
        for column in METHOD_FEATURE_COLUMNS:
            try:
                value = float(row.get(column))
            except (TypeError, ValueError):
                value = np.nan
            payload[column] = round(value, 12) if math.isfinite(value) else None
        rows.append(payload)
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(serialized.encode("utf-8"))


def _refresh_decision_sha256(decision: dict) -> None:
    unsigned = dict(decision)
    unsigned.pop("decision_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    decision["decision_sha256"] = _sha256_bytes(canonical.encode("utf-8"))


def decide_feature_contract(
    terminal: pd.DataFrame,
    *,
    reference_date: Optional[object] = None,
) -> dict:
    """Choose the registered contract using only the frozen quality gate."""
    dates = pd.to_datetime(terminal["event_date"], errors="coerce")
    if dates.isna().any() or terminal.empty:
        raise RecoveryInputError("terminal report needs valid event dates")
    reference = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    reference = reference.normalize()
    window_start = reference - pd.Timedelta(days=RECENT_WINDOW_DAYS - 1)
    recent_mask = dates.between(window_start, reference, inclusive="both")
    established_mask = terminal["established_fighters"].fillna(False).astype(bool)
    recent = terminal[recent_mask & established_mask].copy()

    complete_mask = recent["method_odds_complete"].fillna(False).astype(bool)
    complete_mask &= recent["status"].eq("success")
    for column in METHOD_FEATURE_COLUMNS:
        complete_mask &= pd.to_numeric(recent[column], errors="coerce").notna()
    recent_rows = len(recent)
    complete_rows = int(complete_mask.sum())
    coverage = complete_rows / recent_rows if recent_rows else 0.0
    latest_complete_date = None
    lag_days = None
    if complete_rows:
        complete_dates = pd.to_datetime(recent.loc[complete_mask, "event_date"])
        latest_complete_date = complete_dates.max().normalize()
        lag_days = max(0, int((reference - latest_complete_date).days))

    criteria = {
        "minimum_recent_established_rows": recent_rows >= MIN_RECENT_ESTABLISHED_ROWS,
        "minimum_complete_method_coverage": coverage >= MIN_COMPLETE_COVERAGE,
        "maximum_method_data_lag": lag_days is not None and lag_days <= MAX_DATA_LAG_DAYS,
    }
    targets_total = len(terminal)
    terminal_status = terminal["status"].isin(("success", "failed"))
    targets_terminal = int(terminal_status.sum())
    target_keys = terminal["target_key"].map(_text)
    all_targets_terminal = bool(
        targets_total > 0
        and targets_terminal == targets_total
        and target_keys.ne("").all()
        and target_keys.nunique() == targets_total
    )
    expected_passes = list(RECOVERY_PASS_ORDER)
    if "attempted_sources" in terminal.columns:
        ledger_complete = terminal["attempted_sources"].map(
            lambda value: _attempt_ledger_complete(value, expected_passes)
        )
    else:
        ledger_complete = pd.Series(False, index=terminal.index, dtype=bool)
    attempt_ledger_failures = int((~ledger_complete).sum())
    passes_completed = expected_passes if attempt_ledger_failures == 0 else []
    if "target_recovery_complete" in terminal.columns:
        declared_complete = terminal["target_recovery_complete"].map(_strict_true)
    else:
        declared_complete = pd.Series(False, index=terminal.index, dtype=bool)
    if "recovery_preconditions_met" in terminal.columns:
        preconditions_met = terminal["recovery_preconditions_met"].map(_strict_true)
    else:
        preconditions_met = pd.Series(False, index=terminal.index, dtype=bool)
    target_complete = declared_complete & preconditions_met & ledger_complete
    targets_recovery_complete = int(target_complete.sum())
    all_targets_recovery_complete = bool(
        targets_total > 0 and targets_recovery_complete == targets_total
    )
    precondition_failures = int((~preconditions_met).sum())
    recovery_complete = bool(
        all_targets_terminal
        and all_targets_recovery_complete
        and attempt_ledger_failures == 0
        and passes_completed == list(RECOVERY_PASS_ORDER)
    )
    data_gate_passed = all(criteria.values())
    gate_passed = bool(data_gate_passed and recovery_complete)
    evaluation_contract = CONTRACT_211 if gate_passed else CONTRACT_205
    fullfit_contract = CONTRACT_211_FULLFIT if gate_passed else CONTRACT_205_FULLFIT
    selected_feature_count = 211 if gate_passed else 205
    selected_removed = [] if gate_passed else sorted(V4_METHOD_ODDS_FEATURE_COLS)

    decision = {
        "schema_version": 1,
        "decision_phase": "pre_model_metrics",
        "model_metrics_consulted": False,
        "campaign_cutoff_exclusive": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "reference_date": reference.strftime("%Y-%m-%d"),
        "recent_window_start": window_start.strftime("%Y-%m-%d"),
        "recent_window_days": RECENT_WINDOW_DAYS,
        "recent_established_rows": recent_rows,
        "complete_method_rows": complete_rows,
        "complete_method_coverage": round(coverage, 12),
        "latest_complete_method_event_date": (
            latest_complete_date.strftime("%Y-%m-%d")
            if latest_complete_date is not None
            else None
        ),
        "method_data_lag_days": lag_days,
        "gate_thresholds": {
            "minimum_recent_established_rows": MIN_RECENT_ESTABLISHED_ROWS,
            "minimum_complete_method_coverage": MIN_COMPLETE_COVERAGE,
            "maximum_method_data_lag_days": MAX_DATA_LAG_DAYS,
        },
        "criteria": criteria,
        "data_gate_passed": data_gate_passed,
        "gate_passed": gate_passed,
        "recovery_complete": recovery_complete,
        "all_targets_terminal": all_targets_terminal,
        "all_targets_recovery_complete": all_targets_recovery_complete,
        "passes_completed": passes_completed,
        "targets_total": targets_total,
        "targets_terminal": targets_terminal,
        "targets_recovery_complete": targets_recovery_complete,
        "precondition_failures": precondition_failures,
        "attempt_ledger_failures": attempt_ledger_failures,
        "selected_evaluation_contract": evaluation_contract,
        "selected_spec_name": evaluation_contract,
        "selected_fullfit_contract": fullfit_contract,
        "selected_feature_count": selected_feature_count,
        "selected_removed_features": selected_removed,
        "fallback_exact_removed_features": sorted(V4_METHOD_ODDS_FEATURE_COLS),
        "odds_noise": {"mode": "antithetic", "std": 0.06, "seed": 42},
        "decision_input_sha256": _decision_input_hash(terminal),
        "terminal_report_sha256": _sha256_bytes(
            terminal.to_csv(index=False).encode("utf-8")
        ),
    }
    _refresh_decision_sha256(decision)
    return decision


def publish_outputs(
    output_dir: Path,
    terminal: pd.DataFrame,
    recovered_records: list[dict],
    decision: dict,
    *,
    canonical_decision_path: Optional[Path] = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = output_dir / "method_odds_recovery_terminal.csv"
    snapshot_path = output_dir / "method_odds_recovery_snapshot_v2.json"
    decision_path = canonical_decision_path or (
        output_dir / "method_odds_contract_decision.json"
    )

    terminal_csv = terminal.to_csv(index=False).encode("utf-8")
    terminal_sha256 = _sha256_bytes(terminal_csv)
    if decision.get("terminal_report_sha256") != terminal_sha256:
        raise RecoveryInputError(
            "contract decision is not bound to the exact terminal report bytes"
        )
    try:
        terminal_locator = terminal_path.resolve().relative_to(
            REPO_ROOT.resolve()
        ).as_posix()
    except ValueError:
        terminal_locator = str(terminal_path.resolve())
    decision["terminal_report_path"] = terminal_locator
    _refresh_decision_sha256(decision)
    snapshot = {
        "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": _now_iso(),
        "status": (
            "success"
            if len(recovered_records) == len(terminal)
            else "partial" if recovered_records else "failed"
        ),
        "record_count": len(recovered_records),
        "target_count": len(terminal),
        "campaign_cutoff_exclusive": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "parser_contracts": {"bestfightodds": method_odds.BFO_PARSER_CONTRACT},
        "terminal_report_sha256": terminal_sha256,
        "contract_decision_sha256": decision["decision_sha256"],
        "records": recovered_records,
        "sources": ["direct_bfo", "playwright_bfo", "wayback_bfo"],
    }
    write_csv_atomically(terminal, terminal_path, refuse_empty=True)
    if _sha256_bytes(terminal_path.read_bytes()) != terminal_sha256:
        raise RuntimeError("published terminal report bytes do not match the decision")
    write_json_atomically(snapshot, snapshot_path)
    write_json_atomically(decision, decision_path)
    return {
        "terminal": terminal_path,
        "snapshot": snapshot_path,
        "decision": decision_path,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decision-path",
        type=Path,
        help=(
            "Write the single canonical contract decision here instead of "
            "creating a campaign-local duplicate."
        ),
    )
    parser.add_argument(
        "--v1-snapshot-dir",
        type=Path,
        action="append",
        default=[],
        help="May be repeated. Schema-v1 files are candidates only, never loaded for inference.",
    )
    parser.add_argument(
        "--event-url-map",
        type=Path,
        default=REPO_ROOT / "data/raw/method_odds/bfo_event_url_map.csv",
    )
    parser.add_argument(
        "--line-history-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "May be repeated. Missing starts use only the latest unambiguous exact-pair "
            "match in verified odds_*.csv snapshots."
        ),
    )
    parser.add_argument(
        "--recover-commence-from-odds-api-history",
        action="store_true",
        help=(
            "Fill missing exact starts from one historical Odds API events "
            "snapshot per card date before method recovery."
        ),
    )
    parser.add_argument(
        "--commence-jsonld-url",
        action="append",
        default=[],
        help=(
            "May be repeated. Fill still-missing exact starts only from an "
            "exact-match JSON-LD SportsEvent on this explicit page."
        ),
    )
    parser.add_argument(
        "--run-network-campaign",
        action="store_true",
        help="Required acknowledgement before direct, browser, or Wayback requests run.",
    )
    args = parser.parse_args(argv)

    if not args.run_network_campaign:
        print(
            "Refusing to start network recovery without --run-network-campaign.",
            file=sys.stderr,
        )
        return 2

    line_history_dirs = args.line_history_dir or [REPO_ROOT / "data/raw/line_history"]
    targets = load_targets(
        args.targets,
        event_url_map=args.event_url_map,
        line_history_dirs=line_history_dirs,
    )
    if args.recover_commence_from_odds_api_history:
        if not ODDS_API_KEY:
            print(
                "Cannot recover historical commence times: ODDS_API_KEY is missing.",
                file=sys.stderr,
            )
            return 2
        targets = enrich_commence_times_from_odds_api_history(
            targets,
            api_key=ODDS_API_KEY,
        )
    if args.commence_jsonld_url:
        targets = enrich_commence_times_from_jsonld_pages(
            targets,
            args.commence_jsonld_url,
        )
    snapshot_dirs = args.v1_snapshot_dir or [REPO_ROOT / "data/raw/method_odds"]
    v1_index = load_schema_v1_candidates(snapshot_dirs)

    run_cache = RecoveryRunCache()
    with PlaywrightBfoAcquirer(cache=run_cache) as browser_acquirer:
        terminal, recovered = run_recovery(
            targets,
            v1_index=v1_index,
            acquirers=(
                ("direct_bfo", partial(acquire_direct_bfo, cache=run_cache)),
                ("playwright_bfo", browser_acquirer),
                ("wayback_bfo", partial(acquire_wayback_bfo, cache=run_cache)),
            ),
            run_cache=run_cache,
        )
    decision = decide_feature_contract(terminal)
    paths = publish_outputs(
        args.output_dir,
        terminal,
        recovered,
        decision,
        canonical_decision_path=args.decision_path,
    )
    print(
        f"targets={len(terminal)} recovered={len(recovered)} "
        f"contract={decision['selected_evaluation_contract']} "
        f"terminal={paths['terminal']} decision={paths['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
