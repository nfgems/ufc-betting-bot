"""BetsAPI MMA/UFC ingestion, normalization, and challenger feature helpers."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from src.config import (
    BETSAPI_BASE_URL,
    BETSAPI_MMA_PROCESSED_DIR,
    BETSAPI_MMA_RAW_DIR,
    BETSAPI_MMA_SPORT_ID,
    BETSAPI_TOKEN,
)

logger = logging.getLogger(__name__)

BETSAPI_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
BETSAPI_MAX_ATTEMPTS = 5
BETSAPI_SUMMARY_MAX_WORKERS = 2

RAW_EVENTS_UPCOMING_DIR = BETSAPI_MMA_RAW_DIR / "events_upcoming"
RAW_EVENTS_ENDED_DIR = BETSAPI_MMA_RAW_DIR / "events_ended"
RAW_BET365_UPCOMING_DIR = BETSAPI_MMA_RAW_DIR / "bet365_upcoming"
RAW_BET365_PREMATCH_DIR = BETSAPI_MMA_RAW_DIR / "bet365_prematch"
RAW_ODDS_SUMMARY_DIR = BETSAPI_MMA_RAW_DIR / "odds_summary"
JOIN_AUDIT_DIR = BETSAPI_MMA_PROCESSED_DIR / "join_audit"
MARKET_INVENTORY_DIR = BETSAPI_MMA_PROCESSED_DIR / "market_inventory"
SUMMARY_FILE_RE = re.compile(r"^\d{8}T\d{6}Z_odds_summary_event_(?P<event_id>.+)\.json$")

SUMMARY_ROW_COLUMNS = [
    "event_id",
    "bookmaker",
    "matching_dir",
    "market_phase",
    "market_key",
    "line_id",
    "home_odds",
    "over_odds",
    "draw_odds",
    "away_odds",
    "under_odds",
    "handicap",
    "score_state",
    "selection_added_at_utc",
    "bookmaker_last_update_utc",
    "snapshot_time_utc",
]
EVENT_SNAPSHOT_COLUMNS = [
    "event_id",
    "fight_key",
    "fighter_a",
    "fighter_b",
    "scheduled_start_utc",
    "opening_snapshot_timestamp_utc",
    "opening_bookmakers",
    "opening_home_implied_prob_avg",
    "opening_away_implied_prob_avg",
    "end_snapshot_timestamp_utc",
    "end_bookmakers",
    "end_home_implied_prob_avg",
    "end_away_implied_prob_avg",
    "first_bookmaker_last_update_utc",
    "last_bookmaker_last_update_utc",
]

for directory in [
    RAW_EVENTS_UPCOMING_DIR,
    RAW_EVENTS_ENDED_DIR,
    RAW_BET365_UPCOMING_DIR,
    RAW_BET365_PREMATCH_DIR,
    RAW_ODDS_SUMMARY_DIR,
    JOIN_AUDIT_DIR,
    MARKET_INVENTORY_DIR,
]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _api_root() -> str:
    """Return the BetsAPI root without a pinned version suffix."""
    return re.sub(r"/v\d+$", "", BETSAPI_BASE_URL.rstrip("/"))


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_timestamp(value: object) -> pd.Timestamp:
    if value in (None, "", 0, "0"):
        return pd.NaT
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", utc=True)


def _is_valid_decimal_odds(value: object) -> bool:
    numeric = pd.to_numeric(value, errors="coerce")
    return bool(pd.notna(numeric) and float(numeric) > 1.0)


def _normalize_query_date(value: object) -> pd.Timestamp:
    """Convert input into a normalized date-only timestamp without timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def normalize_mma_text(value: object) -> str:
    """Normalize MMA text into lowercase ASCII tokens for joins."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _entity_name(value: object) -> str:
    """Extract a human name from BetsAPI entity shapes."""
    if isinstance(value, dict):
        return str(value.get("name", "") or "")
    return str(value or "")


def _fight_key(fighter_a: object, fighter_b: object, event_date: object) -> str:
    left = normalize_mma_text(fighter_a)
    right = normalize_mma_text(fighter_b)
    date = pd.to_datetime(event_date, errors="coerce")
    if not left or not right or pd.isna(date):
        return ""
    fighters = sorted([left, right])
    return f"{fighters[0]}__{fighters[1]}__{date.date().isoformat()}"


def _shift_fight_key_date(fight_key: object, *, days: int) -> str:
    """Return the same normalized fighter pairing with the date shifted."""
    if not isinstance(fight_key, str) or not fight_key:
        return ""
    try:
        prefix, date_text = fight_key.rsplit("__", 1)
    except ValueError:
        return ""

    date = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(date):
        return ""
    return f"{prefix}__{(date + pd.Timedelta(days=days)).date().isoformat()}"


def _merge_on_fight_key_with_previous_day_fallback(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    signal_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Merge on fight_key, falling back to the previous UTC day for the right side.

    BetsAPI event timestamps are stored in UTC. UFC cards that start late in the
    evening can roll over to the next UTC day even though the source fight date
    remains the local card date. Prefer exact-date matches, but allow the right
    side to match one day earlier when that is the only available key.
    """
    if left_df.empty or right_df.empty:
        return left_df.copy()

    direct = right_df.copy()
    direct["_match_fight_key"] = direct["fight_key"]
    direct["_match_priority"] = 0

    previous_day = right_df.copy()
    previous_day["_match_fight_key"] = previous_day["fight_key"].map(
        lambda value: _shift_fight_key_date(value, days=-1)
    )
    previous_day["_match_priority"] = 1

    lookup = pd.concat([direct, previous_day], ignore_index=True)
    lookup = lookup[lookup["_match_fight_key"].astype(bool)].copy()
    usable_signal_cols = [column for column in (signal_cols or []) if column in lookup.columns]
    sort_columns = ["_match_priority"]
    ascending = [True]
    helper_cols = ["_match_priority"]
    if usable_signal_cols:
        lookup["_match_signal_count"] = lookup[usable_signal_cols].notna().sum(axis=1)
        sort_columns = ["_match_signal_count", "_match_priority"]
        ascending = [False, True]
        helper_cols.append("_match_signal_count")

    lookup = lookup.sort_values(sort_columns, ascending=ascending).drop_duplicates(
        subset=["_match_fight_key"],
        keep="first",
    )
    lookup = lookup.drop(columns=["fight_key", *helper_cols]).rename(
        columns={"_match_fight_key": "fight_key"}
    )

    return left_df.merge(lookup, on="fight_key", how="left")


def _response_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", [])
    return rows if isinstance(rows, list) else []


def _prematch_result_root(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize Bet365 prematch payloads that may wrap the result in a list."""
    results = payload.get("results")
    if isinstance(results, list):
        first = results[0] if results else {}
        return first if isinstance(first, dict) else {}
    return results if isinstance(results, dict) else {}


def _is_selection_leaf(value: Any) -> bool:
    """Return True only for priced selection rows, not intermediate market containers."""
    if not isinstance(value, dict):
        return False

    odds = value.get("odds")
    if isinstance(odds, (list, dict)):
        return False

    if "header" in value or "handicap" in value:
        return True

    return "id" in value and "name" in value and _safe_float(odds) is not None


def _write_raw_payload(
    directory: Path,
    payload: dict[str, Any],
    stem: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    path = directory / f"{_utc_now()}_{stem}.json"
    body = {"metadata": metadata or {}, "payload": payload}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _write_raw_file(path: Path, payload: dict[str, Any], metadata: Optional[dict[str, Any]] = None) -> Path:
    body = {"metadata": metadata or {}, "payload": payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _load_raw_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(body, dict) and "payload" in body:
        payload = body.get("payload") or {}
        metadata = body.get("metadata") or {}
    else:
        payload = body if isinstance(body, dict) else {}
        metadata = {}
    return payload, metadata


@dataclass(frozen=True)
class BetsApiRequestResult:
    """Minimal structured response metadata for raw MMA requests."""

    endpoint: str
    path: Path
    count: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class MMABackfillArtifacts:
    """Concrete outputs for one historical MMA backfill run."""

    events_csv: Path
    summary_rows_csv: Path
    event_snapshots_csv: Path
    manifest_json: Path


class BetsApiMMAClient:
    """Minimal version-aware BetsAPI client for MMA/UFC discovery."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or BETSAPI_TOKEN
        if not self.token:
            raise RuntimeError("BETSAPI_TOKEN missing")
        self.api_root = _api_root()

    def _get(
        self,
        endpoint: str,
        params: Optional[dict[str, object]] = None,
        *,
        api_version: str,
    ) -> dict[str, Any]:
        query = dict(params or {})
        query["token"] = self.token
        url = f"{self.api_root}/{api_version}/{endpoint}"
        for attempt in range(1, BETSAPI_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(url, params=query, timeout=30)
                if response.status_code in BETSAPI_RETRY_STATUS_CODES and attempt < BETSAPI_MAX_ATTEMPTS:
                    wait_seconds = _betsapi_retry_wait_seconds(attempt, response=response)
                    logger.warning(
                        "Retrying BetsAPI %s after HTTP %s in %.1f seconds (attempt %s/%s)",
                        endpoint,
                        response.status_code,
                        wait_seconds,
                        attempt,
                        BETSAPI_MAX_ATTEMPTS,
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                payload = response.json()
                if int(payload.get("success", 0)) != 1:
                    error = payload.get("error_detail") or payload.get("error") or "unknown error"
                    raise RuntimeError(f"BetsAPI request failed for {endpoint}: {error}")
                return payload
            except requests.RequestException as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt < BETSAPI_MAX_ATTEMPTS and (
                    status_code in BETSAPI_RETRY_STATUS_CODES or status_code is None
                ):
                    wait_seconds = _betsapi_retry_wait_seconds(attempt)
                    logger.warning(
                        "Retrying BetsAPI %s after %s in %.1f seconds (attempt %s/%s)",
                        endpoint,
                        type(exc).__name__,
                        wait_seconds,
                        attempt,
                        BETSAPI_MAX_ATTEMPTS,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise

        raise RuntimeError(f"BetsAPI request failed for {endpoint}: exhausted retries")

    def get_mma_upcoming_events(self, page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, object] = {"sport_id": BETSAPI_MMA_SPORT_ID, "page": page}
        if day:
            params["day"] = day
        return self._get("events/upcoming", params=params, api_version="v3")

    def get_mma_ended_events(self, page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, object] = {"sport_id": BETSAPI_MMA_SPORT_ID, "page": page}
        if day:
            params["day"] = day
        return self._get("events/ended", params=params, api_version="v3")

    def get_bet365_mma_upcoming(self, page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, object] = {"sport_id": BETSAPI_MMA_SPORT_ID, "page": page}
        if day:
            params["day"] = day
        return self._get("bet365/upcoming", params=params, api_version="v1")

    def get_bet365_mma_prematch(self, fi: object) -> dict[str, Any]:
        return self._get("bet365/prematch", params={"FI": fi}, api_version="v4")

    def get_event_odds_summary(self, event_id: object, source: str = "bet365") -> dict[str, Any]:
        return self._get(
            "event/odds/summary",
            params={"event_id": event_id, "source": source},
            api_version="v2",
        )


def get_mma_upcoming_events(page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
    return BetsApiMMAClient().get_mma_upcoming_events(page=page, day=day)


def get_mma_ended_events(page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
    return BetsApiMMAClient().get_mma_ended_events(page=page, day=day)


def get_bet365_mma_upcoming(page: int = 1, day: Optional[str] = None) -> dict[str, Any]:
    return BetsApiMMAClient().get_bet365_mma_upcoming(page=page, day=day)


def get_bet365_mma_prematch(fi: object) -> dict[str, Any]:
    return BetsApiMMAClient().get_bet365_mma_prematch(fi)


def get_event_odds_summary(event_id: object, source: str = "bet365") -> dict[str, Any]:
    return BetsApiMMAClient().get_event_odds_summary(event_id, source=source)


def _betsapi_retry_wait_seconds(attempt: int, response: Optional[requests.Response] = None) -> float:
    retry_after = ""
    if response is not None:
        retry_after = str(getattr(response, "headers", {}).get("Retry-After", "")).strip()
    if retry_after.isdigit():
        return max(float(retry_after), 1.0)
    if getattr(response, "status_code", None) == 429:
        return float(min(10 * attempt, 30))
    return float(min(2 ** (attempt - 1), 8))


def _is_retryable_request_exception(exc: Exception) -> bool:
    if not isinstance(exc, requests.RequestException):
        return False
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return status_code in BETSAPI_RETRY_STATUS_CODES or status_code is None


def _is_ufc_event_row(event: dict[str, Any]) -> bool:
    return "ufc" in normalize_mma_text(_entity_name(event.get("league")))


def normalize_mma_event_rows(
    payload: dict[str, Any],
    *,
    source: str,
    snapshot_time_utc: Optional[object] = None,
) -> pd.DataFrame:
    """Normalize generic or Bet365 event payloads into a flat event frame."""
    snapshot_ts = pd.to_datetime(snapshot_time_utc, errors="coerce", utc=True)
    if pd.isna(snapshot_ts):
        snapshot_ts = pd.Timestamp.now(tz="UTC")

    rows: list[dict[str, object]] = []
    for event in _response_rows(payload):
        league = event.get("league")
        home = event.get("home")
        away = event.get("away")
        fighter_a = _entity_name(home)
        fighter_b = _entity_name(away)
        event_time = event.get("time")
        rows.append(
            {
                "source": source,
                "sport_id": event.get("sport_id", BETSAPI_MMA_SPORT_ID),
                "event_id": str(event.get("id", "")),
                "our_event_id": str(event.get("our_event_id", "")),
                "fi": str(event.get("id", "")) if source == "bet365_upcoming" else "",
                "league_name": _entity_name(league),
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "scheduled_start_utc": _safe_timestamp(event_time),
                "time_status": event.get("time_status"),
                "odds_updated_at_utc": _safe_timestamp(event.get("odds_updated_at")),
                "updated_at_utc": _safe_timestamp(event.get("updated_at")),
                "snapshot_time_utc": snapshot_ts,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "sport_id",
                "event_id",
                "our_event_id",
                "fi",
                "league_name",
                "fighter_a",
                "fighter_b",
                "scheduled_start_utc",
                "time_status",
                "odds_updated_at_utc",
                "updated_at_utc",
                "snapshot_time_utc",
                "fight_key",
            ]
        )

    frame["fight_key"] = frame.apply(
        lambda row: _fight_key(row["fighter_a"], row["fighter_b"], row["scheduled_start_utc"]),
        axis=1,
    )
    return frame


def _selection_iter(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Walk nested prematch market containers and yield selection dicts."""
    results: list[tuple[str, dict[str, Any]]] = []

    if isinstance(value, dict):
        nested_odds = value.get("odds")
        if isinstance(nested_odds, list):
            results.extend(_selection_iter(nested_odds, prefix))
            return results
        if _is_selection_leaf(value):
            results.append((prefix, value))
            return results
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(nested, list):
                for item in nested:
                    if _is_selection_leaf(item):
                        results.append((nested_prefix, item))
                    else:
                        results.extend(_selection_iter(item, nested_prefix))
            else:
                results.extend(_selection_iter(nested, nested_prefix))
    elif isinstance(value, list):
        for item in value:
            results.extend(_selection_iter(item, prefix))

    return results


def _iter_prematch_market_groups(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Yield each prematch market dict, including deferred top-level others blocks."""
    groups: list[tuple[str, dict[str, Any]]] = []

    for container_name in ["main", "schedule"]:
        container = result.get(container_name) or {}
        if not isinstance(container, dict):
            continue
        markets = container.get("sp") or {}
        if isinstance(markets, dict):
            groups.append((container_name, markets))

    others = result.get("others") or []
    if isinstance(others, dict):
        others = [others]
    if isinstance(others, list):
        for container in others:
            if not isinstance(container, dict):
                continue
            markets = container.get("sp") or {}
            if isinstance(markets, dict):
                groups.append(("others", markets))

    return groups


def normalize_bet365_prematch_market_rows(
    payload: dict[str, Any],
    *,
    snapshot_time_utc: Optional[object] = None,
) -> pd.DataFrame:
    """Flatten Bet365 prematch payloads into one row per market selection."""
    snapshot_ts = pd.to_datetime(snapshot_time_utc, errors="coerce", utc=True)
    if pd.isna(snapshot_ts):
        snapshot_ts = pd.Timestamp.now(tz="UTC")

    result = _prematch_result_root(payload)
    fi = str(result.get("FI", ""))
    event_id = str(result.get("event_id", ""))
    rows: list[dict[str, object]] = []

    for container_name, markets in _iter_prematch_market_groups(result):
        for market_key, market_value in markets.items():
            for market_path, selection in _selection_iter(market_value, prefix=market_key):
                rows.append(
                    {
                        "fi": fi,
                        "event_id": event_id,
                        "snapshot_time_utc": snapshot_ts,
                        "market_group": container_name,
                        "market_name": market_path,
                        "selection_id": str(selection.get("id", "")),
                        "selection_name": selection.get("name", ""),
                        "selection_header": selection.get("header", ""),
                        "selection_handicap": selection.get("handicap", ""),
                        "decimal_odds": _safe_float(selection.get("odds")),
                    }
                )

    return pd.DataFrame(rows)


def normalize_mma_odds_summary_rows(
    payload: dict[str, Any],
    *,
    event_id: object,
    snapshot_time_utc: Optional[object] = None,
) -> pd.DataFrame:
    """Flatten BetsAPI event/odds/summary payloads into bookmaker-phase rows."""
    snapshot_ts = pd.to_datetime(snapshot_time_utc, errors="coerce", utc=True)
    if pd.isna(snapshot_ts):
        snapshot_ts = pd.Timestamp.now(tz="UTC")

    rows: list[dict[str, object]] = []
    results = payload.get("results") or {}
    if not isinstance(results, dict):
        return pd.DataFrame(columns=SUMMARY_ROW_COLUMNS)

    for bookmaker, bookmaker_payload in results.items():
        payload_dict = bookmaker_payload if isinstance(bookmaker_payload, dict) else {}
        updates = payload_dict.get("odds_update") or {}
        update_times = [
            ts for ts in (_safe_timestamp(value) for value in updates.values())
            if not pd.isna(ts)
        ]
        bookmaker_last_update = max(update_times) if update_times else pd.NaT
        odds = payload_dict.get("odds") or {}
        matching_dir = pd.to_numeric(payload_dict.get("matching_dir"), errors="coerce")

        for market_phase, market_map in odds.items():
            if not isinstance(market_map, dict):
                continue
            for market_key, market in market_map.items():
                if not isinstance(market, dict):
                    continue
                home_odds = _safe_float(market.get("home_od"))
                over_odds = _safe_float(market.get("over_od"))
                away_odds = _safe_float(market.get("away_od"))
                under_odds = _safe_float(market.get("under_od"))
                draw_odds = _safe_float(market.get("draw_od"))
                if (
                    home_odds is None
                    and away_odds is None
                    and over_odds is None
                    and under_odds is None
                    and draw_odds is None
                ):
                    continue
                rows.append(
                    {
                        "event_id": str(event_id),
                        "bookmaker": str(bookmaker),
                        "matching_dir": matching_dir,
                        "market_phase": str(market_phase),
                        "market_key": str(market_key),
                        "line_id": str(market.get("id", "")),
                        "home_odds": home_odds,
                        "over_odds": over_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "under_odds": under_odds,
                        "handicap": market.get("handicap"),
                        "score_state": market.get("ss"),
                        "selection_added_at_utc": _safe_timestamp(market.get("add_time")),
                        "bookmaker_last_update_utc": bookmaker_last_update,
                        "snapshot_time_utc": snapshot_ts,
                    }
                )

    return pd.DataFrame(rows, columns=SUMMARY_ROW_COLUMNS)


def snapshot_mma_payloads(
    client: Optional[BetsApiMMAClient] = None,
    *,
    max_prematch: int = 10,
    page: int = 1,
    ufc_only: bool = False,
    day: Optional[str] = None,
    hydrate_odds_summary: bool = False,
) -> dict[str, BetsApiRequestResult]:
    """Fetch a minimal MMA snapshot set and persist raw responses."""
    client = client or BetsApiMMAClient()

    upcoming = client.get_mma_upcoming_events(page=page, day=day)
    upcoming_path = _write_raw_payload(
        RAW_EVENTS_UPCOMING_DIR,
        upcoming,
        "events_upcoming",
        {"page": page, "day": day},
    )

    bet365_upcoming = client.get_bet365_mma_upcoming(page=page, day=day)
    bet365_upcoming_path = _write_raw_payload(
        RAW_BET365_UPCOMING_DIR,
        bet365_upcoming,
        "bet365_upcoming",
        {"page": page, "day": day},
    )

    prematch_results: dict[str, BetsApiRequestResult] = {
        "events_upcoming": BetsApiRequestResult(
            endpoint="events/upcoming",
            path=upcoming_path,
            count=len(_response_rows(upcoming)),
            payload=upcoming,
        ),
        "bet365_upcoming": BetsApiRequestResult(
            endpoint="bet365/upcoming",
            path=bet365_upcoming_path,
            count=len(_response_rows(bet365_upcoming)),
            payload=bet365_upcoming,
        ),
    }

    upcoming_rows = _response_rows(bet365_upcoming)
    if ufc_only:
        upcoming_rows = [
            row for row in upcoming_rows
            if "ufc" in normalize_mma_text(_entity_name(row.get("league")))
        ]

    for row in upcoming_rows[:max_prematch]:
        fi = row.get("id")
        payload = client.get_bet365_mma_prematch(fi)
        path = _write_raw_payload(
            RAW_BET365_PREMATCH_DIR,
            payload,
            f"bet365_prematch_fi_{fi}",
            {"fi": fi},
        )
        prematch_results[f"prematch_{fi}"] = BetsApiRequestResult(
            endpoint="bet365/prematch",
            path=path,
            count=len(normalize_bet365_prematch_market_rows(payload)),
            payload=payload,
        )

        if hydrate_odds_summary:
            event_id = _prematch_result_root(payload).get("event_id")
            if event_id:
                summary_payload = client.get_event_odds_summary(event_id)
                summary_path = _write_raw_payload(
                    RAW_ODDS_SUMMARY_DIR,
                    summary_payload,
                    f"odds_summary_event_{event_id}",
                    {"event_id": event_id, "fi": fi},
                )
                prematch_results[f"odds_summary_{event_id}"] = BetsApiRequestResult(
                    endpoint="event/odds/summary",
                    path=summary_path,
                    count=len(_response_rows(summary_payload)),
                    payload=summary_payload,
                )

    return prematch_results


def backfill_mma_ended_payloads(
    client: Optional[BetsApiMMAClient] = None,
    *,
    page_from: int = 1,
    page_to: int = 1,
    day: Optional[str] = None,
    hydrate_odds_summary: bool = False,
    max_summaries: Optional[int] = None,
    ufc_only: bool = False,
) -> list[BetsApiRequestResult]:
    """Backfill ended MMA pages and optionally hydrate odds summaries."""
    client = client or BetsApiMMAClient()
    artifacts: list[BetsApiRequestResult] = []
    summary_count = 0

    for page in range(page_from, page_to + 1):
        payload = client.get_mma_ended_events(page=page, day=day)
        path = _write_raw_payload(
            RAW_EVENTS_ENDED_DIR,
            payload,
            f"events_ended_page_{page}",
            {"page": page, "day": day},
        )
        artifacts.append(
            BetsApiRequestResult(
                endpoint="events/ended",
                path=path,
                count=len(_response_rows(payload)),
                payload=payload,
            )
        )

        if not hydrate_odds_summary:
            continue

        summary_rows = _response_rows(payload)
        if ufc_only:
            summary_rows = [event for event in summary_rows if _is_ufc_event_row(event)]

        for event in summary_rows:
            if max_summaries is not None and summary_count >= max_summaries:
                break
            event_id = event.get("id")
            if not event_id:
                continue
            summary_payload = client.get_event_odds_summary(event_id)
            summary_path = _write_raw_payload(
                RAW_ODDS_SUMMARY_DIR,
                summary_payload,
                f"odds_summary_event_{event_id}",
                {"event_id": event_id, "page": page, "day": day},
            )
            artifacts.append(
                BetsApiRequestResult(
                    endpoint="event/odds/summary",
                    path=summary_path,
                    count=len(_response_rows(summary_payload)),
                    payload=summary_payload,
                )
            )
            summary_count += 1

    return artifacts


def _snapshot_from_filename(path: Path) -> pd.Timestamp:
    match = re.match(r"(?P<stamp>\d{8}T\d{6}Z)_", path.name)
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group("stamp"), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")


def _load_json_files(directory: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    if not directory.exists():
        return loaded

    for path in sorted(directory.glob("*.json")):
        payload, metadata = _load_raw_payload(path)
        loaded.append((path, payload, metadata))
    return loaded


def _betsapi_total_pages(payload: dict[str, Any]) -> int:
    """Infer total BetsAPI pages across documented and observed pager shapes."""
    pager = payload.get("pager") or {}

    total_pages = pd.to_numeric(pager.get("total_page"), errors="coerce")
    if not pd.isna(total_pages) and int(total_pages) > 0:
        return int(total_pages)

    total_results = pd.to_numeric(pager.get("total"), errors="coerce")
    per_page = pd.to_numeric(pager.get("per_page"), errors="coerce")
    if not pd.isna(total_results) and not pd.isna(per_page) and int(per_page) > 0:
        return max((int(total_results) + int(per_page) - 1) // int(per_page), 1)

    current_page = pd.to_numeric(pager.get("page"), errors="coerce")
    if not pd.isna(current_page) and int(current_page) > 0:
        return int(current_page)

    return 1


def _summary_event_context(path: Path, metadata: dict[str, Any]) -> str:
    event_id = metadata.get("event_id")
    if event_id not in (None, ""):
        return str(event_id)
    if path.stem.startswith("event_"):
        return path.stem[len("event_"):]
    match = SUMMARY_FILE_RE.match(path.name)
    return match.group("event_id") if match else ""


def _straight_moneyline_summary_rows(summary_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only straight winner-moneyline rows before odds-unit validation."""
    if summary_rows.empty:
        return summary_rows

    subset = summary_rows.copy()
    subset["home_odds"] = pd.to_numeric(subset["home_odds"], errors="coerce")
    subset["away_odds"] = pd.to_numeric(subset["away_odds"], errors="coerce")
    subset = subset[subset["home_odds"].notna() & subset["away_odds"].notna()].copy()
    subset["handicap_text"] = subset["handicap"].astype(str).str.strip()
    subset = subset[(subset["handicap"].isna()) | (subset["handicap_text"] == "")].copy()
    if subset.empty:
        return subset
    return subset.drop(columns="handicap_text")


def _moneyline_summary_rows(summary_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only straight winner-moneyline rows with decimal-odds semantics."""
    subset = _straight_moneyline_summary_rows(summary_rows)
    if subset.empty:
        return subset

    subset = subset[
        subset["home_odds"].map(_is_valid_decimal_odds)
        & subset["away_odds"].map(_is_valid_decimal_odds)
    ].copy()
    if subset.empty:
        return subset

    subset = subset.sort_values(
        ["event_id", "bookmaker", "market_phase", "selection_added_at_utc", "market_key"],
        na_position="last",
    )
    subset = subset.drop_duplicates(subset=["event_id", "bookmaker", "market_phase"], keep="first")
    subset["home_implied_prob"] = 1.0 / subset["home_odds"]
    subset["away_implied_prob"] = 1.0 / subset["away_odds"]
    return subset


def _market_summary_rows(
    summary_rows: pd.DataFrame,
    *,
    market_key: str,
    left_odds_column: str,
    right_odds_column: str,
    require_blank_handicap: bool = False,
) -> pd.DataFrame:
    """Normalize one odds-summary market into per-bookmaker implied probabilities."""
    if summary_rows.empty:
        return summary_rows

    subset = summary_rows.copy()
    subset = subset[subset["market_key"].astype(str) == str(market_key)].copy()
    subset[left_odds_column] = pd.to_numeric(subset[left_odds_column], errors="coerce")
    subset[right_odds_column] = pd.to_numeric(subset[right_odds_column], errors="coerce")
    subset = subset[
        subset[left_odds_column].notna()
        & subset[right_odds_column].notna()
        & subset[left_odds_column].map(_is_valid_decimal_odds)
        & subset[right_odds_column].map(_is_valid_decimal_odds)
    ].copy()
    if subset.empty:
        return subset

    if require_blank_handicap:
        subset["handicap_text"] = subset["handicap"].astype(str).str.strip()
        subset = subset[(subset["handicap"].isna()) | (subset["handicap_text"] == "")].copy()
        if subset.empty:
            return subset

    subset = subset.sort_values(
        ["event_id", "bookmaker", "market_phase", "selection_added_at_utc", "market_key"],
        na_position="last",
    )
    subset = subset.drop_duplicates(subset=["event_id", "bookmaker", "market_phase"], keep="first")
    subset["left_implied_prob"] = 1.0 / subset[left_odds_column]
    subset["right_implied_prob"] = 1.0 / subset[right_odds_column]
    subset["handicap_value"] = pd.to_numeric(subset["handicap"], errors="coerce")
    return subset


def _sanitize_summary_rows(summary_rows: pd.DataFrame) -> pd.DataFrame:
    """Drop normalized odds-summary rows that do not preserve decimal-odds semantics."""
    if summary_rows.empty:
        return summary_rows

    sanitized = summary_rows.copy()
    odds_columns = [
        column
        for column in ("home_odds", "away_odds", "over_odds", "under_odds", "draw_odds")
        if column in sanitized.columns
    ]
    if not odds_columns:
        return sanitized

    invalid_row_mask = pd.Series(False, index=sanitized.index)
    for column in odds_columns:
        sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce")
        invalid_row_mask |= sanitized[column].notna() & ~sanitized[column].map(_is_valid_decimal_odds)

    return sanitized.loc[~invalid_row_mask].copy()


def _summary_phase_snapshot(summary_rows: pd.DataFrame, market_phase: str, prefix: str) -> pd.DataFrame:
    subset = _moneyline_summary_rows(summary_rows)
    subset = subset[subset["market_phase"] == market_phase].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                f"{prefix}_snapshot_timestamp_utc",
                f"{prefix}_bookmakers",
                f"{prefix}_home_implied_prob_avg",
                f"{prefix}_away_implied_prob_avg",
            ]
        )

    return (
        subset.groupby("event_id", dropna=False)
        .agg(
            **{
                f"{prefix}_snapshot_timestamp_utc": ("selection_added_at_utc", "min"),
                f"{prefix}_bookmakers": ("bookmaker", "nunique"),
                f"{prefix}_home_implied_prob_avg": ("home_implied_prob", "mean"),
                f"{prefix}_away_implied_prob_avg": ("away_implied_prob", "mean"),
            }
        )
        .reset_index()
    )


def _market_phase_snapshot(
    summary_rows: pd.DataFrame,
    *,
    market_key: str,
    market_phase: str,
    prefix: str,
    left_odds_column: str,
    right_odds_column: str,
    require_blank_handicap: bool = False,
) -> pd.DataFrame:
    subset = _market_summary_rows(
        summary_rows,
        market_key=market_key,
        left_odds_column=left_odds_column,
        right_odds_column=right_odds_column,
        require_blank_handicap=require_blank_handicap,
    )
    subset = subset[subset["market_phase"] == market_phase].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                f"{prefix}_snapshot_timestamp_utc",
                f"{prefix}_bookmakers",
                f"{prefix}_left_implied_prob_avg",
                f"{prefix}_right_implied_prob_avg",
                f"{prefix}_handicap_avg",
            ]
        )

    return (
        subset.groupby("event_id", dropna=False)
        .agg(
            **{
                f"{prefix}_snapshot_timestamp_utc": ("selection_added_at_utc", "min"),
                f"{prefix}_bookmakers": ("bookmaker", "nunique"),
                f"{prefix}_left_implied_prob_avg": ("left_implied_prob", "mean"),
                f"{prefix}_right_implied_prob_avg": ("right_implied_prob", "mean"),
                f"{prefix}_handicap_avg": ("handicap_value", "mean"),
            }
        )
        .reset_index()
    )


def build_mma_event_snapshots(events_df: pd.DataFrame, summary_rows_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ended-event odds summaries into one historical row per MMA fight."""
    if events_df.empty:
        return pd.DataFrame(columns=EVENT_SNAPSHOT_COLUMNS)

    base = events_df.copy()
    base["event_id"] = base["event_id"].astype(str)
    base["scheduled_start_utc"] = pd.to_datetime(base["scheduled_start_utc"], errors="coerce", utc=True)
    base = (
        base.sort_values("snapshot_time_utc")
        .groupby("fight_key", dropna=False)
        .tail(1)[["event_id", "fight_key", "fighter_a", "fighter_b", "scheduled_start_utc"]]
        .drop_duplicates(subset=["event_id"])
        .reset_index(drop=True)
    )

    if summary_rows_df.empty:
        for column in EVENT_SNAPSHOT_COLUMNS:
            if column not in base.columns:
                base[column] = pd.NA
        return base[EVENT_SNAPSHOT_COLUMNS].copy()

    summary_rows = summary_rows_df.copy()
    summary_rows["event_id"] = summary_rows["event_id"].astype(str)
    update_summary = (
        summary_rows.groupby("event_id", dropna=False)
        .agg(
            first_bookmaker_last_update_utc=("bookmaker_last_update_utc", "min"),
            last_bookmaker_last_update_utc=("bookmaker_last_update_utc", "max"),
        )
        .reset_index()
    )

    snapshots = base.copy()
    for market_phase, prefix in [("start", "opening"), ("end", "end")]:
        snapshots = snapshots.merge(
            _summary_phase_snapshot(summary_rows, market_phase, prefix),
            on="event_id",
            how="left",
        )
    snapshots = snapshots.merge(update_summary, on="event_id", how="left")
    return snapshots[EVENT_SNAPSHOT_COLUMNS].copy()


def _historical_summary_features(event_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Convert historical event snapshots into conservative production-safe features."""
    if event_snapshots.empty:
        return pd.DataFrame(
            columns=[
                "fight_key",
                "betsapi_hist_opening_a_implied_prob",
                "betsapi_hist_opening_b_implied_prob",
                "betsapi_hist_diff_opening_implied_prob",
                "betsapi_hist_opening_bookmakers",
                "betsapi_hist_opening_lead_hours",
            ]
        )

    snapshots = event_snapshots.copy()
    snapshots["scheduled_start_utc"] = pd.to_datetime(snapshots["scheduled_start_utc"], errors="coerce", utc=True)
    snapshots["opening_snapshot_timestamp_utc"] = pd.to_datetime(
        snapshots["opening_snapshot_timestamp_utc"], errors="coerce", utc=True
    )
    lead_hours = (
        (snapshots["scheduled_start_utc"] - snapshots["opening_snapshot_timestamp_utc"]).dt.total_seconds() / 3600.0
    )

    return pd.DataFrame(
        {
            "fight_key": snapshots["fight_key"],
            "betsapi_hist_opening_a_implied_prob": snapshots["opening_home_implied_prob_avg"],
            "betsapi_hist_opening_b_implied_prob": snapshots["opening_away_implied_prob_avg"],
            "betsapi_hist_diff_opening_implied_prob": (
                snapshots["opening_home_implied_prob_avg"] - snapshots["opening_away_implied_prob_avg"]
            ),
            "betsapi_hist_opening_bookmakers": snapshots["opening_bookmakers"],
            "betsapi_hist_opening_lead_hours": lead_hours,
        }
    )


def _historical_summary_market_features(
    event_snapshots: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build additional generic historical summary-market features beyond straight moneyline."""
    if event_snapshots.empty or summary_rows.empty:
        return pd.DataFrame(columns=["fight_key"])

    base = event_snapshots[["event_id", "fight_key"]].drop_duplicates(subset=["event_id"]).copy()
    summary_rows = summary_rows.copy()
    summary_rows["event_id"] = summary_rows["event_id"].astype(str)
    base["event_id"] = base["event_id"].astype(str)

    market_specs = [
        {
            "market_key": "162_2",
            "left_odds_column": "home_odds",
            "right_odds_column": "away_odds",
            "feature_prefix": "betsapi_hist_m162_2",
            "left_label": "home",
            "right_label": "away",
        },
        {
            "market_key": "162_3",
            "left_odds_column": "over_odds",
            "right_odds_column": "under_odds",
            "feature_prefix": "betsapi_hist_m162_3",
            "left_label": "over",
            "right_label": "under",
        },
    ]

    features = base.copy()
    for spec in market_specs:
        opening = _market_phase_snapshot(
            summary_rows,
            market_key=spec["market_key"],
            market_phase="start",
            prefix="opening",
            left_odds_column=spec["left_odds_column"],
            right_odds_column=spec["right_odds_column"],
        )
        end = _market_phase_snapshot(
            summary_rows,
            market_key=spec["market_key"],
            market_phase="end",
            prefix="end",
            left_odds_column=spec["left_odds_column"],
            right_odds_column=spec["right_odds_column"],
        )
        merged = (
            base.merge(opening, on="event_id", how="left")
            .merge(end, on="event_id", how="left")
            .copy()
        )

        prefix = spec["feature_prefix"]
        left_label = spec["left_label"]
        right_label = spec["right_label"]
        features = features.merge(
            pd.DataFrame(
                {
                    "fight_key": merged["fight_key"],
                    f"{prefix}_opening_{left_label}_implied_prob": merged["opening_left_implied_prob_avg"],
                    f"{prefix}_opening_{right_label}_implied_prob": merged["opening_right_implied_prob_avg"],
                    f"{prefix}_opening_handicap": merged["opening_handicap_avg"],
                    f"{prefix}_end_{left_label}_implied_prob": merged["end_left_implied_prob_avg"],
                    f"{prefix}_end_{right_label}_implied_prob": merged["end_right_implied_prob_avg"],
                    f"{prefix}_end_handicap": merged["end_handicap_avg"],
                    f"{prefix}_prob_move_abs": (
                        (merged["end_left_implied_prob_avg"] - merged["opening_left_implied_prob_avg"]).abs()
                        + (merged["end_right_implied_prob_avg"] - merged["opening_right_implied_prob_avg"]).abs()
                    ),
                    f"{prefix}_handicap_move_abs": (
                        merged["end_handicap_avg"] - merged["opening_handicap_avg"]
                    ).abs(),
                }
            ),
            on="fight_key",
            how="left",
        )
    return features.drop(columns=["event_id"], errors="ignore")


def _historical_expanded_features(
    event_snapshots: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Compute expanded historical features from odds summary data.

    Returns a DataFrame keyed on ``fight_key`` with the following groups:
      - bookmaker_count, overround
      - opening-to-close probability moves (a/b)
      - cross-market agreement (moneyline vs total direction)
      - handicap direction, total direction
      - late update flag
      - market coverage flags (moneyline, totals, handicap)
      - event-level coverage percentage
      - market depth score
    """
    EXPANDED_COLUMNS = [
        "fight_key",
        "betsapi_hist_bookmaker_count",
        "betsapi_hist_overround",
        "betsapi_hist_prob_move_open_close_a",
        "betsapi_hist_prob_move_open_close_b",
        "betsapi_hist_moneyline_total_agreement",
        "betsapi_hist_handicap_direction",
        "betsapi_hist_total_direction",
        "betsapi_hist_has_late_update",
        "betsapi_hist_has_moneyline",
        "betsapi_hist_has_totals",
        "betsapi_hist_has_handicap",
        "betsapi_hist_event_coverage_pct",
        "betsapi_hist_market_depth",
    ]

    if event_snapshots.empty or summary_rows.empty:
        return pd.DataFrame(columns=EXPANDED_COLUMNS)

    snapshots = event_snapshots.copy()
    sr = summary_rows.copy()
    sr["event_id"] = sr["event_id"].astype(str)
    snapshots["event_id"] = snapshots["event_id"].astype(str)

    # Map event_id -> fight_key from snapshots (one fight per event_id)
    eid_to_fk = (
        snapshots[["event_id", "fight_key"]]
        .drop_duplicates(subset=["event_id"])
        .set_index("event_id")["fight_key"]
        .to_dict()
    )

    # ---- Per-event aggregations from summary_rows ----

    # 1. Bookmaker count: distinct bookmakers across all markets for each event
    bookmaker_count = (
        sr.groupby("event_id", dropna=False)["bookmaker"]
        .nunique()
        .rename("betsapi_hist_bookmaker_count")
    )

    # 2. Overround: from moneyline rows (home+away implied prob sum - 1)
    ml_rows = _moneyline_summary_rows(sr)
    if not ml_rows.empty:
        ml_rows = ml_rows.copy()
        ml_rows["overround"] = ml_rows["home_implied_prob"] + ml_rows["away_implied_prob"] - 1.0
        overround = (
            ml_rows.groupby("event_id", dropna=False)["overround"]
            .mean()
            .rename("betsapi_hist_overround")
        )
    else:
        overround = pd.Series(dtype=float, name="betsapi_hist_overround")

    # 3. Opening-to-close probability movement (a=home, b=away)
    #    Difference between end-phase and start-phase average implied probs
    prob_move_a = pd.Series(dtype=float, name="betsapi_hist_prob_move_open_close_a")
    prob_move_b = pd.Series(dtype=float, name="betsapi_hist_prob_move_open_close_b")
    ml_start = pd.DataFrame()
    ml_end = pd.DataFrame()
    if not ml_rows.empty:
        phase_avg = (
            ml_rows.groupby(["event_id", "market_phase"], dropna=False)
            .agg(home_avg=("home_implied_prob", "mean"), away_avg=("away_implied_prob", "mean"))
            .reset_index()
        )
        ml_start = phase_avg[phase_avg["market_phase"] == "start"].set_index("event_id")
        ml_end = phase_avg[phase_avg["market_phase"] == "end"].set_index("event_id")
        if not ml_start.empty and not ml_end.empty:
            prob_move_a = (ml_end["home_avg"] - ml_start["home_avg"]).rename("betsapi_hist_prob_move_open_close_a")
            prob_move_b = (ml_end["away_avg"] - ml_start["away_avg"]).rename("betsapi_hist_prob_move_open_close_b")

    # 4. Market presence flags per event
    #    Detect moneyline by checking which events had valid home_odds+away_odds rows
    ml_event_ids = set(ml_rows["event_id"].unique()) if not ml_rows.empty else set()
    has_moneyline_s = pd.Series(
        {eid: 1 if eid in ml_event_ids else 0 for eid in bookmaker_count.index},
        name="betsapi_hist_has_moneyline",
    )

    totals_rows = sr[sr["market_key"].astype(str) == "162_3"]
    totals_event_ids = set(totals_rows["event_id"].unique()) if not totals_rows.empty else set()
    has_totals_s = pd.Series(
        {eid: 1 if eid in totals_event_ids else 0 for eid in bookmaker_count.index},
        name="betsapi_hist_has_totals",
    )

    handicap_rows = sr[sr["market_key"].astype(str) == "162_2"]
    handicap_event_ids = set(handicap_rows["event_id"].unique()) if not handicap_rows.empty else set()
    has_handicap_s = pd.Series(
        {eid: 1 if eid in handicap_event_ids else 0 for eid in bookmaker_count.index},
        name="betsapi_hist_has_handicap",
    )

    # 5. Market depth score: bookmaker_count * number of distinct market types
    market_type_count = (
        sr.groupby("event_id", dropna=False)["market_key"]
        .nunique()
        .rename("market_type_count")
    )
    market_depth = (bookmaker_count * market_type_count).rename("betsapi_hist_market_depth")

    # 6. Handicap direction: positive handicap means home is underdog (needs more rounds)
    #    Use end-phase handicap market (162_2); >0 = away favored, <0 = home favored
    handicap_direction = pd.Series(dtype=float, name="betsapi_hist_handicap_direction")
    if not handicap_rows.empty:
        hc = handicap_rows.copy()
        hc["handicap_value"] = pd.to_numeric(hc["handicap"], errors="coerce")
        hc_end = hc[hc["market_phase"] == "end"]
        if hc_end.empty:
            hc_end = hc  # fall back to all phases
        handicap_direction = (
            hc_end.groupby("event_id", dropna=False)["handicap_value"]
            .mean()
            .rename("betsapi_hist_handicap_direction")
        )

    # 7. Total direction: average handicap on totals market (162_3)
    #    Higher value = bookmakers expect more rounds/action
    total_direction = pd.Series(dtype=float, name="betsapi_hist_total_direction")
    if not totals_rows.empty:
        tt = totals_rows.copy()
        tt["handicap_value"] = pd.to_numeric(tt["handicap"], errors="coerce")
        tt_end = tt[tt["market_phase"] == "end"]
        if tt_end.empty:
            tt_end = tt
        total_direction = (
            tt_end.groupby("event_id", dropna=False)["handicap_value"]
            .mean()
            .rename("betsapi_hist_total_direction")
        )

    # 8. Cross-market agreement: correlation between moneyline favorite and total direction
    #    Positive = moneyline favorite also expected to go over in totals (aligned)
    #    We approximate by: sign(home_implied - away_implied) * sign(handicap on totals)
    xmarket = pd.Series(dtype=float, name="betsapi_hist_moneyline_total_agreement")
    if not ml_rows.empty and not totals_rows.empty:
        # moneyline end-phase: who is favored?  home_implied > 0.5 means home favored
        if not ml_end.empty:
            ml_direction = (ml_end["home_avg"] - 0.5).rename("ml_dir")  # positive = home favored
        elif not ml_start.empty:
            ml_direction = (ml_start["home_avg"] - 0.5).rename("ml_dir")
        else:
            ml_direction = pd.Series(dtype=float, name="ml_dir")
        # total direction already computed above
        if not ml_direction.empty:
            combined = pd.DataFrame({"ml_dir": ml_direction, "total_dir": total_direction}).dropna()
            if not combined.empty:
                xmarket = (np.sign(combined["ml_dir"]) * np.sign(combined["total_dir"])).rename(
                    "betsapi_hist_moneyline_total_agreement"
                )

    # 9. Late update flag: whether any bookmaker updated odds within 2 hours of scheduled start
    snapshots["scheduled_start_utc"] = pd.to_datetime(
        snapshots["scheduled_start_utc"], errors="coerce", utc=True
    )
    eid_to_start = snapshots.set_index("event_id")["scheduled_start_utc"].to_dict()

    sr["bookmaker_last_update_utc"] = pd.to_datetime(sr["bookmaker_last_update_utc"], errors="coerce", utc=True)
    has_late = {}
    for eid, group in sr.groupby("event_id", dropna=False):
        scheduled = eid_to_start.get(eid)
        if pd.isna(scheduled) or scheduled is pd.NaT:
            has_late[eid] = np.nan
            continue
        last_updates = group["bookmaker_last_update_utc"].dropna()
        if last_updates.empty:
            has_late[eid] = 0
        else:
            hours_before = (scheduled - last_updates).dt.total_seconds() / 3600.0
            has_late[eid] = 1 if (hours_before <= 2.0).any() else 0
    has_late_s = pd.Series(has_late, name="betsapi_hist_has_late_update")

    # 10. Event-level coverage: % of fights on same event card that have BetsAPI odds
    #     We use event_snapshots which has one row per fight; we check which have opening probs
    if "opening_home_implied_prob_avg" in snapshots.columns:
        # Group fights by their event card date
        snapshots["event_date_only"] = pd.to_datetime(
            snapshots["scheduled_start_utc"], errors="coerce"
        ).dt.date
        card_total = snapshots.groupby("event_date_only", dropna=False)["fight_key"].count().rename("total")
        card_covered = (
            snapshots[snapshots["opening_home_implied_prob_avg"].notna()]
            .groupby("event_date_only", dropna=False)["fight_key"]
            .count()
            .rename("covered")
        )
        coverage = (card_covered / card_total).rename("coverage_pct")
        # Map back to event_id
        snap_date_map = snapshots.set_index("event_id")["event_date_only"].to_dict()
        event_coverage = pd.Series(
            {eid: coverage.get(snap_date_map.get(eid)) for eid in bookmaker_count.index},
            name="betsapi_hist_event_coverage_pct",
        )
    else:
        event_coverage = pd.Series(dtype=float, name="betsapi_hist_event_coverage_pct")

    # ---- Assemble per-event DataFrame, then map to fight_key ----
    per_event = pd.DataFrame({"event_id": bookmaker_count.index})
    for series in [
        bookmaker_count,
        overround,
        prob_move_a,
        prob_move_b,
        xmarket,
        handicap_direction,
        total_direction,
        has_late_s,
        has_moneyline_s,
        has_totals_s,
        has_handicap_s,
        event_coverage,
        market_depth,
    ]:
        per_event = per_event.merge(
            series.reset_index().rename(columns={"index": "event_id"}),
            on="event_id",
            how="left",
        )

    per_event["fight_key"] = per_event["event_id"].map(eid_to_fk)
    per_event = per_event.dropna(subset=["fight_key"])

    return per_event[[c for c in EXPANDED_COLUMNS if c in per_event.columns]].copy()


def _artifacts(processed_root: Path) -> MMABackfillArtifacts:
    processed_root.mkdir(parents=True, exist_ok=True)
    return MMABackfillArtifacts(
        events_csv=processed_root / "ended_events.csv",
        summary_rows_csv=processed_root / "odds_summary_rows.csv",
        event_snapshots_csv=processed_root / "event_snapshots.csv",
        manifest_json=processed_root / "backfill_manifest.json",
    )


def _summary_path_inventory(summary_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return canonical and non-canonical odds-summary payload paths."""
    if not summary_dir.exists():
        return [], []

    canonical_paths = sorted(summary_dir.glob("event_*.json"))
    canonical_names = {path.name for path in canonical_paths}
    noncanonical_paths = [
        path
        for path in sorted(summary_dir.glob("*.json"))
        if path.name not in canonical_names
    ]
    return canonical_paths, noncanonical_paths


def _requested_event_ids_from_raw(
    raw_root: Path,
    *,
    ufc_only: bool,
) -> list[str]:
    """Return distinct requested event ids from saved ended-event payloads."""
    requested_event_ids: list[str] = []
    for _path, payload, _metadata in _load_json_files(raw_root / "events_ended"):
        rows = _response_rows(payload)
        if ufc_only:
            rows = [event for event in rows if _is_ufc_event_row(event)]
        requested_event_ids.extend(
            str(event.get("id"))
            for event in rows
            if event.get("id") not in (None, "")
        )
    return list(dict.fromkeys(requested_event_ids))


def _load_existing_manifest(processed_root: Path) -> dict[str, Any] | None:
    manifest_path = processed_root / "backfill_manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _build_processed_manifest(
    *,
    raw_root: Path,
    frames: dict[str, pd.DataFrame],
    existing_manifest: Optional[dict[str, Any]] = None,
    manifest_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a manifest that matches the current processed MMA artifact set."""
    manifest = dict(existing_manifest or {})
    if manifest_overrides:
        manifest.update(manifest_overrides)

    ufc_only = bool(manifest.get("ufc_only"))
    requested_event_ids = _requested_event_ids_from_raw(raw_root, ufc_only=ufc_only)
    canonical_summary_paths, noncanonical_summary_paths = _summary_path_inventory(
        raw_root / "odds_summary"
    )

    manifest.update(
        {
            "source": "betsapi",
            "sport": "mma",
            "raw_event_payload_files": len(list((raw_root / "events_ended").glob("*.json"))),
            "raw_summary_payload_files": len(canonical_summary_paths),
            "raw_summary_payload_files_total": len(canonical_summary_paths)
            + len(noncanonical_summary_paths),
            "raw_summary_payload_files_noncanonical": len(noncanonical_summary_paths),
            "requested_event_rows": len(requested_event_ids),
            "requested_ufc_event_ids": len(
                _requested_event_ids_from_raw(raw_root, ufc_only=True)
            ),
            "hydrated_summaries": len(canonical_summary_paths),
            "event_rows": int(len(frames["events_ended"])),
            "summary_rows": int(len(frames["summary_rows"])),
            "event_snapshots": int(len(frames["event_snapshots"])),
            "events_with_summaries": int(frames["summary_rows"]["event_id"].nunique())
            if not frames["summary_rows"].empty
            else 0,
            "generated_at_utc": _utc_now(),
        }
    )
    return manifest


def _write_processed_betsapi_mma_artifacts(
    *,
    raw_root: Path,
    processed_root: Path,
    frames: dict[str, pd.DataFrame],
    manifest_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, object]:
    """Write processed MMA CSVs plus a manifest that matches them exactly."""
    artifacts = _artifacts(processed_root)
    frames["events_ended"].to_csv(artifacts.events_csv, index=False)
    frames["summary_rows"].to_csv(artifacts.summary_rows_csv, index=False)
    frames["event_snapshots"].to_csv(artifacts.event_snapshots_csv, index=False)

    manifest = _build_processed_manifest(
        raw_root=raw_root,
        frames=frames,
        existing_manifest=_load_existing_manifest(processed_root),
        manifest_overrides=manifest_overrides,
    )
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Saved BetsAPI MMA backfill: %s ended events, %s summary rows, %s event snapshots",
        len(frames["events_ended"]),
        len(frames["summary_rows"]),
        len(frames["event_snapshots"]),
    )
    return {
        "events_path": artifacts.events_csv,
        "summary_rows_path": artifacts.summary_rows_csv,
        "event_snapshots_path": artifacts.event_snapshots_csv,
        "manifest_path": artifacts.manifest_json,
        "event_rows": int(len(frames["events_ended"])),
        "summary_rows": int(len(frames["summary_rows"])),
        "event_snapshots": int(len(frames["event_snapshots"])),
        "events_with_summaries": int(frames["summary_rows"]["event_id"].nunique())
        if not frames["summary_rows"].empty
        else 0,
    }


def backfill_betsapi_mma_history(
    *,
    start_date: str,
    end_date: Optional[str] = None,
    client: Optional[BetsApiMMAClient] = None,
    hydrate_odds_summary: bool = True,
    max_summaries: Optional[int] = None,
    resume: bool = True,
    ufc_only: bool = False,
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    processed_root: Path = BETSAPI_MMA_PROCESSED_DIR,
) -> dict[str, object]:
    """Backfill BetsAPI MMA ended events and optional odds summaries by date range."""
    client = client or BetsApiMMAClient()
    events_dir = raw_root / "events_ended"
    summaries_dir = raw_root / "odds_summary"
    events_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    start_ts = _normalize_query_date(start_date)
    end_ts = (
        _normalize_query_date(end_date)
        if end_date is not None
        else _normalize_query_date(pd.Timestamp.now(tz="UTC"))
    )
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    requested_event_ids: list[str] = []
    total_event_payloads = 0
    total_days = 0

    for day in pd.date_range(start_ts, end_ts, freq="D"):
        total_days += 1
        page = 1
        day_query = day.strftime("%Y%m%d")
        while True:
            raw_path = events_dir / f"{day_query}_page_{page:03d}.json"
            if resume and raw_path.exists():
                payload, _ = _load_raw_payload(raw_path)
            else:
                payload = client.get_mma_ended_events(page=page, day=day_query)
                _write_raw_file(
                    raw_path,
                    payload,
                    metadata={"query_day": day.strftime("%Y-%m-%d"), "query_page": page, "saved_at_utc": _utc_now()},
                )

            summary_rows = _response_rows(payload)
            if ufc_only:
                summary_rows = [event for event in summary_rows if _is_ufc_event_row(event)]
            requested_event_ids.extend(
                [
                    str(event.get("id"))
                    for event in summary_rows
                    if event.get("id") not in (None, "")
                ]
            )
            total_event_payloads += 1
            total_pages = _betsapi_total_pages(payload)
            if page >= total_pages:
                break
            page += 1

    hydrated_summaries = 0
    requested_event_ids = list(dict.fromkeys(requested_event_ids))
    if hydrate_odds_summary:
        pending_event_ids: list[str] = []
        for event_id in requested_event_ids:
            if max_summaries is not None and hydrated_summaries >= max_summaries:
                break
            raw_path = summaries_dir / f"event_{event_id}.json"
            if resume and raw_path.exists():
                _load_raw_payload(raw_path)
                hydrated_summaries += 1
                continue
            pending_event_ids.append(event_id)

        if pending_event_ids:
            worker_count = min(BETSAPI_SUMMARY_MAX_WORKERS, len(pending_event_ids))
            logger.info(
                "Hydrating %s BetsAPI MMA odds summaries with %s workers%s",
                len(pending_event_ids),
                worker_count,
                " (UFC only)" if ufc_only else "",
            )
            deferred_event_ids: list[str] = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_event_id = {
                    executor.submit(client.get_event_odds_summary, event_id): event_id
                    for event_id in pending_event_ids
                }
                for future in as_completed(future_to_event_id):
                    event_id = future_to_event_id[future]
                    try:
                        payload = future.result()
                    except Exception as exc:
                        if _is_retryable_request_exception(exc):
                            logger.warning(
                                "Deferring BetsAPI MMA odds summary %s after concurrent %s; retrying sequentially",
                                event_id,
                                type(exc).__name__,
                            )
                            deferred_event_ids.append(event_id)
                            continue
                        raise
                    raw_path = summaries_dir / f"event_{event_id}.json"
                    _write_raw_file(
                        raw_path,
                        payload,
                        metadata={"event_id": event_id, "saved_at_utc": _utc_now()},
                    )
                    hydrated_summaries += 1

            if deferred_event_ids:
                logger.warning(
                    "Retrying %s deferred BetsAPI MMA odds summaries sequentially after cooldown",
                    len(deferred_event_ids),
                )
                time.sleep(30)
                for event_id in deferred_event_ids:
                    payload = client.get_event_odds_summary(event_id)
                    raw_path = summaries_dir / f"event_{event_id}.json"
                    _write_raw_file(
                        raw_path,
                        payload,
                        metadata={"event_id": event_id, "saved_at_utc": _utc_now()},
                    )
                    hydrated_summaries += 1

    frames = load_saved_betsapi_mma_frames(raw_root=raw_root)
    return _write_processed_betsapi_mma_artifacts(
        raw_root=raw_root,
        processed_root=processed_root,
        frames=frames,
        manifest_overrides={
            "start_date": start_ts.strftime("%Y-%m-%d"),
            "end_date": end_ts.strftime("%Y-%m-%d"),
            "resume": bool(resume),
            "ufc_only": bool(ufc_only),
            "hydrate_odds_summary": bool(hydrate_odds_summary),
            "max_summaries": max_summaries,
            "days_requested": total_days,
            "event_payloads_saved": total_event_payloads,
        },
    )


def rebuild_betsapi_mma_history_from_raw(
    *,
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    processed_root: Path = BETSAPI_MMA_PROCESSED_DIR,
) -> dict[str, object]:
    """Rebuild processed MMA historical artifacts from saved raw payloads only."""
    frames = load_saved_betsapi_mma_frames(raw_root=raw_root)
    return _write_processed_betsapi_mma_artifacts(
        raw_root=raw_root,
        processed_root=processed_root,
        frames=frames,
        manifest_overrides={"rebuild_only": True},
    )


def load_saved_betsapi_mma_frames(
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    *,
    include_noncanonical_summary: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load saved raw MMA payloads into normalized event and market frames."""
    event_frames: list[pd.DataFrame] = []
    bet365_frames: list[pd.DataFrame] = []
    prematch_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for path, payload, metadata in _load_json_files(raw_root / "events_upcoming"):
        event_frames.append(
            normalize_mma_event_rows(
                payload,
                source="events_upcoming",
                snapshot_time_utc=(
                    metadata.get("snapshot_time_utc")
                    or metadata.get("saved_at_utc")
                    or _snapshot_from_filename(path)
                ),
            )
        )

    for path, payload, metadata in _load_json_files(raw_root / "events_ended"):
        event_frames.append(
            normalize_mma_event_rows(
                payload,
                source="events_ended",
                snapshot_time_utc=(
                    metadata.get("snapshot_time_utc")
                    or metadata.get("saved_at_utc")
                    or _snapshot_from_filename(path)
                ),
            )
        )

    for path, payload, metadata in _load_json_files(raw_root / "bet365_upcoming"):
        bet365_frames.append(
            normalize_mma_event_rows(
                payload,
                source="bet365_upcoming",
                snapshot_time_utc=(
                    metadata.get("snapshot_time_utc")
                    or metadata.get("saved_at_utc")
                    or _snapshot_from_filename(path)
                ),
            )
        )

    for path, payload, metadata in _load_json_files(raw_root / "bet365_prematch"):
        prematch = normalize_bet365_prematch_market_rows(
            payload,
            snapshot_time_utc=(
                metadata.get("snapshot_time_utc")
                or metadata.get("saved_at_utc")
                or _snapshot_from_filename(path)
            ),
        )
        if not prematch.empty:
            prematch["source_file"] = path.name
            prematch_frames.append(prematch)

    canonical_summary_paths, noncanonical_summary_paths = _summary_path_inventory(
        raw_root / "odds_summary"
    )
    summary_paths = canonical_summary_paths
    if include_noncanonical_summary:
        summary_paths = sorted(canonical_summary_paths + noncanonical_summary_paths)

    for path in summary_paths:
        payload, metadata = _load_raw_payload(path)
        event_id = _summary_event_context(path, metadata)
        if not event_id:
            continue
        summary = normalize_mma_odds_summary_rows(
            payload,
            event_id=event_id,
            snapshot_time_utc=(
                metadata.get("snapshot_time_utc")
                or metadata.get("saved_at_utc")
                or _snapshot_from_filename(path)
            ),
        )
        if not summary.empty:
            summary["source_file"] = path.name
            summary_frames.append(summary)

    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    bet365_df = pd.concat(bet365_frames, ignore_index=True) if bet365_frames else pd.DataFrame()
    prematch_df = pd.concat(prematch_frames, ignore_index=True) if prematch_frames else pd.DataFrame()
    summary_rows_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_rows_df = _sanitize_summary_rows(summary_rows_df)

    combined_events = (
        pd.concat([events_df, bet365_df], ignore_index=True)
        if not events_df.empty or not bet365_df.empty
        else pd.DataFrame()
    )
    event_snapshots_df = build_mma_event_snapshots(
        events_df[events_df["source"] == "events_ended"].copy() if not events_df.empty else pd.DataFrame(),
        summary_rows_df,
    )

    return {
        "events": combined_events,
        "events_upcoming": events_df[events_df["source"] == "events_upcoming"].copy() if not events_df.empty else pd.DataFrame(),
        "events_ended": events_df[events_df["source"] == "events_ended"].copy() if not events_df.empty else pd.DataFrame(),
        "bet365_upcoming": bet365_df,
        "prematch_markets": prematch_df,
        "summary_rows": summary_rows_df,
        "event_snapshots": event_snapshots_df,
    }


def summarize_saved_betsapi_mma_backfill(
    *,
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    processed_root: Path = BETSAPI_MMA_PROCESSED_DIR,
) -> dict[str, Any]:
    """Summarize raw, processed, and quality signals for the local MMA backfill."""
    requested_event_ids = _requested_event_ids_from_raw(raw_root, ufc_only=True)

    summary_dir = raw_root / "odds_summary"
    canonical_summary_paths, noncanonical_summary_paths = _summary_path_inventory(summary_dir)
    all_summary_paths = canonical_summary_paths + noncanonical_summary_paths
    summary_ids = {
        path.stem.replace("event_", "")
        for path in canonical_summary_paths
    }
    missing_summary_ids = [
        event_id for event_id in requested_event_ids if event_id not in summary_ids
    ]

    manifest_path = processed_root / "backfill_manifest.json"
    def _artifact_ready(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    processed_artifacts = {
        "backfill_manifest.json": _artifact_ready(manifest_path),
        "ended_events.csv": _artifact_ready(processed_root / "ended_events.csv"),
        "odds_summary_rows.csv": _artifact_ready(processed_root / "odds_summary_rows.csv"),
        "event_snapshots.csv": _artifact_ready(processed_root / "event_snapshots.csv"),
    }
    processed_ready = all(processed_artifacts.values())

    manifest = _load_existing_manifest(processed_root)

    processed_event_rows = 0
    processed_summary_rows = 0
    processed_snapshot_rows = 0
    processed_events_with_summaries = 0
    usable_summary_event_ids = 0
    usable_event_snapshot_ids = 0
    invalid_moneyline_rows = 0
    invalid_moneyline_event_ids = 0
    invalid_moneyline_bookmakers: dict[str, int] = {}
    manifest_count_mismatches: dict[str, dict[str, int]] = {}
    manifest_is_fresh: Optional[bool] = None
    quality_errors: list[str] = []

    if processed_ready:
        try:
            ended_events = pd.read_csv(processed_root / "ended_events.csv")
            processed_event_rows = int(len(ended_events))
        except Exception as exc:  # pragma: no cover - defensive guard for local artifact corruption
            quality_errors.append(f"failed to inspect ended_events.csv: {exc}")

        try:
            summary_rows = pd.read_csv(processed_root / "odds_summary_rows.csv")
            processed_summary_rows = int(len(summary_rows))
            usable_summary_event_ids = int(
                summary_rows["event_id"].astype(str).nunique()
            ) if "event_id" in summary_rows.columns else 0
            processed_events_with_summaries = usable_summary_event_ids
            straight_moneyline = _straight_moneyline_summary_rows(summary_rows)
            invalid_moneyline = straight_moneyline[
                (~straight_moneyline["home_odds"].map(_is_valid_decimal_odds))
                | (~straight_moneyline["away_odds"].map(_is_valid_decimal_odds))
            ].copy()
            invalid_moneyline_rows = int(len(invalid_moneyline))
            invalid_moneyline_event_ids = int(
                invalid_moneyline["event_id"].astype(str).nunique()
            ) if not invalid_moneyline.empty else 0
            invalid_moneyline_bookmakers = (
                invalid_moneyline["bookmaker"].astype(str).value_counts().to_dict()
                if not invalid_moneyline.empty
                else {}
            )
        except Exception as exc:  # pragma: no cover - defensive guard for local artifact corruption
            quality_errors.append(f"failed to inspect odds_summary_rows.csv: {exc}")

        try:
            event_snapshots = pd.read_csv(processed_root / "event_snapshots.csv")
            processed_snapshot_rows = int(len(event_snapshots))
            usable_event_snapshot_ids = int(
                event_snapshots["event_id"].astype(str).nunique()
            ) if "event_id" in event_snapshots.columns else 0
        except Exception as exc:  # pragma: no cover - defensive guard for local artifact corruption
            quality_errors.append(f"failed to inspect event_snapshots.csv: {exc}")

        if manifest is not None:
            expected_counts = {
                "event_rows": processed_event_rows,
                "summary_rows": processed_summary_rows,
                "event_snapshots": processed_snapshot_rows,
                "events_with_summaries": processed_events_with_summaries,
                "raw_event_payload_files": len(list((raw_root / "events_ended").glob("*.json"))),
                "raw_summary_payload_files": len(canonical_summary_paths),
                "raw_summary_payload_files_total": len(all_summary_paths),
                "raw_summary_payload_files_noncanonical": len(noncanonical_summary_paths),
                "requested_event_rows": len(
                    _requested_event_ids_from_raw(
                        raw_root,
                        ufc_only=bool(manifest.get("ufc_only")),
                    )
                ),
                "requested_ufc_event_ids": len(requested_event_ids),
                "hydrated_summaries": len(canonical_summary_paths),
            }
            for key, actual in expected_counts.items():
                manifest_value = manifest.get(key)
                if manifest_value is None:
                    continue
                numeric_value = pd.to_numeric(manifest_value, errors="coerce")
                if pd.isna(numeric_value):
                    continue
                if int(numeric_value) != int(actual):
                    manifest_count_mismatches[key] = {
                        "manifest": int(numeric_value),
                        "processed": int(actual),
                    }

        if manifest_path.exists():
            latest_processed_mtime = max(
                (processed_root / filename).stat().st_mtime
                for filename in ("ended_events.csv", "odds_summary_rows.csv", "event_snapshots.csv")
                if (processed_root / filename).exists()
            )
            manifest_is_fresh = manifest_path.stat().st_mtime >= latest_processed_mtime
            if manifest_is_fresh is False:
                quality_errors.append(
                    "backfill_manifest.json is older than the processed CSV artifacts"
                )

    if invalid_moneyline_rows > 0:
        quality_errors.append(
            f"processed odds_summary_rows.csv still contains {invalid_moneyline_rows} non-decimal moneyline rows"
        )
    if manifest_count_mismatches:
        quality_errors.append(
            "backfill_manifest.json count fields do not match the processed MMA artifacts"
        )

    raw_ready = (
        (raw_root / "events_ended").exists()
        and any((raw_root / "events_ended").rglob("*.json"))
        and summary_dir.exists()
        and bool(canonical_summary_paths)
    )
    manifest_ready = bool(
        manifest
        and manifest.get("event_rows", 0) > 0
        and manifest.get("summary_rows", 0) > 0
        and manifest.get("event_snapshots", 0) > 0
        and len(missing_summary_ids) == 0
    )
    usable_ready = usable_summary_event_ids > 0 and usable_event_snapshot_ids > 0
    ready = bool(
        raw_ready
        and processed_ready
        and manifest_ready
        and usable_ready
        and invalid_moneyline_rows == 0
        and not quality_errors
    )

    return {
        "ready": ready,
        "raw_event_payload_files": len(list((raw_root / "events_ended").glob("*.json"))),
        "raw_summary_payload_files": len(canonical_summary_paths),
        "raw_summary_payload_files_total": len(all_summary_paths),
        "raw_summary_payload_files_noncanonical": len(all_summary_paths) - len(canonical_summary_paths),
        "requested_ufc_event_ids": len(requested_event_ids),
        "hydrated_summary_ids": len(summary_ids),
        "missing_summary_ids": len(missing_summary_ids),
        "missing_summary_sample": missing_summary_ids[:25],
        "usable_summary_event_ids": usable_summary_event_ids,
        "usable_event_snapshot_ids": usable_event_snapshot_ids,
        "invalid_moneyline_rows": invalid_moneyline_rows,
        "invalid_moneyline_event_ids": invalid_moneyline_event_ids,
        "invalid_moneyline_bookmakers": invalid_moneyline_bookmakers,
        "processed_row_counts": {
            "event_rows": processed_event_rows,
            "summary_rows": processed_summary_rows,
            "event_snapshots": processed_snapshot_rows,
            "events_with_summaries": processed_events_with_summaries,
        },
        "manifest_matches_processed_counts": not manifest_count_mismatches,
        "manifest_count_mismatches": manifest_count_mismatches,
        "manifest_is_fresh": manifest_is_fresh,
        "processed_artifacts": processed_artifacts,
        "manifest": manifest,
        "quality_errors": quality_errors,
    }


def inventory_saved_prematch_markets(
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    *,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Count saved prematch market frequency by market and header."""
    frames = load_saved_betsapi_mma_frames(raw_root=raw_root)
    prematch_df = frames["prematch_markets"]
    if prematch_df.empty:
        return pd.DataFrame(
            columns=["market_group", "market_name", "selection_header", "rows", "files", "priced_rows"]
        )

    summary = (
        prematch_df.groupby(["market_group", "market_name", "selection_header"], dropna=False)
        .agg(
            rows=("selection_id", "size"),
            files=("source_file", "nunique"),
            priced_rows=("decimal_odds", lambda col: int(col.notna().sum())),
        )
        .reset_index()
        .sort_values(["files", "rows", "market_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    if save_csv:
        summary.to_csv(MARKET_INVENTORY_DIR / "prematch_market_inventory.csv", index=False)

    return summary


def join_betsapi_events_to_fights(
    fights_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    event_snapshots_df: Optional[pd.DataFrame] = None,
    save_artifacts: bool = False,
) -> pd.DataFrame:
    """Join normalized BetsAPI events to UFC fight rows via fighter/date keys."""
    if fights_df.empty:
        return pd.DataFrame()

    fights = fights_df.copy()
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    fights["fight_key"] = fights.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
        axis=1,
    )

    if events_df.empty:
        empty = fights.assign(
            betsapi_match_found=False,
            betsapi_event_id=pd.NA,
            betsapi_fi=pd.NA,
            betsapi_league_name=pd.NA,
        )
        return empty

    events = events_df.copy()
    events["scheduled_start_utc"] = pd.to_datetime(events["scheduled_start_utc"], errors="coerce", utc=True)
    events["fight_key"] = events.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("scheduled_start_utc")),
        axis=1,
    )
    event_latest = (
        events.sort_values("snapshot_time_utc")
        .groupby("fight_key", dropna=False)
        .tail(1)
        [["fight_key", "event_id", "fi", "league_name", "scheduled_start_utc"]]
        .rename(
            columns={
                "event_id": "betsapi_event_id",
                "fi": "betsapi_fi",
                "league_name": "betsapi_league_name",
                "scheduled_start_utc": "betsapi_scheduled_start_utc",
            }
        )
    )

    merge_signal_cols: list[str] = []
    if event_snapshots_df is not None and not event_snapshots_df.empty:
        snapshot_cols = [
            "opening_home_implied_prob_avg",
            "opening_away_implied_prob_avg",
            "opening_bookmakers",
            "end_home_implied_prob_avg",
            "end_away_implied_prob_avg",
            "end_bookmakers",
        ]
        usable_snapshot_cols = [column for column in snapshot_cols if column in event_snapshots_df.columns]
        if usable_snapshot_cols:
            signal_lookup = event_snapshots_df[["event_id"] + usable_snapshot_cols].copy()
            signal_lookup["event_id"] = signal_lookup["event_id"].astype(str)
            signal_lookup = signal_lookup.drop_duplicates(subset=["event_id"])
            rename_map = {column: f"_merge_signal_{column}" for column in usable_snapshot_cols}
            signal_lookup = signal_lookup.rename(columns=rename_map)
            event_latest["betsapi_event_id"] = event_latest["betsapi_event_id"].astype(str)
            event_latest = event_latest.merge(
                signal_lookup,
                left_on="betsapi_event_id",
                right_on="event_id",
                how="left",
            ).drop(columns=["event_id"])
            merge_signal_cols = list(rename_map.values())

    joined = _merge_on_fight_key_with_previous_day_fallback(
        fights,
        event_latest,
        signal_cols=merge_signal_cols,
    )
    if merge_signal_cols:
        joined = joined.drop(columns=merge_signal_cols, errors="ignore")
    joined["betsapi_match_found"] = joined["betsapi_event_id"].notna()

    if save_artifacts:
        joined[joined["betsapi_match_found"]].to_csv(JOIN_AUDIT_DIR / "matched_fights.csv", index=False)
        joined[~joined["betsapi_match_found"]].to_csv(JOIN_AUDIT_DIR / "unmatched_fights.csv", index=False)

    return joined


def _selection_side(selection_name: object, header: object, fighter_a: object, fighter_b: object) -> str:
    selection_norm = normalize_mma_text(selection_name)
    header_norm = normalize_mma_text(header)
    fighter_a_norm = normalize_mma_text(fighter_a)
    fighter_b_norm = normalize_mma_text(fighter_b)

    if selection_norm in {"1", "home"}:
        return "a"
    if selection_norm in {"2", "away"}:
        return "b"
    if fighter_a_norm and fighter_a_norm == selection_norm:
        return "a"
    if fighter_b_norm and fighter_b_norm == selection_norm:
        return "b"
    if header_norm in {"to win", "winner"} and selection_norm.endswith(fighter_a_norm):
        return "a"
    if header_norm in {"to win", "winner"} and selection_norm.endswith(fighter_b_norm):
        return "b"
    return ""


def _moneyline_snapshot_features(events_df: pd.DataFrame, market_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Bet365 moneyline snapshots into challenger-ready fight features."""
    if events_df.empty or market_rows.empty:
        return pd.DataFrame(
            columns=[
                "fight_key",
                "betsapi_snapshot_count",
                "betsapi_a_implied_prob",
                "betsapi_b_implied_prob",
                "betsapi_diff_implied_prob",
                "betsapi_odds_staleness_hours",
                "betsapi_line_move_a",
                "betsapi_line_move_b",
                "betsapi_line_move_abs",
            ]
        )

    event_lookup = (
        events_df.sort_values("snapshot_time_utc")
        .groupby("fi", dropna=False)
        .tail(1)
        [["fi", "fight_key", "fighter_a", "fighter_b", "odds_updated_at_utc", "snapshot_time_utc"]]
        .copy()
    )
    event_lookup["fi"] = event_lookup["fi"].astype(str)

    market_df = market_rows.copy()
    market_df["fi"] = market_df["fi"].astype(str)
    market_df = market_df.merge(event_lookup, on="fi", how="inner", suffixes=("", "_event"))
    market_df = market_df[
        market_df["selection_header"].map(normalize_mma_text).isin({"to win", "winner"})
        | market_df["market_name"].str.contains("fight_lines", case=False, na=False)
    ].copy()
    if market_df.empty:
        return pd.DataFrame()

    market_df["side"] = market_df.apply(
        lambda row: _selection_side(
            row.get("selection_name"),
            row.get("selection_header"),
            row.get("fighter_a"),
            row.get("fighter_b"),
        ),
        axis=1,
    )
    market_df = market_df[market_df["side"].isin(["a", "b"]) & market_df["decimal_odds"].notna()].copy()
    if market_df.empty:
        return pd.DataFrame()

    market_df["implied_prob"] = 1.0 / market_df["decimal_odds"]

    snapshots = (
        market_df.pivot_table(
            index=["fight_key", "snapshot_time_utc", "odds_updated_at_utc"],
            columns="side",
            values="implied_prob",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"a": "a_implied_prob", "b": "b_implied_prob"})
        .sort_values(["fight_key", "snapshot_time_utc"])
    )
    if snapshots.empty:
        return pd.DataFrame()

    grouped_rows: list[dict[str, object]] = []
    for fight_key, group in snapshots.groupby("fight_key", dropna=False):
        latest = group.tail(1).iloc[0]
        first = group.head(1).iloc[0]
        a_last = latest.get("a_implied_prob")
        b_last = latest.get("b_implied_prob")
        a_first = first.get("a_implied_prob")
        b_first = first.get("b_implied_prob")
        staleness_hours = np.nan
        if pd.notna(latest.get("odds_updated_at_utc")) and pd.notna(latest.get("snapshot_time_utc")):
            delta = latest["snapshot_time_utc"] - latest["odds_updated_at_utc"]
            staleness_hours = delta.total_seconds() / 3600.0

        grouped_rows.append(
            {
                "fight_key": fight_key,
                "betsapi_snapshot_count": int(len(group)),
                "betsapi_a_implied_prob": a_last,
                "betsapi_b_implied_prob": b_last,
                "betsapi_diff_implied_prob": (
                    a_last - b_last if pd.notna(a_last) and pd.notna(b_last) else np.nan
                ),
                "betsapi_odds_staleness_hours": staleness_hours,
                "betsapi_line_move_a": (
                    a_last - a_first if pd.notna(a_last) and pd.notna(a_first) else np.nan
                ),
                "betsapi_line_move_b": (
                    b_last - b_first if pd.notna(b_last) and pd.notna(b_first) else np.nan
                ),
                "betsapi_line_move_abs": (
                    abs(a_last - a_first) + abs(b_last - b_first)
                    if pd.notna(a_last) and pd.notna(a_first) and pd.notna(b_last) and pd.notna(b_first)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(grouped_rows)


def augment_features_with_betsapi_mma(
    features_df: pd.DataFrame,
    *,
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    save_artifacts: bool = True,
    include_live_snapshot_features: bool = True,
) -> pd.DataFrame:
    """Augment UFC features with saved historical odds-summary and snapshot features."""
    augmented = features_df.copy()
    stale_betsapi_cols = [column for column in augmented.columns if column.startswith("betsapi_")]
    if stale_betsapi_cols:
        augmented = augmented.drop(columns=stale_betsapi_cols)
    augmented["event_date"] = pd.to_datetime(augmented["event_date"], errors="coerce")
    augmented["fight_key"] = augmented.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
        axis=1,
    )

    frames = load_saved_betsapi_mma_frames(raw_root=raw_root)
    if include_live_snapshot_features:
        bet365_events = frames["bet365_upcoming"]
        moneyline_features = _moneyline_snapshot_features(bet365_events, frames["prematch_markets"])
    else:
        moneyline_features = pd.DataFrame()
    historical_features = _historical_summary_features(frames["event_snapshots"])
    historical_market_features = _historical_summary_market_features(
        frames["event_snapshots"],
        frames["summary_rows"],
    )
    historical_expanded_features = _historical_expanded_features(
        frames["event_snapshots"],
        frames["summary_rows"],
    )

    if not historical_features.empty:
        augmented = _merge_on_fight_key_with_previous_day_fallback(
            augmented,
            historical_features,
            signal_cols=[
                "betsapi_hist_opening_a_implied_prob",
                "betsapi_hist_opening_b_implied_prob",
                "betsapi_hist_opening_bookmakers",
            ],
        )
    else:
        for column in [
            "betsapi_hist_opening_a_implied_prob",
            "betsapi_hist_opening_b_implied_prob",
            "betsapi_hist_diff_opening_implied_prob",
            "betsapi_hist_opening_bookmakers",
            "betsapi_hist_opening_lead_hours",
            ]:
            if column not in augmented.columns:
                augmented[column] = np.nan

    if not historical_market_features.empty:
        augmented = _merge_on_fight_key_with_previous_day_fallback(
            augmented,
            historical_market_features,
            signal_cols=[
                column for column in historical_market_features.columns
                if column != "fight_key"
            ],
        )

    if not historical_expanded_features.empty:
        augmented = _merge_on_fight_key_with_previous_day_fallback(
            augmented,
            historical_expanded_features,
            signal_cols=[
                column for column in historical_expanded_features.columns
                if column != "fight_key"
            ],
        )

    if not moneyline_features.empty:
        augmented = _merge_on_fight_key_with_previous_day_fallback(
            augmented,
            moneyline_features,
            signal_cols=[
                column for column in moneyline_features.columns
                if column != "fight_key"
            ],
        )
    else:
        for column in [
            "betsapi_snapshot_count",
            "betsapi_a_implied_prob",
            "betsapi_b_implied_prob",
            "betsapi_diff_implied_prob",
            "betsapi_odds_staleness_hours",
            "betsapi_line_move_a",
            "betsapi_line_move_b",
            "betsapi_line_move_abs",
        ]:
            if column not in augmented.columns:
                augmented[column] = np.nan

    if (
        "a_implied_prob" in augmented.columns
        and "betsapi_hist_opening_a_implied_prob" in augmented.columns
    ):
        augmented["betsapi_hist_market_disagreement_a"] = (
            augmented["betsapi_hist_opening_a_implied_prob"] - augmented["a_implied_prob"]
        )
    if (
        "b_implied_prob" in augmented.columns
        and "betsapi_hist_opening_b_implied_prob" in augmented.columns
    ):
        augmented["betsapi_hist_market_disagreement_b"] = (
            augmented["betsapi_hist_opening_b_implied_prob"] - augmented["b_implied_prob"]
        )
    if {
        "betsapi_hist_market_disagreement_a",
        "betsapi_hist_market_disagreement_b",
    }.issubset(augmented.columns):
        augmented["betsapi_hist_market_disagreement_abs"] = (
            augmented["betsapi_hist_market_disagreement_a"].abs()
            + augmented["betsapi_hist_market_disagreement_b"].abs()
        )

    if "a_implied_prob" in augmented.columns and "betsapi_a_implied_prob" in augmented.columns:
        augmented["betsapi_market_disagreement_a"] = (
            augmented["betsapi_a_implied_prob"] - augmented["a_implied_prob"]
        )
    if "b_implied_prob" in augmented.columns and "betsapi_b_implied_prob" in augmented.columns:
        augmented["betsapi_market_disagreement_b"] = (
            augmented["betsapi_b_implied_prob"] - augmented["b_implied_prob"]
        )
    if {
        "betsapi_market_disagreement_a",
        "betsapi_market_disagreement_b",
    }.issubset(augmented.columns):
        augmented["betsapi_market_disagreement_abs"] = (
            augmented["betsapi_market_disagreement_a"].abs()
            + augmented["betsapi_market_disagreement_b"].abs()
        )

    joined = join_betsapi_events_to_fights(
        augmented,
        frames["events"],
        event_snapshots_df=frames["event_snapshots"],
        save_artifacts=save_artifacts,
    )
    if len(joined) != len(augmented):
        raise RuntimeError(
            f"join_betsapi_events_to_fights changed row count: "
            f"expected {len(augmented)}, got {len(joined)}"
        )
    for column in [
        "betsapi_match_found",
        "betsapi_event_id",
        "betsapi_fi",
        "betsapi_league_name",
        "betsapi_scheduled_start_utc",
    ]:
        if column in joined.columns:
            augmented[column] = joined[column].to_numpy()

    numeric_exclusions = {
        "betsapi_match_found",
        "betsapi_event_id",
        "betsapi_fi",
        "betsapi_league_name",
        "betsapi_scheduled_start_utc",
    }
    for column in [col for col in augmented.columns if col.startswith("betsapi_") and col not in numeric_exclusions]:
        augmented[column] = pd.to_numeric(augmented[column], errors="coerce")

    if include_live_snapshot_features:
        matched_snapshot_count = int(augmented["betsapi_snapshot_count"].notna().sum())
        if matched_snapshot_count == 0:
            logger.warning(
                "No saved BetsAPI MMA moneyline snapshots matched the feature matrix; "
                "expanded snapshot-only BetsAPI columns will remain null."
            )
        else:
            matched_dates = pd.to_datetime(
                augmented.loc[augmented["betsapi_snapshot_count"].notna(), "event_date"],
                errors="coerce",
            )
            first_matched_date = matched_dates.min()
            earliest_event_date = pd.to_datetime(augmented["event_date"], errors="coerce").min()
            if pd.notna(first_matched_date) and pd.notna(earliest_event_date) and first_matched_date > earliest_event_date:
                logger.warning(
                    "Saved BetsAPI MMA moneyline snapshots first match fights on %s; "
                    "earlier fights will keep snapshot-only BetsAPI columns null.",
                    first_matched_date.date().isoformat(),
                )

    historical_match_count = int(augmented["betsapi_hist_opening_a_implied_prob"].notna().sum())
    if historical_match_count == 0 and not frames["summary_rows"].empty:
        logger.warning(
            "Saved BetsAPI MMA odds_summary rows loaded, but no historical opening odds matched the UFC feature matrix."
        )

    if save_artifacts:
        _write_processed_betsapi_mma_artifacts(
            raw_root=raw_root,
            processed_root=BETSAPI_MMA_PROCESSED_DIR,
            frames=frames,
        )
        augmented.to_csv(BETSAPI_MMA_PROCESSED_DIR / "features_betsapi_augmented.csv", index=False)

    return augmented
