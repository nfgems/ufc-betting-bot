"""Durable, display-only storage for historical UFC model predictions.

The live prediction cache is deliberately short lived because it can feed trade
execution.  This module keeps a separate archive whose rows are never consumed
by the trading path.  The archive accepts older display formats so a sparse
winner pick can survive even when the original breakdown is no longer usable.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from src.data.name_utils import normalize_cross_source_name, normalize_person_name


PREDICTION_HISTORY_FILENAME = "predictions_history.json"
PREDICTION_HISTORY_SCHEMA_VERSION = 1
PREDICTION_HISTORY_MAX_ROWS = 10_000
PREDICTION_HISTORY_CARD_HINT_LOOKAHEAD_DAYS = 180

_HISTORY_LOCK = threading.RLock()
_INITIALIZED_HISTORY_PATHS: set[str] = set()
_STRUCTURED_CARD_HINTS_BY_HISTORY_PATH: dict[
    str, dict[tuple[str, str], list[dict]]
] = {}
_INTERNAL_LIVE_FIELDS = frozenset(
    {
        "cache_key",
        "event_context_snapshot",
        "feature_provenance",
        "method_odds_fingerprint",
        "model_features",
        "odds_snapshot",
        # Legacy schema names retained so older cache rows are still stripped.
        "operator_features",
        "operator_provenance",
        "pair_key",
        "prediction_input_line_features",
        "prediction_input_odds_snapshot",
        "runtime_signature",
    }
)
_COMPLEMENTARY_RECOVERY_FIELDS = (
    "recovered_research_summary",
    "recovered_rationale",
)
_FIGHTER_A_KEYS = (
    "fighter_a",
    "fighterA",
    "fighter_1",
    "fighter1",
    "red_corner",
    "red_fighter",
    "a_fighter",
)
_FIGHTER_B_KEYS = (
    "fighter_b",
    "fighterB",
    "fighter_2",
    "fighter2",
    "blue_corner",
    "blue_fighter",
    "b_fighter",
)
_EXPLICIT_PICK_KEYS = (
    "predicted_winner",
    "model_pick",
    "predicted_pick",
    "pick",
    "prediction",
    "bet_on",
)
_PROBABILITY_KEY_PAIRS = (
    ("prob_a", "prob_b"),
    ("model_prob_a", "model_prob_b"),
    ("a_win_probability", "b_win_probability"),
    ("fighter_a_probability", "fighter_b_probability"),
    ("red_probability", "blue_probability"),
)
_LOG_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ][0-9:.,+-]+)\s+\["
)
_LOG_MATCHUP_RE = re.compile(r"^\s{2}(?P<a>.+?)\s+vs\s+(?P<b>.+?):\s*$")
_LOG_PROBABILITY_RE = re.compile(
    r"^\s*(?P<label>Bookmakers|Model|No-odds):\s*"
    r"(?P<a>.+?)\s+(?P<pa>\d+(?:\.\d+)?)%\s*\|\s*"
    r"(?P<b>.+?)\s+(?P<pb>\d+(?:\.\d+)?)%\s*$"
)
_LOG_CARD_CONTEXT_RE = re.compile(
    r"No official card-row context for (?P<a>.+?)\s+vs\s+(?P<b>.+?) "
    r"on (?P<card_date>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_LOG_FIGHTER_REFERENCE_DATE_RE = re.compile(
    r"Processed live snapshot for (?P<fighter>.+?) may be stale relative to "
    r"(?P<card_date>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp(value: datetime | None = None) -> str:
    current = value or _utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.isoformat()


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _first_text(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _fighter_names(row: dict) -> tuple[str, str]:
    fighter_a = _first_text(row, _FIGHTER_A_KEYS)
    fighter_b = _first_text(row, _FIGHTER_B_KEYS)
    if fighter_a and fighter_b:
        return fighter_a, fighter_b

    fighters = row.get("fighters")
    if isinstance(fighters, (list, tuple)) and len(fighters) >= 2:
        first = str(fighters[0] or "").strip()
        second = str(fighters[1] or "").strip()
        if first and second:
            return first, second

    matchup = str(row.get("matchup") or row.get("fight") or "").strip()
    if matchup:
        parts = re.split(r"\s+vs\.?\s+", matchup, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and all(part.strip() for part in parts):
            return parts[0].strip(), parts[1].strip().rstrip(":")
    return fighter_a, fighter_b


def _normalized_name(value: object) -> str:
    canonical = normalize_cross_source_name(value)
    return "".join(character for character in canonical if character.isalnum())


def _coerce_probability(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed /= 100.0
    return parsed if 0.0 <= parsed <= 1.0 else None


def _winner_from_side(value: object, fighter_a: str, fighter_b: str) -> str | None:
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"a", "fighter_a", "fighter1", "fighter_1", "red", "red_corner"}:
        return fighter_a or None
    if normalized in {"b", "fighter_b", "fighter2", "fighter_2", "blue", "blue_corner"}:
        return fighter_b or None
    return None


def resolve_predicted_winner(row: dict) -> str | None:
    """Resolve a stored pick without inventing probabilities for sparse rows."""
    if not isinstance(row, dict):
        return None
    fighter_a, fighter_b = _fighter_names(row)
    if not fighter_a or not fighter_b:
        return None

    for key in _EXPLICIT_PICK_KEYS:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        side_winner = _winner_from_side(value, fighter_a, fighter_b)
        if side_winner:
            return side_winner
        return value.strip()

    for key in ("predicted_side", "pick_side", "model_side"):
        side_winner = _winner_from_side(row.get(key), fighter_a, fighter_b)
        if side_winner:
            return side_winner

    for key_a, key_b in _PROBABILITY_KEY_PAIRS:
        prob_a = _coerce_probability(row.get(key_a))
        prob_b = _coerce_probability(row.get(key_b))
        if prob_a is not None and prob_b is not None:
            return fighter_a if prob_a >= prob_b else fighter_b
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _timestamp_value_has_time(value: object) -> bool:
    raw = str(value or "").strip()
    return bool(re.search(r"(?:T|\s)\d{1,2}:\d{2}", raw))


def _prediction_is_after_event(generated_value, event_value) -> bool:
    generated_at = _parse_timestamp(generated_value)
    event_at = _parse_timestamp(event_value)
    if generated_at is None or event_at is None:
        return False
    if not _timestamp_value_has_time(event_value):
        event_day = _calendar_token(event_value, convert_timestamp=False)
        try:
            return (
                generated_at.astimezone(_event_timezone()).date()
                > date.fromisoformat(event_day)
            )
        except ValueError:
            return False
    return generated_at > event_at


def _event_timezone():
    name = str(os.getenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York") or "").strip()
    try:
        return ZoneInfo(name) if name else timezone.utc
    except Exception:
        return timezone.utc


def _calendar_token(value: object, *, convert_timestamp: bool = True) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    raw = str(value or "").strip()
    if not raw:
        return ""
    exact = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
    if exact:
        return exact.group(0)
    for date_format in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, date_format).date().isoformat()
        except ValueError:
            pass
    parsed = _parse_timestamp(value)
    if parsed is not None:
        if convert_timestamp:
            parsed = parsed.astimezone(_event_timezone())
        return parsed.date().isoformat()
    return re.sub(r"\s+", " ", raw.casefold())


_MONTH_NUMBER_BY_NAME = {
    month: index
    for index, month in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


def _snapshot_card_date(payload: dict) -> str:
    """Prefer the official UFC event URL date over scraper-local date labels."""
    for key in ("event_url", "event_key"):
        raw = str(payload.get(key) or "").strip().casefold()
        match = re.search(
            r"(?:^|[-/])(" + "|".join(_MONTH_NUMBER_BY_NAME) + r")-(\d{1,2})-(\d{4})(?:$|[/?#])",
            raw,
        )
        if match:
            try:
                return date(
                    int(match.group(3)),
                    _MONTH_NUMBER_BY_NAME[match.group(1)],
                    int(match.group(2)),
                ).isoformat()
            except ValueError:
                pass
    return _calendar_token(
        payload.get("card_date") or payload.get("event_date"),
        convert_timestamp=False,
    )


def prediction_archive_key(row: dict, fallback_timestamp=None) -> str | None:
    """Return a stable card-date + unordered-matchup identity."""
    if not isinstance(row, dict):
        return None
    fighter_a, fighter_b = _fighter_names(row)
    pair = sorted((_normalized_name(fighter_a), _normalized_name(fighter_b)))
    if not pair[0] or not pair[1]:
        return None

    event_token = ""
    for key in (
        "card_date",
        "event_group_date",
        "event_date",
        "commence_time",
        "market_event_date",
    ):
        event_token = _calendar_token(row.get(key))
        if event_token:
            break
    if not event_token:
        recovered_group = _calendar_token(
            row.get("recovered_group_date"), convert_timestamp=False
        )
        fallback = (
            row.get("prediction_generated_at")
            or row.get("generated_at")
            or fallback_timestamp
        )
        fallback_token = recovered_group or _calendar_token(fallback)
        event_token = f"unknown:{fallback_token or 'undated'}"
    return f"{event_token}::{pair[0]}::{pair[1]}"


def _detail_level(row: dict) -> str:
    if any(
        isinstance(row.get(key), (dict, list)) and bool(row.get(key))
        for key in ("method_stats", "fighter_context", "feature_highlights", "shap_values")
    ):
        return "full"
    if any(
        _coerce_probability(row.get(key)) is not None
        for pair in _PROBABILITY_KEY_PAIRS
        for key in pair
    ) or _coerce_probability(row.get("confidence")) is not None:
        return "summary"
    return "pick_only"


def _history_rows_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("predictions", "history", "rows", "picks"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            rows = []
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _prepare_archive_row(
    raw_row: dict,
    *,
    payload_timestamp,
    source_schema_version,
    source: str,
    archived_at: str,
) -> dict | None:
    fighter_a, fighter_b = _fighter_names(raw_row)
    winner = resolve_predicted_winner(raw_row)
    if not fighter_a or not fighter_b or not winner:
        return None

    row = {
        key: copy.deepcopy(value)
        for key, value in raw_row.items()
        if key not in _INTERNAL_LIVE_FIELDS and not key.startswith("archive_")
    }
    row["fighter_a"] = fighter_a
    row["fighter_b"] = fighter_b
    row["predicted_winner"] = winner
    if not row.get("card_date"):
        normalized_event_day = _calendar_token(
            row.get("event_date")
            or row.get("commence_time")
            or row.get("market_event_date")
        )
        if normalized_event_day:
            row["card_date"] = normalized_event_day
    row["detail_level"] = _detail_level(row)
    row["source"] = source
    row["source_schema_version"] = source_schema_version
    row["source_cache_timestamp"] = payload_timestamp
    row["first_archived_at"] = archived_at
    row["last_archived_at"] = archived_at
    row["recovered"] = source.startswith("recovered")
    archive_key = prediction_archive_key(row, fallback_timestamp=payload_timestamp)
    if not archive_key:
        return None
    row["history_key"] = archive_key
    return _json_safe(row)


def _row_timestamp(row: dict) -> float:
    for key in (
        "prediction_generated_at",
        "generated_at",
        "source_cache_timestamp",
        "last_archived_at",
    ):
        parsed = _parse_timestamp(row.get(key))
        if parsed is not None:
            return parsed.timestamp()
    return 0.0


def _detail_score(row: dict) -> tuple[int, int, int]:
    level = {"pick_only": 0, "summary": 1, "full": 2}.get(
        str(row.get("detail_level") or ""),
        0,
    )
    populated = sum(
        value not in (None, "", [], {})
        for key, value in row.items()
        if key not in {"first_archived_at", "last_archived_at"}
    )
    nested = sum(
        len(value)
        for value in row.values()
        if isinstance(value, (list, dict))
    )
    return level, populated, nested


def _candidate_wins(existing: dict, candidate: dict) -> bool:
    existing_timestamp = _row_timestamp(existing)
    candidate_timestamp = _row_timestamp(candidate)
    if candidate_timestamp != existing_timestamp:
        return candidate_timestamp > existing_timestamp
    return _detail_score(candidate) > _detail_score(existing)


def _merge_complementary_recovery_fields(primary: dict, secondary: dict) -> dict:
    merged = dict(primary)
    for field in _COMPLEMENTARY_RECOVERY_FIELDS:
        if merged.get(field) in (None, "", [], {}) and secondary.get(field) not in (
            None,
            "",
            [],
            {},
        ):
            merged[field] = copy.deepcopy(secondary[field])
    return merged


def _history_limit(max_rows: int | None) -> int:
    if max_rows is not None:
        return max(1, int(max_rows))
    try:
        configured = int(
            str(os.getenv("PREDICTION_HISTORY_MAX_ROWS", PREDICTION_HISTORY_MAX_ROWS))
        )
    except (TypeError, ValueError):
        configured = PREDICTION_HISTORY_MAX_ROWS
    return max(100, configured)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_decoded_log_lines(path: Path):
    """Decode mixed historical log rotations without destroying fighter names."""
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                yield raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                yield raw_line.decode("cp1252", errors="replace").rstrip("\r\n")


@contextmanager
def _history_write_lock(path: Path, *, timeout_seconds: float = 30.0):
    """Serialize archive updates across threads and local processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with _HISTORY_LOCK:
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + max(timeout_seconds, 0.0)
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out locking prediction history {path}"
                            )
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _empty_history_payload(status: str) -> dict:
    return {
        "schema_version": PREDICTION_HISTORY_SCHEMA_VERSION,
        "updated_at": None,
        "prediction_count": 0,
        "predictions": [],
        "archive_status": status,
        "error": None,
    }


def load_prediction_history(path: Path | str) -> dict:
    """Load an archive for display, returning an explicit status on failure."""
    history_path = Path(path)
    if not history_path.exists():
        return _empty_history_payload("missing")
    try:
        raw = _read_json(history_path)
    except Exception as exc:
        payload = _empty_history_payload("error")
        payload["error"] = str(exc)
        return payload

    rows = _history_rows_from_payload(raw)
    raw_schema = raw.get("schema_version") if isinstance(raw, dict) else None
    try:
        schema_current = int(raw_schema) == PREDICTION_HISTORY_SCHEMA_VERSION
    except (TypeError, ValueError):
        schema_current = False
    updated_at = raw.get("updated_at") if isinstance(raw, dict) else None
    return {
        "schema_version": raw_schema,
        "updated_at": updated_at,
        "prediction_count": len(rows),
        "predictions": rows,
        "archive_status": "current" if schema_current else "legacy",
        "error": None,
    }


def archive_prediction_payload(
    payload: object,
    history_path: Path | str,
    *,
    source: str = "prediction_cache",
    now: datetime | None = None,
    max_rows: int | None = None,
) -> dict:
    """Merge prediction rows into the display archive using an atomic replace."""
    path = Path(history_path)
    raw_rows = _history_rows_from_payload(payload)
    if not raw_rows:
        return {"added": 0, "updated": 0, "skipped": 0, "total": 0}

    payload_timestamp = payload.get("timestamp") if isinstance(payload, dict) else None
    source_schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    archived_at = _iso_timestamp(now)

    with _history_write_lock(path):
        registered_card_hints = _STRUCTURED_CARD_HINTS_BY_HISTORY_PATH.get(
            str(path.resolve()),
            {},
        )
        if registered_card_hints:
            _apply_snapshot_card_hints(raw_rows, registered_card_hints)

        existing_payload: object = {}
        if path.exists():
            # A malformed archive is not safe to overwrite; leave it available
            # for manual recovery and let the caller log the failure.
            existing_payload = _read_json(path)
        existing_rows = _history_rows_from_payload(existing_payload)
        rows_by_key: dict[str, dict] = {}
        for row in existing_rows:
            key = str(row.get("history_key") or "").strip()
            if not key:
                key = prediction_archive_key(
                    row,
                    fallback_timestamp=row.get("source_cache_timestamp"),
                ) or ""
            if key:
                row["history_key"] = key
                rows_by_key[key] = row

        added = updated = skipped = 0
        for raw_row in raw_rows:
            candidate = _prepare_archive_row(
                raw_row,
                payload_timestamp=payload_timestamp,
                source_schema_version=source_schema_version,
                source=source,
                archived_at=archived_at,
            )
            if candidate is None:
                skipped += 1
                continue
            key = candidate["history_key"]
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = candidate
                added += 1
                continue
            if _candidate_wins(existing, candidate):
                candidate["first_archived_at"] = (
                    existing.get("first_archived_at") or archived_at
                )
                rows_by_key[key] = _merge_complementary_recovery_fields(
                    candidate,
                    existing,
                )
                updated += 1
            else:
                existing = _merge_complementary_recovery_fields(existing, candidate)
                existing["last_archived_at"] = archived_at
                rows_by_key[key] = existing

        rows = sorted(
            rows_by_key.values(),
            key=lambda row: (
                str(row.get("card_date") or row.get("event_date") or ""),
                _row_timestamp(row),
                str(row.get("fighter_a") or ""),
                str(row.get("fighter_b") or ""),
            ),
            reverse=True,
        )[: _history_limit(max_rows)]
        result = {
            "schema_version": PREDICTION_HISTORY_SCHEMA_VERSION,
            "updated_at": archived_at,
            "prediction_count": len(rows),
            "predictions": rows,
        }
        result = _json_safe(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(path)
    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "total": len(rows),
    }


def _merge_rekeyed_archive_rows(primary: dict, secondary: dict) -> dict:
    merged = _merge_complementary_recovery_fields(primary, secondary)
    first_values = [
        value
        for value in (
            primary.get("first_archived_at"),
            secondary.get("first_archived_at"),
        )
        if value
    ]
    last_values = [
        value
        for value in (
            primary.get("last_archived_at"),
            secondary.get("last_archived_at"),
        )
        if value
    ]

    def timestamp_key(value):
        parsed = _parse_timestamp(value)
        return parsed.timestamp() if parsed is not None else 0.0

    if first_values:
        merged["first_archived_at"] = min(first_values, key=timestamp_key)
    if last_values:
        merged["last_archived_at"] = max(last_values, key=timestamp_key)
    return merged


def reconcile_prediction_history_cards(
    history_path: Path | str,
    card_hints: dict[tuple[str, str], list[dict]],
    *,
    now: datetime | None = None,
    max_rows: int | None = None,
) -> dict:
    """Move recoverable undated rows into real cards and collapse stale duplicates."""
    path = Path(history_path)
    if not path.exists():
        return {
            "rekeyed": 0,
            "scheduled": 0,
            "deduplicated": 0,
            "discarded_post_start": 0,
            "total": 0,
        }

    with _history_write_lock(path):
        payload = _read_json(path)
        rows = _history_rows_from_payload(payload)
        if not rows:
            return {
                "rekeyed": 0,
                "scheduled": 0,
                "deduplicated": 0,
                "discarded_post_start": 0,
                "total": 0,
            }

        # A dated sibling from a ledger/live cache can identify an older
        # undated operator/log row even when the raw card snapshot was pruned.
        event_hints = _prediction_event_hints(rows)
        sibling_card_hints = _prediction_card_hints(rows)
        _apply_event_hints_to_recovered_rows(rows, event_hints)
        # Completed-event/snapshot metadata is authoritative. In particular,
        # do not let an already-archived UTC-next-day mistake become its own
        # exact-match hint and preserve itself forever.
        authoritative_matches = _apply_snapshot_card_hints(rows, card_hints)
        _apply_snapshot_card_hints(
            (row for row in rows if id(row) not in authoritative_matches),
            sibling_card_hints,
        )

        rows_by_key: dict[str, dict] = {}
        rekeyed = scheduled = deduplicated = discarded_post_start = 0
        for row in rows:
            old_key = str(row.get("history_key") or "").strip()
            old_was_unscheduled = old_key.startswith("unknown:")
            key = prediction_archive_key(
                row,
                fallback_timestamp=row.get("source_cache_timestamp"),
            )
            if not key:
                continue
            row["history_key"] = key

            fighter_a, fighter_b = _fighter_names(row)
            pair = tuple(
                sorted((_normalized_name(fighter_a), _normalized_name(fighter_b)))
            )
            row_card_date = _calendar_token(
                row.get("card_date"),
                convert_timestamp=False,
            )
            matching_cutoffs = []
            for hint in event_hints.get(pair, []):
                hint_event_date = hint.get("event_date")
                hint_card_date = _calendar_token(
                    hint.get("card_date") or hint_event_date,
                )
                if (
                    hint_event_date
                    and row_card_date
                    and hint_card_date == row_card_date
                ):
                    matching_cutoffs.append(hint_event_date)
            cutoff = (
                max(
                    matching_cutoffs,
                    key=lambda value: (
                        _parse_timestamp(value).timestamp()
                        if _parse_timestamp(value) is not None
                        else 0.0
                    ),
                )
                if matching_cutoffs
                else (
                    row.get("event_date")
                    or row.get("market_event_date")
                    or row.get("card_date")
                )
            )
            if _prediction_is_after_event(
                row.get("prediction_generated_at")
                or row.get("generated_at")
                or row.get("source_cache_timestamp")
                or row.get("recovered_group_date"),
                cutoff,
            ):
                discarded_post_start += 1
                continue
            if old_key and old_key != key:
                rekeyed += 1
            if old_was_unscheduled and not key.startswith("unknown:"):
                scheduled += 1

            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = row
                continue
            deduplicated += 1
            if _candidate_wins(existing, row):
                rows_by_key[key] = _merge_rekeyed_archive_rows(row, existing)
            else:
                rows_by_key[key] = _merge_rekeyed_archive_rows(existing, row)

        reconciled_rows = sorted(
            rows_by_key.values(),
            key=lambda row: (
                str(row.get("card_date") or row.get("event_date") or ""),
                _row_timestamp(row),
                str(row.get("fighter_a") or ""),
                str(row.get("fighter_b") or ""),
            ),
            reverse=True,
        )[: _history_limit(max_rows)]
        updated_at = _iso_timestamp(now)
        result = _json_safe(
            {
                "schema_version": PREDICTION_HISTORY_SCHEMA_VERSION,
                "updated_at": updated_at,
                "prediction_count": len(reconciled_rows),
                "predictions": reconciled_rows,
            }
        )
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(path)

    return {
        "rekeyed": rekeyed,
        "scheduled": scheduled,
        "deduplicated": deduplicated,
        "discarded_post_start": discarded_post_start,
        "total": len(reconciled_rows),
    }


def recover_prediction_rows_from_model_tracker(path: Path | str) -> list[dict]:
    """Recover exact model-side tracker picks from its persistent ledger."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    try:
        payload = _read_json(ledger_path)
    except Exception:
        return []
    bets = payload.get("bets") if isinstance(payload, dict) else None
    if not isinstance(bets, list):
        return []

    recovered: list[dict] = []
    for bet in bets:
        if not isinstance(bet, dict):
            continue
        probability_source = str(bet.get("probability_source") or "model").strip().casefold()
        if probability_source != "model":
            continue
        picked_fighter = str(bet.get("fighter") or bet.get("bet_on") or "").strip()
        opponent = str(bet.get("opponent") or "").strip()
        side = str(bet.get("side") or bet.get("bet_side") or "").strip().casefold()
        model_prob = _coerce_probability(bet.get("model_prob"))
        if not picked_fighter or not opponent or side not in {"a", "b"} or model_prob is None:
            continue
        fighter_a, fighter_b = (
            (picked_fighter, opponent) if side == "a" else (opponent, picked_fighter)
        )
        row = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "predicted_winner": picked_fighter,
            "predicted_side": side,
            "prob_a": model_prob if side == "a" else 1.0 - model_prob,
            "prob_b": model_prob if side == "b" else 1.0 - model_prob,
            "confidence": model_prob,
            "event_date": bet.get("event_date") or bet.get("market_event_date"),
            "market_event_date": bet.get("market_event_date"),
            "card_date": bet.get("card_date"),
            "event_title": bet.get("event_title"),
            "prediction_generated_at": (
                bet.get("placed_at") or bet.get("created_at") or bet.get("timestamp")
            ),
            "recovery_provenance": "model_tracker_ledger",
        }
        market_prob = _coerce_probability(bet.get("market_prob"))
        if market_prob is not None:
            row["a_market_prob"] = market_prob if side == "a" else 1.0 - market_prob
            row["b_market_prob"] = market_prob if side == "b" else 1.0 - market_prob
        recovered.append(_json_safe(row))
    return recovered


def recover_prediction_rows_from_bet_ledgers(
    paths: Iterable[Path | str],
) -> list[dict]:
    """Recover dated model reads retained by the older S/C/B bet ledgers."""
    rows_by_key: dict[str, dict] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        bets = payload.get("bets") if isinstance(payload, dict) else None
        if not isinstance(bets, list):
            continue

        for bet in bets:
            if not isinstance(bet, dict):
                continue
            picked_fighter = str(
                bet.get("fighter") or bet.get("bet_on") or ""
            ).strip()
            opponent = str(bet.get("opponent") or "").strip()
            side = str(
                bet.get("side") or bet.get("bet_side") or ""
            ).strip().casefold()
            model_prob = _coerce_probability(bet.get("model_prob"))
            event_date = bet.get("event_date") or bet.get("market_event_date")
            card_date = bet.get("card_date")
            if (
                not picked_fighter
                or not opponent
                or side not in {"a", "b"}
                or model_prob is None
                or model_prob <= 0.0
                or not (event_date or card_date)
            ):
                continue

            fighter_a, fighter_b = (
                (picked_fighter, opponent)
                if side == "a"
                else (opponent, picked_fighter)
            )
            prob_a = model_prob if side == "a" else 1.0 - model_prob
            prob_b = model_prob if side == "b" else 1.0 - model_prob
            generated_at = (
                bet.get("placed_at")
                or bet.get("created_at")
                or bet.get("timestamp")
            )
            if _prediction_is_after_event(generated_at, event_date):
                continue

            row = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "predicted_winner": fighter_a if prob_a >= prob_b else fighter_b,
                "predicted_side": "a" if prob_a >= prob_b else "b",
                "prob_a": prob_a,
                "prob_b": prob_b,
                "confidence": max(prob_a, prob_b),
                "event_date": event_date,
                "market_event_date": bet.get("market_event_date"),
                "card_date": card_date,
                "event_title": bet.get("event_title"),
                "prediction_generated_at": generated_at,
                "recovery_provenance": f"historical_bet_ledger:{path.name}",
            }
            market_prob = _coerce_probability(bet.get("market_prob"))
            if market_prob is not None:
                row["a_market_prob"] = (
                    market_prob if side == "a" else 1.0 - market_prob
                )
                row["b_market_prob"] = (
                    market_prob if side == "b" else 1.0 - market_prob
                )
            row = _json_safe(row)
            key = prediction_archive_key(row, fallback_timestamp=generated_at)
            if not key:
                continue
            existing = rows_by_key.get(key)
            if existing is None or _row_timestamp(row) > _row_timestamp(existing):
                rows_by_key[key] = row
    return sorted(rows_by_key.values(), key=_row_timestamp, reverse=True)


def _prediction_event_hints(rows: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    def timestamp_score(value: object) -> float:
        parsed = _parse_timestamp(value)
        return parsed.timestamp() if parsed is not None else float("-inf")

    raw_hints: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        event_date = row.get("event_date") or row.get("market_event_date") or row.get("commence_time")
        if not pair[0] or not pair[1] or not event_date:
            continue
        hint = {
            "event_date": event_date,
            "card_date": row.get("card_date"),
            "event_title": row.get("event_title"),
            "observed_at": (
                row.get("prediction_generated_at")
                or row.get("placed_at")
                or row.get("timestamp")
                or row.get("created_at")
            ),
        }
        if hint not in raw_hints.setdefault(pair, []):
            raw_hints[pair].append(hint)

    # Event times occasionally get corrected after the first market snapshot.
    # Treat dates within two weeks as the same scheduled bout and keep the
    # event metadata observed most recently. This prevents an early, incorrect
    # cutoff from discarding valid final pre-start model logs.
    hints: dict[tuple[str, str], list[dict]] = {}
    for pair, pair_hints in raw_hints.items():
        pair_hints.sort(
            key=lambda hint: (
                timestamp_score(hint.get("event_date")),
                timestamp_score(hint.get("observed_at")),
            )
        )
        clusters: list[list[dict]] = []
        for hint in pair_hints:
            event_at = _parse_timestamp(hint.get("event_date"))
            if not clusters:
                clusters.append([hint])
                continue
            previous_event_at = _parse_timestamp(clusters[-1][-1].get("event_date"))
            if (
                event_at is not None
                and previous_event_at is not None
                and abs((event_at - previous_event_at).total_seconds()) <= 14 * 86400
            ):
                clusters[-1].append(hint)
            else:
                clusters.append([hint])
        for cluster in clusters:
            selected = max(
                cluster,
                key=lambda hint: (
                    timestamp_score(hint.get("observed_at")),
                    timestamp_score(hint.get("event_date")),
                ),
            )
            normalized = dict(selected)
            normalized.pop("observed_at", None)
            hints.setdefault(pair, []).append(normalized)
    return hints


def _prediction_card_hints(
    rows: Iterable[dict],
) -> dict[tuple[str, str], list[dict]]:
    hints: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        card_date = _calendar_token(
            row.get("card_date") or row.get("event_group_date"),
            convert_timestamp=False,
        )
        if not pair[0] or not pair[1] or not card_date:
            continue
        hints.setdefault(pair, []).append(
            {
                "card_date": card_date,
                "event_title": row.get("event_title"),
                "observed_at": (
                    row.get("prediction_generated_at")
                    or row.get("source_cache_timestamp")
                    or row.get("last_archived_at")
                ),
            }
        )
    return _merge_card_hint_maps(hints)


def _load_jsonl_event_hint_rows(paths: Iterable[Path | str]) -> list[dict]:
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except OSError:
            continue
    return rows


def _load_operator_cache_hint_rows(paths: Iterable[Path | str]) -> list[dict]:
    """Read date/pair metadata from retired analysis caches, never their picks."""
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for raw_key, raw_entry in payload.items():
            if not isinstance(raw_entry, dict):
                continue
            parts = str(raw_key or "").split("|")
            if len(parts) < 3 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0]):
                continue
            fighter_a = parts[1].strip()
            fighter_b = parts[2].strip()
            if not fighter_a or not fighter_b:
                continue
            response = raw_entry.get("response")
            if not isinstance(response, dict):
                response = {}
            cached_at = raw_entry.get("cached_at")
            if isinstance(cached_at, (int, float)) and math.isfinite(cached_at):
                cached_at = datetime.fromtimestamp(cached_at, tz=timezone.utc).isoformat()
            rows.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "card_date": parts[0],
                    "event_date": (
                        raw_entry.get("event_date")
                        or response.get("event_date")
                    ),
                    "event_title": (
                        raw_entry.get("event_title")
                        or response.get("event_title")
                    ),
                    "prediction_generated_at": cached_at,
                }
            )
    return rows


def _normalized_event_title(value: object) -> str:
    return normalize_person_name(value)


def _load_completed_event_card_hints(
    raw_data_dir: Path | str,
) -> dict[tuple[str, str], list[dict]]:
    """Join completed UFCStats bouts to their authoritative local card dates."""
    directory = Path(raw_data_dir)
    event_dates_path = directory / "ufc-event-dates.csv"
    fight_results_path = directory / "ufc-fight-results.csv"
    if not event_dates_path.is_file() or not fight_results_path.is_file():
        return {}

    event_date_candidates: dict[str, set[str]] = {}
    try:
        with event_dates_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"event_name", "event_date"}.issubset(reader.fieldnames or []):
                return {}
            for row in reader:
                event_title = str(row.get("event_name") or "").strip()
                event_date = _calendar_token(
                    row.get("event_date"),
                    convert_timestamp=False,
                )
                event_key = _normalized_event_title(event_title)
                if event_key and event_date:
                    event_date_candidates.setdefault(event_key, set()).add(event_date)
    except (OSError, UnicodeError, csv.Error):
        return {}
    event_dates = {
        event_key: next(iter(dates))
        for event_key, dates in event_date_candidates.items()
        if len(dates) == 1
    }

    hints: dict[tuple[str, str], dict[str, dict]] = {}
    conflicted_hints: set[tuple[tuple[str, str], str]] = set()
    try:
        with fight_results_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"EVENT", "BOUT"}.issubset(reader.fieldnames or []):
                return {}
            for row in reader:
                event_title = str(row.get("EVENT") or "").strip()
                card_date = event_dates.get(_normalized_event_title(event_title))
                fighter_a, fighter_b = _fighter_names(
                    {"matchup": row.get("BOUT")}
                )
                pair = tuple(
                    sorted(
                        (
                            _normalized_name(fighter_a),
                            _normalized_name(fighter_b),
                        )
                    )
                )
                if not card_date or not pair[0] or not pair[1]:
                    continue
                hint_identity = (pair, card_date)
                if hint_identity in conflicted_hints:
                    continue
                pair_hints = hints.setdefault(pair, {})
                existing = pair_hints.get(card_date)
                if (
                    existing is not None
                    and _normalized_event_title(existing.get("event_title"))
                    != _normalized_event_title(event_title)
                ):
                    pair_hints.pop(card_date, None)
                    conflicted_hints.add(hint_identity)
                    continue
                pair_hints[card_date] = {
                    "card_date": card_date,
                    "event_title": event_title,
                    "authoritative": True,
                    # Stable tie-breaker when another source describes the
                    # same card. No result or winner data enters the archive.
                    "observed_at": card_date,
                }
    except (OSError, UnicodeError, csv.Error):
        return {}

    return {
        pair: sorted(
            pair_hints.values(),
            key=lambda hint: str(hint.get("card_date") or ""),
        )
        for pair, pair_hints in hints.items()
    }


def _load_snapshot_card_hints(directory: Path) -> dict[tuple[str, str], list[dict]]:
    hints_by_card: dict[tuple[str, str], dict[str, dict]] = {}
    if not directory.exists():
        return {}
    for path in directory.glob("*.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("fights"), list):
            continue
        card_date = _snapshot_card_date(payload)
        event_title = str(payload.get("event") or payload.get("event_title") or "").strip()
        observed_at = payload.get("timestamp")
        if not observed_at:
            try:
                observed_at = datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat()
            except OSError:
                observed_at = None
        for fight in payload["fights"]:
            if not isinstance(fight, dict):
                continue
            fighter_a, fighter_b = _fighter_names(fight)
            pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
            if not pair[0] or not pair[1] or not card_date:
                continue
            hint = {
                "card_date": card_date,
                "event_title": event_title,
                "observed_at": observed_at,
            }
            pair_hints = hints_by_card.setdefault(pair, {})
            existing = pair_hints.get(card_date)
            if existing is None or _row_timestamp(
                {"prediction_generated_at": hint.get("observed_at")}
            ) >= _row_timestamp(
                {"prediction_generated_at": existing.get("observed_at")}
            ):
                pair_hints[card_date] = hint
    return {
        pair: sorted(
            pair_hints.values(),
            key=lambda hint: str(hint.get("card_date") or ""),
        )
        for pair, pair_hints in hints_by_card.items()
    }


def _merge_card_hint_maps(
    *hint_maps: dict[tuple[str, str], list[dict]] | None,
) -> dict[tuple[str, str], list[dict]]:
    merged: dict[tuple[str, str], dict[str, dict]] = {}
    for hint_map in hint_maps:
        if not hint_map:
            continue
        for pair, candidates in hint_map.items():
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                card_date = _calendar_token(
                    candidate.get("card_date"),
                    convert_timestamp=False,
                )
                if not card_date:
                    continue
                pair_hints = merged.setdefault(pair, {})
                existing = pair_hints.get(card_date)
                candidate_timestamp = _row_timestamp(
                    {"prediction_generated_at": candidate.get("observed_at")}
                )
                existing_timestamp = _row_timestamp(
                    {
                        "prediction_generated_at": (
                            existing.get("observed_at") if existing else None
                        )
                    }
                )
                candidate_authoritative = bool(candidate.get("authoritative"))
                existing_authoritative = bool(
                    existing and existing.get("authoritative")
                )
                if existing_authoritative and not candidate_authoritative:
                    continue
                if (
                    existing is None
                    or candidate_authoritative
                    or candidate_timestamp >= existing_timestamp
                ):
                    replacement = dict(candidate)
                    if (
                        not replacement.get("event_title")
                        and existing is not None
                        and existing.get("event_title")
                    ):
                        replacement["event_title"] = existing["event_title"]
                    pair_hints[card_date] = replacement
    return {
        pair: sorted(
            candidates.values(),
            key=lambda candidate: str(candidate.get("card_date") or ""),
        )
        for pair, candidates in merged.items()
    }


def _snapshot_hint_date(hint: dict) -> date | None:
    token = _calendar_token(hint.get("card_date"), convert_timestamp=False)
    try:
        return date.fromisoformat(token)
    except ValueError:
        return None


def _select_snapshot_card_hint(row: dict, candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    event_day_token = _calendar_token(
        row.get("card_date")
        or row.get("event_date")
        or row.get("market_event_date")
        or row.get("commence_time")
    )
    try:
        event_day = date.fromisoformat(event_day_token)
    except ValueError:
        event_day = None

    dated_candidates = [
        (candidate_day, candidate)
        for candidate in candidates
        for candidate_day in [_snapshot_hint_date(candidate)]
        if candidate_day is not None
    ]
    if event_day is not None and dated_candidates:
        nearby = [
            (abs((candidate_day - event_day).days), candidate)
            for candidate_day, candidate in dated_candidates
            if abs((candidate_day - event_day).days) <= 2
        ]
        if not nearby:
            return None
        _, selected = min(
            nearby,
            key=lambda item: (
                0 if item[1].get("authoritative") else 1,
                item[0],
                str(item[1].get("card_date") or ""),
            ),
        )
        return selected

    generated_token = _calendar_token(
        row.get("prediction_generated_at")
        or row.get("generated_at")
        or row.get("source_cache_timestamp")
        or row.get("recovered_group_date")
    )
    try:
        generated_day = date.fromisoformat(generated_token)
    except ValueError:
        generated_day = None

    if generated_day is not None and dated_candidates:
        ranked = sorted(
            [
                (
                (candidate_day - generated_day).days,
                candidate,
                )
                for candidate_day, candidate in dated_candidates
            ],
            key=lambda item: (item[0], str(item[1].get("card_date") or "")),
        )
        future = [
            (days_until_card, candidate)
            for days_until_card, candidate in ranked
            if -1 <= days_until_card <= PREDICTION_HISTORY_CARD_HINT_LOOKAHEAD_DAYS
        ]
        if future:
            return min(
                future,
                key=lambda item: (
                    max(item[0], 0),
                    abs(item[0]),
                    0 if item[1].get("authoritative") else 1,
                ),
            )[1]
        # A known observation time that fits none of the candidate cards must
        # stay unresolved. This is common when completed-results data contains
        # an old meeting but the prediction concerns a future rematch.
        return None

    return candidates[0] if len(candidates) == 1 else None


def _apply_snapshot_card_hints(
    rows: Iterable[dict],
    hints: dict[tuple[str, str], list[dict]],
) -> set[int]:
    matched_row_ids: set[int] = set()
    for row in rows:
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        candidates = hints.get(pair, [])
        selected = _select_snapshot_card_hint(row, candidates)
        if selected is None:
            continue
        row["card_date"] = selected["card_date"]
        if selected.get("event_title"):
            row["event_title"] = selected["event_title"]
        matched_row_ids.add(id(row))
    return matched_row_ids


def recover_prediction_rows_from_operator_decisions(
    paths: Iterable[Path | str],
) -> list[dict]:
    """Recover model picks from retained operator decisions.

    ``bet_on`` can be a value candidate instead of the model favorite, so the
    winner is derived from that side's stored model probability rather than
    blindly treating the candidate as the pick.
    """
    recovered: list[dict] = []
    for decision in _load_jsonl_event_hint_rows(paths):
        fighter_a, fighter_b = _fighter_names(decision)
        bet_on = str(decision.get("bet_on") or decision.get("pick") or "").strip()
        model_prob = _coerce_probability(decision.get("model_prob"))
        if not fighter_a or not fighter_b or not bet_on or model_prob is None:
            continue
        normalized_bet_on = _normalized_name(bet_on)
        if normalized_bet_on == _normalized_name(fighter_a):
            bet_side = "a"
        elif normalized_bet_on == _normalized_name(fighter_b):
            bet_side = "b"
        else:
            bet_side = str(decision.get("bet_side") or "").strip().casefold()
        if bet_side not in {"a", "b"}:
            continue

        prob_a = model_prob if bet_side == "a" else 1.0 - model_prob
        prob_b = model_prob if bet_side == "b" else 1.0 - model_prob
        timestamp = decision.get("timestamp") or decision.get("created_at")
        event_date = decision.get("event_date") or decision.get("market_event_date")
        if _prediction_is_after_event(timestamp, event_date):
            continue
        row = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "predicted_winner": fighter_a if prob_a >= prob_b else fighter_b,
            "predicted_side": "a" if prob_a >= prob_b else "b",
            "prob_a": prob_a,
            "prob_b": prob_b,
            "confidence": max(prob_a, prob_b),
            "event_date": event_date,
            "market_event_date": decision.get("market_event_date"),
            "card_date": decision.get("card_date"),
            "event_title": decision.get("event_title"),
            "prediction_generated_at": timestamp,
            "recovery_provenance": "operator_decision_log",
        }
        market_prob = _coerce_probability(decision.get("market_prob"))
        if market_prob is not None:
            row["a_market_prob"] = market_prob if bet_side == "a" else 1.0 - market_prob
            row["b_market_prob"] = market_prob if bet_side == "b" else 1.0 - market_prob
        if isinstance(decision.get("research_summary"), dict):
            row["recovered_research_summary"] = decision["research_summary"]
        rationale = str(decision.get("rationale") or "").strip()
        if rationale:
            row["recovered_rationale"] = rationale
        recovered.append(_json_safe(row))
    return recovered


def _apply_event_hints_to_recovered_rows(
    rows: list[dict],
    event_hints: dict[tuple[str, str], list[dict]],
) -> None:
    for row in rows:
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        candidates = event_hints.get(pair, [])
        if not candidates:
            continue
        generated_at = _parse_timestamp(
            row.get("prediction_generated_at")
            or row.get("generated_at")
            or row.get("source_cache_timestamp")
            or row.get("recovered_group_date")
        )
        if generated_at is None and len(candidates) != 1:
            # Without an observation time, choosing between rematches would be
            # guesswork and can silently move a prediction to the wrong card.
            continue
        ranked = []
        for hint in candidates:
            raw_event_date = hint.get("event_date")
            event_at = _parse_timestamp(raw_event_date)
            if event_at is None:
                continue
            if generated_at is None:
                rank = (0, event_at.timestamp())
            elif not _timestamp_value_has_time(raw_event_date):
                event_day_token = _calendar_token(
                    raw_event_date,
                    convert_timestamp=False,
                )
                try:
                    days_until_event = (
                        date.fromisoformat(event_day_token)
                        - generated_at.astimezone(_event_timezone()).date()
                    ).days
                except ValueError:
                    continue
                if not (
                    0
                    <= days_until_event
                    <= PREDICTION_HISTORY_CARD_HINT_LOOKAHEAD_DAYS
                ):
                    continue
                rank = (0, days_until_event * 86400)
            else:
                seconds_until_event = (event_at - generated_at).total_seconds()
                # A normal live prediction is produced before the bout. Allow a
                # small clock/source skew, but never map it to a distant event.
                if (
                    seconds_until_event < -6 * 3600
                    or seconds_until_event
                    > PREDICTION_HISTORY_CARD_HINT_LOOKAHEAD_DAYS * 86400
                ):
                    continue
                rank = (0 if seconds_until_event >= 0 else 1, abs(seconds_until_event))
            ranked.append((rank, hint, event_at))
        if not ranked:
            continue
        _, selected, _ = min(ranked, key=lambda item: item[0])
        row["event_date"] = selected.get("event_date")
        if selected.get("card_date"):
            row["card_date"] = selected["card_date"]
        if selected.get("event_title"):
            row["event_title"] = selected["event_title"]


def _collapse_recovered_operator_rows(rows: list[dict]) -> list[dict]:
    """Keep the final pre-start operator model read for each recovered bout."""
    collapsed: list[dict] = []
    mapped: dict[str, list[dict]] = {}
    undated: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("event_date") or row.get("card_date"):
            key = prediction_archive_key(row)
            if key:
                mapped.setdefault(key, []).append(row)
                continue
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        if pair[0] and pair[1]:
            undated.setdefault(pair, []).append(row)

    def richest_latest(candidates: list[dict]) -> dict:
        primary = max(candidates, key=lambda row: (_row_timestamp(row), _detail_score(row)))
        for candidate in candidates:
            if candidate is not primary:
                primary = _merge_complementary_recovery_fields(primary, candidate)
        return primary

    for candidates in mapped.values():
        pre_event = [
            row
            for row in candidates
            if not _prediction_is_after_event(
                row.get("prediction_generated_at"),
                row.get("event_date") or row.get("card_date"),
            )
        ]
        if pre_event:
            collapsed.append(richest_latest(pre_event))

    for candidates in undated.values():
        candidates.sort(key=_row_timestamp)
        clusters: list[list[dict]] = []
        for row in candidates:
            if not clusters or (
                _row_timestamp(row) - _row_timestamp(clusters[-1][-1]) > 14 * 86400
            ):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster in clusters:
            selected = richest_latest(cluster)
            selected["recovered_group_date"] = _calendar_token(
                cluster[0].get("prediction_generated_at")
            )
            collapsed.append(selected)
    return sorted(collapsed, key=_row_timestamp, reverse=True)


def recover_prediction_rows_from_logs(
    log_paths: Iterable[Path | str],
    *,
    event_hints: dict[tuple[str, str], list[dict]] | None = None,
    card_hints: dict[tuple[str, str], list[dict]] | None = None,
) -> list[dict]:
    """Recover model-side probabilities from the bot's retained text logs."""
    recovered: list[dict] = []
    log_card_hints: dict[tuple[str, str], list[dict]] = {}

    def add_log_card_hint(
        fighter_a: str,
        fighter_b: str,
        card_date: str,
        observed_at,
    ) -> None:
        pair = tuple(
            sorted((_normalized_name(fighter_a), _normalized_name(fighter_b)))
        )
        if not pair[0] or not pair[1] or not _calendar_token(card_date):
            return
        hint = {
            "card_date": _calendar_token(card_date),
            "event_title": "",
            "observed_at": observed_at,
        }
        log_card_hints.setdefault(pair, []).append(hint)

    for raw_path in log_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        current_timestamp = None
        pending: dict | None = None
        recent_reference_dates: dict[str, tuple[str, int, object]] = {}
        line_index = 0

        def flush_pending():
            nonlocal pending
            if pending and pending.get("prob_a") is not None and pending.get("prob_b") is not None:
                fighter_a_key = _normalized_name(pending["fighter_a"])
                fighter_b_key = _normalized_name(pending["fighter_b"])
                for mapping_key, output_a, output_b in (
                    ("_market_probs", "a_market_prob", "b_market_prob"),
                    ("_no_odds_probs", "no_odds_prob_a", "no_odds_prob_b"),
                ):
                    probability_map = pending.pop(mapping_key, {})
                    if (
                        fighter_a_key in probability_map
                        and fighter_b_key in probability_map
                    ):
                        pending[output_a] = probability_map[fighter_a_key]
                        pending[output_b] = probability_map[fighter_b_key]

                reference_a = recent_reference_dates.get(fighter_a_key)
                reference_b = recent_reference_dates.get(fighter_b_key)
                matchup_line = int(pending.pop("_matchup_line_index", line_index))
                if (
                    reference_a is not None
                    and reference_b is not None
                    and reference_a[0] == reference_b[0]
                    and max(reference_a[1], reference_b[1]) <= matchup_line
                    and matchup_line - min(reference_a[1], reference_b[1]) <= 250
                ):
                    pending["card_date"] = reference_a[0]
                    add_log_card_hint(
                        pending["fighter_a"],
                        pending["fighter_b"],
                        reference_a[0],
                        max(reference_a[2], reference_b[2], key=lambda value: str(value or "")),
                    )
                pending["predicted_winner"] = (
                    pending["fighter_a"]
                    if pending["prob_a"] >= pending["prob_b"]
                    else pending["fighter_b"]
                )
                recovered.append(pending)
            pending = None

        try:
            for line in _iter_decoded_log_lines(path):
                line_index += 1
                timestamp_match = _LOG_TIMESTAMP_RE.match(line)
                line_timestamp = (
                    timestamp_match.group("timestamp")
                    if timestamp_match is not None
                    else current_timestamp
                )
                context_match = _LOG_CARD_CONTEXT_RE.search(line)
                if context_match:
                    add_log_card_hint(
                        context_match.group("a").strip(),
                        context_match.group("b").strip(),
                        context_match.group("card_date"),
                        line_timestamp,
                    )
                reference_match = _LOG_FIGHTER_REFERENCE_DATE_RE.search(line)
                if reference_match:
                    recent_reference_dates[
                        _normalized_name(reference_match.group("fighter"))
                    ] = (
                        reference_match.group("card_date"),
                        line_index,
                        line_timestamp,
                    )
                if timestamp_match:
                    flush_pending()
                    current_timestamp = line_timestamp
                    continue
                matchup_match = _LOG_MATCHUP_RE.match(line)
                if matchup_match:
                    flush_pending()
                    pending = {
                        "fighter_a": matchup_match.group("a").strip(),
                        "fighter_b": matchup_match.group("b").strip(),
                        "prediction_generated_at": current_timestamp,
                        "_matchup_line_index": line_index,
                    }
                    continue
                probability_match = _LOG_PROBABILITY_RE.match(line)
                if pending is None or probability_match is None:
                    continue
                label = probability_match.group("label").casefold()
                fighter_a = probability_match.group("a").strip()
                fighter_b = probability_match.group("b").strip()
                prob_a = float(probability_match.group("pa")) / 100.0
                prob_b = float(probability_match.group("pb")) / 100.0
                probability_map = {
                    _normalized_name(fighter_a): prob_a,
                    _normalized_name(fighter_b): prob_b,
                }
                if label == "model":
                    pending["fighter_a"] = fighter_a
                    pending["fighter_b"] = fighter_b
                    pending["prob_a"] = prob_a
                    pending["prob_b"] = prob_b
                    pending["confidence"] = max(prob_a, prob_b)
                elif label == "bookmakers":
                    pending["_market_probs"] = probability_map
                else:
                    pending["_no_odds_probs"] = probability_map
        except OSError:
            continue
        flush_pending()

    if event_hints:
        _apply_event_hints_to_recovered_rows(recovered, event_hints)
    combined_card_hints = _merge_card_hint_maps(card_hints, log_card_hints)
    if combined_card_hints:
        _apply_snapshot_card_hints(recovered, combined_card_hints)

    # Monitoring may log the same future bout for several consecutive days.
    # Collapse observations within two weeks, while still allowing a later
    # rematch between the same fighters to remain a distinct recovered row.
    clustered: list[dict] = []
    mapped_rows: dict[str, list[dict]] = {}
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in recovered:
        if row.get("event_date") or row.get("card_date"):
            key = prediction_archive_key(row)
            if key:
                mapped_rows.setdefault(key, []).append(row)
                continue
        pair = tuple(sorted((_normalized_name(row["fighter_a"]), _normalized_name(row["fighter_b"]))))
        by_pair.setdefault(pair, []).append(row)
    for rows in mapped_rows.values():
        pre_event_rows = [
            row
            for row in rows
            if not _prediction_is_after_event(
                row.get("prediction_generated_at"),
                row.get("event_date") or row.get("card_date"),
            )
        ]
        if not pre_event_rows:
            # A mapped bout with only post-start model output is not a valid
            # pre-fight prediction and may already contain live/result signal.
            continue
        candidates = pre_event_rows
        clustered.append(max(candidates, key=lambda row: (_row_timestamp(row), _detail_score({**row, "detail_level": _detail_level(row)}))))
    for rows in by_pair.values():
        rows.sort(key=_row_timestamp)
        clusters: list[list[dict]] = []
        for row in rows:
            if not clusters:
                clusters.append([row])
                continue
            previous_timestamp = _row_timestamp(clusters[-1][-1])
            if _row_timestamp(row) - previous_timestamp <= 14 * 86400:
                clusters[-1].append(row)
            else:
                clusters.append([row])
        for cluster in clusters:
            richest_latest = max(
                cluster,
                key=lambda row: (_row_timestamp(row), _detail_score({**row, "detail_level": _detail_level(row)})),
            )
            recovered_row = dict(richest_latest)
            recovered_row["recovered_group_date"] = _calendar_token(
                cluster[0].get("prediction_generated_at")
            )
            clustered.append(recovered_row)
    return sorted(clustered, key=_row_timestamp, reverse=True)


def initialize_prediction_history(
    logs_dir: Path | str,
    *,
    data_dir: Path | str | None = None,
    raw_data_dir: Path | str | None = None,
) -> dict:
    """One-time bootstrap from the old live cache and retained rotating logs."""
    directory = Path(logs_dir)
    history_path = directory / PREDICTION_HISTORY_FILENAME
    process_key = str(history_path.resolve())
    with _HISTORY_LOCK:
        if process_key in _INITIALIZED_HISTORY_PATHS:
            return {
                "initialized": False,
                "reason": "already_initialized_in_process",
                "total": None,
            }
        # Retry on the next process start, but never rescan large retained logs
        # every live cycle when this process encounters a malformed archive.
        _INITIALIZED_HISTORY_PATHS.add(process_key)

    model_tracker_rows = recover_prediction_rows_from_model_tracker(
        directory / "bet_ledger_model_tracker.json"
    )
    historical_ledger_rows = recover_prediction_rows_from_bet_ledgers(
        directory / filename
        for filename in (
            "bet_ledger_single.json",
            "bet_ledger_conviction.json",
            "bet_ledger_trader_b.json",
        )
    )
    operator_directories = [directory / "operator"]
    if data_dir is not None:
        canonical_operator_dir = Path(data_dir) / "operator"
        if canonical_operator_dir not in operator_directories:
            operator_directories.insert(0, canonical_operator_dir)
    decision_paths = tuple(
        operator_dir / filename
        for operator_dir in operator_directories
        for filename in ("decision_log.jsonl", "tracker_decision_log.jsonl")
    )
    operator_cache_paths = tuple(
        operator_dir / filename
        for operator_dir in operator_directories
        for filename in ("gemini_pick_cache.json", "gemini_research_cache.json")
    )
    raw_decision_rows = _load_jsonl_event_hint_rows(decision_paths)
    operator_cache_hint_rows = _load_operator_cache_hint_rows(operator_cache_paths)
    operator_rows = recover_prediction_rows_from_operator_decisions(decision_paths)
    snapshot_directories = [directory / "raw" / "snapshots"]
    if raw_data_dir is not None:
        canonical_snapshot_dir = Path(raw_data_dir) / "snapshots"
        if canonical_snapshot_dir not in snapshot_directories:
            snapshot_directories.insert(0, canonical_snapshot_dir)
    snapshot_card_hints: dict[tuple[str, str], list[dict]] = {}
    for snapshot_directory in snapshot_directories:
        for pair, hints in _load_snapshot_card_hints(snapshot_directory).items():
            for hint in hints:
                if hint not in snapshot_card_hints.setdefault(pair, []):
                    snapshot_card_hints[pair].append(hint)
    completed_event_card_hints = (
        _load_completed_event_card_hints(raw_data_dir)
        if raw_data_dir is not None
        else {}
    )
    structured_card_hints = _merge_card_hint_maps(
        completed_event_card_hints,
        snapshot_card_hints,
        _prediction_card_hints(operator_cache_hint_rows),
    )
    with _HISTORY_LOCK:
        _STRUCTURED_CARD_HINTS_BY_HISTORY_PATH[process_key] = copy.deepcopy(
            structured_card_hints
        )
    _apply_snapshot_card_hints(model_tracker_rows, structured_card_hints)
    _apply_snapshot_card_hints(historical_ledger_rows, structured_card_hints)
    _apply_snapshot_card_hints(raw_decision_rows, structured_card_hints)
    _apply_snapshot_card_hints(operator_rows, structured_card_hints)
    event_hint_rows = [
        *model_tracker_rows,
        *historical_ledger_rows,
        *operator_cache_hint_rows,
        *raw_decision_rows,
    ]
    event_hints = _prediction_event_hints(event_hint_rows)
    _apply_event_hints_to_recovered_rows(operator_rows, event_hints)
    operator_rows = _collapse_recovered_operator_rows(operator_rows)

    log_paths = sorted(
        (path for path in directory.glob("bot.log*") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    recovered_rows = recover_prediction_rows_from_logs(
        log_paths,
        event_hints=event_hints,
        card_hints=structured_card_hints,
    )
    total = 0
    if model_tracker_rows:
        result = archive_prediction_payload(
            {"predictions": model_tracker_rows},
            history_path,
            source="recovered_model_tracker",
        )
        total = result["total"]
    if historical_ledger_rows:
        result = archive_prediction_payload(
            {"predictions": historical_ledger_rows},
            history_path,
            source="recovered_historical_ledger",
        )
        total = result["total"]
    if operator_rows:
        result = archive_prediction_payload(
            {"predictions": operator_rows},
            history_path,
            source="recovered_operator_decision",
        )
        total = result["total"]
    if recovered_rows:
        result = archive_prediction_payload(
            {"predictions": recovered_rows},
            history_path,
            source="recovered_bot_log",
        )
        total = result["total"]

    cache_path = directory / "predictions_cache.json"
    if cache_path.exists():
        cache_payload = _read_json(cache_path)
        result = archive_prediction_payload(
            cache_payload,
            history_path,
            source="legacy_live_cache_bootstrap",
        )
        total = result["total"]
    reconciliation = reconcile_prediction_history_cards(
        history_path,
        structured_card_hints,
    )
    total = reconciliation["total"] or total
    summary = {
        "initialized": (
            bool(model_tracker_rows)
            or bool(historical_ledger_rows)
            or bool(operator_rows)
            or bool(recovered_rows)
            or cache_path.exists()
        ),
        "recovered_from_model_tracker": len(model_tracker_rows),
        "recovered_from_historical_ledgers": len(historical_ledger_rows),
        "operator_cache_card_hints": len(operator_cache_hint_rows),
        "completed_event_card_hints": sum(
            len(hints) for hints in completed_event_card_hints.values()
        ),
        "recovered_from_operator_decisions": len(operator_rows),
        "recovered_from_logs": len(recovered_rows),
        "rekeyed_to_cards": reconciliation["rekeyed"],
        "deduplicated_after_rekey": reconciliation["deduplicated"],
        "discarded_post_start": reconciliation["discarded_post_start"],
        "total": total,
    }
    return summary
