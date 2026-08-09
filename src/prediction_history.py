"""Durable, display-only storage for historical UFC model predictions.

The live prediction cache is deliberately short lived because it can feed trade
execution.  This module keeps a separate archive whose rows are never consumed
by the trading path.  The archive accepts older display formats so a sparse
winner pick can survive even when the original breakdown is no longer usable.
"""

from __future__ import annotations

import copy
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

from src.data.name_utils import normalize_cross_source_name


PREDICTION_HISTORY_FILENAME = "predictions_history.json"
PREDICTION_HISTORY_SCHEMA_VERSION = 1
PREDICTION_HISTORY_MAX_ROWS = 10_000

_HISTORY_LOCK = threading.RLock()
_INITIALIZED_HISTORY_PATHS: set[str] = set()
_INTERNAL_LIVE_FIELDS = frozenset(
    {
        "cache_key",
        "event_context_snapshot",
        "method_odds_fingerprint",
        "odds_snapshot",
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


def _load_snapshot_card_hints(directory: Path) -> dict[tuple[str, str], list[dict]]:
    hints: dict[tuple[str, str], list[dict]] = {}
    if not directory.exists():
        return hints
    for path in directory.glob("*.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("fights"), list):
            continue
        card_date = _calendar_token(payload.get("event_date"), convert_timestamp=False)
        event_title = str(payload.get("event") or payload.get("event_title") or "").strip()
        for fight in payload["fights"]:
            if not isinstance(fight, dict):
                continue
            fighter_a, fighter_b = _fighter_names(fight)
            pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
            if not pair[0] or not pair[1] or not card_date:
                continue
            hint = {"card_date": card_date, "event_title": event_title}
            if hint not in hints.setdefault(pair, []):
                hints[pair].append(hint)
    return hints


def _apply_snapshot_card_hints(
    rows: Iterable[dict],
    hints: dict[tuple[str, str], list[dict]],
) -> None:
    for row in rows:
        fighter_a, fighter_b = _fighter_names(row)
        pair = tuple(sorted((_normalized_name(fighter_a), _normalized_name(fighter_b))))
        candidates = hints.get(pair, [])
        if not candidates:
            continue
        event_day = _calendar_token(
            row.get("event_date") or row.get("market_event_date") or row.get("commence_time")
        )
        selected = None
        if event_day:
            ranked = []
            for candidate in candidates:
                candidate_day = _calendar_token(
                    candidate.get("card_date"), convert_timestamp=False
                )
                try:
                    distance = abs(
                        (date.fromisoformat(candidate_day) - date.fromisoformat(event_day)).days
                    )
                except ValueError:
                    distance = 9999
                ranked.append((distance, candidate))
            if ranked:
                closest_distance, closest = min(ranked, key=lambda item: item[0])
                if closest_distance <= 2:
                    selected = closest
        elif len(candidates) == 1:
            selected = candidates[0]
        if selected is None:
            continue
        row["card_date"] = selected["card_date"]
        if selected.get("event_title"):
            row["event_title"] = selected["event_title"]


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
        generated_at = _parse_timestamp(timestamp)
        event_at = _parse_timestamp(event_date)
        if generated_at is not None and event_at is not None and generated_at > event_at:
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
        generated_at = _parse_timestamp(row.get("prediction_generated_at"))
        ranked = []
        for hint in candidates:
            event_at = _parse_timestamp(hint.get("event_date"))
            if event_at is None:
                continue
            if generated_at is None:
                rank = (0, event_at.timestamp())
            else:
                seconds_until_event = (event_at - generated_at).total_seconds()
                # A normal live prediction is produced before the bout. Allow a
                # small clock/source skew, but never map it to a distant event.
                if seconds_until_event < -6 * 3600 or seconds_until_event > 30 * 86400:
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
        event_at = _parse_timestamp(candidates[0].get("event_date"))
        pre_event = [
            row
            for row in candidates
            if event_at is None
            or _parse_timestamp(row.get("prediction_generated_at")) is None
            or _parse_timestamp(row.get("prediction_generated_at")) <= event_at
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
) -> list[dict]:
    """Recover model-side probabilities from the bot's retained text logs."""
    recovered: list[dict] = []
    for raw_path in log_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        current_timestamp = None
        pending: dict | None = None

        def flush_pending():
            nonlocal pending
            if pending and pending.get("prob_a") is not None and pending.get("prob_b") is not None:
                pending["predicted_winner"] = (
                    pending["fighter_a"]
                    if pending["prob_a"] >= pending["prob_b"]
                    else pending["fighter_b"]
                )
                recovered.append(pending)
            pending = None

        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\r\n")
                    timestamp_match = _LOG_TIMESTAMP_RE.match(line)
                    if timestamp_match:
                        flush_pending()
                        current_timestamp = timestamp_match.group("timestamp")
                        continue
                    matchup_match = _LOG_MATCHUP_RE.match(line)
                    if matchup_match:
                        flush_pending()
                        pending = {
                            "fighter_a": matchup_match.group("a").strip(),
                            "fighter_b": matchup_match.group("b").strip(),
                            "prediction_generated_at": current_timestamp,
                        }
                        continue
                    probability_match = _LOG_PROBABILITY_RE.match(line)
                    if pending is None or probability_match is None:
                        continue
                    label = probability_match.group("label").casefold()
                    prob_a = float(probability_match.group("pa")) / 100.0
                    prob_b = float(probability_match.group("pb")) / 100.0
                    if label == "model":
                        pending["fighter_a"] = probability_match.group("a").strip()
                        pending["fighter_b"] = probability_match.group("b").strip()
                        pending["prob_a"] = prob_a
                        pending["prob_b"] = prob_b
                        pending["confidence"] = max(prob_a, prob_b)
                    elif label == "bookmakers":
                        pending["a_market_prob"] = prob_a
                        pending["b_market_prob"] = prob_b
                    else:
                        pending["no_odds_prob_a"] = prob_a
                        pending["no_odds_prob_b"] = prob_b
        except OSError:
            continue
        flush_pending()

    if event_hints:
        _apply_event_hints_to_recovered_rows(recovered, event_hints)

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
        event_at = _parse_timestamp(rows[0].get("event_date"))
        pre_event_rows = [
            row
            for row in rows
            if event_at is None
            or _parse_timestamp(row.get("prediction_generated_at")) is None
            or _parse_timestamp(row.get("prediction_generated_at")) <= event_at
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
    raw_decision_rows = _load_jsonl_event_hint_rows(decision_paths)
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
    _apply_snapshot_card_hints(model_tracker_rows, snapshot_card_hints)
    _apply_snapshot_card_hints(raw_decision_rows, snapshot_card_hints)
    _apply_snapshot_card_hints(operator_rows, snapshot_card_hints)
    event_hint_rows = [*model_tracker_rows, *raw_decision_rows]
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
    )
    total = 0
    if model_tracker_rows:
        result = archive_prediction_payload(
            {"predictions": model_tracker_rows},
            history_path,
            source="recovered_model_tracker",
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
    summary = {
        "initialized": (
            bool(model_tracker_rows)
            or bool(operator_rows)
            or bool(recovered_rows)
            or cache_path.exists()
        ),
        "recovered_from_model_tracker": len(model_tracker_rows),
        "recovered_from_operator_decisions": len(operator_rows),
        "recovered_from_logs": len(recovered_rows),
        "total": total,
    }
    return summary
