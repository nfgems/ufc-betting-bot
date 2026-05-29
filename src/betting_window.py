"""Shared pre-fight timing rules for live bet execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import MAX_BET_HOURS_BEFORE_EVENT

LIVE_TRADE_START_BUFFER = timedelta(hours=1)


def _current_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_event_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" not in text and ":" not in text:
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
    return parsed


def bet_window_status(
    event_time: object,
    *,
    now: datetime | None = None,
    close_buffer: timedelta | None = None,
) -> dict | None:
    commence = parse_event_timestamp(event_time)
    if commence is None:
        return None

    current_time = now or _current_utc()
    effective_close_buffer = close_buffer or LIVE_TRADE_START_BUFFER
    opens_at = commence - timedelta(hours=MAX_BET_HOURS_BEFORE_EVENT)
    closes_at = commence - effective_close_buffer

    if current_time < opens_at:
        return {
            "open": False,
            "state": "too_early",
            "reason": "Bet window not open",
            "detail": (
                f"Bet window opens at {opens_at.isoformat()} "
                f"({MAX_BET_HOURS_BEFORE_EVENT}h before fight starts at {commence.isoformat()})"
            ),
            "commence_time": commence,
            "window_opens_at": opens_at,
            "window_closes_at": closes_at,
        }

    if current_time >= closes_at:
        return {
            "open": False,
            "state": "too_late",
            "reason": "Too close to event",
            "detail": (
                f"Fight starts at {commence.isoformat()} "
                f"(safety buffer {effective_close_buffer})"
            ),
            "commence_time": commence,
            "window_opens_at": opens_at,
            "window_closes_at": closes_at,
        }

    return {
        "open": True,
        "state": "open",
        "reason": None,
        "detail": None,
        "commence_time": commence,
        "window_opens_at": opens_at,
        "window_closes_at": closes_at,
    }
