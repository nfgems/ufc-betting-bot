"""
LLM Operator — the decision-making brain that receives model outputs,
conducts its own research, and makes final bet/no-bet decisions.

The XGBoost model produces a probability. The operator treats that as ONE
input among many. It runs its own research pipeline, synthesizes everything,
and makes the final call.

Pipeline:
    model.predict() → value.detect_value() → operator.evaluate() → execute_bet()
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from src.config import DATA_DIR
from src.data.name_utils import same_person_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPERATOR_ENABLED = os.getenv("LLM_OPERATOR_ENABLED", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
OPERATOR_MODE: Literal["gate", "advisory"] = (
    "gate" if os.getenv("LLM_OPERATOR_MODE", "gate").strip().lower() == "gate"
    else "advisory"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_OPERATOR_MODEL", "gemini-3.1-pro-preview")
_RETIRED_GEMINI_MODELS = frozenset(
    {
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-3-pro-preview",
        "gemini-3.1-flash-lite-preview",
    }
)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = str(os.getenv(name, default) or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = str(os.getenv(name, default) or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in str(
        os.getenv(
            "GEMINI_OPERATOR_FALLBACK_MODELS",
            "gemini-3.5-flash,gemini-3-flash-preview,gemini-2.5-pro,gemini-2.5-flash",
        ) or ""
    ).split(",")
    if model.strip()
)
# Legacy shared timeout. Grounded research now has its own default ceiling
# because Gemini 3 search requests routinely exceed 30s even when healthy.
GEMINI_TIMEOUT_MS = _env_int("GEMINI_OPERATOR_TIMEOUT_MS", 30000, minimum=1000)
GEMINI_RESEARCH_TIMEOUT_MS = _env_int(
    "GEMINI_OPERATOR_RESEARCH_TIMEOUT_MS",
    60000,
    minimum=1000,
)
GEMINI_SYNTHESIS_TIMEOUT_MS = _env_int(
    "GEMINI_OPERATOR_SYNTHESIS_TIMEOUT_MS",
    GEMINI_TIMEOUT_MS,
    minimum=1000,
)
GEMINI_PRIMARY_MODEL_RETRIES = _env_int(
    "GEMINI_OPERATOR_PRIMARY_MODEL_RETRIES",
    5,
    minimum=1,
)
GEMINI_FALLBACK_RETRIES_PER_MODEL = _env_int(
    "GEMINI_OPERATOR_FALLBACK_RETRIES_PER_MODEL",
    2,
    minimum=1,
)
GEMINI_RETRY_INITIAL_DELAY_SECONDS = _env_float(
    "GEMINI_OPERATOR_RETRY_INITIAL_DELAY_SECONDS",
    4.0,
    minimum=0.0,
)
GEMINI_RETRY_MAX_DELAY_SECONDS = _env_float(
    "GEMINI_OPERATOR_RETRY_MAX_DELAY_SECONDS",
    30.0,
    minimum=0.0,
)
GEMINI_RETRY_JITTER_SECONDS = _env_float(
    "GEMINI_OPERATOR_RETRY_JITTER_SECONDS",
    1.0,
    minimum=0.0,
)
GEMINI_PRIMARY_GROUNDING_RETRIES = _env_int(
    "GEMINI_OPERATOR_PRIMARY_GROUNDING_RETRIES",
    2,
    minimum=0,
)
GEMINI_OVERLOAD_FAILURE_THRESHOLD = _env_int(
    "GEMINI_OPERATOR_OVERLOAD_FAILURE_THRESHOLD",
    2,
    minimum=1,
)
GEMINI_OVERLOAD_COOLDOWN_SECONDS = _env_float(
    "GEMINI_OPERATOR_OVERLOAD_COOLDOWN_SECONDS",
    180.0,
    minimum=0.0,
)
GEMINI_RESEARCH_CACHE_TTL_SECONDS = _env_float(
    "GEMINI_RESEARCH_CACHE_TTL_SECONDS",
    900.0,
    minimum=0.0,
)
LLM_OPERATOR_FAILURE_CACHE_TTL_SECONDS = _env_float(
    "LLM_OPERATOR_FAILURE_CACHE_TTL_SECONDS",
    1800.0,
    minimum=0.0,
)

# Paths
OPERATOR_DIR = DATA_DIR / "operator"
OPERATOR_DIR.mkdir(parents=True, exist_ok=True)
BLIND_SPOTS_PATH = OPERATOR_DIR / "blind_spots.json"
DECISION_LOG_PATH = OPERATOR_DIR / "decision_log.jsonl"  # append-only, one JSON object per line
TRACKER_DECISION_LOG_PATH = OPERATOR_DIR / "tracker_decision_log.jsonl"
_GEMINI_PICK_CACHE_FILE = OPERATOR_DIR / "gemini_pick_cache.json"
_gemini_pick_cache_lock = threading.Lock()
_GEMINI_RESEARCH_CACHE_FILE = OPERATOR_DIR / "gemini_research_cache.json"
_gemini_research_cache_lock = threading.Lock()

# Exposure limits
MAX_BETS_PER_EVENT = 3  # Flag concentration risk above this

# Session-level decision cache: fight_key → (OperatorDecision, epoch)
# Prevents re-evaluating the same fight across loop cycles and across
# value/conviction trader passes within a single cycle.
_decision_cache: dict[str, tuple["OperatorDecision", float]] = {}
_decision_cache_lock = threading.Lock()
# Per-key locks: prevents two threads from evaluating the same fight
# concurrently (they'd both miss the cache and double-call the LLM).
_decision_inflight: dict[str, threading.Lock] = {}
_gemini_client_cache: dict[tuple[str, int], object] = {}
_gemini_client_cache_lock = threading.Lock()
_gemini_runtime_lock = threading.Lock()
_gemini_consecutive_transient_failures = 0
_gemini_circuit_open_until = 0.0

# Historical fallback TTL for entries that do not have a usable event date.
# For upcoming fights we keep one sticky decision until shortly after the
# event, which avoids repeated evaluations and verdict flips on the same card.
CACHE_TTL_SECONDS = float(os.getenv("LLM_OPERATOR_CACHE_TTL", "0"))
POST_EVENT_RETENTION_SECONDS = float(
    os.getenv("LLM_OPERATOR_POST_EVENT_RETENTION_HOURS", "48")
) * 3600.0

# Disk-backed cache file — survives process restarts.
_DECISION_CACHE_FILE = OPERATOR_DIR / "decision_cache.json"
_DECISION_LOCK_DIR = OPERATOR_DIR / "locks"
_DECISION_LOCK_DIR.mkdir(parents=True, exist_ok=True)
_PROCESS_LOCK_TIMEOUT_SECONDS = float(os.getenv("LLM_OPERATOR_LOCK_TIMEOUT_SECONDS", "20"))
_PROCESS_LOCK_STALE_SECONDS = float(os.getenv("LLM_OPERATOR_LOCK_STALE_SECONDS", "300"))


def clear_gemini_runtime_state() -> None:
    """Reset Gemini client/cache runtime state used for retries and circuit breaking."""
    global _gemini_consecutive_transient_failures, _gemini_circuit_open_until
    with _gemini_runtime_lock:
        _gemini_consecutive_transient_failures = 0
        _gemini_circuit_open_until = 0.0
    with _gemini_client_cache_lock:
        _gemini_client_cache.clear()


def _configured_gemini_models() -> list[str]:
    models: list[str] = []
    for model_name in (GEMINI_MODEL, *GEMINI_FALLBACK_MODELS):
        normalized = str(model_name or "").strip()
        if not normalized:
            continue
        if normalized in _RETIRED_GEMINI_MODELS:
            logger.warning("Skipping retired Gemini model configured for operator: %s", normalized)
            continue
        if normalized not in models:
            models.append(normalized)
    return models


def _gemini_attempts_for_model(model_name: str, *, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    if str(model_name or "").strip() == GEMINI_MODEL:
        return GEMINI_PRIMARY_MODEL_RETRIES
    return GEMINI_FALLBACK_RETRIES_PER_MODEL


def _gemini_retry_wait_seconds(attempt: int) -> float:
    base_wait = min(
        GEMINI_RETRY_MAX_DELAY_SECONDS,
        GEMINI_RETRY_INITIAL_DELAY_SECONDS * (2 ** attempt),
    )
    if GEMINI_RETRY_JITTER_SECONDS <= 0:
        return base_wait
    return base_wait + random.uniform(0.0, GEMINI_RETRY_JITTER_SECONDS)


def _gemini_grounding_retry_prompt(prompt: str) -> str:
    return (
        str(prompt or "").rstrip()
        + "\n\n"
        + "GROUNDING RETRY REQUIREMENT:\n"
        + "Your previous response was rejected because it returned text without "
        + "Gemini grounding sources/groundingChunks. You must execute the Google "
        + "Search tool for this matchup before answering. Do not answer from "
        + "memory or from the prompt alone. The application can only accept this "
        + "research if the API response includes grounding metadata. Return the "
        + "same plain-text headings only after web search has run."
    )


def _is_gemini_transient_error(exc: Exception) -> bool:
    message = str(exc or "").upper()
    transient_markers = (
        "408",
        "429",
        "500",
        "503",
        "504",
        "499",
        "CANCELLED",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
        "SERVER DISCONNECTED",
    )
    return any(marker in message for marker in transient_markers)


def _gemini_circuit_blocked_until(now: float | None = None) -> float:
    global _gemini_consecutive_transient_failures, _gemini_circuit_open_until
    current = time.time() if now is None else now
    with _gemini_runtime_lock:
        blocked_until = float(_gemini_circuit_open_until or 0.0)
        if blocked_until and current >= blocked_until:
            _gemini_consecutive_transient_failures = 0
            _gemini_circuit_open_until = 0.0
            return 0.0
        return blocked_until


def _record_gemini_transient_failure(last_error: str) -> None:
    global _gemini_consecutive_transient_failures, _gemini_circuit_open_until
    now = time.time()
    with _gemini_runtime_lock:
        if _gemini_circuit_open_until and now >= _gemini_circuit_open_until:
            _gemini_consecutive_transient_failures = 0
            _gemini_circuit_open_until = 0.0
        _gemini_consecutive_transient_failures += 1
        if (
            GEMINI_OVERLOAD_COOLDOWN_SECONDS > 0
            and _gemini_consecutive_transient_failures >= GEMINI_OVERLOAD_FAILURE_THRESHOLD
        ):
            _gemini_circuit_open_until = now + GEMINI_OVERLOAD_COOLDOWN_SECONDS
            logger.warning(
                "Gemini overload circuit opened for %.0fs after %d consecutive transient failures: %s",
                GEMINI_OVERLOAD_COOLDOWN_SECONDS,
                _gemini_consecutive_transient_failures,
                last_error,
            )


def _record_gemini_success() -> None:
    global _gemini_consecutive_transient_failures, _gemini_circuit_open_until
    with _gemini_runtime_lock:
        _gemini_consecutive_transient_failures = 0
        _gemini_circuit_open_until = 0.0


def _gemini_timeout_ms(*, use_search: bool) -> int:
    return GEMINI_RESEARCH_TIMEOUT_MS if use_search else GEMINI_SYNTHESIS_TIMEOUT_MS


def _is_gemini_3_model(model_name: str) -> bool:
    return str(model_name or "").strip().startswith("gemini-3")


def _gemini_stage_thinking_config(model_name: str, *, use_search: bool) -> dict[str, str] | None:
    # Do not lower Gemini reasoning effort for operator decisions. These calls
    # gate real-money trades, so we let the model use its default thinking
    # behavior even when that costs more latency during overloaded periods.
    return None


def _gemini_operation_label(success_log_label: str) -> str:
    label = str(success_log_label or "").strip()
    return label[7:].strip() if label.lower().startswith("gemini ") else label


def _get_gemini_client(timeout_ms: int | None = None):
    from google import genai
    from google.genai import types

    effective_timeout_ms = int(timeout_ms or GEMINI_SYNTHESIS_TIMEOUT_MS)
    cache_key = (GEMINI_API_KEY, effective_timeout_ms)
    with _gemini_client_cache_lock:
        cached = _gemini_client_cache.get(cache_key)
        if cached is not None:
            return cached

        http_options = types.HttpOptions(
            timeout=effective_timeout_ms,
            # Disable hidden SDK retries so operator latency stays bounded and
            # our own fallback/circuit-breaker policy is the only retry layer.
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_options)
        _gemini_client_cache[cache_key] = client
        return client


def _fight_cache_key(
    fighter_a: str,
    fighter_b: str,
    *,
    event_date: str = "",
    event_title: str = "",
    context: str = "",
) -> str:
    """Canonical cache key for one booked fight (order-independent, event-scoped)."""
    # ``context`` is accepted for older call sites, but deliberately ignored:
    # S/C trader passes should share one operator verdict for the same fight.
    pair = sorted([fighter_a.strip().lower(), fighter_b.strip().lower()])
    event_token = _normalize_event_date(event_date) or _normalize_event_date(event_title)
    return f"{event_token}|{pair[0]}|{pair[1]}" if event_token else f"{pair[0]}|{pair[1]}"


def _operator_research_cache_key(cache_key: str, model_pick: str) -> str:
    pick_token = re.sub(r"\s+", " ", str(model_pick or "").strip().lower())
    return f"{cache_key}|model_pick:{pick_token}" if pick_token else cache_key


def _normalize_event_date(value: object) -> str:
    """Normalize event identifiers down to a durable YYYY-MM-DD key when possible."""
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return text.casefold()
    return parsed.strftime("%Y-%m-%d")


def _event_cache_is_fresh(
    *,
    event_date: str = "",
    event_title: str = "",
    cached_at: float,
    now: float | None = None,
) -> bool:
    now = time.time() if now is None else now
    candidate_event = _normalize_event_date(event_date) or _normalize_event_date(event_title)
    if candidate_event:
        event_ts = pd.to_datetime(candidate_event, utc=True, errors="coerce")
        if not pd.isna(event_ts):
            return now <= float(event_ts.timestamp()) + POST_EVENT_RETENTION_SECONDS
    if CACHE_TTL_SECONDS <= 0:
        return True
    return (now - cached_at) < CACHE_TTL_SECONDS


def _existing_bet_matches_fight(
    existing_bet: dict,
    *,
    fighter_a: str,
    fighter_b: str,
    event_date: str = "",
) -> bool:
    """Check whether an existing ledger row refers to this same fight."""
    existing_a = str(existing_bet.get("fighter") or existing_bet.get("fighter_a") or "").strip()
    existing_b = str(existing_bet.get("opponent") or existing_bet.get("fighter_b") or "").strip()
    if not existing_a or not existing_b:
        return False

    names_match = (
        same_person_name(fighter_a, existing_a) and same_person_name(fighter_b, existing_b)
    ) or (
        same_person_name(fighter_a, existing_b) and same_person_name(fighter_b, existing_a)
    )
    if not names_match:
        return False

    candidate_event_date = _normalize_event_date(event_date)
    existing_event_date = _normalize_event_date(
        existing_bet.get("event_date") or existing_bet.get("market_event_date")
    )
    if candidate_event_date and existing_event_date and candidate_event_date != existing_event_date:
        return False

    return True


def _has_existing_bet_for_fight(
    *,
    fighter_a: str,
    fighter_b: str,
    existing_bets: list[dict] | None,
    event_date: str = "",
) -> bool:
    """Return True when the fight is already present in the current ledgers."""
    return any(
        _existing_bet_matches_fight(
            existing_bet,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            event_date=event_date,
        )
        for existing_bet in (existing_bets or [])
    )


def clear_decision_cache() -> None:
    """Clear the session decision cache (e.g. when a new event starts)."""
    with _decision_cache_lock:
        _decision_cache.clear()
        _decision_inflight.clear()
    try:
        _DECISION_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    for lock_path in _DECISION_LOCK_DIR.glob("*.lock"):
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            continue
    for extra_cache_path in (_GEMINI_PICK_CACHE_FILE, _GEMINI_RESEARCH_CACHE_FILE):
        try:
            extra_cache_path.unlink(missing_ok=True)
        except Exception:
            continue
    logger.info("Operator decision cache cleared")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResearchFindings:
    """Structured output from the research pipeline."""

    recency_flags: list[str] = field(default_factory=list)
    matchup_analysis: str = ""
    motivation_flags: list[str] = field(default_factory=list)
    social_signals: dict = field(default_factory=dict)
    blind_spot_matches: list[str] = field(default_factory=list)
    exposure_warning: str = ""


@dataclass
class OperatorDecision:
    """Final decision from the LLM Operator for a single bet."""

    verdict: Literal["PASS", "BLOCK"]
    confidence: float  # 0.0–1.0, operator's own confidence
    model_prob: float  # what the model said
    operator_prob: float  # operator's adjusted probability
    rationale: str  # written explanation (ALWAYS logged)
    research_summary: dict  # structured research findings
    risk_flags: list[str]  # any flags raised during research
    timestamp: str  # ISO timestamp
    fighter_a: str = ""
    fighter_b: str = ""
    bet_on: str = ""
    bet_side: str = ""
    edge: float = 0.0
    market_prob: float = 0.0
    event_date: str = ""
    event_title: str = ""
    trade_reason: str = ""
    decision_context: str = ""
    decision_key: str = ""
    provenance: dict = field(default_factory=dict)


def _deserialize_operator_decision(data: dict) -> OperatorDecision:
    d_fighter_a = data.get("fighter_a", "")
    d_fighter_b = data.get("fighter_b", "")
    event_date = data.get("event_date", "")
    event_title = data.get("event_title", "")
    return OperatorDecision(
        verdict=data.get("verdict", "PASS"),
        confidence=float(data.get("confidence", 0.0)),
        model_prob=float(data.get("model_prob", 0.5)),
        operator_prob=float(data.get("operator_prob", 0.5)),
        rationale=data.get("rationale", ""),
        research_summary=dict(data.get("research_summary") or {}),
        risk_flags=list(data.get("risk_flags") or []),
        timestamp=data.get("timestamp", ""),
        fighter_a=d_fighter_a,
        fighter_b=d_fighter_b,
        bet_on=data.get("bet_on", ""),
        bet_side=data.get("bet_side", ""),
        edge=float(data.get("edge", 0.0)),
        market_prob=float(data.get("market_prob", 0.0)),
        event_date=event_date,
        event_title=event_title,
        trade_reason=data.get("trade_reason", ""),
        decision_context=data.get("decision_context", ""),
        decision_key=data.get("decision_key")
        or _fight_cache_key(
            d_fighter_a,
            d_fighter_b,
            event_date=event_date,
            event_title=event_title,
            context=data.get("decision_context", ""),
        ),
        provenance=dict(data.get("provenance") or {}),
    )


def _decision_cache_is_fresh(
    decision: OperatorDecision,
    cached_at: float,
    *,
    now: float | None = None,
) -> bool:
    if "llm_unavailable" in (decision.risk_flags or []):
        current = time.time() if now is None else now
        return (current - cached_at) < LLM_OPERATOR_FAILURE_CACHE_TTL_SECONDS
    return _event_cache_is_fresh(
        event_date=decision.event_date,
        event_title=decision.event_title,
        cached_at=cached_at,
        now=now,
    )


def _prune_decision_cache_locked(*, now: float | None = None) -> None:
    now = time.time() if now is None else now
    expired_keys = [
        key
        for key, (decision, cached_at) in _decision_cache.items()
        if not _decision_cache_is_fresh(decision, cached_at, now=now)
    ]
    for key in expired_keys:
        _decision_cache.pop(key, None)


def _load_cached_decision_from_disk(cache_key: str) -> tuple[OperatorDecision, float] | None:
    if not _DECISION_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_DECISION_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to read operator decision cache from disk: %s", exc)
        return None

    entry = data.get(cache_key)
    if not isinstance(entry, dict):
        return None

    try:
        decision = _deserialize_operator_decision(entry.get("decision", {}))
        cached_at = float(entry.get("cached_at", 0))
    except Exception as exc:
        logger.debug("Failed to decode operator decision cache entry %s: %s", cache_key, exc)
        return None

    if not _decision_cache_is_fresh(decision, cached_at):
        return None
    return decision, cached_at


def _get_cached_decision(cache_key: str) -> OperatorDecision | None:
    now = time.time()
    with _decision_cache_lock:
        cached_entry = _decision_cache.get(cache_key)
        if cached_entry is not None:
            decision, cached_at = cached_entry
            if _decision_cache_is_fresh(decision, cached_at, now=now):
                return decision
            _decision_cache.pop(cache_key, None)

    disk_entry = _load_cached_decision_from_disk(cache_key)
    if disk_entry is None:
        return None

    decision, cached_at = disk_entry
    with _decision_cache_lock:
        current = _decision_cache.get(cache_key)
        if current is None or cached_at >= current[1]:
            _decision_cache[cache_key] = (decision, cached_at)
    return decision


def _save_decision_cache_to_disk() -> None:
    """Persist the in-memory decision cache to disk so it survives restarts."""
    try:
        with _decision_cache_lock:
            _prune_decision_cache_locked()
            serializable: dict[str, dict] = {}
            for key, (decision, cached_at) in _decision_cache.items():
                serializable[key] = {
                    "decision": asdict(decision),
                    "cached_at": cached_at,
                }
        _DECISION_CACHE_FILE.write_text(json.dumps(serializable, default=str), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to persist operator decision cache to disk: %s", exc)


def _load_decision_cache_from_disk() -> None:
    """Load persisted decision cache from disk into memory (called once at import)."""
    if not _DECISION_CACHE_FILE.exists():
        return
    try:
        data = json.loads(_DECISION_CACHE_FILE.read_text(encoding="utf-8"))
        now = time.time()
        restored = 0
        with _decision_cache_lock:
            for key, entry in data.items():
                try:
                    cached_at = float(entry.get("cached_at", 0))
                    decision = _deserialize_operator_decision(entry.get("decision", {}))
                except Exception as exc:
                    logger.debug("Skipping unreadable operator cache entry %s: %s", key, exc)
                    continue
                if not _decision_cache_is_fresh(decision, cached_at, now=now):
                    continue
                _decision_cache[key] = (decision, cached_at)
                restored += 1
        if restored:
            logger.info("Restored %d operator decision cache entries from disk", restored)
    except Exception as exc:
        logger.debug("Failed to load operator decision cache from disk: %s", exc)


def _load_gemini_pick_cache_file() -> dict:
    if not _GEMINI_PICK_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(_GEMINI_PICK_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load Gemini pick cache from disk: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _get_cached_gemini_pick(
    cache_key: str,
    *,
    event_date: str = "",
    event_title: str = "",
) -> dict | None:
    with _gemini_pick_cache_lock:
        data = _load_gemini_pick_cache_file()
    entry = data.get(cache_key)
    if not isinstance(entry, dict):
        return None

    response = entry.get("response")
    if not isinstance(response, dict):
        return None

    try:
        cached_at = float(entry.get("cached_at", 0))
    except (TypeError, ValueError):
        return None

    entry_event_date = str(entry.get("event_date") or event_date or "")
    entry_event_title = str(entry.get("event_title") or event_title or "")
    if not _event_cache_is_fresh(
        event_date=entry_event_date,
        event_title=entry_event_title,
        cached_at=cached_at,
    ):
        return None

    return dict(response)


def _save_gemini_pick_cache_entry(
    cache_key: str,
    response: dict,
    *,
    event_date: str = "",
    event_title: str = "",
) -> None:
    try:
        with _gemini_pick_cache_lock:
            data = _load_gemini_pick_cache_file()
            data[cache_key] = {
                "response": response,
                "cached_at": time.time(),
                "event_date": event_date,
                "event_title": event_title,
            }
            _GEMINI_PICK_CACHE_FILE.write_text(
                json.dumps(data, default=str),
                encoding="utf-8",
            )
    except Exception as exc:
        logger.debug("Failed to persist Gemini pick cache to disk: %s", exc)


def _load_gemini_research_cache_file() -> dict:
    if not _GEMINI_RESEARCH_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(_GEMINI_RESEARCH_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load Gemini research cache from disk: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _research_cache_is_fresh(
    cached_at: float,
    *,
    now: float | None = None,
) -> bool:
    current = time.time() if now is None else now
    return (current - cached_at) < GEMINI_RESEARCH_CACHE_TTL_SECONDS


def _get_cached_gemini_research(cache_key: str) -> dict | None:
    with _gemini_research_cache_lock:
        data = _load_gemini_research_cache_file()
    entry = data.get(cache_key)
    if not isinstance(entry, dict):
        return None

    response = entry.get("response")
    if not isinstance(response, dict):
        return None

    try:
        cached_at = float(entry.get("cached_at", 0))
    except (TypeError, ValueError):
        return None

    if not _research_cache_is_fresh(cached_at):
        return None

    return dict(response)


def _save_gemini_research_cache_entry(cache_key: str, response: dict) -> None:
    if GEMINI_RESEARCH_CACHE_TTL_SECONDS <= 0:
        return
    try:
        with _gemini_research_cache_lock:
            data = _load_gemini_research_cache_file()
            data[cache_key] = {
                "response": response,
                "cached_at": time.time(),
            }
            _GEMINI_RESEARCH_CACHE_FILE.write_text(
                json.dumps(data, default=str),
                encoding="utf-8",
            )
    except Exception as exc:
        logger.debug("Failed to persist Gemini research cache to disk: %s", exc)


def _decision_lock_path(cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return _DECISION_LOCK_DIR / f"{digest}.lock"


def _acquire_process_lock(cache_key: str) -> tuple[int | None, Path]:
    lock_path = _decision_lock_path(cache_key)
    deadline = time.time() + max(_PROCESS_LOCK_TIMEOUT_SECONDS, 0.0)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = f"{os.getpid()}|{time.time()}|{cache_key}".encode("utf-8")
            os.write(fd, payload)
            return fd, lock_path
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                age = 0.0
            if age >= _PROCESS_LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            if time.time() >= deadline:
                logger.warning(
                    "Timed out waiting for operator cross-process lock on %s; proceeding without it",
                    cache_key,
                )
                return None, lock_path
            time.sleep(0.2)


def _release_process_lock(fd: int | None, lock_path: Path) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


# Load persisted cache from disk at import time so process restarts don't
# lose cached decisions (the most common cause of duplicate API calls).
_load_decision_cache_from_disk()


# ---------------------------------------------------------------------------
# 1. Recency Context
# ---------------------------------------------------------------------------

def _coerce_calendar_timestamp(value: object) -> pd.Timestamp | None:
    """Parse a timestamp-like value into a UTC-normalized pandas Timestamp."""
    text = str(value or "").strip()
    if not text:
        return None

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _format_calendar_date(value: pd.Timestamp) -> str:
    return value.strftime("%B %d, %Y")


def _format_layoff_summary(
    fighter_name: str,
    days: float,
    *,
    event_ts: pd.Timestamp | None = None,
) -> str:
    day_count = int(round(days))
    if event_ts is None:
        return (
            f"{fighter_name} returning after {day_count} days layoff "
            f"({day_count / 365:.1f} years)"
        )

    last_fight_ts = event_ts - pd.Timedelta(days=day_count)
    return (
        f"{fighter_name} last fought on {_format_calendar_date(last_fight_ts)} "
        f"and returns on {_format_calendar_date(event_ts)} after {day_count} days "
        f"({day_count / 365:.1f} years)"
    )


def _build_prompt_date_anchor(
    features: dict,
    fighter_a: str,
    fighter_b: str,
    *,
    event_date: str = "",
) -> list[str]:
    event_ts = _coerce_calendar_timestamp(event_date)
    if event_ts is None:
        return []

    lines = [
        f"Scheduled fight date: {_format_calendar_date(event_ts)}.",
        "Treat layoff math as anchored to the scheduled fight date, not to today's date.",
    ]
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        days = features.get(f"{side}_days_since_last_fight")
        if days is None:
            continue
        try:
            days = float(days)
        except (TypeError, ValueError):
            continue
        lines.append(_format_layoff_summary(name, days, event_ts=event_ts) + ".")
    return lines

def _check_recency_context(
    features: dict,
    fighter_a: str,
    fighter_b: str,
    event_date: str = "",
) -> list[str]:
    """Flag regime changes that rolling averages can't capture."""
    flags = []

    event_ts = _coerce_calendar_timestamp(event_date)

    # Long layoffs (2+ years)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        days_key = f"{side}_days_since_last_fight"
        days = features.get(days_key)
        if days is not None:
            try:
                days = float(days)
            except (TypeError, ValueError):
                continue
            layoff_summary = _format_layoff_summary(name, days, event_ts=event_ts)
            if days > 730:
                flags.append(
                    f"{layoff_summary} — long layoff risk"
                )
            elif days > 365:
                flags.append(layoff_summary)

    # Short-notice replacements
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        notice_key = f"{side}_is_short_notice"
        if features.get(notice_key):
            flags.append(f"{name} is a short-notice replacement")

    # Weight class changes (check if fighting at unusual weight)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        wc_change_key = f"{side}_weight_class_change"
        if features.get(wc_change_key):
            flags.append(f"{name} is fighting at a new weight class")

    # Debut fighters
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        debut_key = f"{side}_is_debut"
        num_fights_key = f"{side}_num_fights"
        is_debut = features.get(debut_key, False)
        num_fights = features.get(num_fights_key, 0)
        try:
            num_fights = int(num_fights) if num_fights is not None else 0
        except (TypeError, ValueError):
            num_fights = 0
        if is_debut or num_fights == 0:
            flags.append(f"{name} is making their UFC debut — limited data")

    return flags


# ---------------------------------------------------------------------------
# 2. Style Matchup Reasoning (extracted from feature vector)
# ---------------------------------------------------------------------------

def _analyze_matchup_from_features(
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> str:
    """Build a matchup narrative from the feature vector for LLM synthesis."""
    lines = []

    def _get(key, default=None):
        val = features.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _fmt_pct(val) -> str:
        if val is None:
            return "unknown"
        try:
            val = float(val)
        except (TypeError, ValueError):
            return "unknown"
        if 0.0 <= val <= 1.0:
            val *= 100.0
        return f"{val:.0f}%"

    # Striking differential
    a_slpm = _get("a_roll_slpm")
    b_slpm = _get("b_roll_slpm")
    a_str_acc = _get("a_roll_str_acc")
    b_str_acc = _get("b_roll_str_acc")
    if a_slpm and b_slpm:
        acc_a = _fmt_pct(a_str_acc)
        acc_b = _fmt_pct(b_str_acc)
        lines.append(
            f"Striking output: {fighter_a} {a_slpm:.1f} SLpM "
            f"({acc_a} acc) vs {fighter_b} {b_slpm:.1f} SLpM "
            f"({acc_b} acc)"
        )

    # Grappling differential
    a_td_avg = _get("a_roll_td_avg")
    b_td_avg = _get("b_roll_td_avg")
    a_td_acc = _get("a_roll_td_acc")
    b_td_acc = _get("b_roll_td_acc")
    a_td_def = _get("a_roll_td_def")
    b_td_def = _get("b_roll_td_def")
    if a_td_avg and b_td_avg:
        td_acc_a = _fmt_pct(a_td_acc)
        td_acc_b = _fmt_pct(b_td_acc)
        td_def_a = _fmt_pct(a_td_def)
        td_def_b = _fmt_pct(b_td_def)
        lines.append(
            f"Takedowns: {fighter_a} {a_td_avg:.1f}/fight "
            f"({td_acc_a} acc, {td_def_a} def) vs "
            f"{fighter_b} {b_td_avg:.1f}/fight "
            f"({td_acc_b} acc, {td_def_b} def)"
        )

    # Wrestler vs striker mismatch (td_def is in 0-100 range)
    if a_td_avg and b_td_avg:
        if a_td_avg > 3.0 and b_td_def is not None and b_td_def < 55:
            lines.append(
                f"MISMATCH: {fighter_a} is an active wrestler vs "
                f"{fighter_b}'s weak TDD ({b_td_def:.0f}%)"
            )
        if b_td_avg > 3.0 and a_td_def is not None and a_td_def < 55:
            lines.append(
                f"MISMATCH: {fighter_b} is an active wrestler vs "
                f"{fighter_a}'s weak TDD ({a_td_def:.0f}%)"
            )

    # Stance matchup
    a_stance = features.get("a_stance", "")
    b_stance = features.get("b_stance", "")
    if a_stance and b_stance and a_stance != b_stance:
        lines.append(f"Stance matchup: {fighter_a} ({a_stance}) vs {fighter_b} ({b_stance})")

    # Win method profiles (rates are in 0-100 range)
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        ko_rate = _get(f"{side}_ko_rate")
        sub_rate = _get(f"{side}_sub_rate")
        dec_rate = _get(f"{side}_dec_rate")
        if ko_rate is not None and sub_rate is not None:
            lines.append(
                f"{name} finishes: KO {_fmt_pct(ko_rate)}, Sub {_fmt_pct(sub_rate)}, "
                f"Dec {_fmt_pct(dec_rate)}" if dec_rate is not None
                else f"{name} finishes: KO {_fmt_pct(ko_rate)}, Sub {_fmt_pct(sub_rate)}"
            )

    # Pre-UFC career context
    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        org_tier = features.get(f"{side}_pre_ufc_org_tier")
        pre_record = features.get(f"{side}_pre_ufc_record_depth")
        if org_tier is not None:
            lines.append(f"{name} pre-UFC org tier: {org_tier}")
        if pre_record is not None:
            lines.append(f"{name} pre-UFC record depth: {pre_record}")

    return "\n".join(lines) if lines else "Insufficient feature data for matchup analysis."


# ---------------------------------------------------------------------------
# 3. Motivation / Stakes Signals
# ---------------------------------------------------------------------------

def _check_motivation_signals(
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> list[str]:
    """Flag motivation-related signals from available features."""
    flags = []

    for side, name in [("a", fighter_a), ("b", fighter_b)]:
        # Losing streak — potential desperation or contract fight
        streak = features.get(f"{side}_lose_streak", 0)
        try:
            streak = int(streak) if streak is not None else 0
        except (TypeError, ValueError):
            streak = 0
        if streak >= 3:
            flags.append(
                f"{name} is on a {streak}-fight losing streak — "
                "potential contract fight / must-win"
            )
        elif streak == 2:
            flags.append(
                f"{name} has lost 2 straight — possible urgency"
            )

        # Win streak — riding momentum
        w_streak = features.get(f"{side}_current_win_streak", 0)
        try:
            w_streak = int(w_streak) if w_streak is not None else 0
        except (TypeError, ValueError):
            w_streak = 0
        if w_streak >= 5:
            flags.append(
                f"{name} is on a {w_streak}-fight win streak — "
                "high momentum, likely motivated"
            )

        # Age concerns
        age = features.get(f"{side}_age")
        if age is not None:
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = None
            if age is not None and age >= 38:
                flags.append(
                    f"{name} is {age:.0f} years old — "
                    "potential age/retirement factor"
                )

    return flags


# ---------------------------------------------------------------------------
# 4. Historical Model Blind Spots
# ---------------------------------------------------------------------------

def load_blind_spots() -> list[dict]:
    """Load known model failure patterns from disk."""
    if not BLIND_SPOTS_PATH.exists():
        return []
    try:
        with open(BLIND_SPOTS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load blind spots: %s", exc)
        return []


def save_blind_spots(blind_spots: list[dict]) -> None:
    """Persist blind spot patterns to disk."""
    with open(BLIND_SPOTS_PATH, "w") as f:
        json.dump(blind_spots, f, indent=2)


def _check_blind_spots(
    features: dict,
    fighter_a: str,
    fighter_b: str,
    model_prob_a: float,
    market_prob_a: float,
) -> list[str]:
    """Check if the current fight matches any known model blind spots."""
    blind_spots = load_blind_spots()
    if not blind_spots:
        return []

    matches = []
    for spot in blind_spots:
        pattern = spot.get("pattern", {})
        matched = True

        for key, condition in pattern.items():
            feature_val = features.get(key)
            if feature_val is None:
                matched = False
                break

            if isinstance(condition, dict):
                # Threshold conditions: {"op": "gt", "value": 3.0}
                op = condition.get("op", "eq")
                threshold = condition.get("value")
                try:
                    feature_val = float(feature_val)
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    matched = False
                    break

                if op == "gt" and not (feature_val > threshold):
                    matched = False
                elif op == "lt" and not (feature_val < threshold):
                    matched = False
                elif op == "gte" and not (feature_val >= threshold):
                    matched = False
                elif op == "lte" and not (feature_val <= threshold):
                    matched = False
                elif op == "eq" and feature_val != threshold:
                    matched = False
            else:
                # Plain value — equality check (supports strings and numbers)
                try:
                    if float(feature_val) != float(condition):
                        matched = False
                except (TypeError, ValueError):
                    if str(feature_val) != str(condition):
                        matched = False

        if matched:
            matches.append(
                f"Blind spot match: {spot.get('description', 'unknown pattern')} "
                f"(historical accuracy: {spot.get('accuracy', 'N/A')})"
            )

    return matches


# ---------------------------------------------------------------------------
# 6. Correlated Exposure Check
# ---------------------------------------------------------------------------

def _check_correlated_exposure(
    event_title: str,
    existing_bets: list[dict],
    event_date: str = "",
) -> str:
    """Flag concentration risk when multiple bets are on the same event.

    Matches on event_title OR event_date — each compared against its own
    field on the existing bet, not cross-compared.
    """
    if not existing_bets:
        return ""

    # Normalise the candidate event date to YYYY-MM-DD for comparison.
    candidate_date = _normalize_event_date(event_date) or _normalize_event_date(event_title)
    candidate_title = (event_title or "").strip()

    same_event_bets = []
    for b in existing_bets:
        bet_title = (b.get("event_title") or "").strip()
        bet_date = _normalize_event_date(b.get("event_date") or b.get("market_event_date"))
        if candidate_title and bet_title and candidate_title == bet_title:
            same_event_bets.append(b)
        elif candidate_date and bet_date and candidate_date == bet_date:
            same_event_bets.append(b)
        # Don't double-count if both matched
    # Deduplicate (a bet could match on both title and date)
    same_event_bets = list({id(b): b for b in same_event_bets}.values())

    count = len(same_event_bets)
    if count >= MAX_BETS_PER_EVENT:
        return (
            f"CONCENTRATION RISK: Already {count} bets on this event. "
            f"One bad judging night or doctor stoppage affects all. "
            f"Consider reducing position size."
        )
    elif count >= 2:
        return (
            f"Moderate exposure: {count} existing bets on this event. "
            f"Adding another increases correlated risk."
        )
    return ""


# ---------------------------------------------------------------------------
# Research aggregator
# ---------------------------------------------------------------------------

def run_research_pipeline(
    *,
    features: dict,
    fighter_a: str,
    fighter_b: str,
    model_prob_a: float,
    market_prob_a: float,
    event_title: str = "",
    event_date: str = "",
    existing_bets: list[dict] | None = None,
) -> ResearchFindings:
    """Run all research layers and aggregate findings."""
    findings = ResearchFindings()

    # 1. Recency context
    findings.recency_flags = _check_recency_context(
        features,
        fighter_a,
        fighter_b,
        event_date=event_date,
    )

    # 2. Matchup analysis
    findings.matchup_analysis = _analyze_matchup_from_features(
        features, fighter_a, fighter_b
    )

    # 3. Motivation signals
    findings.motivation_flags = _check_motivation_signals(
        features, fighter_a, fighter_b
    )

    # 4. Blind spot matching
    findings.blind_spot_matches = _check_blind_spots(
        features, fighter_a, fighter_b, model_prob_a, market_prob_a
    )

    # 6. Correlated exposure
    findings.exposure_warning = _check_correlated_exposure(
        event_title, existing_bets or [], event_date=event_date,
    )

    return findings


# ---------------------------------------------------------------------------
# LLM Synthesis — the "brain"
# ---------------------------------------------------------------------------

def _build_grounded_research_system_prompt() -> str:
    """System prompt for the grounded research stage."""
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""\
You are an expert MMA fight researcher preparing a compact evidence bundle for \
an UPCOMING UFC matchup.

TODAY'S DATE: {today}. You have access to WEB SEARCH and you must use it. Your \
job in this stage is research only, not final machine-formatted decision output.

Search for:
1. Both fighters' current records and recent form
2. The scheduled matchup and whether it is still upcoming
3. Each fighter's last 3 fights: opponent quality, result, method, and what the fight showed stylistically
4. Level of competition: UFC/ranked opponents, step-up or step-down spots, and quality of recent wins/losses
5. Style matchup: striking, wrestling, takedown defense, submission threat, durability, cardio, pace, and physicality
6. Recent news: injuries, layoffs, weight issues, camp changes, or cancellations
7. UFC rankings or clear class differences when available

Return plain text only with these exact headings:
FIGHT STATUS:
RESEARCH MEMO:
RECENT FORM:
LEVEL OF COMPETITION:
STYLE MATCHUP:
PATHS TO VICTORY:
CONCERNS FOR MODEL PICK:
VERIFIED RECORDS:
KEY FLAGS:

Do not return JSON. Do not use markdown fences. Keep the memo concise and factual.
"""


def _build_operator_synthesis_system_prompt() -> str:
    """System prompt for the operator schema-only synthesis stage."""
    return """\
You are an expert MMA fight analyst acting as a sanity check for a data-driven \
UFC betting model.

This call has NO web access. Use only the grounded research bundle provided in \
the prompt plus the local stats/features. The stats are verified model inputs. \
Do not claim the stats are fabricated or corrupted.

The model already handled price, edge, and whether the market number is good. \
Do not re-price the bet. Your job is to check whether the model's fighter read \
is missing fight-context information: level of competition, recent form, style \
matchup, paths to victory, injuries/news, or a stale matchup assumption.

Decision rules:
- Default to PASS. Only BLOCK when the grounded evidence shows something the \
model clearly cannot see.
- Do not block close fights just because the edge is small or uncertainty exists.
- Do not block just because the opponent has a viable path to victory, a style \
advantage in one phase, or normal MMA variance. Those belong in \
model_read_concerns while still returning PASS.
- Richer research should improve explanation quality, not lower the BLOCK \
threshold. If evidence is mixed, speculative, or already represented by the \
local features, return PASS with concerns.
- BLOCK only for high-confidence, material fight-context evidence that clearly \
contradicts the model's fighter read and is not already represented in the \
local stats/features.
- Explain why the model's read might be right and why it might be wrong before \
choosing PASS or BLOCK.
- Copy the exact stat-reference values into stats_confirmed. Do not substitute \
numbers from the research memo.

Return JSON that matches the provided response schema.
"""


def _build_standalone_pick_synthesis_system_prompt() -> str:
    """System prompt for the standalone pick schema-only synthesis stage."""
    return """\
You are an expert MMA fight analyst making a standalone outright winner pick.

This call has NO web access. Use only the grounded research bundle provided in \
the prompt. If the research says the fight is completed, cancelled, or cannot be \
verified as upcoming, return pick as null.

Return JSON that matches the provided response schema.
"""


def _build_grounded_research_request(
    *,
    fighter_a: str,
    fighter_b: str,
    model_pick: str = "",
    weight_class: str = "",
    event_date: str = "",
    event_title: str = "",
) -> str:
    event_ts = _coerce_calendar_timestamp(event_date)
    fight_label = f"{fighter_a} vs {fighter_b}"
    if weight_class:
        fight_label += f" ({weight_class})"

    lines = [f"Fight: {fight_label}"]
    if model_pick:
        lines.append(f"Model pick to scrutinize: {model_pick}")
    if event_title:
        lines.append(f"Event: {event_title}")
    if event_ts is not None:
        lines.append(f"Scheduled date: {_format_calendar_date(event_ts)}")
    elif event_date:
        lines.append(f"Scheduled date: {event_date}")

    lines.extend(
        [
            "",
            "Research the matchup above with Google Search grounding.",
            "Return plain text using these exact headings:",
            "FIGHT STATUS:",
            "RESEARCH MEMO:",
            "RECENT FORM:",
            "LEVEL OF COMPETITION:",
            "STYLE MATCHUP:",
            "PATHS TO VICTORY:",
            "CONCERNS FOR MODEL PICK:",
            "VERIFIED RECORDS:",
            "KEY FLAGS:",
            "",
            "Under FIGHT STATUS, write exactly one of: upcoming, completed, cancelled, unverified.",
            "Under RESEARCH MEMO, write 3-5 concise sentences with the overall fight context and any meaningful news.",
            "Under RECENT FORM, summarize each fighter's last 3 fights: opponent, result, method, opponent quality/ranking if known, and what the fight showed.",
            "Under LEVEL OF COMPETITION, compare recent opponent quality, UFC/ranked experience, step-up/step-down spots, and whether records are inflated by weaker opposition.",
            "Under STYLE MATCHUP, compare striking, wrestling, takedown defense, submission threat, durability/chin, cardio/pace, size/reach, and physicality.",
            "Under PATHS TO VICTORY, give each fighter's cleanest realistic path to winning this matchup.",
            "Under CONCERNS FOR MODEL PICK, focus only on reasons the model's preferred fighter read might be stale, incomplete, or contradicted by fight context. If no model pick is listed above, cover concerns for both fighters. Do not evaluate market price or edge.",
            "Under VERIFIED RECORDS, include these exact lines:",
            f"fighter_a: <record for {fighter_a} or unknown>",
            f"fighter_b: <record for {fighter_b} or unknown>",
            f"fighter_a_ranking: <ranking or unranked or unknown for {fighter_a}>",
            f"fighter_b_ranking: <ranking or unranked or unknown for {fighter_b}>",
            "source: <main source used for records/rankings>",
            "Under KEY FLAGS, provide short bullet points for concrete risks or leave a single bullet of `- none`.",
        ]
    )
    return "\n".join(lines)


def _fval(features: dict, key: str, fmt: str = ".1f") -> str:
    """Format a feature value, returning 'N/A' for missing data."""
    val = features.get(key)
    if val is None or str(val) in ("", "nan", "None"):
        return "N/A"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return "N/A"


def _fpct(features: dict, key: str) -> str:
    """Format a feature value as a percentage string."""
    val = features.get(key)
    if val is None or str(val) in ("", "nan", "None"):
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if 0.0 <= val <= 1.0:
        val *= 100.0
    return f"{val:.0f}"


def _build_fighter_narrative(features: dict, prefix: str, name: str) -> str:
    """Build a plain-English summary of what the model sees for one fighter."""
    parts = []

    # Experience
    num_fights = _fval(features, f"{prefix}_num_fights", ".0f")
    if num_fights != "N/A":
        parts.append(f"{num_fights} UFC fights on record")

    # Streaks — only mention if notable
    w_streak = features.get(f"{prefix}_current_win_streak")
    l_streak = features.get(f"{prefix}_lose_streak")
    try:
        w = int(float(w_streak)) if w_streak is not None else 0
    except (TypeError, ValueError):
        w = 0
    try:
        l = int(float(l_streak)) if l_streak is not None else 0
    except (TypeError, ValueError):
        l = 0
    if w >= 3:
        parts.append(f"on a {w}-fight win streak")
    elif l >= 2:
        parts.append(f"on a {l}-fight losing streak")

    # Striking — characterise the style, not the numbers
    slpm = _fval(features, f"{prefix}_roll_slpm")
    str_acc = _fpct(features, f"{prefix}_roll_str_acc")
    if slpm != "N/A":
        try:
            slpm_f = float(slpm)
            vol = "high-volume" if slpm_f >= 5.0 else "moderate-volume" if slpm_f >= 3.0 else "low-volume"
            parts.append(f"{vol} striker ({slpm} SLpM, {str_acc}% acc)")
        except (TypeError, ValueError):
            pass

    # Grappling — characterise the threat level
    td_avg = _fval(features, f"{prefix}_roll_td_avg")
    td_acc = _fpct(features, f"{prefix}_roll_td_acc")
    td_def = _fpct(features, f"{prefix}_roll_td_def")
    if td_avg != "N/A":
        try:
            td_f = float(td_avg)
            if td_f >= 3.0:
                parts.append(f"active wrestler ({td_avg} TD/fight, {td_acc}% acc)")
            elif td_f >= 1.0:
                parts.append(f"moderate grappling threat ({td_avg} TD/fight)")
            else:
                parts.append(f"primarily a striker on the feet ({td_avg} TD/fight)")
        except (TypeError, ValueError):
            pass
    if td_def != "N/A":
        try:
            td_def_f = float(td_def)
            if td_def_f < 55:
                parts.append(f"vulnerable to takedowns ({td_def}% TDD)")
            elif td_def_f >= 85:
                parts.append(f"strong takedown defense ({td_def}% TDD)")
        except (TypeError, ValueError):
            pass

    # Age — only if notable
    age = _fval(features, f"{prefix}_age", ".0f")
    if age != "N/A":
        try:
            age_f = float(age)
            if age_f >= 37:
                parts.append(f"age {age} (potential decline factor)")
            elif age_f <= 24:
                parts.append(f"age {age} (young, still developing)")
        except (TypeError, ValueError):
            pass

    # Layoff
    days = features.get(f"{prefix}_days_since_last_fight")
    if days is not None:
        try:
            d = int(float(days))
            if d > 365:
                parts.append(f"hasn't fought in {d} days ({d / 365:.1f} years)")
        except (TypeError, ValueError):
            pass

    if not parts:
        return f"**{name}:** Limited data available."
    return f"**{name}:** " + ". ".join(p.capitalize() if i == 0 else p for i, p in enumerate(parts)) + "."


def _build_model_narrative(features: dict, fighter_a: str, fighter_b: str) -> str:
    """Build a combined narrative of both fighters from the model's perspective."""
    lines = [
        _build_fighter_narrative(features, "a", fighter_a),
        _build_fighter_narrative(features, "b", fighter_b),
    ]

    # Key matchup note — only flag clear mismatches
    a_td_avg = features.get("a_roll_td_avg")
    b_td_def = features.get("b_roll_td_def")
    b_td_avg = features.get("b_roll_td_avg")
    a_td_def = features.get("a_roll_td_def")
    try:
        if (a_td_avg and b_td_def and
                float(a_td_avg) > 3.0 and float(b_td_def) < 55):
            lines.append(
                f"**Key mismatch:** The model sees {fighter_a} as an active "
                f"wrestler against {fighter_b}'s weak takedown defense."
            )
        elif (b_td_avg and a_td_def and
                float(b_td_avg) > 3.0 and float(a_td_def) < 55):
            lines.append(
                f"**Key mismatch:** The model sees {fighter_b} as an active "
                f"wrestler against {fighter_a}'s weak takedown defense."
            )
    except (TypeError, ValueError):
        pass

    return "\n\n".join(lines)


def _build_stat_reference(features: dict, fighter_a: str, fighter_b: str) -> str:
    """Compact stat block for the stats_confirmed echo check."""
    lines = []
    for prefix, name in [("a", fighter_a), ("b", fighter_b)]:
        str_acc = _fpct(features, f"{prefix}_roll_str_acc")
        td_acc = _fpct(features, f"{prefix}_roll_td_acc")
        td_def = _fpct(features, f"{prefix}_roll_td_def")
        lines.append(f"{name}: str_acc={str_acc}%, td_acc={td_acc}%, td_def={td_def}%")
    return "\n".join(lines)


def _build_model_matchup_signals(features: dict, fighter_a: str, fighter_b: str) -> str:
    """Highlight fight-context features without asking the operator to re-price."""
    lines = []

    def _pair(label: str, a_key: str, b_key: str, *, fmt: str = ".2f") -> None:
        a_val = _fval(features, a_key, fmt)
        b_val = _fval(features, b_key, fmt)
        if a_val == "N/A" and b_val == "N/A":
            return
        lines.append(f"{label}: {fighter_a} {a_val} vs {fighter_b} {b_val}")

    _pair("UFC experience", "a_num_fights", "b_num_fights", fmt=".0f")
    _pair("Recent opponent strength", "a_opp_strength", "b_opp_strength")
    _pair("Weight-class rank feature", "a_wc_rank_feat", "b_wc_rank_feat", fmt=".0f")
    _pair("Pound-for-pound rank feature", "a_pfp_rank_feat", "b_pfp_rank_feat", fmt=".0f")
    _pair("Striker-edge interaction", "a_striker_edge", "b_striker_edge")
    _pair("Grappler-edge interaction", "a_grappler_edge", "b_grappler_edge")
    _pair("Pre-UFC best org tier", "a_pre_ufc_org_tier_best", "b_pre_ufc_org_tier_best", fmt=".0f")
    _pair("Pre-UFC total fights", "a_pre_ufc_total_fights", "b_pre_ufc_total_fights", fmt=".0f")
    _pair("Fight pace", "a_fight_pace", "b_fight_pace")
    _pair("Control efficiency", "a_ctrl_efficiency", "b_ctrl_efficiency")

    for label, key in [
        ("Opponent strength differential", "diff_opp_strength"),
        ("Striker-edge differential", "diff_striker_edge"),
        ("Grappler-edge differential", "diff_grappler_edge"),
        ("Ranking differential", "diff_wc_rank"),
        ("UFC experience differential", "diff_num_fights"),
        ("Pre-UFC org-tier differential", "diff_pre_ufc_org_tier_best"),
    ]:
        value = _fval(features, key, ".2f")
        if value != "N/A":
            lines.append(f"{label}: {value}")

    return "\n".join(f"- {line}" for line in lines) if lines else "- No additional model matchup signals available."


def _build_synthesis_prompt(
    *,
    fighter_a: str,
    fighter_b: str,
    bet_on: str,
    bet_side: str,
    model_prob: float,
    market_prob: float,
    blended_prob: float,
    edge: float,
    features: dict,
    findings: ResearchFindings,
    weight_class: str = "",
    event_date: str = "",
) -> str:
    """Build the local-only context block used by the synthesis stage."""
    sections = []

    wc_label = f" ({weight_class})" if weight_class else ""
    event_ts = _coerce_calendar_timestamp(event_date)
    date_label = (
        f" — scheduled for {_format_calendar_date(event_ts)}"
        if event_ts is not None
        else ""
    )
    sections.append(f"## Fight: {fighter_a} vs {fighter_b}{wc_label}{date_label}")
    sections.append(
        f"The model wants to bet on **{bet_on}**.\n"
        f"- Model probability: {model_prob:.1%}\n"
        f"- Market probability: {market_prob:.1%}\n"
        f"- Blended probability: {blended_prob:.1%}\n"
        f"- Edge: {edge:.1%}"
    )

    date_anchor = _build_prompt_date_anchor(
        features,
        fighter_a,
        fighter_b,
        event_date=event_date,
    )
    if date_anchor:
        sections.append("## Date Anchor")
        sections.extend(date_anchor)

    # -- Model's view: narrative summary of what the model "sees" --
    sections.append("## What the Model Sees")
    sections.append(
        "The stats below are the rolling averages the model used to make its "
        "prediction. Your job is NOT to re-analyze these numbers — the model "
        "already did that. Your job is to check whether the model's picture of "
        "these fighters matches REALITY based on your web research."
    )
    sections.append(_build_model_narrative(features, fighter_a, fighter_b))

    sections.append("## Local Matchup Analysis")
    sections.append(findings.matchup_analysis or "No local matchup analysis available.")

    sections.append("## Model Matchup Signals")
    sections.append(_build_model_matchup_signals(features, fighter_a, fighter_b))

    # Compact stat reference (for stats_confirmed echo — do not remove)
    sections.append("## Stat Reference (for confirmation)")
    sections.append(_build_stat_reference(features, fighter_a, fighter_b))

    # Recency flags
    if findings.recency_flags:
        sections.append("## Context Flags")
        for flag in findings.recency_flags:
            sections.append(f"- {flag}")

    # Motivation signals
    if findings.motivation_flags:
        sections.append("## Motivation Signals")
        for flag in findings.motivation_flags:
            sections.append(f"- {flag}")

    # Blind spots
    if findings.blind_spot_matches:
        sections.append("## Known Model Blind Spots Matched")
        for match in findings.blind_spot_matches:
            sections.append(f"- {match}")

    # Exposure
    if findings.exposure_warning:
        sections.append("## Exposure Warning")
        sections.append(findings.exposure_warning)

    return "\n\n".join(sections)


_OPERATOR_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "stats_confirmed": {
            "type": "object",
            "properties": {
                "fighter_a_str_acc": {"type": "number"},
                "fighter_a_td_acc": {"type": "number"},
                "fighter_a_td_def": {"type": "number"},
                "fighter_b_str_acc": {"type": "number"},
                "fighter_b_td_acc": {"type": "number"},
                "fighter_b_td_def": {"type": "number"},
            },
            "required": [
                "fighter_a_str_acc",
                "fighter_a_td_acc",
                "fighter_a_td_def",
                "fighter_b_str_acc",
                "fighter_b_td_acc",
                "fighter_b_td_def",
            ],
        },
        "verified_records": {
            "type": "object",
            "properties": {
                "fighter_a": {"type": "string"},
                "fighter_b": {"type": "string"},
                "fighter_a_ranking": {"type": "string"},
                "fighter_b_ranking": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": [
                "fighter_a",
                "fighter_b",
                "fighter_a_ranking",
                "fighter_b_ranking",
                "source",
            ],
        },
        "verdict": {"type": "string", "enum": ["PASS", "BLOCK"]},
        "rationale": {"type": "string"},
        "fighter_assessment": {"type": "string"},
        "level_of_competition_summary": {"type": "string"},
        "style_matchup_summary": {"type": "string"},
        "path_to_victory_for_model_pick": {"type": "string"},
        "path_to_victory_for_opponent": {"type": "string"},
        "model_read_support": {
            "type": "array",
            "items": {"type": "string"},
        },
        "model_read_concerns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "stats_confirmed",
        "verified_records",
        "verdict",
        "rationale",
        "fighter_assessment",
        "level_of_competition_summary",
        "style_matchup_summary",
        "path_to_victory_for_model_pick",
        "path_to_victory_for_opponent",
        "model_read_support",
        "model_read_concerns",
        "risk_flags",
    ],
}


def _build_standalone_pick_schema(fighter_a: str, fighter_b: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "pick": {
                "anyOf": [
                    {"type": "string", "enum": [fighter_a, fighter_b]},
                    {"type": "null"},
                ]
            },
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "fighter_assessment": {"type": "string"},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "verified_records": {
                "type": "object",
                "properties": {
                    "fighter_a": {"type": "string"},
                    "fighter_b": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["fighter_a", "fighter_b", "source"],
            },
        },
        "required": [
            "pick",
            "confidence",
            "rationale",
            "fighter_assessment",
            "risk_flags",
            "verified_records",
        ],
    }


def _build_grounded_research_section(research_bundle: dict) -> str:
    sections = ["## Grounded Research Bundle"]
    sections.append(
        f"Fight status from grounded research: {research_bundle.get('fight_status') or 'unverified'}"
    )

    memo_text = str(research_bundle.get("memo_text") or "").strip()
    if memo_text:
        sections.append("### Research Memo")
        sections.append(memo_text)

    for key, heading in [
        ("recent_form", "Recent Form"),
        ("level_of_competition", "Level Of Competition"),
        ("style_matchup", "Style Matchup"),
        ("paths_to_victory", "Paths To Victory"),
        ("model_pick_concerns", "Concerns For Model Pick"),
    ]:
        value = str(research_bundle.get(key) or "").strip()
        if value:
            sections.append(f"### {heading}")
            sections.append(value)

    verified_records = dict(research_bundle.get("verified_records") or {})
    if any(str(value or "").strip() for value in verified_records.values()):
        sections.append("### Verified Records From Search")
        sections.append(
            "\n".join(
                [
                    f"fighter_a: {verified_records.get('fighter_a', '')}",
                    f"fighter_b: {verified_records.get('fighter_b', '')}",
                    f"fighter_a_ranking: {verified_records.get('fighter_a_ranking', '')}",
                    f"fighter_b_ranking: {verified_records.get('fighter_b_ranking', '')}",
                    f"source: {verified_records.get('source', '')}",
                ]
            )
        )

    key_flags = [str(flag or "").strip() for flag in research_bundle.get("key_flags") or [] if str(flag or "").strip()]
    if key_flags:
        sections.append("### Key Flags")
        sections.extend(f"- {flag}" for flag in key_flags)

    sources = [str(source or "").strip() for source in research_bundle.get("sources") or [] if str(source or "").strip()]
    if sources:
        sections.append("### Grounding Sources")
        sections.extend(f"- {source}" for source in sources)

    return "\n\n".join(sections)


def _build_operator_synthesis_prompt_from_research(base_prompt: str, research_bundle: dict) -> str:
    return "\n\n".join(
        [
            base_prompt,
            _build_grounded_research_section(research_bundle),
            "## Your Task\n"
            "Use ONLY the grounded research bundle above plus the local stats/features above. "
            "Do not browse the web or invent sources. Do not evaluate market price or edge; "
            "the model already handled that. Decide whether the model's fighter read should PASS "
            "or BLOCK based on fight context, level of competition, style matchup, and paths to victory. "
            "Normal opponent upside, viable opponent paths, and speculative concerns should be written as "
            "model_read_concerns but should still PASS unless they clearly invalidate the model's fighter read.",
        ]
    )


def _build_standalone_pick_prompt_from_research(
    *,
    fighter_a: str,
    fighter_b: str,
    weight_class: str = "",
    event_date: str = "",
    event_title: str = "",
    research_bundle: dict,
) -> str:
    event_ts = _coerce_calendar_timestamp(event_date)
    fight_label = f"{fighter_a} vs {fighter_b}"
    if weight_class:
        fight_label += f" ({weight_class})"

    sections = [f"## Fight: {fight_label}"]
    if event_title:
        sections.append(f"Event: {event_title}")
    if event_ts is not None:
        sections.append(f"Scheduled date: {_format_calendar_date(event_ts)}")
    elif event_date:
        sections.append(f"Scheduled date: {event_date}")

    sections.append(_build_grounded_research_section(research_bundle))
    sections.append(
        "## Your Task\n"
        f"Choose the more likely winner between {fighter_a} and {fighter_b} using only the grounded research bundle above. "
        "If the fight does not look verifiably upcoming, return pick as null."
    )
    return "\n\n".join(sections)


def _operator_passthrough_result(
    rationale: str,
    *,
    risk_flags: list[str],
    research_bundle: dict | None = None,
    stage_telemetry: dict | None = None,
    full_prompt: str = "",
) -> dict:
    return {
        "verdict": "PASS",
        "rationale": rationale,
        "fighter_assessment": "",
        "level_of_competition_summary": "",
        "style_matchup_summary": "",
        "path_to_victory_for_model_pick": "",
        "path_to_victory_for_opponent": "",
        "model_read_support": [],
        "model_read_concerns": [],
        "risk_flags": risk_flags,
        "_research_bundle": dict(research_bundle or {}),
        "_stage_telemetry": dict(stage_telemetry or {}),
        "_full_synthesis_prompt": full_prompt,
    }


def _normalize_operator_synthesis_payload(payload: dict) -> dict:
    normalized = dict(payload or {})
    normalized.setdefault("level_of_competition_summary", "")
    normalized.setdefault("style_matchup_summary", "")
    normalized.setdefault("path_to_victory_for_model_pick", "")
    normalized.setdefault("path_to_victory_for_opponent", "")
    for key in ("model_read_support", "model_read_concerns", "risk_flags"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [str(item) for item in value if str(item).strip()]
        elif value:
            normalized[key] = [str(value)]
        else:
            normalized[key] = []
    return normalized


def _call_llm_synthesis(
    prompt: str,
    *,
    research_prompt: str,
    research_cache_key: str,
) -> dict:
    """Run grounded research first, then schema-only operator synthesis."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured — operator passthrough")
        return _operator_passthrough_result(
            "Operator passthrough: GEMINI_API_KEY not configured",
            risk_flags=["llm_unavailable", "llm_not_configured"],
            stage_telemetry={
                "research": {"failure_class": "not_configured"},
                "synthesis": {},
            },
        )

    research_bundle, research_telemetry = _call_gemini_research(
        research_prompt,
        cache_key=research_cache_key,
        success_log_label="Gemini operator research",
    )
    if research_bundle is None:
        logger.warning("Gemini grounded research failed after retries — operator passthrough")
        return _operator_passthrough_result(
            "Operator passthrough: Gemini grounded research failed after retries",
            risk_flags=["llm_unavailable", "llm_failed_after_retries", "llm_research_failed"],
            research_bundle={
                "fight_status": "unverified",
                "memo_text": "",
                "verified_records": {},
                "key_flags": [],
                "sources": [],
                "failure_class": research_telemetry.get("failure_class", ""),
            },
            stage_telemetry={
                "research": research_telemetry,
                "synthesis": {},
            },
        )

    full_prompt = _build_operator_synthesis_prompt_from_research(prompt, research_bundle)
    result, synthesis_telemetry = _call_gemini_synthesis_from_research(
        full_prompt,
        system_instruction=_build_operator_synthesis_system_prompt(),
        response_json_schema=_OPERATOR_SYNTHESIS_SCHEMA,
        fallback_json_key="verdict",
        success_log_label="Gemini operator synthesis",
    )
    if result is None:
        logger.warning("Gemini synthesis failed after retries — operator passthrough PASS")
        return _operator_passthrough_result(
            "Operator passthrough: Gemini synthesis failed after retries",
            risk_flags=["llm_unavailable", "llm_failed_after_retries", "llm_synthesis_failed"],
            research_bundle=research_bundle,
            stage_telemetry={
                "research": research_telemetry,
                "synthesis": synthesis_telemetry,
            },
            full_prompt=full_prompt,
        )

    payload = _normalize_operator_synthesis_payload(dict(result))
    payload["_research_bundle"] = research_bundle
    payload["_stage_telemetry"] = {
        "research": research_telemetry,
        "synthesis": synthesis_telemetry,
    }
    payload["_full_synthesis_prompt"] = full_prompt
    return payload


def _gemini_field(obj, *names: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj.get(name)
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    for dumper_name in ("to_json_dict", "model_dump"):
        dumper = getattr(obj, dumper_name, None)
        if not callable(dumper):
            continue
        try:
            dumped = dumper()
        except Exception:
            continue
        if isinstance(dumped, dict):
            return _gemini_field(dumped, *names)
    return None


def _extract_gemini_grounding_sources(response) -> list[str]:
    candidates = _gemini_field(response, "candidates") or []
    if not isinstance(candidates, (list, tuple)):
        return []

    sources: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        grounding = _gemini_field(candidate, "grounding_metadata", "groundingMetadata")
        chunks = _gemini_field(grounding, "grounding_chunks", "groundingChunks") or []
        if not isinstance(chunks, (list, tuple)):
            continue

        for chunk in chunks:
            source_obj = (
                _gemini_field(chunk, "web")
                or _gemini_field(chunk, "retrieved_context", "retrievedContext")
                or chunk
            )
            uri = str(
                _gemini_field(source_obj, "uri", "url", "source_uri", "sourceUri")
                or ""
            ).strip()
            if uri and uri not in seen:
                seen.add(uri)
                sources.append(uri)
    return sources


def _parse_gemini_json_response(
    text: str,
    *,
    fallback_json_key: str,
) -> dict:
    decoder = json.JSONDecoder()
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", cleaned):
        try:
            candidate, _ = decoder.raw_decode(cleaned, idx=match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and fallback_json_key in candidate:
            return candidate

    raise json.JSONDecodeError("No JSON object found in response", cleaned, 0)


def _classify_gemini_failure(exc: Exception | None = None, *, text: str = "") -> str:
    message = str(exc or text or "").upper()
    if not message.strip():
        return "empty_response"
    if any(marker in message for marker in ("429", "RESOURCE_EXHAUSTED", "OVERLOAD", "RATE LIMIT")):
        return "overload"
    if any(marker in message for marker in ("408", "TIMEOUT", "TIMED OUT", "DEADLINE_EXCEEDED")):
        return "timeout"
    if "JSON" in message or "MALFORMED" in message:
        return "malformed_json"
    if "UNAVAILABLE" in message or "SERVER DISCONNECTED" in message:
        return "unavailable"
    if "SEARCH" in message and "FAILED" in message:
        return "search_failed"
    return "api_error"


def _parse_grounded_research_response(text: str) -> dict:
    normalized = str(text or "").replace("\r\n", "\n").strip()
    headings = {
        "FIGHT STATUS": "fight_status",
        "RESEARCH MEMO": "memo_text",
        "RECENT FORM": "recent_form",
        "LEVEL OF COMPETITION": "level_of_competition",
        "STYLE MATCHUP": "style_matchup",
        "PATHS TO VICTORY": "paths_to_victory",
        "CONCERNS FOR MODEL PICK": "model_pick_concerns",
        "VERIFIED RECORDS": "verified_records_raw",
        "KEY FLAGS": "key_flags_raw",
    }
    sections: dict[str, str] = {}
    current_key = ""
    buffer: list[str] = []

    def _commit() -> None:
        if current_key:
            sections[current_key] = "\n".join(buffer).strip()

    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        matched_heading = False
        for heading, section_key in headings.items():
            prefix = f"{heading}:"
            upper = stripped.upper()
            if upper == prefix:
                _commit()
                current_key = section_key
                buffer = []
                matched_heading = True
                break
            if upper.startswith(prefix):
                _commit()
                current_key = section_key
                inline_value = stripped[len(prefix):].strip()
                buffer = [inline_value] if inline_value else []
                matched_heading = True
                break
        if matched_heading:
            continue
        if current_key:
            buffer.append(raw_line)
    _commit()

    verified_records: dict[str, str] = {
        "fighter_a": "",
        "fighter_b": "",
        "fighter_a_ranking": "",
        "fighter_b_ranking": "",
        "source": "",
    }
    for line in sections.get("verified_records_raw", "").splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if ":" not in cleaned:
            continue
        key, value = cleaned.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in verified_records:
            verified_records[normalized_key] = value.strip()

    key_flags = [
        line.strip().lstrip("-").strip()
        for line in sections.get("key_flags_raw", "").splitlines()
        if line.strip()
    ]
    fight_status = (sections.get("fight_status") or "unverified").strip().lower()
    if not fight_status:
        fight_status = "unverified"

    memo_text = sections.get("memo_text", "").strip() or normalized
    return {
        "fight_status": fight_status,
        "memo_text": memo_text,
        "recent_form": sections.get("recent_form", "").strip(),
        "level_of_competition": sections.get("level_of_competition", "").strip(),
        "style_matchup": sections.get("style_matchup", "").strip(),
        "paths_to_victory": sections.get("paths_to_victory", "").strip(),
        "model_pick_concerns": sections.get("model_pick_concerns", "").strip(),
        "verified_records": verified_records,
        "key_flags": key_flags,
        "raw_text": normalized,
    }


def _call_gemini_stage(
    prompt: str,
    *,
    system_instruction: str,
    success_log_label: str,
    use_search: bool,
    parse_response: Callable[[str], object],
    require_sources: bool = False,
    response_mime_type: str | None = None,
    response_json_schema: dict | None = None,
    _max_retries: int | None = None,
) -> tuple[object | None, list[str], dict]:
    """Call Gemini with bounded retries, model fallback, and outage circuit breaking."""
    text = ""
    telemetry = {
        "models_attempted": [],
        "model_used": "",
        "fallback_reached": False,
        "search_enabled": use_search,
        "search_success": None if not use_search else False,
        "schema_mode": bool(response_json_schema),
        "schema_parse_success": None if response_json_schema is None else False,
        "failure_class": "",
        "grounding_retry_count": 0,
    }
    operation_label = _gemini_operation_label(success_log_label)
    blocked_until = _gemini_circuit_blocked_until()
    if blocked_until:
        logger.warning(
            "Gemini circuit open until %s — skipping %s",
            datetime.fromtimestamp(blocked_until, tz=timezone.utc).isoformat(),
            success_log_label,
        )
        telemetry["failure_class"] = "circuit_open"
        return None, [], telemetry

    try:
        client = _get_gemini_client(timeout_ms=_gemini_timeout_ms(use_search=use_search))
        model_chain = _configured_gemini_models()
        if not model_chain:
            logger.warning("No Gemini models configured")
            telemetry["failure_class"] = "not_configured"
            return None, [], telemetry

        last_exc: Exception | None = None
        saw_transient_error = False

        for model_idx, model_name in enumerate(model_chain):
            if model_name not in telemetry["models_attempted"]:
                telemetry["models_attempted"].append(model_name)
            attempts_for_model = _gemini_attempts_for_model(
                model_name,
                override=_max_retries,
            )
            grounding_retries_allowed = (
                GEMINI_PRIMARY_GROUNDING_RETRIES
                if use_search and require_sources and model_name == GEMINI_MODEL
                else 0
            )
            if grounding_retries_allowed and _max_retries is None:
                attempts_for_model = max(attempts_for_model, grounding_retries_allowed + 1)
            grounding_retry_count = 0
            for attempt in range(attempts_for_model):
                try:
                    request_prompt = (
                        _gemini_grounding_retry_prompt(prompt)
                        if grounding_retry_count
                        else prompt
                    )
                    config: dict[str, object] = {
                        "system_instruction": system_instruction,
                    }
                    thinking_config = _gemini_stage_thinking_config(model_name, use_search=use_search)
                    if thinking_config is not None:
                        config["thinking_config"] = thinking_config
                    elif not _is_gemini_3_model(model_name):
                        config["temperature"] = 0.3
                    if use_search:
                        config["tools"] = [{"google_search": {}}]
                    if response_mime_type:
                        config["response_mime_type"] = response_mime_type
                    if response_json_schema is not None:
                        config["response_json_schema"] = response_json_schema
                    response = client.models.generate_content(
                        model=model_name,
                        contents=request_prompt,
                        config=config,
                    )
                    text = str(getattr(response, "text", "") or "").strip()
                    sources = _extract_gemini_grounding_sources(response)
                    if use_search:
                        telemetry["search_success"] = bool(sources)

                    if not text:
                        telemetry["failure_class"] = "empty_response"
                        next_model = (
                            model_chain[model_idx + 1]
                            if model_idx + 1 < len(model_chain)
                            else None
                        )
                        if next_model:
                            logger.warning(
                                "Gemini %s returned an empty response on %s — trying fallback model %s",
                                operation_label,
                                model_name,
                                next_model,
                            )
                        else:
                            logger.warning(
                                "Gemini %s returned an empty response on %s",
                                operation_label,
                                model_name,
                            )
                        break

                    if require_sources and not sources:
                        telemetry["failure_class"] = "search_failed"
                        if (
                            grounding_retry_count < grounding_retries_allowed
                            and attempt + 1 < attempts_for_model
                        ):
                            grounding_retry_count += 1
                            telemetry["grounding_retry_count"] = (
                                int(telemetry.get("grounding_retry_count") or 0) + 1
                            )
                            logger.warning(
                                "Gemini %s returned no grounding sources on %s "
                                "(grounding retry %d/%d) — retrying same model "
                                "with stricter Google Search instructions",
                                operation_label,
                                model_name,
                                grounding_retry_count,
                                grounding_retries_allowed,
                            )
                            continue
                        next_model = (
                            model_chain[model_idx + 1]
                            if model_idx + 1 < len(model_chain)
                            else None
                        )
                        if next_model:
                            logger.warning(
                                "Gemini %s returned no grounding sources on %s — trying fallback model %s",
                                operation_label,
                                model_name,
                                next_model,
                            )
                        else:
                            logger.warning(
                                "Gemini %s returned no grounding sources on %s",
                                operation_label,
                                model_name,
                            )
                        break

                    try:
                        parsed = parse_response(text)
                    except json.JSONDecodeError as exc:
                        raw_preview = text[:300] if text else "(empty)"
                        next_model = (
                            model_chain[model_idx + 1]
                            if model_idx + 1 < len(model_chain)
                            else None
                        )
                        telemetry["failure_class"] = "malformed_json"
                        if next_model:
                            logger.warning(
                                "Gemini %s returned malformed JSON on %s — trying fallback model %s: %s — raw: %s",
                                operation_label,
                                model_name,
                                next_model,
                                exc,
                                raw_preview,
                            )
                            last_exc = exc
                            break

                        logger.warning(
                            "Gemini %s returned malformed JSON on %s: %s — raw: %s",
                            operation_label,
                            model_name,
                            exc,
                            raw_preview,
                        )
                        last_exc = exc
                        break
                    if response_json_schema is not None:
                        telemetry["schema_parse_success"] = True
                    telemetry["model_used"] = model_name
                    telemetry["fallback_reached"] = model_idx > 0
                    if sources:
                        logger.info(
                            "%s used %d web sources via %s: %s",
                            success_log_label,
                            len(sources),
                            model_name,
                            "; ".join(sources[:3]) + ("..." if len(sources) > 3 else ""),
                        )
                    elif model_idx > 0:
                        logger.info("%s succeeded via fallback model %s", success_log_label, model_name)
                    telemetry["failure_class"] = ""
                    _record_gemini_success()
                    return parsed, sources, telemetry
                except Exception as exc:
                    last_exc = exc
                    telemetry["failure_class"] = _classify_gemini_failure(exc)
                    is_transient = _is_gemini_transient_error(exc)
                    if is_transient:
                        saw_transient_error = True
                    has_retry = attempt + 1 < attempts_for_model
                    next_model = (
                        model_chain[model_idx + 1]
                        if model_idx + 1 < len(model_chain)
                        else None
                    )
                    if is_transient and has_retry:
                        wait = _gemini_retry_wait_seconds(attempt)
                        logger.warning(
                            "Gemini transient error on %s for %s (attempt %d/%d) — retrying in %.1fs: %s",
                            model_name,
                            success_log_label,
                            attempt + 1,
                            attempts_for_model,
                            wait,
                            exc,
                        )
                        time.sleep(wait)
                        continue

                    if next_model:
                        logger.warning(
                            "Gemini %s failed on %s after %d attempt(s) — trying fallback model %s: %s",
                            operation_label,
                            model_name,
                            attempts_for_model,
                            next_model,
                            exc,
                        )
                    else:
                        logger.warning(
                            "Gemini %s unavailable on %s after %d attempt(s): %s",
                            operation_label,
                            model_name,
                            attempts_for_model,
                            exc,
                        )
                    break

        if saw_transient_error and last_exc is not None:
            _record_gemini_transient_failure(str(last_exc))
        if not telemetry["failure_class"] and last_exc is not None:
            telemetry["failure_class"] = _classify_gemini_failure(last_exc)
        return None, [], telemetry

    except ImportError:
        logger.warning("google-genai package not installed")
        telemetry["failure_class"] = "not_installed"
        return None, [], telemetry
    except json.JSONDecodeError as exc:
        raw_preview = text[:300] if text else "(empty)"
        logger.warning("Failed to parse Gemini response as JSON: %s — raw: %s", exc, raw_preview)
        telemetry["failure_class"] = "malformed_json"
        return None, [], telemetry
    except Exception as exc:
        logger.warning("Gemini API error: %s", exc)
        telemetry["failure_class"] = _classify_gemini_failure(exc)
        return None, [], telemetry


def _call_gemini_research(
    prompt: str,
    *,
    cache_key: str,
    success_log_label: str,
    _max_retries: int | None = None,
) -> tuple[dict | None, dict]:
    """Call Gemini for grounded fight research and return a compact research bundle."""
    cached = _get_cached_gemini_research(cache_key)
    if cached is not None:
        telemetry = {
            "models_attempted": [str(cached.get("model_used") or "")] if cached.get("model_used") else [],
            "model_used": str(cached.get("model_used") or ""),
            "fallback_reached": False,
            "search_enabled": True,
            "search_success": bool(cached.get("sources")),
            "schema_mode": False,
            "schema_parse_success": None,
            "failure_class": "",
            "cached": True,
        }
        cached_bundle = dict(cached)
        cached_bundle["cached"] = True
        return cached_bundle, telemetry

    parsed, sources, telemetry = _call_gemini_stage(
        prompt,
        system_instruction=_build_grounded_research_system_prompt(),
        success_log_label=success_log_label,
        use_search=True,
        parse_response=_parse_grounded_research_response,
        require_sources=True,
        _max_retries=_max_retries,
    )
    telemetry["cached"] = False
    if parsed is None or not isinstance(parsed, dict):
        return None, telemetry

    research_bundle = dict(parsed)
    research_bundle["sources"] = sources
    research_bundle["model_used"] = telemetry.get("model_used", "")
    research_bundle["failure_class"] = ""
    research_bundle["cached"] = False
    _save_gemini_research_cache_entry(
        cache_key,
        {
            "fight_status": research_bundle.get("fight_status", ""),
            "memo_text": research_bundle.get("memo_text", ""),
            "recent_form": research_bundle.get("recent_form", ""),
            "level_of_competition": research_bundle.get("level_of_competition", ""),
            "style_matchup": research_bundle.get("style_matchup", ""),
            "paths_to_victory": research_bundle.get("paths_to_victory", ""),
            "model_pick_concerns": research_bundle.get("model_pick_concerns", ""),
            "verified_records": research_bundle.get("verified_records", {}),
            "key_flags": research_bundle.get("key_flags", []),
            "sources": research_bundle.get("sources", []),
            "model_used": research_bundle.get("model_used", ""),
        },
    )
    return research_bundle, telemetry


def _call_gemini_synthesis_from_research(
    prompt: str,
    *,
    system_instruction: str,
    response_json_schema: dict,
    fallback_json_key: str,
    success_log_label: str,
    _max_retries: int | None = None,
) -> tuple[dict | None, dict]:
    """Call Gemini for schema-only synthesis using an existing research bundle."""
    parsed, _sources, telemetry = _call_gemini_stage(
        prompt,
        system_instruction=system_instruction,
        success_log_label=success_log_label,
        use_search=False,
        parse_response=lambda text: _parse_gemini_json_response(
            text,
            fallback_json_key=fallback_json_key,
        ),
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
        _max_retries=_max_retries,
    )
    return parsed if isinstance(parsed, dict) else None, telemetry


_STATS_CONFIRMED_MAP = {
    "fighter_a_str_acc": "a_roll_str_acc",
    "fighter_a_td_acc": "a_roll_td_acc",
    "fighter_a_td_def": "a_roll_td_def",
    "fighter_b_str_acc": "b_roll_str_acc",
    "fighter_b_td_acc": "b_roll_td_acc",
    "fighter_b_td_def": "b_roll_td_def",
}

# Maximum allowed absolute difference between the LLM's echoed stat and the
# real feature value.  Accounts for rounding (profile shows "43%" for 43.15).
_STATS_TOLERANCE = 3.0


def _check_stats_confirmed(synthesis: dict, features: dict) -> list[str]:
    """Compare LLM-echoed stats against real features.

    Returns a list of mismatch descriptions.  Empty list means the LLM read
    the data correctly (or didn't return the field at all).
    """
    confirmed = synthesis.get("stats_confirmed")
    if not isinstance(confirmed, dict):
        return []
    mismatches = []
    for llm_key, feature_key in _STATS_CONFIRMED_MAP.items():
        llm_val = confirmed.get(llm_key)
        real_val = features.get(feature_key)
        if llm_val is None or real_val is None:
            continue
        try:
            llm_val = float(llm_val)
            real_val = float(real_val)
        except (TypeError, ValueError):
            continue
        if 0.0 <= llm_val <= 1.0 and real_val > 1.0:
            llm_val *= 100.0
        if abs(llm_val - real_val) > _STATS_TOLERANCE:
            mismatches.append(
                f"{llm_key}: LLM said {llm_val:.1f}, actual {real_val:.1f}"
            )
    return mismatches


def _build_correction_prompt(
    original_prompt: str,
    original_rationale: str,
    mismatches: list[str],
    features: dict,
    fighter_a: str,
    fighter_b: str,
) -> str:
    """Build a follow-up prompt that corrects the LLM with the real stats."""
    # Build a verified stats block from the real features.
    def _fmt(key: str) -> str:
        val = features.get(key)
        if val is None or str(val) in ("", "nan", "None"):
            return "N/A"
        try:
            v = float(val)
            return f"{v:.1f}%"
        except (TypeError, ValueError):
            return "N/A"

    stats_block = (
        f"## CORRECTION — Your previous response used WRONG stats\n\n"
        f"Your previous answer contained these errors:\n"
        + "\n".join(f"- {m}" for m in mismatches)
        + f"\n\nHere are the VERIFIED stats from the model's feature pipeline. "
        f"These are the real numbers. Use ONLY these:\n\n"
        f"**{fighter_a}:**\n"
        f"- Striking accuracy: {_fmt('a_roll_str_acc')}\n"
        f"- Takedown accuracy: {_fmt('a_roll_td_acc')}\n"
        f"- Takedown defense: {_fmt('a_roll_td_def')}\n"
        f"- SLpM: {_fmt('a_roll_slpm')}\n"
        f"- TD avg/fight: {_fmt('a_roll_td_avg')}\n\n"
        f"**{fighter_b}:**\n"
        f"- Striking accuracy: {_fmt('b_roll_str_acc')}\n"
        f"- Takedown accuracy: {_fmt('b_roll_td_acc')}\n"
        f"- Takedown defense: {_fmt('b_roll_td_def')}\n"
        f"- SLpM: {_fmt('b_roll_slpm')}\n"
        f"- TD avg/fight: {_fmt('b_roll_td_avg')}\n\n"
        f"Now re-evaluate this bet using the CORRECT stats above combined with "
        f"the grounded research bundle already provided. Your previous verdict was based on wrong data so "
        f"start fresh. Respond with the same JSON format."
    )

    return f"{original_prompt}\n\n{stats_block}"


def _guard_data_hallucination(
    synthesis: dict,
    features: dict,
    fighter_a: str,
    fighter_b: str,
    *,
    original_prompt: str = "",
    _retry: bool = False,
) -> dict:
    """Validate the LLM read real stats.  If it didn't, retry with corrections.

    If the LLM's echoed stats don't match reality, we re-call the LLM with the
    correct stats explicitly injected so it can make a properly informed
    decision.  We never auto-pass — the fight may genuinely deserve a BLOCK.
    """
    mismatches = _check_stats_confirmed(synthesis, features)

    if not mismatches:
        # Stats match (or weren't returned) — accept the verdict as-is.
        return synthesis

    if _retry:
        # Already retried once — accept whatever the retry returned but annotate.
        logger.warning(
            "Operator for %s vs %s still misread stats on retry — accepting "
            "verdict with annotation: %s",
            fighter_a,
            fighter_b,
            "; ".join(mismatches),
        )
        synthesis = dict(synthesis)
        synthesis.setdefault("risk_flags", []).append("stats_mismatch_after_retry")
        return synthesis

    # First attempt had wrong stats — retry with explicit corrections.
    detail_str = "; ".join(mismatches)
    logger.warning(
        "Operator for %s vs %s echoed wrong stats (%s) — retrying with "
        "corrected data",
        fighter_a,
        fighter_b,
        detail_str,
    )

    correction_prompt = _build_correction_prompt(
        original_prompt=original_prompt,
        original_rationale=synthesis.get("rationale", ""),
        mismatches=mismatches,
        features=features,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
    )
    research_bundle = dict(synthesis.get("_research_bundle") or {})
    stage_telemetry = dict(synthesis.get("_stage_telemetry") or {})
    retry_result, retry_telemetry = _call_gemini_synthesis_from_research(
        correction_prompt,
        system_instruction=_build_operator_synthesis_system_prompt(),
        response_json_schema=_OPERATOR_SYNTHESIS_SCHEMA,
        fallback_json_key="verdict",
        success_log_label="Gemini operator synthesis retry",
    )
    if retry_result is None:
        retry_synthesis = _operator_passthrough_result(
            "Operator passthrough: Gemini synthesis retry failed after retries",
            risk_flags=["llm_unavailable", "llm_failed_after_retries", "llm_synthesis_failed"],
            research_bundle=research_bundle,
            stage_telemetry={
                "research": stage_telemetry.get("research", {}),
                "synthesis": retry_telemetry,
            },
            full_prompt=correction_prompt,
        )
    else:
        retry_synthesis = _normalize_operator_synthesis_payload(dict(retry_result))
        retry_synthesis["_research_bundle"] = research_bundle
        retry_synthesis["_stage_telemetry"] = {
            "research": stage_telemetry.get("research", {}),
            "synthesis": retry_telemetry,
        }
        retry_synthesis["_full_synthesis_prompt"] = correction_prompt

    # Validate the retry too (but don't recurse further).
    retry_synthesis = _guard_data_hallucination(
        retry_synthesis,
        features,
        fighter_a,
        fighter_b,
        original_prompt=correction_prompt,
        _retry=True,
    )
    retry_synthesis = dict(retry_synthesis)
    retry_synthesis.setdefault("risk_flags", []).append("stats_corrected_retry")
    return retry_synthesis




# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------

def _log_decision(decision: OperatorDecision) -> None:
    """Append decision to the persistent audit log (JSONL — one record per line)."""
    try:
        with open(DECISION_LOG_PATH, "a") as f:
            f.write(json.dumps(asdict(decision), default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to log operator decision: %s", exc)


def load_decision_log() -> list[dict]:
    """Read all operator decisions from the JSONL audit log."""
    if not DECISION_LOG_PATH.exists():
        return []
    decisions = []
    with open(DECISION_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return decisions


def log_tracker_decision(record: dict) -> None:
    """Append a tracker decision/outcome record to the persistent JSONL log."""
    try:
        with open(TRACKER_DECISION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to log tracker decision: %s", exc)


def load_tracker_decision_log() -> list[dict]:
    """Read tracker decision/outcome records from the JSONL audit log."""
    if not TRACKER_DECISION_LOG_PATH.exists():
        return []
    records = []
    with open(TRACKER_DECISION_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _grounded_research_public_view(research_bundle: dict | None) -> dict:
    """Project a grounded research bundle into the stable shape persisted to logs.

    Shared by the operator decision log and the Gemini tracker pick so both
    surface identical research fields on the dashboard.
    """
    bundle = research_bundle or {}
    return {
        "fight_status": bundle.get("fight_status", ""),
        "memo_text": bundle.get("memo_text", ""),
        "recent_form": bundle.get("recent_form", ""),
        "level_of_competition": bundle.get("level_of_competition", ""),
        "style_matchup": bundle.get("style_matchup", ""),
        "paths_to_victory": bundle.get("paths_to_victory", ""),
        "model_pick_concerns": bundle.get("model_pick_concerns", ""),
        "verified_records": bundle.get("verified_records", {}),
        "key_flags": bundle.get("key_flags", []),
        "sources": bundle.get("sources", []),
        "model_used": bundle.get("model_used", ""),
        "cached": bool(bundle.get("cached")),
        "failure_class": bundle.get("failure_class", ""),
    }


def gemini_standalone_pick(
    *,
    fighter_a: str,
    fighter_b: str,
    weight_class: str = "",
    event_date: str = "",
    event_title: str = "",
) -> dict:
    """Return Gemini's standalone outright winner pick plus grounding sources."""
    cache_key = _fight_cache_key(
        fighter_a,
        fighter_b,
        event_date=event_date,
        event_title=event_title,
    )
    cached = _get_cached_gemini_pick(
        cache_key,
        event_date=event_date,
        event_title=event_title,
    )
    if cached is not None:
        cached_result = dict(cached)
        cached_result["cached"] = True
        return cached_result

    if not GEMINI_API_KEY:
        return {
            "pick": None,
            "confidence": 0.0,
            "rationale": "Gemini standalone pick unavailable: GEMINI_API_KEY not configured",
            "fighter_assessment": "",
            "risk_flags": ["llm_unavailable", "llm_not_configured"],
            "verified_records": {},
            "sources": [],
            "decision_key": cache_key,
            "cached": False,
        }

    research_prompt = _build_grounded_research_request(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        weight_class=weight_class,
        event_date=event_date,
        event_title=event_title,
    )
    research_bundle, research_telemetry = _call_gemini_research(
        research_prompt,
        cache_key=cache_key,
        success_log_label="Gemini standalone research",
    )
    if research_bundle is None:
        return {
            "pick": None,
            "confidence": 0.0,
            "rationale": "Gemini standalone pick failed during grounded research",
            "fighter_assessment": "",
            "risk_flags": ["llm_unavailable", "llm_failed_after_retries", "llm_research_failed"],
            "verified_records": {},
            "sources": [],
            "decision_key": cache_key,
            "cached": False,
            "stage_telemetry": {
                "research": research_telemetry,
                "synthesis": {},
            },
        }

    prompt = _build_standalone_pick_prompt_from_research(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        weight_class=weight_class,
        event_date=event_date,
        event_title=event_title,
        research_bundle=research_bundle,
    )
    result, synthesis_telemetry = _call_gemini_synthesis_from_research(
        prompt,
        system_instruction=_build_standalone_pick_synthesis_system_prompt(),
        response_json_schema=_build_standalone_pick_schema(fighter_a, fighter_b),
        fallback_json_key="pick",
        success_log_label="Gemini standalone pick",
    )
    if result is None:
        return {
            "pick": None,
            "confidence": 0.0,
            "rationale": "Gemini standalone pick failed during schema synthesis",
            "fighter_assessment": "",
            "risk_flags": ["llm_unavailable", "llm_failed_after_retries", "llm_synthesis_failed"],
            "verified_records": dict(research_bundle.get("verified_records") or {}),
            "sources": list(research_bundle.get("sources") or []),
            "grounded_research": _grounded_research_public_view(research_bundle),
            "decision_key": cache_key,
            "cached": False,
            "stage_telemetry": {
                "research": research_telemetry,
                "synthesis": synthesis_telemetry,
            },
        }

    payload = dict(result)
    raw_pick = str(payload.get("pick") or "").strip()
    if raw_pick:
        if same_person_name(raw_pick, fighter_a):
            payload["pick"] = fighter_a
        elif same_person_name(raw_pick, fighter_b):
            payload["pick"] = fighter_b

    payload.setdefault("confidence", 0.0)
    payload.setdefault("rationale", "")
    payload.setdefault("fighter_assessment", "")
    payload.setdefault("risk_flags", [])
    payload.setdefault("verified_records", dict(research_bundle.get("verified_records") or {}))
    payload["sources"] = list(research_bundle.get("sources") or [])
    payload["grounded_research"] = _grounded_research_public_view(research_bundle)
    payload["decision_key"] = cache_key
    payload["cached"] = False
    payload["stage_telemetry"] = {
        "research": research_telemetry,
        "synthesis": synthesis_telemetry,
    }

    _save_gemini_pick_cache_entry(
        cache_key,
        payload,
        event_date=event_date,
        event_title=event_title,
    )
    return payload


# ---------------------------------------------------------------------------
# Public API — evaluate a single bet
# ---------------------------------------------------------------------------

def evaluate_bet(
    *,
    fighter_a: str,
    fighter_b: str,
    bet_on: str,
    bet_side: str,
    model_prob: float,
    blended_prob: float,
    market_prob: float,
    edge: float,
    features: dict,
    provenance: dict | None = None,
    weight_class: str = "",
    event_title: str = "",
    event_date: str = "",
    existing_bets: list[dict] | None = None,
    trade_reason: str = "",
    decision_context: str = "",
) -> OperatorDecision:
    """
    Run the full operator pipeline for a single bet candidate.

    Returns an OperatorDecision with the verdict and rationale.
    The operator NEVER crashes the trading loop — any unhandled error
    results in a PASS (let the model's bet through).
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if _has_existing_bet_for_fight(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        existing_bets=existing_bets,
        event_date=event_date,
    ):
        skip_cache_key = _fight_cache_key(
            fighter_a,
            fighter_b,
            event_date=event_date,
            event_title=event_title,
            context=decision_context,
        )
        logger.info(
            "Operator skip for %s vs %s — fight already has a recorded bet/order",
            fighter_a,
            fighter_b,
        )
        return OperatorDecision(
            verdict="PASS",
            confidence=1.0,
            model_prob=model_prob,
            operator_prob=model_prob,
            rationale="Operator skipped: fight already has a recorded bet/order",
            research_summary={},
            risk_flags=["existing_bet"],
            timestamp=timestamp,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            bet_on=bet_on,
            bet_side=bet_side,
            edge=edge,
            market_prob=market_prob,
            event_date=event_date,
            event_title=event_title,
            trade_reason=trade_reason,
            decision_context=decision_context,
            decision_key=skip_cache_key,
            provenance=dict(provenance or {}),
        )

    # Check cache — decisions are sticky per event/fight so market drift
    # does not trigger repeated evaluations or verdict flips.
    cache_key = _fight_cache_key(
        fighter_a,
        fighter_b,
        event_date=event_date,
        event_title=event_title,
        context=decision_context,
    )
    cached_decision = _get_cached_decision(cache_key)
    if cached_decision is not None:
        logger.info(
            "Operator cache hit for %s vs %s — reusing %s verdict",
            fighter_a,
            fighter_b,
            cached_decision.verdict,
        )
        return cached_decision

    # Acquire a per-key lock so only one thread evaluates a given fight.
    with _decision_cache_lock:
        if cache_key not in _decision_inflight:
            _decision_inflight[cache_key] = threading.Lock()
        key_lock = _decision_inflight[cache_key]

    with key_lock:
        cached_decision = _get_cached_decision(cache_key)
        if cached_decision is not None:
            logger.info(
                "Operator cache hit for %s vs %s — reusing %s verdict",
                fighter_a,
                fighter_b,
                cached_decision.verdict,
            )
            return cached_decision

        process_lock_fd, process_lock_path = _acquire_process_lock(cache_key)
        try:
            cached_decision = _get_cached_decision(cache_key)
            if cached_decision is not None:
                logger.info(
                    "Operator disk cache hit for %s vs %s — reusing %s verdict",
                    fighter_a,
                    fighter_b,
                    cached_decision.verdict,
                )
                return cached_decision

            try:
                # Run research
                findings = run_research_pipeline(
                    features=features,
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    model_prob_a=model_prob if bet_side == "a" else 1 - model_prob,
                    market_prob_a=market_prob if bet_side == "a" else 1 - market_prob,
                    event_title=event_title,
                    event_date=event_date,
                    existing_bets=existing_bets,
                )

                research_prompt = _build_grounded_research_request(
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    model_pick=bet_on,
                    weight_class=weight_class,
                    event_date=event_date,
                    event_title=event_title,
                )

                # Build synthesis prompt and call LLM
                prompt = _build_synthesis_prompt(
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    bet_on=bet_on,
                    bet_side=bet_side,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    blended_prob=blended_prob,
                    edge=edge,
                    features=features,
                    findings=findings,
                    weight_class=weight_class,
                    event_date=event_date,
                )

                synthesis = _call_llm_synthesis(
                    prompt,
                    research_prompt=research_prompt,
                    research_cache_key=_operator_research_cache_key(cache_key, bet_on),
                )
                synthesis = _normalize_operator_synthesis_payload(synthesis)

                # Guard: if the LLM misread the stats, retry with corrections
                # so it can make a properly informed decision.
                synthesis = _guard_data_hallucination(
                    synthesis, features, fighter_a, fighter_b,
                    original_prompt=str(synthesis.get("_full_synthesis_prompt") or prompt),
                )
                synthesis = _normalize_operator_synthesis_payload(synthesis)

                grounded_research = dict(synthesis.get("_research_bundle") or {})
                stage_telemetry = dict(synthesis.get("_stage_telemetry") or {})

                # Build decision — PASS/BLOCK only
                verdict = synthesis.get("verdict", "PASS").upper()
                if verdict not in ("PASS", "BLOCK"):
                    logger.warning("Invalid verdict %r from operator — defaulting to PASS", verdict)
                    verdict = "PASS"

                research_summary = asdict(findings) if findings else {}
                if grounded_research:
                    research_summary["grounded_research"] = _grounded_research_public_view(
                        grounded_research
                    )
                if synthesis.get("verified_records"):
                    research_summary["verified_records"] = synthesis["verified_records"]
                if synthesis.get("fighter_assessment"):
                    research_summary["fighter_assessment"] = synthesis["fighter_assessment"]
                for summary_key in [
                    "level_of_competition_summary",
                    "style_matchup_summary",
                    "path_to_victory_for_model_pick",
                    "path_to_victory_for_opponent",
                    "model_read_support",
                    "model_read_concerns",
                ]:
                    value = synthesis.get(summary_key)
                    if value:
                        research_summary[summary_key] = value

                decision_provenance = dict(provenance or {})
                if stage_telemetry:
                    decision_provenance["llm_stage_telemetry"] = stage_telemetry

                decision = OperatorDecision(
                    verdict=verdict,
                    confidence=1.0,
                    model_prob=model_prob,
                    operator_prob=model_prob,
                    rationale=synthesis.get("rationale", "No rationale provided"),
                    research_summary=research_summary,
                    risk_flags=synthesis.get("risk_flags", []),
                    timestamp=timestamp,
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    bet_on=bet_on,
                    bet_side=bet_side,
                    edge=edge,
                    market_prob=market_prob,
                    event_date=event_date,
                    event_title=event_title,
                    trade_reason=trade_reason,
                    decision_context=decision_context,
                    decision_key=cache_key,
                    provenance=decision_provenance,
                )

            except Exception as exc:
                # Operator must NEVER crash the trading loop
                logger.error(
                    "Operator pipeline error for %s vs %s (defaulting to PASS): %s",
                    fighter_a, fighter_b, exc,
                )
                decision = OperatorDecision(
                    verdict="PASS",
                    confidence=1.0,
                    model_prob=model_prob,
                    operator_prob=model_prob,
                    rationale=f"Operator error (defaulting to PASS): {exc}",
                    research_summary={},
                    risk_flags=["operator_error"],
                    timestamp=timestamp,
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    bet_on=bet_on,
                    bet_side=bet_side,
                    edge=edge,
                    market_prob=market_prob,
                    event_date=event_date,
                    event_title=event_title,
                    trade_reason=trade_reason,
                    decision_context=decision_context,
                    decision_key=cache_key,
                    provenance=dict(provenance or {}),
                )

            # Always log
            _log_decision(decision)

            with _decision_cache_lock:
                _decision_cache[cache_key] = (decision, time.time())
            _save_decision_cache_to_disk()
        finally:
            _release_process_lock(process_lock_fd, process_lock_path)

    logger.info(
        "Operator verdict for %s: %s (flags: %s, bundle=%s, model_spec=%s, processed=%s, sources=%s/%s)",
        bet_on,
        decision.verdict,
        ", ".join(decision.risk_flags) if decision.risk_flags else "none",
        decision.provenance.get("bundle_id", "n/a"),
        decision.provenance.get("model_spec_name", "n/a"),
        decision.provenance.get("processed_snapshot_max_event_date", "n/a"),
        decision.provenance.get("fighter_a_source", "n/a"),
        decision.provenance.get("fighter_b_source", "n/a"),
    )

    return decision


# ---------------------------------------------------------------------------
# Batch evaluation — process a DataFrame of bet candidates
# ---------------------------------------------------------------------------

def evaluate_bets(
    bets: pd.DataFrame,
    *,
    features_by_fight: dict[str, dict] | None = None,
    provenance_by_fight: dict[str, dict] | None = None,
    event_title: str = "",
    existing_bets: list[dict] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str = "bets",
    decision_context: str = "",
) -> pd.DataFrame:
    """
    Evaluate a DataFrame of bet candidates through the operator.

    Args:
        bets: DataFrame from find_value_bets or find_conviction_bets
        features_by_fight: mapping of "fighterA|fighterB" → feature dict
        provenance_by_fight: mapping of "fighterA|fighterB" → runtime/source metadata
        event_title: current event name for exposure checks
        existing_bets: list of already-placed bets for exposure check

    Returns:
        Filtered DataFrame with only PASS bets, plus operator columns added.
        Sizing is unchanged — the operator only gates, never adjusts size.
    """
    if bets.empty:
        return bets

    if not OPERATOR_ENABLED:
        logger.debug("LLM Operator is disabled — passing all bets through")
        bets = bets.copy()
        bets["operator_verdict"] = "PASS"
        bets["operator_rationale"] = "Operator disabled"
        bets.attrs["operator_decisions_by_key"] = {}
        bets.attrs["operator_prepared_keys"] = []
        return bets

    features_by_fight = features_by_fight or {}
    provenance_by_fight = provenance_by_fight or {}
    decisions_by_key: dict[str, OperatorDecision] = {}
    approved_rows = []

    def _report_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(message)
        except Exception as exc:
            logger.debug("Operator progress callback failed: %s", exc)

    prepared_rows: list[tuple[pd.Series, str]] = []
    unique_rows: list[tuple[pd.Series, str]] = []
    seen_keys: set[str] = set()
    for _, bet in bets.iterrows():
        fighter_a = bet.get("fighter_a", "")
        fighter_b = bet.get("fighter_b", "")
        bet_event_title = str(bet.get("event_title", "") or event_title or "")
        bet_event_date = str(bet.get("market_event_date") or bet.get("event_date") or "")
        decision_key = _fight_cache_key(
            fighter_a,
            fighter_b,
            event_date=bet_event_date,
            event_title=bet_event_title,
            context=decision_context,
        )
        prepared_rows.append((bet, decision_key))
        if decision_key in seen_keys:
            continue
        seen_keys.add(decision_key)
        unique_rows.append((bet, decision_key))

    total_unique_bets = len(unique_rows)
    for position, (bet, decision_key) in enumerate(unique_rows, start=1):
        fighter_a = bet.get("fighter_a", "")
        fighter_b = bet.get("fighter_b", "")
        fight_key = f"{fighter_a}|{fighter_b}"
        _report_progress(
            f"Cycle active: operator evaluating {progress_label} {position}/{total_unique_bets}: {fighter_a} vs {fighter_b}"
        )

        features = features_by_fight.get(fight_key, {})
        provenance = provenance_by_fight.get(fight_key, {})

        decision = evaluate_bet(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            bet_on=bet.get("bet_on", ""),
            bet_side=bet.get("bet_side", ""),
            model_prob=float(bet.get("model_prob", 0.5)),
            blended_prob=float(bet.get("blended_prob", 0.5)),
            market_prob=float(bet.get("market_prob", 0.5)),
            edge=float(bet.get("edge", 0.0)),
            features=features,
            provenance=provenance,
            weight_class=str(bet.get("weight_class", "")),
            event_title=str(bet.get("event_title", "") or event_title or ""),
            event_date=str(bet.get("market_event_date") or bet.get("event_date") or ""),
            existing_bets=existing_bets,
            trade_reason=str(bet.get("reason", "") or ""),
            decision_context=decision_context,
        )

        decisions_by_key[decision_key] = decision

    for bet, decision_key in prepared_rows:
        decision = decisions_by_key[decision_key]
        if decision.verdict == "BLOCK":
            logger.info(
                "Operator BLOCKED bet on %s: %s",
                bet.get("bet_on", "?"),
                decision.rationale[:100],
            )
            if OPERATOR_MODE == "gate":
                continue

        row = bet.copy()
        row["operator_verdict"] = decision.verdict
        row["operator_rationale"] = decision.rationale
        row["operator_risk_flags"] = ", ".join(decision.risk_flags)

        approved_rows.append(row)

    if not approved_rows:
        cols = list(bets.columns) + ["operator_verdict", "operator_rationale", "operator_risk_flags"]
        result = pd.DataFrame(columns=cols)
        result.attrs["operator_decisions_by_key"] = decisions_by_key
        result.attrs["operator_prepared_keys"] = prepared_rows
        return result

    result = pd.DataFrame(approved_rows)
    result.attrs["operator_decisions_by_key"] = decisions_by_key
    result.attrs["operator_prepared_keys"] = prepared_rows
    blocked = sum(1 for _, decision_key in prepared_rows if decisions_by_key[decision_key].verdict == "BLOCK")
    logger.info(
        "Operator: %d/%d bets passed, %d blocked",
        len(approved_rows),
        len(prepared_rows),
        blocked,
    )
    return result
