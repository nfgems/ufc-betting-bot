"""
Production entrypoint â€” runs the web dashboard + background monitor together.

Used by Railway/Docker for always-on deployment.
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    BTC5M_LIVE_LEDGER_DIR,
    BTC5M_LIVE_PROFILES,
    BTC5M_POLL_SECONDS,
    BTC5M_PROFILES,
    LOGS_DIR,
    MIN_EDGE_THRESHOLD,
)
from src.live_control import (
    LIVE_MODE_DRY_RUN,
    LIVE_MODE_ENV,
    LIVE_MODE_OFF,
    LIVE_MODE_REAL,
    assert_polymarket_real_trading_allowed,
    evaluate_live_startup,
    evaluate_polymarket_live_startup,
    resolve_live_model_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)

# Mirror WARNING/ERROR/CRITICAL to a durable, time-bounded alerts.jsonl so the
# Activity dashboard can surface them for the full retention window instead of
# losing them to INFO-volume scroll-out in bot.log.
from src.web.alert_store import install_alert_handler

install_alert_handler(LOGS_DIR)

logger = logging.getLogger(__name__)

UFC_REFRESH_DEFAULT_INTERVAL_HOURS = 24.0
UFC_REFRESH_MAX_INTERVAL_HOURS = 24.0
UFC_REFRESH_MIN_INTERVAL_HOURS = 1.0
_UFC_REFRESH_CYCLE_LOCK = threading.Lock()
_UFC_REFRESH_CYCLE_IN_PROGRESS = False


def _resolve_hosted_bundle_startup_summary() -> dict | None:
    try:
        from src.model.production_bundle import (
            is_hosted_runtime,
            load_production_bundle,
            validate_production_bundle,
        )
    except Exception:
        return None

    if not is_hosted_runtime():
        return None

    bundle = load_production_bundle()
    summary = validate_production_bundle(bundle)
    logger.info(
        "Hosted production bundle ready: bundle_id=%s manifest=%s model=%s no_odds=%s spec=%s processed_dir=%s processed_max_event_date=%s built_at=%s git_sha=%s",
        summary["bundle_id"],
        summary["manifest_path"],
        summary["model_path"],
        summary["no_odds_model_path"],
        summary["model_spec_name"],
        summary["processed_dir"],
        summary["processed_snapshot_max_event_date"],
        summary["built_at"],
        summary["git_sha"],
    )
    return summary


def _auto_redeem_enabled() -> bool:
    raw = str(os.getenv("POLYMARKET_AUTO_REDEEM", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _ufc_refresh_enabled() -> bool:
    raw = str(os.getenv("UFC_REFRESH_ENABLED", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _mark_ufc_refresh_cycle_started() -> None:
    global _UFC_REFRESH_CYCLE_IN_PROGRESS
    with _UFC_REFRESH_CYCLE_LOCK:
        _UFC_REFRESH_CYCLE_IN_PROGRESS = True


def _mark_ufc_refresh_cycle_finished() -> None:
    global _UFC_REFRESH_CYCLE_IN_PROGRESS
    with _UFC_REFRESH_CYCLE_LOCK:
        _UFC_REFRESH_CYCLE_IN_PROGRESS = False


def _ufc_refresh_cycle_in_progress() -> bool:
    with _UFC_REFRESH_CYCLE_LOCK:
        return _UFC_REFRESH_CYCLE_IN_PROGRESS


def _runtime_bundle_freshness_blocked(exc: Exception) -> bool:
    return "Runtime bundle freshness guard blocked live trading" in str(exc)


def _runtime_bundle_refresh_blocked(exc: Exception) -> bool:
    message = str(exc)
    if _runtime_bundle_freshness_blocked(exc):
        return True
    return (
        "Production bundle processed " in message
        and (
            "snapshot hash mismatch" in message
            or "snapshot size mismatch" in message
        )
    )


def _ufc_refresh_interval_hours() -> float:
    raw = str(os.getenv("UFC_REFRESH_INTERVAL_HOURS", str(UFC_REFRESH_DEFAULT_INTERVAL_HOURS)) or "").strip()
    try:
        value = float(raw)
    except Exception:
        return UFC_REFRESH_DEFAULT_INTERVAL_HOURS
    if value > UFC_REFRESH_MAX_INTERVAL_HOURS:
        logger.warning(
            "UFC_REFRESH_INTERVAL_HOURS=%.2f exceeds the hosted freshness-safe max %.2f; "
            "using %.2f hours.",
            value,
            UFC_REFRESH_MAX_INTERVAL_HOURS,
            UFC_REFRESH_MAX_INTERVAL_HOURS,
        )
        return UFC_REFRESH_MAX_INTERVAL_HOURS
    return max(value, UFC_REFRESH_MIN_INTERVAL_HOURS)


def _ufc_refresh_initial_delay_seconds() -> float:
    raw = str(os.getenv("UFC_REFRESH_INITIAL_DELAY_MINUTES", "30") or "").strip()
    try:
        minutes = float(raw)
    except Exception:
        return 30.0 * 60.0
    return max(minutes, 0.0) * 60.0


def _ufc_refresh_limit_fighters() -> int | None:
    raw = str(os.getenv("UFC_REFRESH_LIMIT_FIGHTERS", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _btc5m_live_profile_names() -> tuple[list[str], list[str]]:
    raw_items = _split_csv(os.getenv("BTC5M_LIVE_PROFILES", BTC5M_LIVE_PROFILES))
    names: list[str] = []
    errors: list[str] = []
    for raw in raw_items:
        name = str(raw or "").strip().lower()
        if not name:
            continue
        if name == "all":
            for profile_name in BTC5M_PROFILES:
                if profile_name not in names:
                    names.append(profile_name)
            continue
        if name not in BTC5M_PROFILES:
            allowed = ", ".join(sorted(BTC5M_PROFILES))
            errors.append(f"Unknown BTC5M_LIVE_PROFILES entry {name!r}. Allowed profiles: {allowed}")
            continue
        if name not in names:
            names.append(name)
    return names, errors


def _btc5m_live_ledger_dir() -> Path:
    return Path(os.getenv("BTC5M_LIVE_LEDGER_DIR", str(BTC5M_LIVE_LEDGER_DIR))).expanduser()


def _btc5m_live_ledger_path(profile_name: str) -> Path:
    return _btc5m_live_ledger_dir() / f"{profile_name}.json"


def _btc5m_component_name(profile_name: str) -> str:
    return f"btc5m_loop:{profile_name}"


def _btc5m_result_reason(result: dict) -> str:
    return str(result.get("reason") or result.get("signal", {}).get("reason") or "")


def _btc5m_result_reason_code(result: dict) -> str:
    return str(
        result.get("reason_code")
        or result.get("risk", {}).get("reason_code")
        or result.get("signal", {}).get("reason_code")
        or ""
    )


def _btc5m_next_utc_midnight(current: datetime) -> datetime:
    current_utc = current.astimezone(timezone.utc)
    next_day = current_utc.date() + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, tzinfo=timezone.utc)


def _btc5m_daily_loss_metadata(
    *,
    reason_code: str,
    result: dict,
    profile,
    hit_at: str | None,
    current: datetime,
) -> dict:
    base = {
        "daily_loss_limit_hit_at": None,
        "daily_loss_limit_auto_resume_at": None,
        "daily_loss_limit_usd": None,
        "daily_loss_realized_today": None,
    }
    if reason_code != "daily_loss_limit":
        return base
    risk = result.get("risk") if isinstance(result.get("risk"), dict) else {}
    return {
        "daily_loss_limit_hit_at": hit_at,
        "daily_loss_limit_auto_resume_at": _btc5m_next_utc_midnight(current).isoformat(),
        "daily_loss_limit_usd": getattr(profile, "daily_loss_limit_usd", None),
        "daily_loss_realized_today": risk.get("realized_loss_today"),
    }


def _parse_btc5m_settlement_timestamp(value) -> datetime | None:
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
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _btc5m_ledger_has_due_settlement(ledger, *, now: datetime) -> bool:
    open_bets = getattr(ledger, "get_open_bets", None)
    if not callable(open_bets):
        return False
    for bet in open_bets(fresh=True):
        if not isinstance(bet, dict):
            continue
        window_end = _parse_btc5m_settlement_timestamp(
            bet.get("window_end") or bet.get("event_date")
        )
        if window_end is not None and now >= window_end:
            return True
    return False


def _btc5m_stop_status() -> dict:
    from src.web.app import btc5m_emergency_stop_status

    return btc5m_emergency_stop_status()


def _btc5m_stop_requested() -> bool:
    try:
        return bool(_btc5m_stop_status().get("active"))
    except Exception as exc:
        logger.warning("BTC 5m emergency stop status check failed: %s", exc)
        return False


def _btc5m_sleep_until_next_poll(seconds: float) -> bool:
    deadline = time.monotonic() + max(float(seconds), 1.0)
    while True:
        if _btc5m_stop_requested():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _btc5m_stop_requested()
        time.sleep(min(remaining, 1.0))


def _ufc_refresh_pct_threshold(env_name: str) -> float | None:
    raw = str(os.getenv(env_name, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value


def _ufc_refresh_positive_int_threshold(env_name: str, default: int | None) -> int | None:
    raw = str(os.getenv(env_name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else None


def _ufc_refresh_positive_pct_threshold(env_name: str, default: float | None) -> float | None:
    raw = str(os.getenv(env_name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return value if value > 0 else None


def _pct_from_metric(metric: object) -> float | None:
    if not isinstance(metric, dict):
        return None
    raw = metric.get("pct")
    try:
        return float(raw)
    except Exception:
        return None


def _coverage_snapshot_from_refresh_summary(summary: dict | None) -> dict[str, float | int | bool | None]:
    refresh_summary = summary or {}
    audit = refresh_summary.get("profile_audit") or {}
    audit_alert_summary = refresh_summary.get("profile_audit_alert_summary") or {}
    overall = audit.get("overall_summary") or {}
    source_limited_missing_fields = audit_alert_summary.get("source_limited_missing_fields") or {}
    split_summary = (
        audit_alert_summary.get("split_summary_alias_aware")
        or audit_alert_summary.get("split_summary_official_name")
        or audit.get("split_summary_alias_aware")
        or audit.get("split_summary_official_name")
        or {}
    )
    newly_added = split_summary.get("newly_added_active_roster") or {}
    new_fighter_alert_context = audit_alert_summary.get("newly_added_active_roster") or {}
    reach_source_limited = source_limited_missing_fields.get("reach") or {}
    stance_source_limited = source_limited_missing_fields.get("stance") or {}
    full_physical_source_limited = source_limited_missing_fields.get("full_physical_bundle") or {}
    return {
        "active_roster_rows": audit.get("active_roster_rows"),
        "overall_full_physical_pct": _pct_from_metric(overall.get("full_physical_bundle_present")),
        "overall_reach_pct": _pct_from_metric(overall.get("reach_present")),
        "overall_stance_pct": _pct_from_metric(overall.get("stance_present")),
        "new_fighter_full_physical_pct": _pct_from_metric(newly_added.get("full_physical_bundle_present")),
        "new_fighter_reach_pct": _pct_from_metric(newly_added.get("reach_present")),
        "new_fighter_stance_pct": _pct_from_metric(newly_added.get("stance_present")),
        "new_fighter_rows_total": new_fighter_alert_context.get("rows_total"),
        "new_fighter_rows_alert_eligible": new_fighter_alert_context.get("rows_alert_eligible"),
        "new_fighter_rows_in_grace": new_fighter_alert_context.get("rows_in_grace"),
        "new_fighter_grace_days": audit_alert_summary.get("new_fighter_grace_days"),
        "new_fighter_identity_match_method": audit_alert_summary.get("identity_match_method"),
        "new_fighter_reach_rows_missing": reach_source_limited.get("rows_missing"),
        "new_fighter_reach_rows_source_limited": reach_source_limited.get("rows_source_limited"),
        "new_fighter_reach_source_limited_only": bool(reach_source_limited.get("source_limited_only")),
        "new_fighter_stance_rows_missing": stance_source_limited.get("rows_missing"),
        "new_fighter_stance_rows_source_limited": stance_source_limited.get("rows_source_limited"),
        "new_fighter_stance_source_limited_only": bool(stance_source_limited.get("source_limited_only")),
        "new_fighter_full_physical_rows_missing": full_physical_source_limited.get("rows_missing"),
        "new_fighter_full_physical_rows_source_limited": full_physical_source_limited.get("rows_source_limited"),
        "new_fighter_full_physical_source_limited_only": bool(
            full_physical_source_limited.get("source_limited_only")
        ),
    }


def _ufc_refresh_coverage_alerts(coverage_snapshot: dict[str, float | int | bool | None]) -> list[str]:
    checks = [
        (
            "UFC_REFRESH_MIN_OVERALL_REACH_PCT",
            "overall reach coverage",
            coverage_snapshot.get("overall_reach_pct"),
        ),
        (
            "UFC_REFRESH_MIN_OVERALL_STANCE_PCT",
            "overall stance coverage",
            coverage_snapshot.get("overall_stance_pct"),
        ),
        (
            "UFC_REFRESH_MIN_NEW_FIGHTER_REACH_PCT",
            "new active-fighter reach coverage",
            coverage_snapshot.get("new_fighter_reach_pct"),
        ),
        (
            "UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT",
            "new active-fighter stance coverage",
            coverage_snapshot.get("new_fighter_stance_pct"),
        ),
        (
            "UFC_REFRESH_MIN_NEW_FIGHTER_FULL_PHYSICAL_PCT",
            "new active-fighter full physical coverage",
            coverage_snapshot.get("new_fighter_full_physical_pct"),
        ),
    ]
    source_limited_only_flags = {
        "UFC_REFRESH_MIN_NEW_FIGHTER_REACH_PCT": "new_fighter_reach_source_limited_only",
        "UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT": "new_fighter_stance_source_limited_only",
        "UFC_REFRESH_MIN_NEW_FIGHTER_FULL_PHYSICAL_PCT": "new_fighter_full_physical_source_limited_only",
    }

    alerts: list[str] = []
    for env_name, label, observed in checks:
        threshold = _ufc_refresh_pct_threshold(env_name)
        if threshold is None or observed is None:
            continue
        source_limited_key = source_limited_only_flags.get(env_name)
        if source_limited_key and coverage_snapshot.get(source_limited_key):
            continue
        if float(observed) < threshold:
            alerts.append(
                f"{label} dropped to {float(observed):.2f}% below configured floor {threshold:.2f}%"
            )
    return alerts


def _ufc_refresh_coverage_notes(coverage_snapshot: dict[str, float | int | bool | None]) -> list[str]:
    notes: list[str] = []
    field_configs = [
        (
            "new_fighter_reach",
            "new active-fighter reach coverage floor skipped",
        ),
        (
            "new_fighter_stance",
            "new active-fighter stance coverage floor skipped",
        ),
        (
            "new_fighter_full_physical",
            "new active-fighter full physical coverage floor skipped",
        ),
    ]
    for prefix, label in field_configs:
        if not coverage_snapshot.get(f"{prefix}_source_limited_only"):
            continue
        rows_missing = int(coverage_snapshot.get(f"{prefix}_rows_missing") or 0)
        rows_source_limited = int(coverage_snapshot.get(f"{prefix}_rows_source_limited") or 0)
        if rows_missing <= 0:
            continue
        notes.append(
            f"{label} because all {rows_source_limited}/{rows_missing} alert-eligible missing rows are source-limited"
        )
    return notes


def _retained_missing_live_roster_detail(roster_sync: dict) -> str | None:
    retained_missing_live_rows = int(roster_sync.get("retained_missing_live_rows") or 0)
    if not retained_missing_live_rows:
        return None
    retained_fighters = roster_sync.get("retained_missing_live_fighters") or []
    names = [
        str(row.get("official_name") or row.get("official_athlete_url") or "").strip()
        for row in retained_fighters
        if isinstance(row, dict) and str(row.get("official_name") or row.get("official_athlete_url") or "").strip()
    ]
    preview = ", ".join(names[:10])
    detail = (
        "official UFC roster live sync omitted "
        f"{retained_missing_live_rows} previously tracked row(s); retained cached rows"
    )
    if preview:
        detail += f": {preview}"
        if retained_missing_live_rows > len(names[:10]):
            detail += ", ..."
    return detail


def _retained_missing_live_roster_exceeds_alert_threshold(roster_sync: dict) -> bool:
    retained_missing_live_rows = int(roster_sync.get("retained_missing_live_rows") or 0)
    if retained_missing_live_rows <= 0:
        return False

    max_rows = _ufc_refresh_positive_int_threshold(
        "UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS",
        50,
    )
    if max_rows is not None and retained_missing_live_rows > max_rows:
        return True

    max_pct = _ufc_refresh_positive_pct_threshold(
        "UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT",
        5.0,
    )
    roster_rows = int(roster_sync.get("rows") or 0)
    if max_pct is not None and roster_rows > 0:
        retained_pct = (retained_missing_live_rows / roster_rows) * 100.0
        if retained_pct > max_pct:
            return True

    return False


def _ufc_refresh_operational_notes(summary: dict | None) -> list[str]:
    refresh_summary = summary or {}
    roster_sync = refresh_summary.get("roster_sync") or {}
    notes: list[str] = []
    discarded_suspicious_cached_rows = int(roster_sync.get("discarded_suspicious_cached_rows") or 0)
    if discarded_suspicious_cached_rows:
        notes.append(
            f"discarded {discarded_suspicious_cached_rows} stale cached UFC roster row(s) after live active-roster sync"
        )
    retained_detail = _retained_missing_live_roster_detail(roster_sync)
    if (
        retained_detail
        and not _retained_missing_live_roster_exceeds_alert_threshold(roster_sync)
    ):
        notes.append(retained_detail)
    identity_audit_rows = int(roster_sync.get("identity_audit_rows") or 0)
    counts = roster_sync.get("identity_audit_action_counts") or {}
    if identity_audit_rows and isinstance(counts, dict):
        inactive_excluded_rows = int(counts.get("excluded_inactive_profile_status") or 0)
        if inactive_excluded_rows:
            notes.append(
                f"excluded {inactive_excluded_rows} inactive/non-fighting UFC profile row(s) from active roster sync"
            )
    return notes


def _ufc_refresh_operational_alerts(summary: dict | None) -> list[str]:
    refresh_summary = summary or {}
    alerts: list[str] = []
    roster_sync = refresh_summary.get("roster_sync") or {}
    if roster_sync.get("used_cached_fallback"):
        cached_snapshot_mtime = str(roster_sync.get("cached_snapshot_mtime_utc") or "").strip()
        sync_error = str(roster_sync.get("sync_error") or "").strip()
        detail = "official UFC roster sync fell back to the last cached roster snapshot"
        if cached_snapshot_mtime:
            detail += f" from {cached_snapshot_mtime}"
        if sync_error:
            detail += f" after live sync error: {sync_error}"
        alerts.append(detail)
    retained_detail = _retained_missing_live_roster_detail(roster_sync)
    if (
        retained_detail
        and _retained_missing_live_roster_exceeds_alert_threshold(roster_sync)
    ):
        detail = retained_detail
        max_rows = _ufc_refresh_positive_int_threshold(
            "UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS",
            50,
        )
        max_pct = _ufc_refresh_positive_pct_threshold(
            "UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT",
            5.0,
        )
        roster_rows = int(roster_sync.get("rows") or 0)
        thresholds: list[str] = []
        retained_missing_live_rows = int(roster_sync.get("retained_missing_live_rows") or 0)
        if max_rows is not None and retained_missing_live_rows > max_rows:
            thresholds.append(f"row threshold {max_rows}")
        if max_pct is not None and roster_rows > 0:
            retained_pct = (retained_missing_live_rows / roster_rows) * 100.0
            if retained_pct > max_pct:
                thresholds.append(f"{max_pct:.2f}% threshold")
        if thresholds:
            detail += f" (exceeds {' and '.join(thresholds)})"
        alerts.append(detail)
    identity_audit_rows = int(roster_sync.get("identity_audit_rows") or 0)
    test_rows = 0
    if identity_audit_rows:
        counts = roster_sync.get("identity_audit_action_counts") or {}
        if isinstance(counts, dict) and counts:
            detail_parts: list[str] = []
            test_rows = int(counts.get("excluded_test_profile") or 0)
            inactive_excluded_rows = int(counts.get("excluded_inactive_profile_status") or 0)
            full_quarantine_rows = int(counts.get("quarantined_untrusted_url_identity") or 0)
            slug_alias_rows = int(counts.get("quarantined_untrusted_slug_alias") or 0)
            other_rows = max(
                0,
                identity_audit_rows
                - test_rows
                - inactive_excluded_rows
                - full_quarantine_rows
                - slug_alias_rows,
            )
            if test_rows:
                detail_parts.append(f"{test_rows} test/staging excluded")
            if inactive_excluded_rows:
                detail_parts.append(f"{inactive_excluded_rows} inactive/non-fighting excluded")
            if full_quarantine_rows:
                detail_parts.append(f"{full_quarantine_rows} URL identity quarantined")
            if slug_alias_rows:
                detail_parts.append(f"{slug_alias_rows} slug aliases suppressed")
            if other_rows:
                detail_parts.append(f"{other_rows} other flagged")
            detail = ", ".join(detail_parts)
            if full_quarantine_rows or other_rows:
                alerts.append(f"official UFC roster identity audit flagged {identity_audit_rows} row(s): {detail}")
        else:
            alerts.append(f"official UFC roster identity audit flagged {identity_audit_rows} row(s)")

    row_drop_guard = refresh_summary.get("row_drop_guard") or {}
    discarded_suspicious_cached_rows = int(roster_sync.get("discarded_suspicious_cached_rows") or 0)
    for violation in row_drop_guard.get("violations") or []:
        artifact = str(violation.get("artifact") or "")
        try:
            rows_lost = int(violation.get("rows_lost") or 0)
        except Exception:
            rows_lost = 0
        if artifact == "ufc_active_roster_official" and rows_lost > 0 and rows_lost <= test_rows:
            continue
        if (
            artifact == "ufc_active_roster_official"
            and rows_lost > 0
            and discarded_suspicious_cached_rows >= rows_lost
        ):
            continue
        alerts.append(
            "row-count guard detected a drop in "
            f"{violation.get('artifact')}: {violation.get('pre_rows')} -> "
            f"{violation.get('post_rows')} rows"
        )
    return alerts


def _ufc_refresh_coverage_skip_reason(summary: dict | None) -> str:
    refresh_summary = summary or {}
    if not refresh_summary.get("partial_refresh"):
        return ""
    limit = refresh_summary.get("limit_fighters")
    if limit is None:
        return "coverage floors skipped because this was a partial refresh"
    return (
        "coverage floors skipped because this refresh ran in partial smoke-test mode "
        f"(limit_fighters={limit})"
    )


def _run_ufc_refresh_cycle(
    *,
    limit_fighters: int | None = None,
) -> dict:
    from scripts.run_scheduled_ufc_refresh import run_scheduled_refresh

    return run_scheduled_refresh(limit_fighters=limit_fighters)


def run_background_ufc_refresh_loop(
    interval_hours: float = UFC_REFRESH_DEFAULT_INTERVAL_HOURS,
    *,
    initial_delay_seconds: float = 1800.0,
    limit_fighters: int | None = None,
):
    """Refresh UFC data inside the hosted service so Railway uses the same volume."""
    from src.web.app import get_runtime_status, set_runtime_status, update_runtime_component

    heartbeat_window = max(1800.0, interval_hours * 3600 * 2.5)
    now = datetime.now(timezone.utc)
    first_run_at = (now + timedelta(seconds=max(initial_delay_seconds, 0.0))).isoformat()
    update_runtime_component(
        "ufc_refresh_loop",
        "starting" if initial_delay_seconds > 0 else "running",
        (
            f"Scheduled UFC refresh loop waiting for first run at {first_run_at}."
            if initial_delay_seconds > 0
            else "Scheduled UFC refresh loop active."
        ),
        stale_after_seconds=heartbeat_window,
        consecutive_failures=0,
        last_successful_refresh_at=None,
        next_planned_refresh_at=first_run_at,
        last_error=None,
    )
    if initial_delay_seconds > 0:
        time.sleep(initial_delay_seconds)

    update_runtime_component(
        "ufc_refresh_loop",
        "running",
        "Scheduled UFC refresh loop active.",
        stale_after_seconds=heartbeat_window,
        consecutive_failures=0,
        last_successful_refresh_at=None,
        next_planned_refresh_at=datetime.now(timezone.utc).isoformat(),
        last_error=None,
    )
    logger.info(
        "Scheduled UFC refresh loop started (every %.2fh, limit_fighters=%s)",
        interval_hours,
        limit_fighters,
    )

    consecutive_failures = 0
    while True:
        cycle_started_at = datetime.now(timezone.utc).isoformat()
        update_runtime_component(
            "ufc_refresh_loop",
            "running",
            f"Refresh cycle started at {cycle_started_at}",
            consecutive_failures=consecutive_failures,
            last_cycle_started_at=cycle_started_at,
        )
        _mark_ufc_refresh_cycle_started()

        try:
            summary = _run_ufc_refresh_cycle(limit_fighters=limit_fighters)
            consecutive_failures = 0
            cycle_completed_at = datetime.now(timezone.utc).isoformat()
            next_planned_refresh_at = (
                datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            ).isoformat()
            outputs = ((summary.get("rebuild") or {}).get("outputs") or [])
            fight_rows = outputs[0].get("fight_rows") if outputs else None
            refreshed_bundle = (summary.get("rebuild") or {}).get("production_bundle")
            coverage_snapshot = _coverage_snapshot_from_refresh_summary(summary)
            coverage_skip_reason = _ufc_refresh_coverage_skip_reason(summary)
            operational_notes = _ufc_refresh_operational_notes(summary)
            coverage_notes = [
                *operational_notes,
                *_ufc_refresh_coverage_notes(coverage_snapshot),
            ]
            operational_alerts = _ufc_refresh_operational_alerts(summary)
            coverage_alerts = (
                []
                if coverage_skip_reason
                else _ufc_refresh_coverage_alerts(coverage_snapshot)
            )
            refresh_alerts = [*operational_alerts, *coverage_alerts]
            if isinstance(refreshed_bundle, dict):
                runtime_status = get_runtime_status()
                runtime_status["production_bundle"] = dict(refreshed_bundle)
                set_runtime_status(runtime_status)
            update_runtime_component(
                "ufc_refresh_loop",
                "degraded" if refresh_alerts else "running",
                (
                    f"Last UFC refresh completed at {cycle_completed_at}; "
                    f"next run in {interval_hours} hours"
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
                last_successful_refresh_at=cycle_completed_at,
                next_planned_refresh_at=next_planned_refresh_at,
                last_error=None,
                last_summary=summary,
                fight_rows=fight_rows,
                coverage_snapshot=coverage_snapshot,
                coverage_alerts=refresh_alerts,
                coverage_skip_reason=coverage_skip_reason,
                coverage_notes=coverage_notes,
            )
            logger.info(
                "Scheduled UFC refresh completed: roster_rows=%s new_results=%s new_stats=%s coverage=%s",
                (summary.get("roster_sync") or {}).get("rows"),
                (summary.get("ufcstats_backfill") or {}).get("new_result_rows"),
                (summary.get("ufcstats_backfill") or {}).get("new_stat_rows"),
                coverage_snapshot,
            )
            if summary.get("resolved_paths"):
                logger.info("Scheduled UFC refresh resolved paths: %s", summary["resolved_paths"])
            if summary.get("seed_stale_scraped_fighters"):
                logger.info("Scheduled UFC refresh seed result: %s", summary["seed_stale_scraped_fighters"])
            audit_source_files = (
                (summary.get("profile_audit") or {}).get("source_files") or {}
            )
            if audit_source_files:
                logger.info("Scheduled UFC refresh audit source files: %s", audit_source_files)
            if coverage_skip_reason:
                logger.info("Scheduled UFC refresh coverage thresholds skipped: %s", coverage_skip_reason)
            if coverage_notes:
                logger.info("Scheduled UFC refresh coverage notes: %s", " | ".join(coverage_notes))
            if refresh_alerts:
                logger.warning("Scheduled UFC refresh alerts: %s", " | ".join(refresh_alerts))
        except Exception as exc:
            consecutive_failures += 1
            cycle_failed_at = datetime.now(timezone.utc).isoformat()
            next_planned_refresh_at = (
                datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            ).isoformat()
            update_runtime_component(
                "ufc_refresh_loop",
                "degraded",
                (
                    f"Last UFC refresh failed at {cycle_failed_at}; "
                    f"retry in {interval_hours} hours: {exc}"
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_failed_at=cycle_failed_at,
                next_planned_refresh_at=next_planned_refresh_at,
                last_error=str(exc),
                coverage_alerts=[f"refresh failure: {exc}"],
            )
            logger.error("Scheduled UFC refresh failed: %s", exc, exc_info=True)
        finally:
            _mark_ufc_refresh_cycle_finished()

        time.sleep(interval_hours * 3600)


def _log_auto_redeem_summary(summary: dict, *, wait: bool) -> None:
    if summary.get("reason") == "redeemer_not_configured":
        return
    if summary.get("reason") == "redeem_submission_pending":
        pending = summary.get("pending_transactions", [])
        logger.info(
            "Auto-redeem skipped because %s redeem transaction(s) are still pending",
            len(pending),
        )
        return
    if summary.get("reason") == "auto_redeem_cooldown":
        remaining = int(summary.get("cooldown_remaining_seconds") or 0)
        logger.info(
            "Auto-redeem cooling down for another %s second(s)",
            remaining,
        )
        return
    if summary.get("errors"):
        logger.warning(
            "Auto-redeem completed with %s error(s)",
            len(summary["errors"]),
        )
    if wait:
        if summary.get("redeemed_conditions"):
            logger.info(
                "Auto-redeemed %s position(s) across %s condition(s)",
                summary["redeemed_positions"],
                summary["redeemed_conditions"],
            )
        return
    if summary.get("submitted_conditions"):
        logger.info(
            "Auto-submitted redeem for %s position(s) across %s condition(s)",
            summary["submitted_positions"],
            summary["submitted_conditions"],
        )


def run_live_betting_loop(
    interval_minutes: float = 10.0,
    min_edge: float = MIN_EDGE_THRESHOLD,
    *,
    trading_mode: str = LIVE_MODE_DRY_RUN,
    model_name: str = "xgboost",
):
    """Run the live betting bot in a background loop."""
    import argparse
    from src.web.app import get_clob_client, update_runtime_component

    # Wait for the web server to start
    time.sleep(15)
    heartbeat_window = max(300.0, interval_minutes * 60 * 2.5)
    consecutive_failures = 0

    if trading_mode not in {LIVE_MODE_DRY_RUN, LIVE_MODE_REAL}:
        update_runtime_component(
            "betting_loop",
            "disabled",
            f"Trading mode '{trading_mode}' does not start the betting loop.",
            stale_after_seconds=heartbeat_window,
            consecutive_failures=0,
        )
        logger.info("Live betting loop disabled: unsupported trading mode %s", trading_mode)
        return

    update_runtime_component(
        "betting_loop",
        "running",
        f"Trading mode: {trading_mode}",
        stale_after_seconds=heartbeat_window,
        consecutive_failures=0,
    )
    logger.info(
        f"Live betting loop started in {trading_mode} mode "
        f"(every {interval_minutes}m, min_edge={min_edge:.1%}, model={model_name})"
    )

    while True:
        cycle_started_at = datetime.now(timezone.utc).isoformat()
        cycle_succeeded = True
        cycle_deferred_reason: str | None = None
        shared_clob = get_clob_client()

        def _heartbeat(message: str, **metadata) -> None:
            update_runtime_component(
                "betting_loop",
                "degraded" if consecutive_failures >= 3 else "running",
                message,
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                **metadata,
            )

        _heartbeat(f"Cycle started at {cycle_started_at}")

        # Cancel any limit bids for fights that have already started
        try:
            _heartbeat("Cycle active: cancelling stale pre-fight limit bids")
            from src.polymarket.executor import cancel_all_stale_limit_bids

            cancelled = cancel_all_stale_limit_bids(clob_client=shared_clob)
            if cancelled:
                logger.info(f"Pre-cycle cleanup: cancelled {cancelled} stale limit bid(s)")
        except Exception as e:
            logger.error(f"Limit bid cancellation error: {e}", exc_info=True)

        # Snapshot odds + Polymarket prices before each betting cycle so
        # Sharp Money Tracker has fresh line data every cycle
        try:
            _heartbeat("Cycle active: snapshotting odds and Polymarket prices")
            from src.data.line_tracker import snapshot_odds, snapshot_polymarket_prices

            snapshot_odds()
            snapshot_polymarket_prices()
        except Exception as e:
            logger.warning(f"Pre-cycle line snapshot failed (non-fatal): {e}")

        try:
            _heartbeat("Cycle active: running live prediction and operator sweep")
            from src.bot import cmd_duo_live

            args = argparse.Namespace(
                dry_run=trading_mode != LIVE_MODE_REAL,
                real=trading_mode == LIVE_MODE_REAL,
                model=model_name,
                min_edge=min_edge,
                progress_callback=_heartbeat,
                clob_client=shared_clob,
            )
            result = cmd_duo_live(args)
            result_status = result.get("status") if isinstance(result, dict) else None
            if result_status in ("error", "degraded"):
                cycle_succeeded = False
                consecutive_failures += 1
                failed_at = datetime.now(timezone.utc).isoformat()
                update_runtime_component(
                    "betting_loop",
                    "degraded" if consecutive_failures >= 3 else "running",
                    (
                        f"Live cycle reported {result_status} ({consecutive_failures} consecutive): "
                        f"{result.get('reason', 'unknown error')}"
                    ),
                    consecutive_failures=consecutive_failures,
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_failed_at=failed_at,
                )
            else:
                consecutive_failures = 0
        except Exception as e:
            cycle_succeeded = False
            failed_at = datetime.now(timezone.utc).isoformat()
            if _runtime_bundle_refresh_blocked(e) and _ufc_refresh_cycle_in_progress():
                cycle_deferred_reason = str(e)
                update_runtime_component(
                    "betting_loop",
                    "running",
                    (
                        "Live trading paused while scheduled UFC refresh rebuilds the "
                        f"processed snapshot: {e}"
                    ),
                    consecutive_failures=consecutive_failures,
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_failed_at=failed_at,
                )
                logger.warning(
                    "Live betting paused while scheduled UFC refresh rebuilds processed data: %s",
                    e,
                )
            else:
                consecutive_failures += 1
                update_runtime_component(
                    "betting_loop",
                    "degraded" if consecutive_failures >= 3 else "running",
                    f"Live cycle failed ({consecutive_failures} consecutive): {e}",
                    consecutive_failures=consecutive_failures,
                    last_cycle_started_at=cycle_started_at,
                    last_cycle_failed_at=failed_at,
                )
                logger.error(f"Live betting error: {e}", exc_info=True)

        # Auto-settle any resolved markets each cycle
        try:
            _heartbeat("Cycle active: reconciling settled markets")
            from src.polymarket.tracker import (
                BetLedger,
                auto_reconcile_sold_positions,
                auto_settle_from_polymarket,
            )
            from src.strategy.duo_trader import get_all_trader_ledgers

            total_settled = 0
            for label, path in get_all_trader_ledgers():
                if Path(path).exists():
                    ledger = BetLedger(path=path)
                    settled = auto_settle_from_polymarket(ledger)
                    if settled:
                        logger.info(f"Auto-settled {settled} bets for Trader {label}")
                        total_settled += settled
            if total_settled:
                logger.info(f"Auto-settled {total_settled} bets total")

            # Reconcile positions sold on Polymarket but still "open" in ledger
            try:
                from src.polymarket.monitor import PositionMonitor
                monitor = PositionMonitor()
                live_positions = monitor.get_positions()
                live_tids = {
                    p.get("asset", p.get("token_id", ""))
                    for p in live_positions
                    if float(p.get("size", 0)) > 0
                }
                for label, path in get_all_trader_ledgers():
                    if Path(path).exists():
                        ledger = BetLedger(path=path)
                        reconciled = auto_reconcile_sold_positions(ledger, live_tids)
                        if reconciled:
                            logger.info(f"Reconciled {reconciled} sold positions for Trader {label}")
            except Exception as e:
                logger.warning(f"Sold-position reconciliation error: {e}")
        except Exception as e:
            logger.error(f"Auto-settle error: {e}")

        cycle_completed_at = datetime.now(timezone.utc).isoformat()
        if cycle_succeeded:
            update_runtime_component(
                "betting_loop",
                "degraded" if consecutive_failures >= 3 else "running",
                (
                    f"Last cycle completed at {cycle_completed_at}; "
                    f"next run in {interval_minutes} minutes"
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
            )
        elif cycle_deferred_reason is not None:
            update_runtime_component(
                "betting_loop",
                "running",
                (
                    f"Last cycle paused for scheduled UFC refresh before {cycle_completed_at}; "
                    f"next retry in {interval_minutes} minutes"
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
            )
        else:
            update_runtime_component(
                "betting_loop",
                "degraded" if consecutive_failures >= 3 else "running",
                (
                    f"Last cycle failed before {cycle_completed_at}; "
                    f"next retry in {interval_minutes} minutes"
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
            )
        logger.info(f"Next betting cycle in {interval_minutes} minutes")
        time.sleep(interval_minutes * 60)


def run_btc5m_live_loop(
    *,
    profile_name: str,
    ledger_path: Path,
    poll_seconds: float = BTC5M_POLL_SECONDS,
    trading_mode: str = LIVE_MODE_DRY_RUN,
    market_slug: str | None = None,
    host: str = "127.0.0.1",
    startup_delay_seconds: float = 10.0,
):
    """Run one promoted BTC 5m profile in a hosted background loop."""
    from src.polymarket import btc_5m
    from src.web.app import get_clob_client, update_runtime_component

    profile_name = str(profile_name or "").strip().lower()
    ledger_path = Path(ledger_path)
    component = _btc5m_component_name(profile_name or "unknown")
    heartbeat_window = max(60.0, float(poll_seconds) * 3.0)

    def _update(state: str, message: str, **metadata) -> None:
        update_runtime_component(
            component,
            state,
            message,
            profile=profile_name,
            ledger_path=str(ledger_path),
            trading_mode=trading_mode,
            poll_seconds=float(poll_seconds),
            stale_after_seconds=heartbeat_window,
            **metadata,
        )

    def _issue_message(prefix: str, *, reason: str, reason_code: str = "", status_code=None, endpoint: str = "") -> str:
        if reason_code == "upstream_rate_limited":
            detail = f"HTTP {status_code}" if status_code else "upstream rate limit"
            if endpoint:
                detail = f"{detail} from {endpoint}"
            return f"{prefix} rate-limited: {detail}"
        if reason_code == "upstream_geo_blocked":
            detail = f"HTTP {status_code}" if status_code else "upstream geoblock"
            if endpoint:
                detail = f"{detail} from {endpoint}"
            return f"{prefix} blocked by upstream access check: {detail}"
        return f"{prefix}: {reason or 'unknown error'}"

    def _http_exception_metadata(exc: Exception) -> dict:
        status_code = btc_5m._http_error_status(exc)
        if status_code is None:
            return {}

        response = getattr(exc, "response", None)
        endpoint = btc_5m._diagnostic_endpoint(getattr(response, "url", "") or "")
        reason_code = (
            "upstream_geo_blocked"
            if status_code in btc_5m.BTC5M_GEO_BLOCK_STATUSES
            else "upstream_rate_limited"
            if status_code in btc_5m.BTC5M_RATE_LIMIT_STATUSES
            else "upstream_http_error"
        )
        metadata = {
            "last_http_status": status_code,
            "last_http_endpoint": endpoint,
            "last_result_reason_code": reason_code,
        }
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After")
        if retry_after:
            metadata["last_retry_after"] = retry_after
        return metadata

    def _pause_while_stop_active(phase: str, **metadata) -> bool:
        paused = False
        while True:
            try:
                stop_status = _btc5m_stop_status()
            except Exception as exc:
                logger.warning("BTC 5m emergency stop status check failed for %s: %s", profile_name, exc)
                return paused
            if not stop_status.get("active"):
                if paused:
                    _update(
                        "running",
                        f"BTC 5m profile {profile_name} resumed after emergency stop.",
                        consecutive_failures=consecutive_failures,
                        emergency_stop=stop_status,
                        **metadata,
                    )
                return paused
            if not paused:
                _update(
                    "paused",
                    f"BTC 5m profile {profile_name} paused by emergency request {phase}.",
                    consecutive_failures=consecutive_failures,
                    emergency_stop=stop_status,
                    **metadata,
                )
                logger.warning(
                    "BTC 5m hosted loop paused by emergency request (profile=%s, phase=%s)",
                    profile_name,
                    phase,
                )
                paused = True
            time.sleep(1.0)

    if startup_delay_seconds > 0:
        time.sleep(startup_delay_seconds)

    if profile_name not in BTC5M_PROFILES:
        _update("disabled", f"BTC 5m profile {profile_name!r} is not configured.")
        logger.error("BTC 5m loop disabled: unknown profile %s", profile_name)
        return

    if trading_mode not in {LIVE_MODE_DRY_RUN, LIVE_MODE_REAL}:
        _update(
            "disabled",
            f"Trading mode '{trading_mode}' does not start BTC 5m profile {profile_name}.",
            consecutive_failures=0,
        )
        logger.info("BTC 5m loop disabled for %s: unsupported trading mode %s", profile_name, trading_mode)
        return

    dry_run = trading_mode != LIVE_MODE_REAL
    if not dry_run:
        try:
            assert_polymarket_real_trading_allowed(
                host=host,
                startup_source=f"serve:btc5m:{profile_name}",
                ledger_paths=(ledger_path,),
            )
        except Exception as exc:
            _update(
                "disabled",
                f"Real BTC 5m trading blocked for {profile_name}: {exc}",
                consecutive_failures=0,
                last_error=str(exc),
            )
            logger.error("BTC 5m real loop blocked for %s: %s", profile_name, exc)
            return

    consecutive_failures = 0
    _pause_while_stop_active("before startup")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _update(
        "running",
        f"BTC 5m profile {profile_name} active in {trading_mode} mode.",
        consecutive_failures=0,
        dry_run=dry_run,
        clob_available=get_clob_client() is not None,
    )
    logger.info(
        "BTC 5m hosted loop started (profile=%s, mode=%s, ledger=%s, poll=%.1fs)",
        profile_name,
        trading_mode,
        ledger_path,
        float(poll_seconds),
    )

    runner = None
    cycle = 0
    daily_loss_limit_hit_at: str | None = None
    while True:
        _pause_while_stop_active("before the next cycle")

        cycle += 1
        cycle_started = datetime.now(timezone.utc)
        cycle_started_at = cycle_started.isoformat()
        shared_clob = get_clob_client()

        if not dry_run and shared_clob is None:
            _update(
                "degraded",
                f"BTC 5m profile {profile_name} waiting for shared CLOB client before real execution.",
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                clob_available=False,
            )
            if _btc5m_sleep_until_next_poll(poll_seconds):
                _pause_while_stop_active(
                    "while waiting for shared CLOB before real execution",
                    last_cycle_started_at=cycle_started_at,
                    clob_available=False,
                    dry_run=dry_run,
                )
            continue

        try:
            if runner is None:
                runner = btc_5m.Btc5mRunner(
                    profile=btc_5m.resolve_btc5m_profile(profile_name),
                    ledger_path=ledger_path,
                    clob_client=shared_clob,
                    record_signal_snapshots=True,
                )
            elif shared_clob is not None and runner.clob_client is not shared_clob:
                runner.clob_client = shared_clob

            settled_count = 0
            ledger = getattr(runner, "ledger", None)
            if _btc5m_ledger_has_due_settlement(ledger, now=cycle_started):
                try:
                    from src.polymarket.tracker import auto_settle_from_polymarket

                    settled_count = auto_settle_from_polymarket(ledger)
                    if settled_count:
                        logger.info(
                            "BTC 5m auto-settled %s bet(s) for profile %s",
                            settled_count,
                            profile_name,
                        )
                except Exception as exc:
                    logger.warning(
                        "BTC 5m auto-settle failed for profile %s: %s",
                        profile_name,
                        exc,
                    )

            _update(
                "running",
                f"BTC 5m profile {profile_name} cycle started at {cycle_started_at}.",
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_settled_count=settled_count,
                clob_available=shared_clob is not None,
            )
            result = runner.run_once(dry_run=dry_run, market_slug=market_slug)
            cycle_completed_at = datetime.now(timezone.utc).isoformat()
            status = str(result.get("status") or "unknown")
            reason = _btc5m_result_reason(result)
            reason_code = _btc5m_result_reason_code(result)
            if reason_code == "daily_loss_limit":
                daily_loss_limit_hit_at = daily_loss_limit_hit_at or cycle_completed_at
            else:
                daily_loss_limit_hit_at = None
            daily_loss_metadata = _btc5m_daily_loss_metadata(
                reason_code=reason_code,
                result=result,
                profile=runner.profile,
                hit_at=daily_loss_limit_hit_at,
                current=cycle_started,
            )
            if status == "error":
                consecutive_failures += 1
                state = "degraded"
                status_code = result.get("http_status")
                endpoint = str(result.get("http_endpoint") or "")
                message = _issue_message(
                    f"BTC 5m profile {profile_name} cycle error ({consecutive_failures} consecutive)",
                    reason=reason,
                    reason_code=reason_code,
                    status_code=status_code,
                    endpoint=endpoint,
                )
            else:
                consecutive_failures = 0
                state = "running"
                message = (
                    f"BTC 5m profile {profile_name} last cycle {status} at "
                    f"{cycle_completed_at}; next poll in {float(poll_seconds):g}s"
                )
            _update(
                state,
                message,
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
                last_result_status=status,
                last_result_reason=reason,
                last_result_reason_code=reason_code,
                last_error=reason if status == "error" else None,
                last_http_status=result.get("http_status"),
                last_http_endpoint=result.get("http_endpoint"),
                last_retry_after=result.get("retry_after"),
                last_market_slug=result.get("market_slug"),
                last_order_count=len(result.get("orders", []) or []),
                last_settled_count=settled_count,
                clob_available=shared_clob is not None,
                dry_run=dry_run,
                **daily_loss_metadata,
            )
            if _pause_while_stop_active(
                "after completing the current cycle",
                last_cycle_started_at=cycle_started_at,
                last_cycle_completed_at=cycle_completed_at,
                last_result_status=status,
                last_result_reason=reason,
                last_result_reason_code=reason_code,
                last_market_slug=result.get("market_slug"),
                last_order_count=len(result.get("orders", []) or []),
                last_settled_count=settled_count,
                clob_available=shared_clob is not None,
                dry_run=dry_run,
                **daily_loss_metadata,
            ):
                continue
            logger.info(
                "BTC 5m hosted cycle %s/%s: status=%s reason=%s market=%s",
                profile_name,
                cycle,
                status,
                reason,
                result.get("market_slug"),
            )
        except Exception as exc:
            consecutive_failures += 1
            failed_at = datetime.now(timezone.utc).isoformat()
            issue_metadata = _http_exception_metadata(exc)
            reason_code = str(issue_metadata.get("last_result_reason_code") or "")
            status_code = issue_metadata.get("last_http_status")
            endpoint = str(issue_metadata.get("last_http_endpoint") or "")
            _update(
                "degraded",
                _issue_message(
                    f"BTC 5m profile {profile_name} cycle failed ({consecutive_failures} consecutive)",
                    reason=str(exc),
                    reason_code=reason_code,
                    status_code=status_code,
                    endpoint=endpoint,
                ),
                consecutive_failures=consecutive_failures,
                last_cycle_started_at=cycle_started_at,
                last_cycle_failed_at=failed_at,
                last_error=str(exc),
                clob_available=shared_clob is not None,
                dry_run=dry_run,
                **issue_metadata,
            )
            logger.error("BTC 5m hosted loop error for %s: %s", profile_name, exc, exc_info=True)

        if _btc5m_sleep_until_next_poll(poll_seconds):
            _pause_while_stop_active("while waiting for the next cycle")
            continue


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _method_odds_runtime_metadata(signals: dict) -> tuple[str, dict]:
    snapshot = signals.get("method_odds_snapshot") if isinstance(signals, dict) else None
    if not isinstance(snapshot, dict) or not snapshot:
        return "", {}

    metadata = {"method_odds_snapshot": snapshot}
    status = str(snapshot.get("status", "") or "unknown").lower()
    record_count = _safe_int(snapshot.get("record_count"), 0)
    if status == "success":
        metadata["method_odds_effective_status"] = "current"
        metadata["method_odds_status_message"] = f"Method-odds refresh succeeded ({record_count} records)."
        return f"; method odds success ({record_count} records)", metadata

    unavailable = status == "unavailable"
    state_message = "Method-odds props are not currently published" if unavailable else "Method-odds refresh failed"
    latest_usable = snapshot.get("latest_usable_snapshot")
    if isinstance(latest_usable, dict) and latest_usable:
        fallback_time = latest_usable.get("snapshot_time") or "unknown time"
        fallback_count = _safe_int(latest_usable.get("record_count"), 0)
        is_stale = bool(latest_usable.get("is_stale"))
        metadata["method_odds_effective_status"] = "stale_fallback" if is_stale else "fresh_fallback"
        freshness = "stale" if is_stale else "fresh"
        message = (
            f"{state_message}; "
            f"latest usable snapshot is {freshness} from {fallback_time} ({fallback_count} records)."
        )
        metadata["method_odds_status_message"] = message
        return f"; {message[:-1].lower()}", metadata

    message = f"{state_message}; no matching snapshot is available."
    metadata["method_odds_effective_status"] = "unavailable"
    metadata["method_odds_status_message"] = message
    return f"; {message[:-1].lower()}", metadata


def run_background_monitor(interval_hours: float = 6.0):
    """Run the monitor + line tracker in a background loop."""
    from src.web.app import update_runtime_component, write_market_intel_artifact

    # Wait for the web server to start before doing anything heavy
    time.sleep(10)
    heartbeat_window = max(600.0, interval_hours * 3600 * 2.5)
    line_tracking_watchdog_seconds = min(300.0, max(60.0, heartbeat_window / 3.0))

    update_runtime_component(
        "monitor_loop",
        "running",
        "Background monitor active.",
        stale_after_seconds=heartbeat_window,
    )
    logger.info(f"Background monitor started (every {interval_hours}h)")

    while True:
        cycle_started_at = datetime.now(timezone.utc).isoformat()

        def _heartbeat(message: str, *, state: str = "running", **metadata) -> None:
            update_runtime_component(
                "monitor_loop",
                state,
                message,
                last_cycle_started_at=cycle_started_at,
                **metadata,
            )

        _heartbeat(f"Monitor cycle started at {cycle_started_at}")
        try:
            _heartbeat("Monitor cycle active: reconciling settled markets")
            from src.polymarket.tracker import (
                BetLedger,
                auto_reconcile_sold_positions,
                auto_redeem_positions_from_polymarket,
                auto_settle_from_polymarket,
            )
            from src.strategy.duo_trader import get_all_trader_ledgers

            total_settled = 0
            for label, path in get_all_trader_ledgers():
                if Path(path).exists():
                    ledger = BetLedger(path=path)
                    settled = auto_settle_from_polymarket(ledger)
                    if settled:
                        logger.info(f"Auto-settled {settled} bets for Trader {label}")
                        total_settled += settled
            if total_settled:
                logger.info(f"Auto-settled {total_settled} bets total")

            # Reconcile positions sold on Polymarket but still "open" in ledger
            try:
                from src.polymarket.monitor import PositionMonitor
                monitor = PositionMonitor()
                live_positions = monitor.get_positions()
                live_tids = {
                    p.get("asset", p.get("token_id", ""))
                    for p in live_positions
                    if float(p.get("size", 0)) > 0
                }
                total_reconciled = 0
                for label, path in get_all_trader_ledgers():
                    if Path(path).exists():
                        ledger = BetLedger(path=path)
                        reconciled = auto_reconcile_sold_positions(ledger, live_tids)
                        if reconciled:
                            logger.info(f"Reconciled {reconciled} sold positions for Trader {label}")
                            total_reconciled += reconciled
                if total_reconciled:
                    logger.info(f"Reconciled {total_reconciled} sold positions total")
            except Exception as e:
                logger.warning(f"Sold-position reconciliation error: {e}")

            if _auto_redeem_enabled():
                redeem_summary = auto_redeem_positions_from_polymarket(
                    wait=False,
                    source="auto",
                )
                _log_auto_redeem_summary(redeem_summary, wait=False)
        except Exception as e:
            logger.error(f"Auto-settle error: {e}")
            _heartbeat(f"Auto-settle error: {e}", state="degraded")

        try:
            from src.data.live_monitor import run_monitoring_pass

            _heartbeat("Monitor cycle active: scanning live events")
            signals = run_monitoring_pass()
            logger.info(f"Monitor pass: {len(signals.get('events', []))} events")
            method_odds_suffix, method_odds_metadata = _method_odds_runtime_metadata(signals)
            _heartbeat(
                "Monitor cycle active: monitoring pass complete "
                f"({len(signals.get('events', []))} events){method_odds_suffix}; starting line tracking",
                **method_odds_metadata,
            )
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            _heartbeat(f"Monitor error: {e}", state="degraded")

        try:
            from src.data.line_tracker import run_line_tracking_pass

            line_progress_lock = threading.Lock()
            line_progress_message = "Line tracking: starting"
            line_tracking_stop = threading.Event()

            def _line_tracking_progress(message: str) -> None:
                nonlocal line_progress_message
                with line_progress_lock:
                    line_progress_message = message
                _heartbeat(message)

            def _line_tracking_watchdog() -> None:
                while not line_tracking_stop.wait(line_tracking_watchdog_seconds):
                    with line_progress_lock:
                        message = line_progress_message
                    _heartbeat(f"{message} (still running)", line_tracking_watchdog=True)

            watchdog_thread = threading.Thread(
                target=_line_tracking_watchdog,
                name="line-tracking-heartbeat",
                daemon=True,
            )
            watchdog_thread.start()
            try:
                line_summary = run_line_tracking_pass(progress_callback=_line_tracking_progress)
            finally:
                line_tracking_stop.set()
                watchdog_thread.join(timeout=1.0)
            write_market_intel_artifact(line_summary)
            logger.info(f"Line tracking: {line_summary.get('sharp_moves', 0)} sharp moves")
        except Exception as e:
            logger.error(f"Line tracking error: {e}")
            _heartbeat(f"Line tracking error: {e}", state="degraded")

        cycle_completed_at = datetime.now(timezone.utc).isoformat()
        _heartbeat(
            f"Last monitor cycle completed at {cycle_completed_at}",
            last_cycle_completed_at=cycle_completed_at,
        )
        time.sleep(interval_hours * 3600)


def main():
    port = int(os.environ.get("PORT", 5050))
    # Hosted deployments must bind all interfaces so Railway can reach the process.
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    monitor_interval = float(os.environ.get("MONITOR_INTERVAL_HOURS", "6"))
    bet_interval = float(os.environ.get("BET_INTERVAL_MINUTES", "10"))
    min_edge = float(os.environ.get("MIN_EDGE", str(MIN_EDGE_THRESHOLD)))
    model_name = resolve_live_model_name(os.environ.get("LIVE_MODEL"))
    bundle_summary = None

    logger.info(f"Starting on {host}:{port}")

    try:
        bundle_summary = _resolve_hosted_bundle_startup_summary()
    except Exception as exc:
        logger.error("Hosted production bundle validation failed: %s", exc)
        raise SystemExit(1) from exc

    from src.web.app import (
        btc5m_emergency_stop_status,
        register_runtime_thread,
        set_clob_client,
        set_runtime_status,
        start_server,
        update_runtime_component,
    )

    runtime_status = evaluate_live_startup(
        requested_mode=os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF),
        host=host,
        model_name=model_name,
        startup_source="serve",
    )
    btc5m_profile_names, btc5m_profile_errors = _btc5m_live_profile_names()
    btc5m_ledger_paths = tuple(_btc5m_live_ledger_path(profile) for profile in btc5m_profile_names)
    btc5m_startup_status = None
    btc5m_stop_status = btc5m_emergency_stop_status()
    if btc5m_profile_names and not btc5m_profile_errors:
        try:
            _btc5m_live_ledger_dir().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            btc5m_profile_errors.append(f"Could not create BTC5M_LIVE_LEDGER_DIR: {exc}")
        if not btc5m_profile_errors:
            btc5m_startup_status = evaluate_polymarket_live_startup(
                requested_mode=os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF),
                host=host,
                startup_source="serve:btc5m",
                ledger_paths=btc5m_ledger_paths,
            )
    btc5m_trading_enabled = bool(
        btc5m_startup_status
        and btc5m_startup_status.get("trading_enabled")
        and not btc5m_profile_errors
    )
    if btc5m_profile_errors:
        btc5m_loop_state = "disabled"
        btc5m_loop_message = "BTC 5m loop disabled by invalid BTC5M_LIVE_PROFILES."
    elif not btc5m_profile_names:
        btc5m_loop_state = "disabled"
        btc5m_loop_message = "No BTC5M_LIVE_PROFILES configured; BTC 5m hosted loop dormant."
    elif btc5m_trading_enabled:
        if btc5m_stop_status.get("active"):
            btc5m_loop_state = "paused"
            btc5m_loop_message = "BTC 5m emergency stop is active; hosted profile loops will pause until resume."
        else:
            btc5m_loop_state = "starting"
            btc5m_loop_message = (
                f"BTC 5m profiles starting in {btc5m_startup_status['effective_live_mode']} mode: "
                + ", ".join(btc5m_profile_names)
            )
    else:
        btc5m_loop_state = "disabled"
        requested_btc_mode = (
            (btc5m_startup_status or {}).get("requested_live_mode")
            or os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF)
        )
        btc5m_loop_message = (
            "BTC 5m loop disabled by LIVE_TRADING_MODE=off."
            if requested_btc_mode == LIVE_MODE_OFF
            else "BTC 5m loop blocked by Polymarket readiness checks."
        )

    runtime_status["components"] = {
        "web": {"state": "starting", "message": f"Binding {host}:{port}"},
        "monitor_loop": {"state": "starting", "message": "Monitor thread booting."},
        "clob": {"state": "starting", "message": "CLOB initialization pending."},
        "ufc_refresh_loop": {
            "state": "starting" if _ufc_refresh_enabled() else "disabled",
            "message": (
                "Hosted UFC refresh loop booting."
                if _ufc_refresh_enabled()
                else "Hosted UFC refresh loop disabled."
            ),
        },
        "betting_loop": {
            "state": "starting" if runtime_status["trading_enabled"] else "disabled",
            "message": (
                f"Trading mode: {runtime_status['effective_live_mode']}"
                if runtime_status["trading_enabled"]
                else (
                    "Trading loop disabled by policy."
                    if runtime_status["requested_live_mode"] == LIVE_MODE_OFF
                    else "Trading loop blocked by readiness checks."
                )
            ),
        },
        "btc5m_loop": {
            "state": btc5m_loop_state,
            "message": btc5m_loop_message,
            "profiles": btc5m_profile_names,
            "ledger_dir": str(_btc5m_live_ledger_dir()),
            "errors": btc5m_profile_errors,
            "emergency_stop": btc5m_stop_status if btc5m_stop_status.get("active") else None,
        },
    }
    for profile_name in btc5m_profile_names:
        runtime_status["components"][_btc5m_component_name(profile_name)] = {
            "state": (
                "paused"
                if btc5m_trading_enabled and btc5m_stop_status.get("active")
                else "starting" if btc5m_trading_enabled
                else "disabled"
            ),
            "message": (
                "BTC 5m profile paused by emergency stop."
                if btc5m_trading_enabled and btc5m_stop_status.get("active")
                else f"BTC 5m profile {profile_name} thread booting."
                if btc5m_trading_enabled
                else (
                    "BTC 5m profile not started because emergency stop is active."
                    if btc5m_stop_status.get("active")
                    else "BTC 5m profile not started."
                )
            ),
            "profile": profile_name,
            "ledger_path": str(_btc5m_live_ledger_path(profile_name)),
            "emergency_stop": btc5m_stop_status if btc5m_stop_status.get("active") else None,
        }
    runtime_status["btc5m_live_profiles"] = btc5m_profile_names
    runtime_status["btc5m_live_ledger_dir"] = str(_btc5m_live_ledger_dir())
    runtime_status["btc5m_live_startup"] = btc5m_startup_status
    runtime_status["btc5m_emergency_stop"] = btc5m_stop_status
    if bundle_summary is not None:
        runtime_status["production_bundle"] = bundle_summary
    set_runtime_status(runtime_status)

    if runtime_status["errors"]:
        logger.error(
            "Hosted trading startup is not ready: %s",
            " | ".join(runtime_status["errors"]),
        )
    if runtime_status["warnings"]:
        logger.warning(
            "Hosted trading startup warnings: %s",
            " | ".join(runtime_status["warnings"]),
        )
    if btc5m_profile_errors:
        logger.error("BTC 5m startup profile errors: %s", " | ".join(btc5m_profile_errors))
    if btc5m_startup_status and btc5m_startup_status.get("errors"):
        logger.error(
            "Hosted BTC 5m startup is not ready: %s",
            " | ".join(btc5m_startup_status["errors"]),
        )
    if btc5m_startup_status and btc5m_startup_status.get("warnings"):
        logger.warning(
            "Hosted BTC 5m startup warnings: %s",
            " | ".join(btc5m_startup_status["warnings"]),
        )

    if _ufc_refresh_enabled():
        refresh_thread = threading.Thread(
            target=run_background_ufc_refresh_loop,
            kwargs={
                "interval_hours": _ufc_refresh_interval_hours(),
                "initial_delay_seconds": _ufc_refresh_initial_delay_seconds(),
                "limit_fighters": _ufc_refresh_limit_fighters(),
            },
            daemon=True,
        )
        register_runtime_thread("ufc_refresh_loop", refresh_thread)
        refresh_thread.start()
    else:
        update_runtime_component(
            "ufc_refresh_loop",
            "disabled",
            "Hosted UFC refresh loop not started.",
        )

    if runtime_status["trading_enabled"]:
        betting_thread = threading.Thread(
            target=run_live_betting_loop,
            kwargs={
                "interval_minutes": bet_interval,
                "min_edge": min_edge,
                "trading_mode": runtime_status["effective_live_mode"],
                "model_name": model_name,
            },
            daemon=True,
        )
        register_runtime_thread("betting_loop", betting_thread)
        betting_thread.start()
    else:
        update_runtime_component(
            "betting_loop",
            "disabled",
            "Trading loop not started.",
        )

    if btc5m_trading_enabled and btc5m_startup_status is not None:
        startup_stop_active = bool(btc5m_stop_status.get("active"))
        update_runtime_component(
            "btc5m_loop",
            "paused" if startup_stop_active else "running",
            (
                "BTC 5m emergency stop is active; hosted profile loops are paused until resume."
                if startup_stop_active
                else (
                    f"BTC 5m hosted loop running {len(btc5m_profile_names)} profile(s) "
                    f"in {btc5m_startup_status['effective_live_mode']} mode."
                )
            ),
            profiles=btc5m_profile_names,
            ledger_dir=str(_btc5m_live_ledger_dir()),
            emergency_stop=btc5m_stop_status if startup_stop_active else None,
        )
        for profile_name, ledger_path in zip(btc5m_profile_names, btc5m_ledger_paths):
            btc_thread = threading.Thread(
                target=run_btc5m_live_loop,
                kwargs={
                    "profile_name": profile_name,
                    "ledger_path": ledger_path,
                    "poll_seconds": BTC5M_POLL_SECONDS,
                    "trading_mode": btc5m_startup_status["effective_live_mode"],
                    "host": host,
                },
                daemon=True,
            )
            component_name = _btc5m_component_name(profile_name)
            register_runtime_thread(component_name, btc_thread)
            btc_thread.start()
    else:
        update_runtime_component(
            "btc5m_loop",
            btc5m_loop_state,
            btc5m_loop_message,
            profiles=btc5m_profile_names,
            ledger_dir=str(_btc5m_live_ledger_dir()),
            errors=btc5m_profile_errors,
            emergency_stop=btc5m_stop_status if btc5m_stop_status.get("active") else None,
        )

    monitor_thread = threading.Thread(
        target=run_background_monitor,
        args=(monitor_interval,),
        daemon=True,
    )
    register_runtime_thread("monitor_loop", monitor_thread)
    monitor_thread.start()

    def _init_clob():
        time.sleep(2)  # let Flask bind the port first
        try:
            from src.polymarket.client import ClobClientWrapper

            clob = ClobClientWrapper()
            set_clob_client(clob)
            update_runtime_component("clob", "running", "Connected to Polymarket CLOB.")
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            update_runtime_component("clob", "degraded", str(e))
            logger.warning(f"Running without CLOB (no live prices): {e}")

    clob_thread = threading.Thread(target=_init_clob, daemon=True)
    clob_thread.start()

    update_runtime_component("web", "running", f"Serving on http://{host}:{port}")
    start_server(port=port, debug=False, clob_client=None, host=host)


if __name__ == "__main__":
    main()
