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

from src.config import LOGS_DIR, MIN_EDGE_THRESHOLD
from src.live_control import (
    LIVE_MODE_DRY_RUN,
    LIVE_MODE_ENV,
    LIVE_MODE_OFF,
    LIVE_MODE_REAL,
    evaluate_live_startup,
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
logger = logging.getLogger(__name__)


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


def _ufc_refresh_interval_hours() -> float:
    raw = str(os.getenv("UFC_REFRESH_INTERVAL_HOURS", "168") or "").strip()
    try:
        value = float(raw)
    except Exception:
        return 168.0
    return max(value, 1.0)


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


def _ufc_refresh_pct_threshold(env_name: str) -> float | None:
    raw = str(os.getenv(env_name, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value


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
    identity_audit_rows = int(roster_sync.get("identity_audit_rows") or 0)
    if identity_audit_rows:
        counts = roster_sync.get("identity_audit_action_counts") or {}
        if isinstance(counts, dict) and counts:
            detail_parts: list[str] = []
            test_rows = int(counts.get("excluded_test_profile") or 0)
            full_quarantine_rows = int(counts.get("quarantined_untrusted_url_identity") or 0)
            slug_alias_rows = int(counts.get("quarantined_untrusted_slug_alias") or 0)
            other_rows = max(
                0,
                identity_audit_rows - test_rows - full_quarantine_rows - slug_alias_rows,
            )
            if test_rows:
                detail_parts.append(f"{test_rows} test/staging excluded")
            if full_quarantine_rows:
                detail_parts.append(f"{full_quarantine_rows} URL identity quarantined")
            if slug_alias_rows:
                detail_parts.append(f"{slug_alias_rows} slug aliases suppressed")
            if other_rows:
                detail_parts.append(f"{other_rows} other flagged")
            detail = ", ".join(detail_parts)
            alerts.append(f"official UFC roster identity audit flagged {identity_audit_rows} row(s): {detail}")
        else:
            alerts.append(f"official UFC roster identity audit flagged {identity_audit_rows} row(s)")

    row_drop_guard = refresh_summary.get("row_drop_guard") or {}
    for violation in row_drop_guard.get("violations") or []:
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
    interval_hours: float = 168.0,
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
            coverage_notes = _ufc_refresh_coverage_notes(coverage_snapshot)
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
    from src.web.app import update_runtime_component

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

            cancelled = cancel_all_stale_limit_bids()
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
            )
            result = cmd_duo_live(args)
            if isinstance(result, dict) and result.get("status") == "error":
                cycle_succeeded = False
                consecutive_failures += 1
                failed_at = datetime.now(timezone.utc).isoformat()
                update_runtime_component(
                    "betting_loop",
                    "degraded" if consecutive_failures >= 3 else "running",
                    (
                        f"Live cycle reported failure ({consecutive_failures} consecutive): "
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
            consecutive_failures += 1
            failed_at = datetime.now(timezone.utc).isoformat()
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


def run_background_monitor(interval_hours: float = 6.0):
    """Run the monitor + line tracker in a background loop."""
    from src.web.app import update_runtime_component

    # Wait for the web server to start before doing anything heavy
    time.sleep(10)
    heartbeat_window = max(600.0, interval_hours * 3600 * 2.5)

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
            _heartbeat(
                "Monitor cycle active: monitoring pass complete "
                f"({len(signals.get('events', []))} events); starting line tracking"
            )
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            _heartbeat(f"Monitor error: {e}", state="degraded")

        try:
            from src.data.line_tracker import run_line_tracking_pass

            line_summary = run_line_tracking_pass(progress_callback=_heartbeat)
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
    }
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

    monitor_thread = threading.Thread(
        target=run_background_monitor,
        args=(monitor_interval,),
        daemon=True,
    )
    register_runtime_thread("monitor_loop", monitor_thread)
    monitor_thread.start()

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
