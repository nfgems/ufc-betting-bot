"""Live-trading policy and deploy-readiness checks."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOGS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

LIVE_MODE_OFF = "off"
LIVE_MODE_DRY_RUN = "dry-run"
LIVE_MODE_REAL = "real"
VALID_LIVE_MODES = {LIVE_MODE_OFF, LIVE_MODE_DRY_RUN, LIVE_MODE_REAL}

# Hard disable — set to True to kill all live trading immediately.
LIVE_TRADING_DISABLED = False

LIVE_MODE_ENV = "LIVE_TRADING_MODE"
LIVE_MODEL_ENV = "LIVE_MODEL"
REAL_TRADING_ARM_ENV = "LIVE_TRADING_ARMED"
REAL_TRADING_CONFIRM_ENV = "LIVE_TRADING_CONFIRMATION"
REAL_TRADING_CONFIRM_VALUE = "REAL_TRADING_ENABLED"

_LOCAL_ONLY_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_truthy(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on", "armed"}


def is_public_bind(host: str | None) -> bool:
    normalized = str(host or "").strip().lower()
    return normalized not in _LOCAL_ONLY_HOSTS


def normalize_live_mode(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "0", "false", "disabled", "none"}:
        return LIVE_MODE_OFF
    if normalized in {LIVE_MODE_OFF, "no", "stop"}:
        return LIVE_MODE_OFF
    if normalized in {LIVE_MODE_DRY_RUN, "dry_run", "dryrun", "paper", "paper-trading"}:
        return LIVE_MODE_DRY_RUN
    if normalized in {LIVE_MODE_REAL, "live"}:
        return LIVE_MODE_REAL
    return None


def resolve_live_mode_from_env() -> str:
    raw_value = os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF)
    normalized = normalize_live_mode(raw_value)
    return normalized if normalized is not None else LIVE_MODE_OFF


def resolve_live_model_name(model_name: str | None = None) -> str:
    resolved = str(model_name or os.environ.get(LIVE_MODEL_ENV, "xgboost") or "").strip()
    return resolved or "xgboost"


def _explicit_model_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.suffix == ".pkl" or candidate.is_absolute() or any(sep in model_name for sep in ("/", "\\")):
        return candidate
    return None


def _resolve_model_artifact_path(model_name: str) -> Path:
    explicit = _explicit_model_path(model_name)
    if explicit is not None:
        return explicit
    return MODELS_DIR / f"{model_name}_model.pkl"


def _resolve_no_odds_model_name(model_name: str) -> str | None:
    explicit = _explicit_model_path(model_name)
    if explicit is None:
        return "xgboost_no_odds"

    sibling = explicit.with_name("xgboost_no_odds_model.pkl")
    if sibling.exists():
        return str(sibling)
    return None


def _base_status(
    *,
    startup_source: str,
    requested_live_mode_raw: str,
    requested_live_mode: str,
    effective_live_mode: str,
    model_name: str,
    host: str,
    ready: bool,
    trading_enabled: bool,
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
) -> dict:
    return {
        "service": "ufc-betting-bot",
        "startup_source": startup_source,
        "requested_live_mode_raw": requested_live_mode_raw,
        "requested_live_mode": requested_live_mode,
        "effective_live_mode": effective_live_mode,
        "trading_enabled": trading_enabled,
        "trading_live": effective_live_mode == LIVE_MODE_REAL,
        "model_name": model_name,
        "host": host,
        "public_bind": is_public_bind(host),
        "armed_for_real": _is_truthy(os.environ.get(REAL_TRADING_ARM_ENV)),
        "ready": ready,
        "errors": list(errors),
        "warnings": list(warnings),
        "checks": list(checks),
        "components": {},
        "updated_at": _utcnow_iso(),
    }


def _record_check(
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
    *,
    name: str,
    ok: bool,
    message: str,
    severity: str = "error",
    **extra,
) -> None:
    entry = {
        "name": name,
        "ok": ok,
        "message": message,
        "severity": severity,
    }
    entry.update(extra)
    checks.append(entry)
    if ok:
        return
    if severity == "warning":
        warnings.append(message)
        return
    errors.append(message)


def _check_env_present(
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
    *,
    env_name: str,
    severity: str = "error",
    message: str | None = None,
) -> None:
    configured = str(os.environ.get(env_name, "") or "").strip()
    _record_check(
        checks,
        errors,
        warnings,
        name=env_name.lower(),
        ok=bool(configured),
        message=message or f"{env_name} is not configured.",
        severity=severity,
        env=env_name,
    )


def _check_exact_env_value(
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
    *,
    env_name: str,
    expected_value: str,
    message: str,
) -> None:
    configured = str(os.environ.get(env_name, "") or "").strip()
    _record_check(
        checks,
        errors,
        warnings,
        name=env_name.lower(),
        ok=configured == expected_value,
        message=message,
        severity="error",
        env=env_name,
        expected_value=expected_value,
    )


def _check_model_artifact(
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
    *,
    label: str,
    model_name: str,
) -> None:
    artifact_path = _resolve_model_artifact_path(model_name)
    _record_check(
        checks,
        errors,
        warnings,
        name=label,
        ok=artifact_path.exists() and artifact_path.is_file(),
        message=f"Missing required model artifact: {artifact_path}",
        severity="error",
        path=str(artifact_path),
        model_name=model_name,
    )


def _check_writable_file_path(
    checks: list[dict],
    errors: list[str],
    warnings: list[str],
    *,
    label: str,
    path: Path,
) -> None:
    parent = path.parent
    if not parent.exists():
        _record_check(
            checks,
            errors,
            warnings,
            name=label,
            ok=False,
            message=f"Missing parent directory for {path}",
            severity="error",
            path=str(path),
        )
        return

    if path.exists() and path.is_dir():
        _record_check(
            checks,
            errors,
            warnings,
            name=label,
            ok=False,
            message=f"Expected a file path but found a directory: {path}",
            severity="error",
            path=str(path),
        )
        return

    target = path if path.exists() else parent
    _record_check(
        checks,
        errors,
        warnings,
        name=label,
        ok=os.access(target, os.W_OK),
        message=f"Path is not writable: {path}",
        severity="error",
        path=str(path),
    )


def evaluate_live_startup(
    *,
    requested_mode: str | None = None,
    host: str = "127.0.0.1",
    model_name: str | None = None,
    startup_source: str = "serve",
) -> dict:
    """Return hosted/CLI live-trading readiness for the requested mode."""
    if LIVE_TRADING_DISABLED:
        logger.warning("Live trading is disabled (LIVE_TRADING_DISABLED=True in live_control.py). No trades will execute.")
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw="disabled",
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name=resolve_live_model_name(model_name),
            host=host,
            ready=True,
            trading_enabled=False,
            checks=[],
            errors=[],
            warnings=["Live trading is disabled via LIVE_TRADING_DISABLED flag in live_control.py."],
        )

    logger.info(
        "Evaluating live startup: mode=%s host=%s model=%s source=%s",
        requested_mode, host, model_name, startup_source,
    )
    requested_raw = str(
        requested_mode if requested_mode is not None else os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF)
    ).strip()
    normalized_mode = normalize_live_mode(requested_raw)
    resolved_model = resolve_live_model_name(model_name)

    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    if normalized_mode is None:
        message = (
            f"Invalid {LIVE_MODE_ENV} value '{requested_raw}'. "
            f"Allowed values: off, dry-run, real."
        )
        _record_check(
            checks,
            errors,
            warnings,
            name="live_mode",
            ok=False,
            message=message,
            severity="error",
            env=LIVE_MODE_ENV,
        )
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw=requested_raw,
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name=resolved_model,
            host=host,
            ready=False,
            trading_enabled=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    if normalized_mode == LIVE_MODE_OFF:
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw=requested_raw or LIVE_MODE_OFF,
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name=resolved_model,
            host=host,
            ready=True,
            trading_enabled=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    _check_env_present(
        checks,
        errors,
        warnings,
        env_name="ODDS_API_KEY",
        message="ODDS_API_KEY is required before the live trading loop may start.",
    )
    _check_model_artifact(
        checks,
        errors,
        warnings,
        label="primary_model_artifact",
        model_name=resolved_model,
    )

    no_odds_model = _resolve_no_odds_model_name(resolved_model)
    if no_odds_model is not None:
        _check_model_artifact(
            checks,
            errors,
            warnings,
            label="no_odds_model_artifact",
            model_name=no_odds_model,
        )

    _check_writable_file_path(
        checks,
        errors,
        warnings,
        label="bot_log_path",
        path=LOGS_DIR / "bot.log",
    )
    _check_writable_file_path(
        checks,
        errors,
        warnings,
        label="predictions_cache_path",
        path=LOGS_DIR / "predictions_cache.json",
    )
    _check_writable_file_path(
        checks,
        errors,
        warnings,
        label="single_ledger_path",
        path=LOGS_DIR / "bet_ledger_single.json",
    )
    _check_writable_file_path(
        checks,
        errors,
        warnings,
        label="conviction_ledger_path",
        path=LOGS_DIR / "bet_ledger_conviction.json",
    )
    public_bind = is_public_bind(host)
    dashboard_token = str(os.environ.get("WEB_DASHBOARD_TOKEN", "") or "").strip()
    if public_bind:
        _record_check(
            checks,
            errors,
            warnings,
            name="dashboard_token",
            ok=bool(dashboard_token),
            message=(
                "WEB_DASHBOARD_TOKEN must be set on public binds before the hosted service "
                f"may run in {normalized_mode} mode."
            ),
            severity="error" if normalized_mode == LIVE_MODE_REAL else "warning",
            env="WEB_DASHBOARD_TOKEN",
        )

    if normalized_mode == LIVE_MODE_REAL:
        _check_env_present(
            checks,
            errors,
            warnings,
            env_name="POLYMARKET_PRIVATE_KEY",
            message="POLYMARKET_PRIVATE_KEY is required for real-money trading.",
        )
        _check_exact_env_value(
            checks,
            errors,
            warnings,
            env_name=REAL_TRADING_ARM_ENV,
            expected_value="1",
            message=f"Set {REAL_TRADING_ARM_ENV}=1 to arm real-money trading.",
        )
        _check_exact_env_value(
            checks,
            errors,
            warnings,
            env_name=REAL_TRADING_CONFIRM_ENV,
            expected_value=REAL_TRADING_CONFIRM_VALUE,
            message=(
                f"Set {REAL_TRADING_CONFIRM_ENV}={REAL_TRADING_CONFIRM_VALUE} "
                "to confirm real-money startup."
            ),
        )
        _check_env_present(
            checks,
            errors,
            warnings,
            env_name="POLYMARKET_FUNDER_ADDRESS",
            severity="warning",
            message=(
                "POLYMARKET_FUNDER_ADDRESS is not configured. The bot will rely on "
                "proxy-wallet auto-discovery at runtime."
            ),
        )

    ready = not errors
    effective_mode = normalized_mode if ready else LIVE_MODE_OFF

    return _base_status(
        startup_source=startup_source,
        requested_live_mode_raw=requested_raw or normalized_mode,
        requested_live_mode=normalized_mode,
        effective_live_mode=effective_mode,
        model_name=resolved_model,
        host=host,
        ready=ready,
        trading_enabled=ready and normalized_mode in {LIVE_MODE_DRY_RUN, LIVE_MODE_REAL},
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def evaluate_polymarket_live_startup(
    *,
    requested_mode: str | None = None,
    host: str = "127.0.0.1",
    startup_source: str = "cli",
    ledger_paths: tuple[Path, ...] = (),
) -> dict:
    """Return readiness for non-UFC Polymarket execution.

    This keeps the same real-money arming controls as the UFC live bot, while
    avoiding UFC-specific dependencies such as ODDS_API_KEY and model artifacts.
    """
    if LIVE_TRADING_DISABLED:
        logger.warning("Live trading is disabled (LIVE_TRADING_DISABLED=True in live_control.py). No trades will execute.")
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw="disabled",
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name="polymarket",
            host=host,
            ready=True,
            trading_enabled=False,
            checks=[],
            errors=[],
            warnings=["Live trading is disabled via LIVE_TRADING_DISABLED flag in live_control.py."],
        )

    requested_raw = str(
        requested_mode if requested_mode is not None else os.environ.get(LIVE_MODE_ENV, LIVE_MODE_OFF)
    ).strip()
    normalized_mode = normalize_live_mode(requested_raw)

    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    if normalized_mode is None:
        message = (
            f"Invalid {LIVE_MODE_ENV} value '{requested_raw}'. "
            f"Allowed values: off, dry-run, real."
        )
        _record_check(
            checks,
            errors,
            warnings,
            name="live_mode",
            ok=False,
            message=message,
            severity="error",
            env=LIVE_MODE_ENV,
        )
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw=requested_raw,
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name="polymarket",
            host=host,
            ready=False,
            trading_enabled=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    if normalized_mode == LIVE_MODE_OFF:
        return _base_status(
            startup_source=startup_source,
            requested_live_mode_raw=requested_raw or LIVE_MODE_OFF,
            requested_live_mode=LIVE_MODE_OFF,
            effective_live_mode=LIVE_MODE_OFF,
            model_name="polymarket",
            host=host,
            ready=True,
            trading_enabled=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    for index, path in enumerate(ledger_paths, start=1):
        _check_writable_file_path(
            checks,
            errors,
            warnings,
            label=f"ledger_path_{index}",
            path=path,
        )

    public_bind = is_public_bind(host)
    dashboard_token = str(os.environ.get("WEB_DASHBOARD_TOKEN", "") or "").strip()
    if public_bind:
        _record_check(
            checks,
            errors,
            warnings,
            name="dashboard_token",
            ok=bool(dashboard_token),
            message=(
                "WEB_DASHBOARD_TOKEN must be set on public binds before hosted "
                f"Polymarket execution may run in {normalized_mode} mode."
            ),
            severity="error" if normalized_mode == LIVE_MODE_REAL else "warning",
            env="WEB_DASHBOARD_TOKEN",
        )

    if normalized_mode == LIVE_MODE_REAL:
        _check_env_present(
            checks,
            errors,
            warnings,
            env_name="POLYMARKET_PRIVATE_KEY",
            message="POLYMARKET_PRIVATE_KEY is required for real-money Polymarket trading.",
        )
        _check_exact_env_value(
            checks,
            errors,
            warnings,
            env_name=REAL_TRADING_ARM_ENV,
            expected_value="1",
            message=f"Set {REAL_TRADING_ARM_ENV}=1 to arm real-money trading.",
        )
        _check_exact_env_value(
            checks,
            errors,
            warnings,
            env_name=REAL_TRADING_CONFIRM_ENV,
            expected_value=REAL_TRADING_CONFIRM_VALUE,
            message=(
                f"Set {REAL_TRADING_CONFIRM_ENV}={REAL_TRADING_CONFIRM_VALUE} "
                "to confirm real-money startup."
            ),
        )
        _check_env_present(
            checks,
            errors,
            warnings,
            env_name="POLYMARKET_FUNDER_ADDRESS",
            severity="warning",
            message=(
                "POLYMARKET_FUNDER_ADDRESS is not configured. The bot will rely on "
                "proxy-wallet auto-discovery at runtime."
            ),
        )

    ready = not errors
    effective_mode = normalized_mode if ready else LIVE_MODE_OFF
    return _base_status(
        startup_source=startup_source,
        requested_live_mode_raw=requested_raw or normalized_mode,
        requested_live_mode=normalized_mode,
        effective_live_mode=effective_mode,
        model_name="polymarket",
        host=host,
        ready=ready,
        trading_enabled=ready and normalized_mode in {LIVE_MODE_DRY_RUN, LIVE_MODE_REAL},
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def assert_polymarket_real_trading_allowed(
    *,
    host: str = "127.0.0.1",
    startup_source: str = "cli",
    ledger_paths: tuple[Path, ...] = (),
) -> dict:
    """Raise unless generic Polymarket real-money execution is explicitly armed."""
    status = evaluate_polymarket_live_startup(
        requested_mode=LIVE_MODE_REAL,
        host=host,
        startup_source=startup_source,
        ledger_paths=ledger_paths,
    )
    if status["effective_live_mode"] == LIVE_MODE_REAL:
        return status

    details = "; ".join(status["errors"]) or "real-money trading is not armed."
    raise RuntimeError(
        "Real-money Polymarket trading blocked. "
        f"{details} "
        f"Set {REAL_TRADING_ARM_ENV}=1 and "
        f"{REAL_TRADING_CONFIRM_ENV}={REAL_TRADING_CONFIRM_VALUE} only after verifying readiness."
    )


def assert_real_trading_allowed(
    *,
    model_name: str | None = None,
    host: str = "127.0.0.1",
    startup_source: str = "cli",
) -> dict:
    """Raise with a concrete message unless real-money trading is explicitly armed."""
    logger.info("Checking real trading authorization: model=%s host=%s", model_name, host)
    status = evaluate_live_startup(
        requested_mode=LIVE_MODE_REAL,
        host=host,
        model_name=model_name,
        startup_source=startup_source,
    )
    if status["effective_live_mode"] == LIVE_MODE_REAL:
        return status

    details = "; ".join(status["errors"]) or "real-money trading is not armed."
    raise RuntimeError(
        "Real-money live trading blocked. "
        f"{details} "
        f"Set {REAL_TRADING_ARM_ENV}=1 and "
        f"{REAL_TRADING_CONFIRM_ENV}={REAL_TRADING_CONFIRM_VALUE} only after verifying readiness."
    )
