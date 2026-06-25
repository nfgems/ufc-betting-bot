"""Configuration settings for the UFC betting bot."""

import logging as _logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = str(os.getenv(name, "") or "").strip()
    return Path(raw) if raw else default


def _safe_float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Invalid value for %s=%r, using default %s", name, raw, default
        )
        return float(default)


def _safe_int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Invalid value for %s=%r, using default %s", name, raw, default
        )
        return int(default)


def _is_truthy_env(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _has_model_artifacts(path: Path) -> bool:
    try:
        return any(path.glob("*_model.pkl"))
    except OSError:
        return False


def _resolve_default_models_dir(
    project_root: Path,
    data_dir: Path,
    *,
    hosted_project_root: Path = Path("/app"),
) -> Path:
    """Use image-bundled hosted models unless an explicit override is provided."""
    legacy_models_dir = project_root / "models"

    if project_root != hosted_project_root:
        return legacy_models_dir
    return legacy_models_dir


def _railway_volume_mount_path() -> Path | None:
    raw = str(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "") or "").strip()
    return Path(raw) if raw else None


def _resolve_default_logs_dir(
    project_root: Path,
    data_dir: Path,
    *,
    hosted_project_root: Path = Path("/app"),
    hosted_volume_mount: Path | None = None,
) -> Path:
    default_logs_dir = data_dir / "logs"
    if project_root != hosted_project_root:
        return default_logs_dir
    return hosted_volume_mount if hosted_volume_mount is not None else default_logs_dir


DATA_DIR = _path_from_env("UFC_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = _path_from_env("UFC_MODELS_DIR", _resolve_default_models_dir(PROJECT_ROOT, DATA_DIR))
LOGS_DIR = _path_from_env(
    "UFC_LOGS_DIR",
    _resolve_default_logs_dir(
        PROJECT_ROOT,
        DATA_DIR,
        hosted_volume_mount=_railway_volume_mount_path(),
    ),
)
BETSAPI_RAW_DIR = RAW_DATA_DIR / "betsapi"
BETSAPI_MMA_RAW_DIR = BETSAPI_RAW_DIR / "mma"
BETSAPI_MMA_PROCESSED_DIR = PROCESSED_DATA_DIR / "betsapi" / "mma"

# Ensure directories exist
_config_logger = _logging.getLogger(__name__)
_required_dir_failures = []

for _d in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    BETSAPI_RAW_DIR,
    BETSAPI_MMA_RAW_DIR,
    BETSAPI_MMA_PROCESSED_DIR,
]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError as _e:
        _required_dir_failures.append((_d, _e))

if _required_dir_failures:
    _failure_summary = "; ".join(
        f"{_path}: {_error}" for _path, _error in _required_dir_failures
    )
    _config_logger.error("Required directory initialization failed: %s", _failure_summary)
    raise RuntimeError(f"Failed to create required project directories: {_failure_summary}")

# UFC Stats scraper
# Note: UFCStats dropped TLS support — HTTP is intentional here.
# See commit d7151b2 for context.
UFCSTATS_DOMAIN = "http://ufcstats.com"
UFCSTATS_BASE_URL = f"{UFCSTATS_DOMAIN}/statistics/events/completed"
UFCSTATS_FIGHTER_URL = f"{UFCSTATS_DOMAIN}/fighter-details/"
UFCSTATS_EVENT_URL = f"{UFCSTATS_DOMAIN}/event-details/"
UFCSTATS_FIGHT_URL = f"{UFCSTATS_DOMAIN}/fight-details/"
UFCSTATS_FIGHTER_SEARCH_URL = f"{UFCSTATS_DOMAIN}/statistics/fighters"
UFCSTATS_UPCOMING_URL = f"{UFCSTATS_DOMAIN}/statistics/events/upcoming"

# Fallback fighter data sources (when UFCStats has no data)
SHERDOG_BASE_URL = "https://www.sherdog.com"
SHERDOG_SEARCH_URL = "https://www.sherdog.com/stats/fightfinder"
TAPOLOGY_BASE_URL = "https://www.tapology.com"
TAPOLOGY_SEARCH_URL = "https://www.tapology.com/search"
TAPOLOGY_PROXY_URL = os.getenv("TAPOLOGY_PROXY_URL", "").strip()
TAPOLOGY_BROWSER_FALLBACK_ENABLED = _is_truthy_env("TAPOLOGY_BROWSER_FALLBACK_ENABLED", "0")
TAPOLOGY_BROWSER_BINARY = os.getenv("TAPOLOGY_BROWSER_BINARY", "").strip()
TAPOLOGY_CHROMEDRIVER_BINARY = os.getenv("TAPOLOGY_CHROMEDRIVER_BINARY", "").strip()
TAPOLOGY_XVFB_BINARY = os.getenv("TAPOLOGY_XVFB_BINARY", "").strip()
TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS = _safe_int_env(
    "TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS", "20"
)
TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS = _safe_float_env(
    "TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS", "20"
)
TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS = _safe_float_env(
    "TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS", "3"
)
TAPOLOGY_READER_FALLBACK_ENABLED = _is_truthy_env("TAPOLOGY_READER_FALLBACK_ENABLED", "1")
TAPOLOGY_READER_BASE_URL = os.getenv("TAPOLOGY_READER_BASE_URL", "https://r.jina.ai/").strip()
TAPOLOGY_READER_TIMEOUT_SECONDS = _safe_float_env("TAPOLOGY_READER_TIMEOUT_SECONDS", "45")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
BRAVE_SEARCH_API_URL = os.getenv(
    "BRAVE_SEARCH_API_URL",
    "https://api.search.brave.com/res/v1/web/search",
).strip()
BRAVE_SEARCH_HTML_FALLBACK_ENABLED = _is_truthy_env(
    "BRAVE_SEARCH_HTML_FALLBACK_ENABLED",
    "0",
)
BRAVE_SEARCH_TIMEOUT_SECONDS = _safe_float_env("BRAVE_SEARCH_TIMEOUT_SECONDS", "12")
DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED = _is_truthy_env(
    "DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED",
    "1",
)
DUCKDUCKGO_SEARCH_HTML_URL = os.getenv(
    "DUCKDUCKGO_SEARCH_HTML_URL",
    "https://html.duckduckgo.com/html/",
).strip()
MARTIALBOT_BASE_URL = "https://www.martialbot.com"
# MartialBot is a client-rendered app; fighter search is served by its JSON API
# (requires both `name` and `sport`), and profile bios come from the React Router
# single-fetch ".data" route appended to the profile URL.
MARTIALBOT_SEARCH_URL = "https://www.martialbot.com/api/fighters-search"
FIGHTDX_BASE_URL = "https://fightdx.com/person"
FIGHTDX_REQUEST_TIMEOUT_SECONDS = max(
    _safe_float_env("FIGHTDX_REQUEST_TIMEOUT_SECONDS", "8"), 1.0
)
FIGHTDX_FAILURE_COOLDOWN_SECONDS = max(
    _safe_float_env("FIGHTDX_FAILURE_COOLDOWN_SECONDS", "180"), 0.0
)

# The Odds API
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "mma_mixed_martial_arts"
METHOD_ODDS_BFO_REQUEST_TIMEOUT_SECONDS = max(
    _safe_float_env("METHOD_ODDS_BFO_REQUEST_TIMEOUT_SECONDS", "12"), 1.0
)
METHOD_ODDS_BFO_MAX_RETRIES = max(
    _safe_int_env("METHOD_ODDS_BFO_MAX_RETRIES", "1"), 0
)
METHOD_ODDS_BFO_RETRY_BACKOFF_SECONDS = max(
    _safe_float_env("METHOD_ODDS_BFO_RETRY_BACKOFF_SECONDS", "2"), 0.0
)
METHOD_ODDS_BFO_FAILURE_BUDGET = max(
    _safe_int_env("METHOD_ODDS_BFO_FAILURE_BUDGET", "2"), 1
)
BETSAPI_TOKEN = os.getenv("BETSAPI_TOKEN", "")
BETSAPI_BASE_URL = "https://api.b365api.com/v3"
BETSAPI_MMA_SPORT_ID = 162


BETSAPI_REQUEST_MIN_INTERVAL_SECONDS = _safe_float_env("BETSAPI_REQUEST_MIN_INTERVAL_SECONDS", "1")
BETSAPI_429_RETRY_MIN_SECONDS = _safe_float_env("BETSAPI_429_RETRY_MIN_SECONDS", "15")

# Activity monitor durable alert retention — every WARNING/ERROR/CRITICAL is
# mirrored to a dedicated alerts.jsonl (independent of bot.log's INFO volume) so
# warnings/errors persist in the dashboard's Activity view for at least this many
# hours instead of scrolling out of the recent-log window. Defaults to 72h so an
# overnight or weekend error is still visible the next time you check.
ACTIVITY_ALERT_RETENTION_HOURS = max(
    _safe_float_env("ACTIVITY_ALERT_RETENTION_HOURS", "72"), 1.0
)

# Polymarket
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")  # Proxy wallet shown on Polymarket profile
POLYMARKET_CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))  # Polygon
POLYMARKET_CLOB_URL = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_API_URL = "https://data-api.polymarket.com"

# BTC 5-minute Polymarket runner. These controls are intentionally conservative
# because the markets resolve quickly and crypto taker fees are enabled.
BTC5M_LEDGER_PATH = _path_from_env("BTC5M_LEDGER_PATH", LOGS_DIR / "bet_ledger_btc5m.json")
BTC5M_SIGNAL_LOG_PATH = _path_from_env("BTC5M_SIGNAL_LOG_PATH", LOGS_DIR / "btc5m_signal_snapshots.jsonl")
BTC5M_SIGNAL_LOG_ENABLED = _is_truthy_env("BTC5M_SIGNAL_LOG_ENABLED", "0")
BTC5M_EXIT_SHADOW_LOG_PATH = _path_from_env("BTC5M_EXIT_SHADOW_LOG_PATH", LOGS_DIR / "btc5m_exit_shadow.jsonl")
BTC5M_EXIT_SHADOW_ENABLED = _is_truthy_env("BTC5M_EXIT_SHADOW_ENABLED", "0")
BTC5M_PAPER_LEDGER_DIR = _path_from_env("BTC5M_PAPER_LEDGER_DIR", LOGS_DIR / "btc5m_paper")
BTC5M_LIVE_LEDGER_DIR = _path_from_env("BTC5M_LIVE_LEDGER_DIR", LOGS_DIR / "btc5m_live")
BTC5M_LIVE_PROFILES = os.getenv("BTC5M_LIVE_PROFILES", "")
BTC5M_PAPER_PROFILES = os.getenv(
    "BTC5M_PAPER_PROFILES",
    "conservative,balanced,late,confidence,cheap_side,late_capture",
)
BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS = max(
    _safe_float_env("BTC5M_PAPER_SETTLEMENT_DELAY_SECONDS", "45"),
    0.0,
)
BTC5M_PAPER_SETTLEMENT_SOURCE = os.getenv("BTC5M_PAPER_SETTLEMENT_SOURCE", "polymarket").strip().lower()
BTC5M_OPPORTUNITY_TARGET_MARKETS = max(
    _safe_int_env("BTC5M_OPPORTUNITY_TARGET_MARKETS", "500"),
    1,
)
BTC5M_MARKET_SLUG_PREFIX = os.getenv("BTC5M_MARKET_SLUG_PREFIX", "btc-updown-5m")
BTC5M_BINANCE_API_BASE_URL = os.getenv(
    "BTC5M_BINANCE_API_BASE_URL",
    "https://data-api.binance.vision",
).rstrip("/")
BTC5M_BINANCE_SYMBOL = os.getenv("BTC5M_BINANCE_SYMBOL", "BTCUSDT")
BTC5M_COINBASE_API_BASE_URL = os.getenv(
    "BTC5M_COINBASE_API_BASE_URL",
    "https://api.exchange.coinbase.com",
).rstrip("/")
BTC5M_COINBASE_PRODUCT_ID = os.getenv("BTC5M_COINBASE_PRODUCT_ID", "BTC-USD")
BTC5M_HYPERLIQUID_API_BASE_URL = os.getenv(
    "BTC5M_HYPERLIQUID_API_BASE_URL",
    "https://api.hyperliquid.xyz",
).rstrip("/")
BTC5M_PRICE_SOURCE = os.getenv("BTC5M_PRICE_SOURCE", "binance").strip().lower()
BTC5M_REQUEST_TIMEOUT_SECONDS = max(_safe_float_env("BTC5M_REQUEST_TIMEOUT_SECONDS", "12"), 1.0)
BTC5M_REQUEST_RETRIES = max(_safe_int_env("BTC5M_REQUEST_RETRIES", "2"), 0)
BTC5M_DEFAULT_PROFILE = os.getenv("BTC5M_DEFAULT_PROFILE", "conservative").strip().lower()
BTC5M_POLL_SECONDS = max(_safe_float_env("BTC5M_POLL_SECONDS", "10"), 1.0)
BTC5M_COOLDOWN_WINDOWS_AFTER_TRADE = max(_safe_int_env("BTC5M_COOLDOWN_WINDOWS_AFTER_TRADE", "0"), 0)
BTC5M_ENTRY_SECONDS_LEFT = max(_safe_float_env("BTC5M_ENTRY_SECONDS_LEFT", "120"), 1.0)
BTC5M_ENTRY_TOLERANCE_SECONDS = max(_safe_float_env("BTC5M_ENTRY_TOLERANCE_SECONDS", "45"), 0.0)
BTC5M_MIN_MOVE_USD = max(_safe_float_env("BTC5M_MIN_MOVE_USD", "70"), 0.0)
BTC5M_MIN_SUPPORTING_PROB = min(
    max(_safe_float_env("BTC5M_MIN_SUPPORTING_PROB", "0.52"), 0.0),
    1.0,
)
BTC5M_MAX_ENTRY_PRICE = min(
    max(_safe_float_env("BTC5M_MAX_ENTRY_PRICE", "0.80"), 0.01),
    0.99,
)
BTC5M_MAX_SPREAD = min(max(_safe_float_env("BTC5M_MAX_SPREAD", "0.08"), 0.0), 1.0)
BTC5M_MIN_TOP_ASK_NOTIONAL = max(_safe_float_env("BTC5M_MIN_TOP_ASK_NOTIONAL", "5"), 0.0)
BTC5M_MIN_TOTAL_ASK_NOTIONAL = max(_safe_float_env("BTC5M_MIN_TOTAL_ASK_NOTIONAL", "20"), 0.0)
BTC5M_TRADE_NOTIONAL_USD = max(_safe_float_env("BTC5M_TRADE_NOTIONAL_USD", "5"), 0.0)
BTC5M_MAX_NOTIONAL_PER_TRADE = max(_safe_float_env("BTC5M_MAX_NOTIONAL_PER_TRADE", "10"), 0.0)
BTC5M_MAX_OPEN_EXPOSURE_USD = max(_safe_float_env("BTC5M_MAX_OPEN_EXPOSURE_USD", "25"), 0.0)
BTC5M_MAX_TRADES_PER_DAY = max(_safe_int_env("BTC5M_MAX_TRADES_PER_DAY", "4"), 0)
BTC5M_DAILY_LOSS_LIMIT_USD = max(_safe_float_env("BTC5M_DAILY_LOSS_LIMIT_USD", "25"), 0.0)
BTC5M_ALLOCATION_FRACTION = min(
    max(_safe_float_env("BTC5M_ALLOCATION_FRACTION", "0.02"), 0.0),
    1.0,
)
BTC5M_ENABLE_EXTREME_SKEW_HEDGE = _is_truthy_env("BTC5M_ENABLE_EXTREME_SKEW_HEDGE", "0")
BTC5M_HEDGE_SKEW_THRESHOLD = min(
    max(_safe_float_env("BTC5M_HEDGE_SKEW_THRESHOLD", "0.95"), 0.5),
    0.99,
)
BTC5M_HEDGE_NOTIONAL_USD = max(_safe_float_env("BTC5M_HEDGE_NOTIONAL_USD", "2"), 0.0)
BTC5M_HEDGE_MAX_PRICE = min(
    max(_safe_float_env("BTC5M_HEDGE_MAX_PRICE", "0.08"), 0.01),
    0.49,
)

BTC5M_LATE_CAPTURE_PROFILE = {
    "trade_notional_usd": 5.0,
    "max_notional_per_trade": 10.0,
    "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
    "allocation_fraction": 0.02,
    "max_trades_per_day": 0,  # disabled
    "daily_loss_limit_usd": 25.0,
    "entry_seconds_left": 32.5,
    "entry_tolerance_seconds": 27.5,
    "min_move_usd": 0.0,
    "min_supporting_prob": 0.85,
    "min_entry_price": 0.85,
    "max_entry_price": 0.95,
    "max_spread": 0.03,
    "max_spread_fraction": 0.02,
    "min_top_ask_notional": 5.0,
    "min_total_ask_notional": 20.0,
    "strategy_style": "probability_capture",
    "cooldown_windows_after_trade": 0,  # disabled
    "enable_extreme_skew_hedge": False,
    "hedge_skew_threshold": 0.95,
    "hedge_notional_usd": 2.0,
    "hedge_max_price": 0.08,
}

BTC5M_CHEAP_SIDE_PROFILE = {
    "trade_notional_usd": 5.0,
    "max_notional_per_trade": 10.0,
    "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
    "allocation_fraction": 0.02,
    "max_trades_per_day": 0,  # disabled
    "daily_loss_limit_usd": 25.0,
    "entry_seconds_left": 195.0,
    "entry_tolerance_seconds": 105.0,
    "min_move_usd": 0.0,
    "min_supporting_prob": 0.0,
    "min_entry_price": 0.15,
    "max_entry_price": 0.45,
    "max_spread": 0.10,
    "min_top_ask_notional": 5.0,
    "min_total_ask_notional": 20.0,
    "strategy_style": "cheap_side",
    "cooldown_windows_after_trade": 0,  # disabled
    "enable_extreme_skew_hedge": False,
    "hedge_skew_threshold": 0.95,
    "hedge_notional_usd": 2.0,
    "hedge_max_price": 0.08,
}

BTC5M_PROFILES = {
    "conservative": {
        "trade_notional_usd": BTC5M_TRADE_NOTIONAL_USD,
        "max_notional_per_trade": BTC5M_MAX_NOTIONAL_PER_TRADE,
        "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
        "allocation_fraction": BTC5M_ALLOCATION_FRACTION,
        "max_trades_per_day": 0,  # disabled
        "daily_loss_limit_usd": BTC5M_DAILY_LOSS_LIMIT_USD,
        "entry_seconds_left": BTC5M_ENTRY_SECONDS_LEFT,
        "entry_tolerance_seconds": BTC5M_ENTRY_TOLERANCE_SECONDS,
        "min_move_usd": BTC5M_MIN_MOVE_USD,
        "min_supporting_prob": BTC5M_MIN_SUPPORTING_PROB,
        "max_entry_price": BTC5M_MAX_ENTRY_PRICE,
        "max_spread": BTC5M_MAX_SPREAD,
        "min_top_ask_notional": BTC5M_MIN_TOP_ASK_NOTIONAL,
        "min_total_ask_notional": BTC5M_MIN_TOTAL_ASK_NOTIONAL,
        "cooldown_windows_after_trade": 0,  # disabled
        "enable_extreme_skew_hedge": BTC5M_ENABLE_EXTREME_SKEW_HEDGE,
        "hedge_skew_threshold": BTC5M_HEDGE_SKEW_THRESHOLD,
        "hedge_notional_usd": BTC5M_HEDGE_NOTIONAL_USD,
        "hedge_max_price": BTC5M_HEDGE_MAX_PRICE,
    },
    "balanced": {
        "trade_notional_usd": 10.0,
        "max_notional_per_trade": 25.0,
        "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
        "allocation_fraction": 0.05,
        "max_trades_per_day": 0,  # disabled
        "daily_loss_limit_usd": 50.0,
        "entry_seconds_left": 120.0,
        "entry_tolerance_seconds": 60.0,
        "min_move_usd": 70.0,
        "min_supporting_prob": 0.50,
        "max_entry_price": 0.88,
        "max_spread": 0.10,
        "min_top_ask_notional": 5.0,
        "min_total_ask_notional": 25.0,
        "cooldown_windows_after_trade": 0,
        "enable_extreme_skew_hedge": False,
        "hedge_skew_threshold": 0.95,
        "hedge_notional_usd": 2.0,
        "hedge_max_price": 0.08,
    },
    "late": {
        "trade_notional_usd": 5.0,
        "max_notional_per_trade": 10.0,
        "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
        "allocation_fraction": 0.02,
        "max_trades_per_day": 0,  # disabled
        "daily_loss_limit_usd": 25.0,
        "entry_seconds_left": 15.0,
        "entry_tolerance_seconds": 10.0,
        "min_move_usd": 10.0,
        "min_supporting_prob": 0.50,
        "max_entry_price": 0.75,
        "max_spread": 0.10,
        "min_top_ask_notional": 5.0,
        "min_total_ask_notional": 20.0,
        "cooldown_windows_after_trade": 0,  # disabled
        "enable_extreme_skew_hedge": False,
        "hedge_skew_threshold": 0.95,
        "hedge_notional_usd": 2.0,
        "hedge_max_price": 0.08,
    },
    "confidence": {
        "trade_notional_usd": 5.0,
        "max_notional_per_trade": 5.0,
        "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
        "allocation_fraction": 0.02,
        "max_trades_per_day": 0,  # disabled
        "daily_loss_limit_usd": 25.0,
        "entry_seconds_left": 120.0,
        "entry_tolerance_seconds": 120.0,
        "min_move_usd": 10.0,
        "min_supporting_prob": 0.70,
        "max_entry_price": 0.83,
        "max_spread": 0.08,
        "min_top_ask_notional": 5.0,
        "min_total_ask_notional": 20.0,
        "order_style": "maker_limit",
        "cooldown_windows_after_trade": 0,
        "enable_extreme_skew_hedge": False,
        "hedge_skew_threshold": 0.95,
        "hedge_notional_usd": 2.0,
        "hedge_max_price": 0.08,
    },
    "cheap_side": {**BTC5M_CHEAP_SIDE_PROFILE},
    "cheap_below30": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "min_entry_price": 0.05,
        "max_entry_price": 0.30,
    },
    "cheap_below20": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "min_entry_price": 0.05,
        "max_entry_price": 0.20,
    },
    "cheap_below10": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "min_entry_price": 0.03,
        "max_entry_price": 0.10,
    },
    "cheap_below20_early": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "entry_seconds_left": 240.0,
        "entry_tolerance_seconds": 60.0,
        "min_entry_price": 0.05,
        "max_entry_price": 0.20,
    },
    "cheap_below10_early": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "entry_seconds_left": 240.0,
        "entry_tolerance_seconds": 60.0,
        "min_entry_price": 0.03,
        "max_entry_price": 0.10,
    },
    "cheap_below20_liq": {
        **BTC5M_CHEAP_SIDE_PROFILE,
        "min_entry_price": 0.05,
        "max_entry_price": 0.20,
        "min_top_ask_notional": 10.0,
        "min_total_ask_notional": 40.0,
    },
    "late_capture": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
    },
    "late_capture_min86": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.86,
    },
    "late_capture_min88": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.88,
    },
    "late_capture_min90": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.90,
    },
    "late_capture_min92": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.92,
    },
    "late_capture_gap005": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "trade_notional_usd": 50.0,
        "max_notional_per_trade": 50.0,
        "allocation_fraction": 1.0,
        "daily_loss_limit_usd": 200.0,
        "max_entry_price": 0.97,
        "max_entry_support_gap": 0.005,
    },
    "late_capture_cap94": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.94,
    },
    "late_capture_cap93": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.93,
    },
    "late_capture_30_60": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 45.0,
        "entry_tolerance_seconds": 15.0,
    },
    "late_capture_45_60": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 52.5,
        "entry_tolerance_seconds": 7.5,
    },
    "late_capture_v2": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 45.0,
        "entry_tolerance_seconds": 15.0,
        "min_supporting_prob": 0.88,
        "max_entry_price": 0.94,
        "max_entry_support_gap": 0.005,
    },
    "late_capture_mid_gap005": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "trade_notional_usd": 25.0,
        "max_notional_per_trade": 25.0,
        "daily_loss_limit_usd": 100.0,
        "entry_seconds_left": 90.0,
        "entry_tolerance_seconds": 60.0,
        "max_entry_price": 0.97,
        "max_entry_support_gap": 0.005,
    },
    "late_capture_mid_min88": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 90.0,
        "entry_tolerance_seconds": 60.0,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.88,
    },
    "late_capture_full_gap005": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "max_entry_support_gap": 0.005,
    },
    "late_capture_full_gap010": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "max_entry_support_gap": 0.010,
    },
    "late_capture_full_min88": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.88,
    },
    "late_capture_full_min90": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "min_supporting_prob": 0.90,
    },
    "late_capture_gap005_min88": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "trade_notional_usd": 50.0,
        "max_notional_per_trade": 50.0,
        "allocation_fraction": 1.0,
        "daily_loss_limit_usd": 200.0,
        "max_entry_price": 0.97,
        "max_entry_support_gap": 0.005,
        "min_supporting_prob": 0.88,
    },
    "late_capture_gap010_min88": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "max_entry_price": 0.97,
        "max_entry_support_gap": 0.010,
        "min_supporting_prob": 0.88,
    },
    "late_capture_full_min88_liq": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "max_entry_price": 0.97,
        "min_supporting_prob": 0.88,
        "min_top_ask_notional": 10.0,
        "min_total_ask_notional": 40.0,
    },
    "late_capture_full_gap005_liq": {
        **BTC5M_LATE_CAPTURE_PROFILE,
        "entry_seconds_left": 150.0,
        "entry_tolerance_seconds": 150.0,
        "max_entry_support_gap": 0.005,
        "min_top_ask_notional": 10.0,
        "min_total_ask_notional": 40.0,
    },
    "aggressive": {
        "trade_notional_usd": 25.0,
        "max_notional_per_trade": 100.0,
        "max_open_exposure_usd": 1_000_000_000.0,  # disabled (unbounded)
        "allocation_fraction": 0.10,
        "max_trades_per_day": 0,  # disabled
        "daily_loss_limit_usd": 150.0,
        "entry_seconds_left": 120.0,
        "entry_tolerance_seconds": 75.0,
        "min_move_usd": 50.0,
        "min_supporting_prob": 0.50,
        "max_entry_price": 0.95,
        "max_spread": 0.15,
        "min_top_ask_notional": 5.0,
        "min_total_ask_notional": 30.0,
        "cooldown_windows_after_trade": 0,
        "enable_extreme_skew_hedge": True,
        "hedge_skew_threshold": 0.95,
        "hedge_notional_usd": 2.0,
        "hedge_max_price": 0.08,
    },
}

BTC5M_ALT_5M_ASSETS = {
    "eth": {
        "asset_symbol": "ETH",
        "asset_name": "Ethereum",
        "market_slug_prefix": "eth-updown-5m",
        "price_source": "binance",
        "price_source_fallbacks": ["coinbase", "hyperliquid"],
        "binance_symbol": "ETHUSDT",
        "coinbase_product_id": "ETH-USD",
        "hyperliquid_coin": "ETH",
    },
    "sol": {
        "asset_symbol": "SOL",
        "asset_name": "Solana",
        "market_slug_prefix": "sol-updown-5m",
        "price_source": "binance",
        "price_source_fallbacks": ["coinbase", "hyperliquid"],
        "binance_symbol": "SOLUSDT",
        "coinbase_product_id": "SOL-USD",
        "hyperliquid_coin": "SOL",
    },
    "xrp": {
        "asset_symbol": "XRP",
        "asset_name": "XRP",
        "market_slug_prefix": "xrp-updown-5m",
        "price_source": "binance",
        "price_source_fallbacks": ["coinbase", "hyperliquid"],
        "binance_symbol": "XRPUSDT",
        "coinbase_product_id": "XRP-USD",
        "hyperliquid_coin": "XRP",
    },
    "doge": {
        "asset_symbol": "DOGE",
        "asset_name": "Dogecoin",
        "market_slug_prefix": "doge-updown-5m",
        "price_source": "binance",
        "price_source_fallbacks": ["coinbase", "hyperliquid"],
        "binance_symbol": "DOGEUSDT",
        "coinbase_product_id": "DOGE-USD",
        "hyperliquid_coin": "DOGE",
    },
    "hype": {
        "asset_symbol": "HYPE",
        "asset_name": "Hyperliquid",
        "market_slug_prefix": "hype-updown-5m",
        "price_source": "hyperliquid",
        "hyperliquid_coin": "@107",
        "binance_symbol": "HYPEUSDT",
        "coinbase_product_id": "HYPE-USD",
    },
    "bnb": {
        "asset_symbol": "BNB",
        "asset_name": "BNB",
        "market_slug_prefix": "bnb-updown-5m",
        "price_source": "binance",
        "price_source_fallbacks": ["coinbase", "hyperliquid"],
        "binance_symbol": "BNBUSDT",
        "coinbase_product_id": "BNB-USD",
        "hyperliquid_coin": "BNB",
    },
}

BTC5M_ALT_5M_PROFILE_OVERRIDES = {
    "trade_notional_usd": 50.0,
    "max_notional_per_trade": 50.0,
    "allocation_fraction": 1.0,
}

for _asset_key, _asset_overrides in BTC5M_ALT_5M_ASSETS.items():
    BTC5M_PROFILES[f"{_asset_key}_late_capture_gap005"] = {
        **BTC5M_PROFILES["late_capture_gap005"],
        **BTC5M_ALT_5M_PROFILE_OVERRIDES,
        **_asset_overrides,
    }
    BTC5M_PROFILES[f"{_asset_key}_late_capture_gap005_min88"] = {
        **BTC5M_PROFILES["late_capture_gap005_min88"],
        **BTC5M_ALT_5M_PROFILE_OVERRIDES,
        **_asset_overrides,
    }

# Buy-cheap-early x take-profit sweep matrix.
# Cross product of {price band} x {early-entry timing} x {take-profit lock level}
# built on the cheap_side base so we can compare, in one run, how far each cheap
# band should let a winner run before locking it. Naming: cheap_<band>_<timing>_<tp>.
#   band:   b10 = 0.03-0.10, b20 = 0.05-0.20, b30 = 0.05-0.30 (the cheap ask range to buy)
#   timing: ve = 270+/-30 (enters 5:00-4:00 left), e = 240+/-60 (5:00-3:00),
#           em = 210+/-90 (5:00-2:00) -- higher seconds-left = earlier in the window
#   tp:     tp05/tp08/tp12 = profit_target +0.05/+0.08/+0.12 over entry, trail-arm and
#           giveback scaled with it. tp08 reproduces the historical cheap-side defaults.
# Take-profit fields only affect the take_profit_exit policy ranking; profiles that
# share a (band, timing) have identical entries and identical hold_to_settlement rows.
_CHEAP_SWEEP_BANDS = {
    "b10": (0.03, 0.10),
    "b20": (0.05, 0.20),
    "b30": (0.05, 0.30),
}
_CHEAP_SWEEP_TIMINGS = {
    "ve": (270.0, 30.0),
    "e": (240.0, 60.0),
    "em": (210.0, 90.0),
}
_CHEAP_SWEEP_TPS = {
    "tp05": {"tp_profit_target": 0.05, "tp_trail_arm": 0.07, "tp_trail_giveback": 0.04},
    "tp08": {"tp_profit_target": 0.08, "tp_trail_arm": 0.10, "tp_trail_giveback": 0.06},
    "tp12": {"tp_profit_target": 0.12, "tp_trail_arm": 0.14, "tp_trail_giveback": 0.08},
}
for _band_key, (_band_min, _band_max) in _CHEAP_SWEEP_BANDS.items():
    for _timing_key, (_entry_secs, _entry_tol) in _CHEAP_SWEEP_TIMINGS.items():
        for _tp_key, _tp_overrides in _CHEAP_SWEEP_TPS.items():
            BTC5M_PROFILES[f"cheap_{_band_key}_{_timing_key}_{_tp_key}"] = {
                **BTC5M_CHEAP_SIDE_PROFILE,
                "min_entry_price": _band_min,
                "max_entry_price": _band_max,
                "entry_seconds_left": _entry_secs,
                "entry_tolerance_seconds": _entry_tol,
                **_tp_overrides,
            }

# Model settings
ROLLING_WINDOW = 5  # Number of recent fights for rolling averages
EWM_HALFLIFE = 3  # Exponential weighted mean halflife (in fights) for rolling stats
# Kept for backward compatibility. Some UFC helper APIs and tests still import
# these values even though current UFC training specs no longer depend on Elo.
ELO_INITIAL = 1500
ELO_K_FACTOR = 32
TRAIN_CUTOFF_DATE = "2022-01-01"  # Train on fights before this date

# Strategy settings
BLEND_WEIGHT = 0.30  # Model weight in market-model blend (0.3 model + 0.7 market)
MIN_EDGE_THRESHOLD = 0.02  # 2% minimum blended edge to place a bet
NEAR_MISS_MIN_EDGE = 0.01  # 1% — lower bound for near-miss limit order eligibility
KELLY_FRACTION = 0.25  # Quarter Kelly
MAX_BET_FRACTION = 0.04  # Never risk more than 4% of bankroll
STOP_LOSS_FRACTION = 0.60  # Stop if bankroll drops 60%
INITIAL_BANKROLL = 500.00  # Default starting bankroll in USD for backtests and dry-run fallback
REQUIRE_MODEL_AGREEMENT = True  # Both models must agree on the bet direction
MODEL_AGREEMENT_MIN_EDGE = 0.01  # No-odds model must show at least 1% edge

# Dynamic blend weight — adjust model weight based on conviction
BLEND_WEIGHT_MIN = 0.15  # Floor: low-confidence predictions defer to market
BLEND_WEIGHT_MAX = 0.50  # Ceiling: high-conviction predictions get more weight
BLEND_CONFIDENCE_THRESHOLD = 0.65  # Model confidence above this starts increasing weight
BLEND_AGREEMENT_BOOST = 0.10  # Extra weight when no-odds model strongly agrees (>5% edge)

# Line movement filter — advisory only in production live trading
LINE_MOVEMENT_FILTER = False  # Do not reject bets solely because the line moved against us
LINE_AGAINST_EXTRA_EDGE = 0.02  # Historical/backtest extra-edge setting if the filter is re-enabled
LINE_SHARP_BLOCK = False  # Historical/backtest hard-block setting if the filter is re-enabled

# Time-decay training — weight recent fights more heavily
TIME_DECAY_ENABLED = True
TIME_DECAY_HALF_LIFE_DAYS = 730  # 2 years: fights 2 years old get half the weight

# Fighter experience filter — skip fights with inexperienced fighters
MIN_FIGHTER_FIGHTS = 2  # Production default: allow 2-fight UFC samples, still skip true sub-2 newcomers
NEWBIE_FIGHTS_THRESHOLD = 3  # Tiered newbie handling applies below this UFC-fight count
NEWBIE_MAJOR_EXTRA_EDGE = 0.02  # Tier-1 pre-UFC orgs need +2% extra edge
NEWBIE_FEEDER_EXTRA_EDGE = 0.03  # Tier-2 pre-UFC orgs need +3% extra edge
NEWBIE_REGIONAL_EXTRA_EDGE = 0.04  # Tier-3/unknown orgs need +4% extra edge
NEWBIE_SIZE_MULTIPLIER = 0.50  # Half-size all tiered-rule newbie bets
NEWBIE_SKIP_ZERO_FIGHTS_REGIONAL = True  # Keep skipping true debutants without major/feeder history

# Underdog safeguards
MIN_MODEL_PROB = 0.40  # Don't bet on fighters below 40% blended probability
MAX_DECIMAL_ODDS = 3.0  # Skip anything above 3.0 decimal odds (+200)
EDGE_SCALING_BASE = 0.02  # Base edge at even money (2.0 odds)
EDGE_SCALING_RATE = 0.02  # Extra edge required per 1.0 increase in odds above 2.0

# Polymarket liquidity / slippage guardrails
MIN_BOOK_LIQUIDITY = 50.0  # Minimum USD available in the orderbook at or near our price
MAX_SLIPPAGE = 0.03  # Max 3% price slippage — skip if filling our size moves price more
MAX_BET_VS_BOOK_RATIO = 0.25  # Never take more than 25% of available book liquidity
MAX_BET_HOURS_BEFORE_EVENT = int(os.getenv("MAX_BET_HOURS_BEFORE_EVENT", "48"))  # Only place bets inside the final 48h before the fight

# Limit bid TTL — cancel and re-evaluate resting bids after this many hours
LIMIT_BID_TTL_HOURS = 24
LIMIT_REPRICE_TICK_THRESHOLD = 2  # Require at least a 2-tick mismatch before repricing
LIMIT_REPRICE_MIN_AGE_MINUTES = 30  # Don't surrender queue priority too quickly
LIMIT_REPRICE_MAX_UPDATES = 2  # Cap upward reprices per market/fighter to avoid chasing
# Pre-event cancellation — pull all limit bids this many hours before the event starts
LIMIT_BID_PRE_EVENT_HOURS = 2

# Injury/cancellation detection — extreme odds shifts are advisory market intel
INJURY_MOVE_THRESHOLD = 0.15  # 15% probability shift = large move warning
INJURY_PRICE_FLOOR = 0.05  # If either side drops below 5¢, warn about possible fight-breaking news
INJURY_BLOCK_BETS = False  # Injury/line-move detector is advisory; never block fights by itself

# Incremental prediction cache — reuse live predictions until inputs move enough
PREDICTION_CACHE_SCHEMA_VERSION = 2
PREDICTION_ODDS_CHANGE_THRESHOLD = 0.03  # Re-predict if consensus odds shift by >3pp
PREDICTION_MAX_AGE_HOURS = 12  # Force a refresh even if the fight inputs look unchanged

# Closing odds leakage mitigation — add noise to odds features during training
# to simulate the gap between current/opening odds and closing odds
ODDS_NOISE_STD = 0.04  # Std dev of Gaussian noise added to implied probabilities (4%)

# Duo-trader system — S (Single) gets full bankroll, C (Conviction) gets remaining
TRADER_C_SHARE = 1.0     # Conviction gets 100% of remaining bankroll after Single bets
# Deprecated: trackers now follow the shared 48h bet window and limit-order pre-event pull.
TRACKER_MIN_HOURS_BEFORE_EVENT = int(os.getenv("TRACKER_MIN_HOURS_BEFORE_EVENT", "24"))
GEMINI_TRACKER_CONFIDENCE_CAP = min(
    max(_safe_float_env("GEMINI_TRACKER_CONFIDENCE_CAP", "0.85"), 0.5),
    1.0,
)

# Trader C (Conviction) — bets on fighters all signals agree will win,
# regardless of whether odds offer traditional "value"
CONVICTION_MIN_MODEL_PROB = 0.65    # Model must be ≥65% confident
CONVICTION_MIN_NO_ODDS_PROB = 0.50  # No-odds model must independently agree (≥50%)
CONVICTION_BET_FRACTION = 0.05      # Flat 5% of bankroll per conviction bet
CONVICTION_CONFIDENCE_BONUS = 0.01  # Extra 1% sizing per 5% model prob above threshold
CONVICTION_MAX_BET_FRACTION = 0.08  # Hard cap at 8% of bankroll per bet

# Auto-retrain — retrain models if they're older than this many months
MODEL_RETRAIN_MONTHS = 1  # Retrain monthly before predict/live
