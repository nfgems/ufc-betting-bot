"""Configuration settings for the UFC betting bot."""

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
import logging as _logging
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
