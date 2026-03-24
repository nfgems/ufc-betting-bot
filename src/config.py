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
MARTIALBOT_BASE_URL = "https://www.martialbot.com"
MARTIALBOT_SEARCH_URL = "https://www.martialbot.com/mma/search"
FIGHTDX_BASE_URL = "https://fightdx.com/person"

# The Odds API
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "mma_mixed_martial_arts"
TENNIS_ODDS_SPORTS_PREFIX = ["tennis_atp_", "tennis_wta_"]
BETSAPI_TOKEN = os.getenv("BETSAPI_TOKEN", "")
BETSAPI_BASE_URL = "https://api.b365api.com/v3"
BETSAPI_MMA_SPORT_ID = 162


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


def _is_truthy_env(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

BETSAPI_REQUEST_MIN_INTERVAL_SECONDS = _safe_float_env("BETSAPI_REQUEST_MIN_INTERVAL_SECONDS", "1")
BETSAPI_429_RETRY_MIN_SECONDS = _safe_float_env("BETSAPI_429_RETRY_MIN_SECONDS", "15")

# Polymarket
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")  # Proxy wallet shown on Polymarket profile
POLYMARKET_CHAIN_ID = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))  # Polygon
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_API_URL = "https://data-api.polymarket.com"

# Model settings
ROLLING_WINDOW = 5  # Number of recent fights for rolling averages
EWM_HALFLIFE = 3  # Exponential weighted mean halflife (in fights) for rolling stats
# Kept for backward compatibility. Some UFC helper APIs and tests still import
# these values even though current UFC training specs no longer depend on Elo.
ELO_INITIAL = 1500
ELO_K_FACTOR = 32
TENNIS_ELO_K_FACTOR = 32
TENNIS_ELO_SURFACE_K = 24
TENNIS_MIN_MATCHES = 3
DEFAULT_TENNIS_MODEL_NAME = "lean_hybrid"
TENNIS_TRAINING_START_DATE = "2022-01-01"
TENNIS_OOS_START_DATE = "2023-01-01"
TENNIS_OOS_TEST_WINDOW_MONTHS = 6
TRAIN_CUTOFF_DATE = "2022-01-01"  # Train on fights before this date

# Strategy settings
BLEND_WEIGHT = 0.30  # Model weight in market-model blend (0.3 model + 0.7 market)
MIN_EDGE_THRESHOLD = 0.02  # 2% minimum blended edge to place a bet
TENNIS_BLEND_WEIGHT = 0.30  # Tennis model blend weight for discovery and dry-run analysis
TENNIS_MIN_EDGE_THRESHOLD = 0.02  # Tennis minimum edge threshold for experimental execution decisions
TENNIS_MIN_BOOKMAKERS = 3  # Minimum consensus bookmaker count for tennis market-comparison decisions
TENNIS_REFERENCE_EDGE_FLOOR = 0.0  # Require the core model to be at least non-negative vs bookmaker consensus
TENNIS_CONFIDENCE_PENALTY_THRESHOLD = 0.60  # Add edge penalty when the selected side is below this confidence
TENNIS_LOW_CONFIDENCE_EDGE_PENALTY = 0.01  # Extra required edge for low-confidence tennis decisions
TENNIS_LOW_HISTORY_EDGE_PENALTY = 0.02  # Extra required edge when a side is still in the 3-5 prior-match bucket
TENNIS_MEDIUM_HISTORY_EDGE_PENALTY = 0.01  # Extra required edge in the 6-10 prior-match bucket
TENNIS_SUSPICIOUS_REFERENCE_EDGE_THRESHOLD = 0.35  # Auto-skip extreme model-vs-market gaps that look structurally suspicious
TENNIS_SECOND_SOURCE_CONFIRMATION_GAP = 0.10  # Max chosen-side probability gap to treat a second market source as confirmatory
TENNIS_SECOND_SOURCE_CONTRADICTION_GAP = 0.20  # Gap large enough to auto-skip as unresolved market disagreement
NEAR_MISS_MIN_EDGE = 0.01  # 1% — lower bound for near-miss limit order eligibility
KELLY_FRACTION = 0.25  # Quarter Kelly
TENNIS_KELLY_FRACTION = 0.25  # Tennis sizing fraction for dry-run reporting and the experimental trader
TENNIS_TRADER_ENABLED = _is_truthy_env("TENNIS_TRADER_ENABLED", "0")  # Enable experimental tennis trader in shared-wallet portfolio runs
TENNIS_PORTFOLIO_SHARE = _safe_float_env("TENNIS_PORTFOLIO_SHARE", "0.25")  # Share of wallet equity/cash reserved for tennis when enabled
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

# Line movement filter — penalize bets where sharp money disagrees
LINE_MOVEMENT_FILTER = True  # Enable line movement filter
LINE_AGAINST_EXTRA_EDGE = 0.02  # Require 2% more edge if line moves against our bet
LINE_SHARP_BLOCK = True  # Block bets where sharp/steam move is against us

# Time-decay training — weight recent fights more heavily
TIME_DECAY_ENABLED = True
TIME_DECAY_HALF_LIFE_DAYS = 730  # 2 years: fights 2 years old get half the weight

# Fighter experience filter — skip fights with inexperienced fighters
MIN_FIGHTER_FIGHTS = 3  # Don't bet when either fighter has fewer than this many UFC fights

# Underdog safeguards
MIN_MODEL_PROB = 0.40  # Don't bet on fighters below 40% blended probability
MAX_DECIMAL_ODDS = 3.0  # Skip anything above 3.0 decimal odds (+200)
EDGE_SCALING_BASE = 0.02  # Base edge at even money (2.0 odds)
EDGE_SCALING_RATE = 0.02  # Extra edge required per 1.0 increase in odds above 2.0

# Polymarket liquidity / slippage guardrails
MIN_BOOK_LIQUIDITY = 50.0  # Minimum USD available in the orderbook at or near our price
MAX_SLIPPAGE = 0.03  # Max 3% price slippage — skip if filling our size moves price more
MAX_BET_VS_BOOK_RATIO = 0.25  # Never take more than 25% of available book liquidity

# Limit bid TTL — cancel and re-evaluate resting bids after this many hours
LIMIT_BID_TTL_HOURS = 24
LIMIT_REPRICE_TICK_THRESHOLD = 2  # Require at least a 2-tick mismatch before repricing
LIMIT_REPRICE_MIN_AGE_MINUTES = 30  # Don't surrender queue priority too quickly
LIMIT_REPRICE_MAX_UPDATES = 2  # Cap upward reprices per market/fighter to avoid chasing
# Pre-event cancellation — pull all limit bids this many hours before the event starts
LIMIT_BID_PRE_EVENT_HOURS = 1

# Injury/cancellation detection — extreme odds shifts signal fight-breaking news
INJURY_MOVE_THRESHOLD = 0.15  # 15% probability shift = likely injury/cancellation
INJURY_PRICE_FLOOR = 0.05  # If either side drops below 5¢, fight is likely off
INJURY_BLOCK_BETS = True  # Block all bets on fights with suspected injury/cancellation

# Closing odds leakage mitigation — add noise to odds features during training
# to simulate the gap between current/opening odds and closing odds
ODDS_NOISE_STD = 0.04  # Std dev of Gaussian noise added to implied probabilities (4%)

# Duo-trader system — S (Single) gets full bankroll, C (Conviction) gets remaining
TRADER_C_SHARE = 1.0     # Conviction gets 100% of remaining bankroll after Single bets

# Trader C (Conviction) — bets on fighters all signals agree will win,
# regardless of whether odds offer traditional "value"
CONVICTION_MIN_MODEL_PROB = 0.65    # Model must be ≥65% confident
CONVICTION_MIN_NO_ODDS_PROB = 0.50  # No-odds model must independently agree (≥50%)
CONVICTION_BET_FRACTION = 0.05      # Flat 5% of bankroll per conviction bet
CONVICTION_CONFIDENCE_BONUS = 0.01  # Extra 1% sizing per 5% model prob above threshold
CONVICTION_MAX_BET_FRACTION = 0.08  # Hard cap at 8% of bankroll per bet

# Auto-retrain — retrain models if they're older than this many months
MODEL_RETRAIN_MONTHS = 1  # Retrain monthly before predict/live
