"""Configuration settings for the UFC betting bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# UFC Stats scraper
UFCSTATS_BASE_URL = "http://ufcstats.com/statistics/events/completed"
UFCSTATS_FIGHTER_URL = "http://ufcstats.com/fighter-details/"
UFCSTATS_EVENT_URL = "http://ufcstats.com/event-details/"
UFCSTATS_FIGHT_URL = "http://ufcstats.com/fight-details/"

# The Odds API
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "mma_mixed_martial_arts"

# Polymarket
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_CHAIN_ID = 137  # Polygon
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# Model settings
ROLLING_WINDOW = 5  # Number of recent fights for rolling averages
ELO_INITIAL = 1500
ELO_K_FACTOR = 32
TRAIN_CUTOFF_DATE = "2022-01-01"  # Train on fights before this date

# Strategy settings
MIN_EDGE_THRESHOLD = 0.04  # 4% minimum edge to place a bet
KELLY_FRACTION = 0.25  # Quarter Kelly
MAX_BET_FRACTION = 0.05  # Never risk more than 5% of bankroll
STOP_LOSS_FRACTION = 0.30  # Stop if bankroll drops 30%
INITIAL_BANKROLL = 1000.0  # Default starting bankroll in USD
