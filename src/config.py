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
BLEND_WEIGHT = 0.30  # Model weight in market-model blend (0.3 model + 0.7 market)
MIN_EDGE_THRESHOLD = 0.03  # 3% minimum blended edge to place a bet
KELLY_FRACTION = 0.25  # Quarter Kelly
MAX_BET_FRACTION = 0.04  # Never risk more than 4% of bankroll
STOP_LOSS_FRACTION = 0.30  # Stop if bankroll drops 30%
INITIAL_BANKROLL = 1000.0  # Default starting bankroll in USD
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

# Underdog safeguards
MIN_MODEL_PROB = 0.40  # Don't bet on fighters below 40% blended probability
MAX_DECIMAL_ODDS = 3.0  # Skip anything above 3.0 decimal odds (+200)
EDGE_SCALING_BASE = 0.03  # Base edge at even money (2.0 odds)
EDGE_SCALING_RATE = 0.02  # Extra edge required per 1.0 increase in odds above 2.0
