# UFC Betting Bot

A machine-learning-powered UFC fight prediction and automated betting bot. It scrapes historical fight data, engineers 90+ features, trains ensemble models, and executes value bets on [Polymarket](https://polymarket.com) prediction markets.

## How It Works

1. **Data Collection** — Scrapes fight stats from UFCStats.com and fetches live odds from The Odds API
2. **Feature Engineering** — Builds 90+ features including ELO ratings, rolling averages, finish rates, style matchups, and historical odds
3. **Model Training** — Trains an XGBoost model (primary) and a logistic regression model with time-decay sample weighting
4. **Value Detection** — Blends model predictions with market odds (30/70 split) to find edges, with dynamic blend weights based on model confidence
5. **Risk Management** — Sizes bets using quarter-Kelly criterion with stop-loss protection and underdog safeguards
6. **Execution** — Places trades on Polymarket UFC prediction markets via the CLOB API

### Key Safeguards

- **Dual-model agreement** — Both the odds-aware and odds-free models must agree on bet direction
- **Line movement filter** — Blocks bets where sharp money moves against the position
- **Fighter experience filter** — Skips fights where either fighter has fewer than 3 UFC bouts
- **Underdog safeguards** — Minimum 40% blended probability, max 3.0 decimal odds
- **Bankroll protection** — Max 4% per bet, 30% drawdown stop-loss

## Setup

### Prerequisites

- Python 3.11+
- A free API key from [The Odds API](https://the-odds-api.com)
- (Optional) A Polygon wallet private key for live Polymarket trading

### Installation

```bash
git clone https://github.com/nfgems/ufc-betting-bot.git
cd ufc-betting-bot
pip install -r requirements.txt
```

### Environment Variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```
ODDS_API_KEY=your_odds_api_key_here
POLYMARKET_PRIVATE_KEY=your_polygon_private_key_here
```

## Usage

All commands are run from the project root:

```bash
# Scrape latest fight data from UFCStats
python -m src.bot scrape

# Train the models
python -m src.bot train

# Evaluate model performance
python -m src.bot evaluate

# Run backtest with walk-forward validation
python -m src.bot backtest

# Sensitivity analysis to find optimal parameters
python -m src.bot sensitivity

# Predict upcoming fights
python -m src.bot predict

# Run live bot (dry run — no real trades)
python -m src.bot live --dry-run

# Run live bot with real money
python -m src.bot live

# Monitor upcoming events continuously
python -m src.bot monitor

# Track line movement over time
python -m src.bot track-lines

# Check pre-fight signals for the upcoming card
python -m src.bot signals
```

### Recommended Workflow

1. `scrape` — Pull the latest fight data
2. `train` — Train/retrain models (auto-retrains every 3 months)
3. `evaluate` — Check model accuracy and calibration
4. `backtest` — Validate the strategy on historical data
5. `predict` — See predictions for the next card
6. `live --dry-run` — Paper trade before risking real money

## Project Structure

```
ufc-betting-bot/
├── src/
│   ├── bot.py              # Main CLI orchestrator
│   ├── config.py           # All settings and parameters
│   ├── data/               # Scraping, odds fetching, line tracking
│   ├── features/           # Feature engineering (90+ features)
│   ├── model/              # Training, prediction, evaluation
│   ├── strategy/           # Value detection, backtesting, bankroll mgmt
│   └── polymarket/         # Polymarket API client and trade execution
├── data/
│   ├── raw/                # Raw scraped data and odds
│   └── processed/          # Cleaned features and datasets
├── models/                 # Trained model artifacts (.pkl)
├── logs/                   # Bot logs
├── requirements.txt
├── Dockerfile
└── railway.toml            # Railway deployment config
```

## Configuration

All strategy parameters live in `src/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `BLEND_WEIGHT` | 0.30 | Model weight in model-market blend |
| `MIN_EDGE_THRESHOLD` | 3% | Minimum edge to place a bet |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly bet sizing |
| `MAX_BET_FRACTION` | 4% | Max bankroll risked per bet |
| `STOP_LOSS_FRACTION` | 30% | Stop trading after this drawdown |
| `MIN_FIGHTER_FIGHTS` | 3 | Min UFC fights for both fighters |
| `TIME_DECAY_HALF_LIFE_DAYS` | 730 | 2-year half-life for training weights |
| `MODEL_RETRAIN_MONTHS` | 3 | Auto-retrain interval |

## Deployment

The bot can be deployed as a Docker container on [Railway](https://railway.app) or any container platform:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env ufc-betting-bot
```

## Disclaimer

This project is for educational and research purposes. Sports betting involves risk — never bet more than you can afford to lose. Past model performance does not guarantee future results.
