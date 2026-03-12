# UFC Betting Bot

A machine-learning-powered UFC fight prediction and automated betting bot. It scrapes historical fight data, engineers 175+ features (selecting the top 40 via SHAP), trains Optuna-tuned ensemble models, and executes value bets on [Polymarket](https://polymarket.com) prediction markets.

## How It Works

1. **Data Collection** — Scrapes fight stats from UFCStats.com (with Sherdog fallback for non-UFC fighters) and fetches live odds from The Odds API
2. **Feature Engineering** — Builds 175+ features including ELO ratings, rolling averages, finish rates, style matchups, fight pace, cage time efficiency, and historical odds
3. **Feature Selection** — SHAP-based selection reduces to the top 40 most predictive features to avoid overfitting
4. **Model Training** — Trains an Optuna-tuned XGBoost model (primary) and a logistic regression model with time-decay sample weighting and TimeSeriesSplit calibration to prevent temporal leakage (auto-retrains monthly)
5. **Value Detection** — Blends model predictions with market odds to find edges, with dynamic blend weights based on model confidence
6. **Duo-Trader System** — Single (value) and Conviction traders run in parallel with coordinated bankroll management
7. **Risk Management** — Sizes bets using quarter-Kelly criterion with stop-loss protection and underdog safeguards
8. **Execution** — Places trades on Polymarket UFC prediction markets via the CLOB API

### Key Safeguards

- **Dual-model agreement** — Both the odds-aware and odds-free models must agree on bet direction
- **Line movement filter** — Blocks bets where sharp money moves against the position
- **Injury/cancellation detection** — Extreme odds shifts (>15%) or near-zero prices auto-block bets
- **Liquidity checks** — Verifies orderbook depth, caps slippage at 3%, limits order size to 25% of available book
- **Fighter experience filter** — Skips fights where either fighter has fewer than 3 UFC bouts
- **Underdog safeguards** — Minimum 40% blended probability, max 3.0 decimal odds
- **Bankroll protection** — Max 4% per value bet, 8% per conviction bet, 60% drawdown stop-loss
- **Limit order TTL** — Stale limit bids auto-cancel after 24 hours and re-evaluate

## Duo-Trader System

The bot runs two independent trading strategies on a single wallet, each with its own ledger:

| Trader | Style | Blend Weight | Bankroll | Description |
|---|---|---|---|---|
| **S** (Single) | Value | 0.30 | Full balance | Kelly-sized value bets — blends model with market |
| **C** (Conviction) | Model agreement | N/A | Remaining after S | Bets when XGBoost (≥65%) and no-odds model (≥50%) both agree, 3+ UFC fights per fighter — ignores market odds and edge |

Coordination rules:
- S evaluates first with the full wallet balance
- C gets the remaining bankroll after S's bets are placed
- If S bets a fight, C skips that fight entirely (no double-betting)
- Each trader has its own persistent ledger for independent P&L tracking

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
POLYMARKET_FUNDER_ADDRESS=your_polymarket_proxy_wallet_address
```

The funder address is the Gnosis Safe proxy wallet shown on your Polymarket profile. If set, the bot auto-detects your live USDC balance instead of using a hardcoded bankroll.

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
python -m src.bot live --real

# Monitor upcoming events continuously
python -m src.bot monitor

# Track line movement over time
python -m src.bot track-lines

# Check pre-fight signals for the upcoming card
python -m src.bot signals

# Show current Polymarket positions and P&L
python -m src.bot positions

# Live-updating terminal dashboard (refreshes every 30s)
python -m src.bot dashboard
python -m src.bot dashboard --refresh 10 --real-only

# Settle bets (auto-settle from resolved Polymarket markets)
python -m src.bot settle --auto

# Manually settle a bet
python -m src.bot settle --bet-id 3 --result win

# Launch web dashboard (Flask)
python -m src.web.serve

# Compare full model vs no-odds baseline backtest
python -m src.bot backtest-compare

# Backfill historical odds from The Odds API
python -m src.bot backfill-odds
python -m src.bot backfill-odds --offsets 7,3,1 --fresh

# Walk-forward backtest with periodic retraining
python -m src.bot walkforward
python -m src.bot walkforward --retrain-months 6 --initial-years 5
```

### Recommended Workflow

1. `scrape` — Pull the latest fight data
2. `train` — Train/retrain models (auto-retrains monthly)
3. `evaluate` — Check model accuracy and calibration
4. `backtest` — Validate the strategy on historical data
5. `predict` — See predictions for the next card
6. `live --dry-run` — Paper trade before risking real money

## Project Structure

```
ufc-betting-bot/
├── src/
│   ├── bot.py                # Main CLI orchestrator
│   ├── config.py             # All settings and parameters
│   ├── data/
│   │   ├── scraper.py        # UFCStats.com scraper
│   │   ├── fallback_scrapers.py # Sherdog/Tapology fallback for non-UFC fighters
│   │   ├── fighter_lookup.py # Fighter lookup and caching
│   │   ├── odds_client.py    # The Odds API client
│   │   ├── historical_backfill.py # Historical odds backfill
│   │   ├── kaggle_loader.py  # Kaggle dataset loading
│   │   ├── line_tracker.py   # Line movement snapshot tracking
│   │   ├── live_monitor.py   # Live event monitoring
│   │   └── prefight_signals.py # Pre-fight signal detection
│   ├── features/             # Feature engineering (175+ features, SHAP-selected top 40)
│   ├── model/
│   │   ├── train.py          # Optuna-tuned XGBoost + logistic regression training
│   │   ├── train_experimental.py # Experimental training variants
│   │   ├── predict.py        # Prediction pipeline
│   │   ├── evaluate.py       # Model evaluation and calibration
│   │   ├── feature_selection.py # SHAP-based feature selection
│   │   ├── compare.py        # Model comparison utilities
│   │   └── hyperparam_search.py # Hyperparameter optimization
│   ├── strategy/
│   │   ├── duo_trader.py     # S+C duo-trader coordination
│   │   ├── value.py          # Value detection, conviction bets, edge calculation
│   │   ├── bankroll.py       # Kelly criterion sizing
│   │   ├── backtest.py       # Backtesting framework
│   │   ├── model_lab.py      # Model experimentation lab
│   │   ├── lab_stats.py      # Statistics utilities for model lab
│   │   ├── model_variants.py # Model variation experiments
│   │   └── triple_trader_backtest.py # Experimental triple-trader backtest
│   ├── polymarket/
│   │   ├── client.py         # CLOB API wrapper (orders, cancellation)
│   │   ├── executor.py       # Order execution, liquidity checks, limit bids
│   │   ├── markets.py        # UFC market discovery
│   │   ├── monitor.py        # Position monitoring
│   │   └── tracker.py        # Bet ledger (add/settle/cancel)
│   └── web/
│       ├── app.py            # Flask routes and API endpoints
│       ├── serve.py          # Production live loop + background monitor
│       └── templates/        # HTML templates
├── tests/                    # Unit and integration tests
├── data/
│   ├── raw/                  # Raw scraped data and odds
│   └── processed/            # Cleaned features and datasets
├── models/                   # Trained model artifacts (.pkl)
├── logs/                     # Bot logs, ledgers, and plots
├── entrypoint.sh             # Docker entrypoint script
├── requirements.txt
├── Dockerfile
└── railway.toml              # Railway deployment config
```

## Configuration

All strategy parameters live in `src/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `BLEND_WEIGHT` | 0.30 | Model weight in model-market blend |
| `MIN_EDGE_THRESHOLD` | 2% | Minimum edge to place a bet |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly bet sizing |
| `MAX_BET_FRACTION` | 4% | Max bankroll risked per value bet |
| `STOP_LOSS_FRACTION` | 60% | Stop trading after this drawdown |
| `MIN_FIGHTER_FIGHTS` | 3 | Min UFC fights for both fighters |
| `TIME_DECAY_HALF_LIFE_DAYS` | 730 | 2-year half-life for training weights |
| `MODEL_RETRAIN_MONTHS` | 1 | Auto-retrain interval (monthly) |
| `MIN_BOOK_LIQUIDITY` | $50 | Minimum orderbook depth to place a bet |
| `MAX_SLIPPAGE` | 3% | Max price slippage before skipping |
| `MAX_BET_VS_BOOK_RATIO` | 25% | Never take more than this % of available book |
| `LIMIT_BID_TTL_HOURS` | 24 | Auto-cancel stale limit bids after this |
| `INJURY_MOVE_THRESHOLD` | 15% | Line shift that triggers injury alert |
| `INJURY_PRICE_FLOOR` | 5¢ | Price below this signals fight is likely off |
| `ODDS_NOISE_STD` | 4% | Noise added to odds features during training |
| `REQUIRE_MODEL_AGREEMENT` | true | Both models must agree on bet direction |
| `MODEL_AGREEMENT_MIN_EDGE` | 1% | No-odds model must show at least this edge |
| `MIN_MODEL_PROB` | 40% | Don't bet on fighters below this blended probability |
| `MAX_DECIMAL_ODDS` | 3.0 | Skip anything above this decimal odds |
| `EDGE_SCALING_BASE` | 2% | Base edge required at even money |
| `EDGE_SCALING_RATE` | 2% | Extra edge per 1.0 increase in odds above 2.0 |
| `LINE_MOVEMENT_FILTER` | true | Enable line movement filter |
| `LINE_AGAINST_EXTRA_EDGE` | 2% | Extra edge required if line moves against position |
| `LINE_SHARP_BLOCK` | true | Block bets where sharp/steam move is against us |
| `BLEND_WEIGHT_MIN` | 0.15 | Blend weight floor for low-confidence predictions |
| `BLEND_WEIGHT_MAX` | 0.50 | Blend weight ceiling for high-conviction predictions |
| `BLEND_CONFIDENCE_THRESHOLD` | 0.65 | Model confidence above this increases blend weight |
| `BLEND_AGREEMENT_BOOST` | 0.10 | Extra blend weight when no-odds model strongly agrees |
| `CONVICTION_MIN_MODEL_PROB` | 65% | Model confidence floor for Trader C |
| `CONVICTION_MIN_NO_ODDS_PROB` | 50% | No-odds model agreement floor for Trader C |
| `CONVICTION_BET_FRACTION` | 5% | Flat bankroll % per conviction bet |
| `CONVICTION_CONFIDENCE_BONUS` | 1% | Extra sizing per 5% model prob above 75% |
| `CONVICTION_MAX_BET_FRACTION` | 8% | Hard cap per conviction bet |

## Web Dashboard

The bot includes a Flask web dashboard for monitoring positions and P&L in the browser:

```bash
python -m src.web.serve
```

The dashboard runs on port 5050 by default (set `PORT` env var to change) and includes:
- Wallet balance and portfolio value display
- Live P&L summary, bet history, and portfolio chart
- Per-trader breakdown (individual P&L, win rate, ROI for each trader)
- Trader performance race visualization
- Upcoming UFC events from monitoring snapshots
- Fight predictions with search, sort, and detailed view
- Injury/cancellation alerts
- Line movement tracking
- Open limit order management
- Filter funnel analysis (see why bets were skipped)
- Recent bot activity log viewer with significant action highlights
- Expandable position details and price alerts
- Mobile-responsive layout
- Background live betting loop (configurable interval via `BET_INTERVAL_MINUTES`, default 10m)
- Background monitor thread that auto-settles resolved markets and tracks line movement

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/summary` | GET | Portfolio summary (balance, P&L, stats) |
| `/api/bets` | GET | All bets from the ledger |
| `/api/pnl-history` | GET | P&L over time for charting |
| `/api/positions` | GET | Current open positions |
| `/api/trade-history` | GET | Completed trade history |
| `/api/balance` | GET | Wallet USDC balance |
| `/api/predictions` | GET | Fight predictions for upcoming card |
| `/api/predictions-detail` | GET | Detailed prediction breakdown |
| `/api/trader-breakdown` | GET | Per-trader P&L, win rate, ROI |
| `/api/trader-race` | GET | Trader performance comparison |
| `/api/upcoming-events` | GET | Upcoming UFC events |
| `/api/bot-activity` | GET | Recent bot activity log |
| `/api/bot-activity-snapshot` | GET | Activity log snapshot |
| `/api/significant-actions` | GET | Notable trading actions |
| `/api/open-limit-orders` | GET | Open limit orders and status |
| `/api/injury-alerts` | GET | Injury/cancellation alerts |
| `/api/filter-funnel` | GET | Filter pipeline analysis |
| `/api/line-movements` | GET | Line movement data |
| `/api/refresh-prices` | POST | Refresh market prices |
| `/api/settle-auto` | POST | Auto-settle resolved bets |
| `/api/settle/<bet_id>/<result>` | POST | Manually settle a bet |

For always-on deployment, this is the entrypoint used by Railway/Docker. Environment variables:
- `PORT` — web server port (default 5050)
- `BET_INTERVAL_MINUTES` — how often to run the betting cycle (default 10)
- `MIN_EDGE` — minimum edge override (default 0.02)
- `MONITOR_INTERVAL_HOURS` — background monitor interval (default 6)

## Deployment

The bot can be deployed as a Docker container on [Railway](https://railway.app) or any container platform:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env ufc-betting-bot
```

## Disclaimer

This project is for educational and research purposes. Sports betting involves risk — never bet more than you can afford to lose. Past model performance does not guarantee future results.
