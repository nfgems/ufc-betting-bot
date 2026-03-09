# UFC Betting Bot

A machine-learning-powered UFC fight prediction and automated betting bot. It scrapes historical fight data, engineers 90+ features, trains ensemble models, and executes value bets on [Polymarket](https://polymarket.com) prediction markets.

## How It Works

1. **Data Collection** — Scrapes fight stats from UFCStats.com and fetches live odds from The Odds API
2. **Feature Engineering** — Builds 90+ features including ELO ratings, rolling averages, finish rates, style matchups, and historical odds
3. **Model Training** — Trains an XGBoost model (primary) and a logistic regression model with time-decay sample weighting (auto-retrains monthly)
4. **Value Detection** — Blends model predictions with market odds to find edges, with dynamic blend weights based on model confidence
5. **Triple-Trader System** — Three independent strategies run in parallel with coordinated bankroll splitting and conflict resolution
6. **Risk Management** — Sizes bets using quarter-Kelly criterion with stop-loss protection and underdog safeguards
7. **Execution** — Places trades on Polymarket UFC prediction markets via the CLOB API

### Key Safeguards

- **Dual-model agreement** — Both the odds-aware and odds-free models must agree on bet direction
- **Line movement filter** — Blocks bets where sharp money moves against the position
- **Injury/cancellation detection** — Extreme odds shifts (>15%) or near-zero prices auto-block bets
- **Liquidity checks** — Verifies orderbook depth, caps slippage at 3%, limits order size to 25% of available book
- **Fighter experience filter** — Skips fights where either fighter has fewer than 3 UFC bouts
- **Underdog safeguards** — Minimum 40% blended probability, max 3.0 decimal odds
- **Bankroll protection** — Max 4% per bet, 60% drawdown stop-loss
- **Cross-trader conflict resolution** — Never bets opposite sides of the same fight across traders

## Triple-Trader System

The bot runs three independent trading strategies on a single wallet, each with its own bankroll slice and ledger:

| Trader | Style | Blend Weight | Bankroll Share | Description |
|---|---|---|---|---|
| **A** (Conservative) | Value | 0.20 | 40% | Fewer, higher-conviction value bets — trusts the market more |
| **B** (Aggressive) | Value | 0.40 | 40% | More value bets — trusts the model more |
| **C** (Conviction) | Model agreement | N/A | 20% | Bets when XGBoost (>75%) and no-odds model (>60%) both agree, 3+ UFC fights per fighter — ignores market odds and edge |

Coordination rules:
- Wallet balance is auto-detected and split 40/40/20 across traders
- If multiple traders want the same side, the one with higher edge/conviction takes it
- If traders disagree on sides, the bet is blocked entirely
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
│   ├── bot.py              # Main CLI orchestrator
│   ├── config.py           # All settings and parameters
│   ├── data/               # Scraping, odds fetching, line tracking
│   ├── features/           # Feature engineering (90+ features)
│   ├── model/              # Training, prediction, evaluation
│   ├── strategy/           # Value detection, conviction bets, triple-trader coordination, backtesting
│   ├── polymarket/         # Polymarket API client, trade execution, position tracking
│   └── web/                # Flask web dashboard with live P&L
├── data/
│   ├── raw/                # Raw scraped data and odds
│   └── processed/          # Cleaned features and datasets
├── models/                 # Trained model artifacts (.pkl)
├── logs/                   # Bot logs and plots
├── blend_weight_test.py    # Blend weight experiments
├── compare_models.py       # Model comparison script
├── test_bet.py             # Bet execution tests
├── test_triple_trader.py   # Triple-trader system tests
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
| `STOP_LOSS_FRACTION` | 60% | Stop trading after this drawdown |
| `MIN_FIGHTER_FIGHTS` | 3 | Min UFC fights for both fighters |
| `TIME_DECAY_HALF_LIFE_DAYS` | 730 | 2-year half-life for training weights |
| `MODEL_RETRAIN_MONTHS` | 1 | Auto-retrain interval (monthly) |
| `MIN_BOOK_LIQUIDITY` | $50 | Minimum orderbook depth to place a bet |
| `MAX_SLIPPAGE` | 3% | Max price slippage before skipping |
| `INJURY_MOVE_THRESHOLD` | 15% | Line shift that triggers injury alert |
| `ODDS_NOISE_STD` | 4% | Noise added to odds features during training |
| `TRADER_A_BLEND` | 0.20 | Conservative trader blend weight |
| `TRADER_B_BLEND` | 0.40 | Aggressive trader blend weight |
| `TRADER_A_SHARE` | 40% | Bankroll share for Trader A |
| `TRADER_B_SHARE` | 40% | Bankroll share for Trader B |
| `TRADER_C_SHARE` | 20% | Bankroll share for Trader C |
| `CONVICTION_MIN_MODEL_PROB` | 75% | Model confidence floor for Trader C |
| `CONVICTION_MIN_NO_ODDS_PROB` | 60% | No-odds model agreement floor for Trader C |
| `CONVICTION_BET_FRACTION` | 5% | Flat bankroll % per conviction bet |
| `CONVICTION_CONFIDENCE_BONUS` | 1% | Extra sizing per 5% model prob above 75% |
| `CONVICTION_MAX_BET_FRACTION` | 8% | Hard cap per conviction bet |
| `BLEND_WEIGHT_MIN` | 0.15 | Blend weight floor for low-confidence predictions |
| `BLEND_WEIGHT_MAX` | 0.50 | Blend weight ceiling for high-conviction predictions |
| `BLEND_CONFIDENCE_THRESHOLD` | 0.65 | Model confidence above this increases blend weight |
| `BLEND_AGREEMENT_BOOST` | 0.10 | Extra blend weight when no-odds model strongly agrees |
| `REQUIRE_MODEL_AGREEMENT` | true | Both models must agree on bet direction |
| `MODEL_AGREEMENT_MIN_EDGE` | 1% | No-odds model must show at least this edge |
| `MIN_MODEL_PROB` | 40% | Don't bet on fighters below this blended probability |
| `MAX_DECIMAL_ODDS` | 3.0 | Skip anything above this decimal odds |
| `EDGE_SCALING_BASE` | 3% | Base edge required at even money |
| `EDGE_SCALING_RATE` | 2% | Extra edge per 1.0 increase in odds above 2.0 |
| `MAX_BET_VS_BOOK_RATIO` | 25% | Never take more than this % of available book |
| `INJURY_PRICE_FLOOR` | 5¢ | Price below this signals fight is likely off |
| `INJURY_BLOCK_BETS` | true | Block all bets on suspected injury/cancellation |
| `LINE_MOVEMENT_FILTER` | true | Enable line movement filter |
| `LINE_AGAINST_EXTRA_EDGE` | 2% | Extra edge required if line moves against position |
| `LINE_SHARP_BLOCK` | true | Block bets where sharp/steam move is against us |

## Web Dashboard

The bot includes a Flask web dashboard for monitoring positions and P&L in the browser:

```bash
python -m src.web.serve
```

The dashboard runs on port 5050 by default (set `PORT` env var to change) and includes:
- Wallet balance and portfolio value display
- Live P&L summary, bet history, and portfolio chart
- Per-trader breakdown (individual P&L, win rate, ROI for each trader)
- Upcoming UFC events from monitoring snapshots
- Recent bot activity log viewer
- Expandable position details and price alerts
- Mobile-responsive layout
- API endpoints (`/api/summary`, `/api/bets`, `/api/pnl-history`, `/api/balance`, `/api/bot-activity`, `/api/upcoming-events`, `/api/trader-breakdown`)
- Background live betting loop (configurable interval via `BET_INTERVAL_MINUTES`, default 10m)
- Background monitor thread that auto-settles resolved markets and tracks line movement

For always-on deployment, this is the entrypoint used by Railway/Docker. Environment variables:
- `PORT` — web server port (default 5050)
- `BET_INTERVAL_MINUTES` — how often to run the betting cycle (default 10)
- `MIN_EDGE` — minimum edge override (default 0.03)
- `MONITOR_INTERVAL_HOURS` — background monitor interval (default 6)

## Deployment

The bot can be deployed as a Docker container on [Railway](https://railway.app) or any container platform:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env ufc-betting-bot
```

## Disclaimer

This project is for educational and research purposes. Sports betting involves risk — never bet more than you can afford to lose. Past model performance does not guarantee future results.
