# UFC Betting Bot

A machine-learning-powered UFC fight prediction and automated betting bot, with experimental ATP/WTA tennis discovery and dry-run tooling. It scrapes historical fight data, engineers 150 features across 13 categories, trains calibrated models with a promoted training spec system, and executes UFC bets on [Polymarket](https://polymarket.com) prediction markets.

## How It Works

1. **Data collection** — Scrapes fight stats from UFCStats.com, with Sherdog and Tapology fallback coverage for fighters missing UFCStats history, and fetches live odds from The Odds API.
2. **Feature engineering** — Builds 150 features including Elo ratings, Elo momentum, strength of schedule, EWM rolling averages, rematch/H2H history, line movement signals, UFC rankings, method-of-victory odds, finish rates, style matchups, and physical attributes.
3. **Training specs** — A spec-driven training system (`training_spec.py`) defines reproducible feature contracts. The current production spec is `rematch_features_v1`. All features have live acquisition paths — zero training/inference gap.
4. **Model training** — Trains a calibrated XGBoost model, a logistic regression baseline, and a no-odds XGBoost baseline with time-decay sample weighting and isotonic calibration via 5-fold timeseries cross-validation.
5. **Value detection** — Blends model probabilities with market probabilities using dynamic blend weights that respond to model confidence and cross-model agreement.
6. **Duo-trader system** — Runs a value trader (S) and a conviction trader (C) on the same wallet with coordinated bankroll usage.
7. **Risk management** — Sizes bets with fractional Kelly logic, market-quality filters, stop-loss rules, line movement filters, and underdog safeguards.
8. **Execution** — Places UFC market orders and limit orders on Polymarket through the CLOB API.

### Key Safeguards

- **Dual-model agreement** — The live UFC strategy can require the odds-aware and no-odds models to agree on bet direction.
- **Line movement filter** — Blocks or penalizes bets when sharp or steam moves go against the position.
- **Injury/cancellation detection** — Extreme line shifts or near-zero prices can block betting on a fight entirely.
- **Liquidity checks** — Verifies book depth, caps slippage at 3%, and limits order size relative to available liquidity.
- **Fighter experience filter** — Skips UFC fights where either fighter has too little prior UFC history.
- **Underdog safeguards** — Requires minimum blended win probability and caps long-shot exposure.
- **Bankroll protection** — Caps per-bet exposure and stops trading after deep drawdowns.
- **Limit order management** — Supports near-miss limit orders, conservative repricing, stale-order TTL cleanup, and pre-event cancellation.

## Feature Engineering

The model uses 150 live-compatible features organized into 13 categories:

| Category | Examples | Source |
|---|---|---|
| **Differential stats** | `diff_elo`, `diff_roll_slpm`, `diff_win_pct`, `diff_striker_edge` | Computed from per-fighter stats |
| **Individual rolling stats** | `a_roll_slpm`, `b_roll_str_acc`, `a_roll_td_avg` | EWM (halflife=3 fights) rolling averages |
| **Career stats** | Win streaks, fight count, finish rates, title bouts | UFCStats career records |
| **Physical attributes** | Height, reach, weight, age, cage rust, layoff | UFCStats + computed |
| **Elo ratings** | `a_elo`, `b_elo`, `diff_elo` | Custom Elo system (K=32, initial=1500) |
| **Elo momentum** | `a_elo_momentum`, `b_elo_momentum` | Slope of Elo over last 5 fights |
| **Strength of schedule** | `a_sos`, `b_sos`, `diff_sos` | Mean opponent Elo of last 5 opponents |
| **Rematch / H2H** | `is_rematch`, `h2h_record_diff` | Chronological H2H history |
| **Line movement** | `line_movement`, `line_is_sharp`, `line_steam_move` | Historical odds snapshots (7/3/1 day offsets) |
| **Rankings** | `a_wc_rank_feat`, `b_pfp_rank_feat`, `diff_wc_rank` | Snapshot-backed UFC.com/ESPN rankings |
| **Method odds** | `a_ko_odds_prob`, `b_sub_odds_prob`, `a_dec_odds_prob` | The Odds API method-of-victory markets |
| **Odds-derived** | `a_implied_prob`, `b_implied_prob`, `diff_implied_prob` | The Odds API moneyline odds |
| **Event context** | `is_title_bout`, `num_rounds_feat`, `weight_class_enc` | Event metadata |

All features are live-compatible. `TRAINING_ONLY_FEATURES` is an empty set — there is no training/inference feature gap.

## Duo-Trader System

The bot runs two UFC trading strategies on one wallet, each with its own ledger:

| Trader | Style | Blend Weight | Bankroll | Description |
|---|---|---|---|---|
| **S** (Single) | Value | 0.30 | Full balance | Kelly-sized value bets that blend model and market probabilities |
| **C** (Conviction) | Model agreement | N/A | Remaining after S | Bets when the primary and no-odds models both support the same side with enough confidence |

Coordination rules:

- S evaluates first with the full wallet balance.
- C receives the remaining bankroll after S places bets.
- If S already bets a fight, C skips that fight entirely.
- Each trader keeps a separate persistent ledger for independent P&L tracking.

## Setup

### Prerequisites

- Python 3.11+
- A free API key from [The Odds API](https://the-odds-api.com)
- An optional Polygon wallet private key for live Polymarket trading

### Installation

```bash
git clone https://github.com/nfgems/ufc-betting-bot.git
cd ufc-betting-bot
pip install -r requirements.txt
```

### Windows PowerShell

If PowerShell blocks `.\.venv\Scripts\Activate.ps1` with a "running scripts is disabled" error, you do not need to change your machine-wide policy to use this repo.

Use the included Windows wrappers from the project root instead:

```powershell
.\python-venv.cmd -m pip install -r requirements.txt
.\bot.cmd predict
.\bot.cmd web --port 8080
```

If you want an activated shell without using `Activate.ps1`, run:

```powershell
.\venv-shell.cmd
```

That opens a `cmd.exe` session with the project's virtual environment activated, using either `.venv` or `venv`.

If you still prefer PowerShell activation, use a temporary process-scoped bypass:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Environment Variables

Copy the example env file and fill in the values you need:

```bash
cp .env.example .env
```

```dotenv
ODDS_API_KEY=your_odds_api_key_here
POLYMARKET_PRIVATE_KEY=your_polygon_private_key_here
POLYMARKET_FUNDER_ADDRESS=your_polymarket_proxy_wallet_address
CLOB_PROXY_URL=http://user:pass@host:port
```

Notes:

- `POLYMARKET_FUNDER_ADDRESS` is an optional override for your Polymarket proxy wallet. If it is omitted, the bot attempts to auto-discover the proxy wallet from Polymarket using your private key.
- `CLOB_PROXY_URL` is optional. If set, the Polymarket CLOB client routes its traffic through that proxy and the dashboard can report live geoblock diagnostics.

## Usage

### Core UFC commands

All commands are run from the project root:

```bash
# Scrape latest UFC fighter and fight data
python -m src.bot scrape

# Train the UFC models
python -m src.bot train

# Evaluate model performance
python -m src.bot evaluate

# Run backtest with walk-forward validation by default
python -m src.bot backtest

# Compare full model vs no-odds baseline
python -m src.bot backtest-compare

# Run walk-forward backtest explicitly
python -m src.bot walkforward
python -m src.bot walkforward --retrain-months 6 --initial-years 5

# Sensitivity analysis
python -m src.bot sensitivity

# Predict upcoming UFC fights
python -m src.bot predict

# Backfill historical odds from The Odds API
python -m src.bot backfill-odds
python -m src.bot backfill-odds --offsets 7,3,1 --fresh

# Run live duo-trader system in dry-run mode
python -m src.bot live --dry-run

# Run live duo-trader system with real money
python -m src.bot live --real

# Monitor upcoming events continuously
python -m src.bot monitor

# Track line movement over time
python -m src.bot track-lines

# Check pre-fight signals for the upcoming card
python -m src.bot signals

# Show current Polymarket positions and P&L
python -m src.bot positions

# Terminal dashboard
python -m src.bot dashboard
python -m src.bot dashboard --refresh 10 --real-only

# Settle bets
python -m src.bot settle --auto
python -m src.bot settle --bet-id 3 --result win
```

### Local web dashboard

Use the local Flask dashboard command when you only want the web UI:

```bash
python -m src.bot web
python -m src.bot web --port 8080 --offline
```

This starts the dashboard only. It does **not** start the production live betting loop or background monitor threads.

### Experimental tennis commands

The repo also includes experimental ATP/WTA singles discovery, model training, prediction, and dry-run edge logging. Real-money tennis trading is **not** implemented.

```bash
# Discover live tennis bookmaker feeds and Polymarket markets
python -m src.bot tennis-discover

# Download ATP/WTA history, build features, and train the tennis baseline
python -m src.bot tennis-train

# Predict live ATP/WTA singles matches
python -m src.bot tennis-predict

# Run the tennis dry-run pipeline
python -m src.bot tennis-live

# Audit bookmaker coverage for tennis
python -m src.bot tennis-bookmaker-audit
```

### Recommended UFC workflow

1. `scrape` — Pull the latest fight data.
2. `train` — Train or retrain the UFC models.
3. `evaluate` — Check accuracy and calibration.
4. `backtest` — Validate the strategy on historical data.
5. `predict` — Review the next card.
6. `live --dry-run` — Paper trade before risking real money.

## Project Structure

```text
ufc-betting-bot/
|-- src/
|   |-- bot.py                        # Main CLI orchestrator
|   |-- config.py                     # Settings and runtime paths
|   |-- data/
|   |   |-- scraper.py                # UFCStats scraper
|   |   |-- fallback_scrapers.py      # Sherdog/Tapology fallback coverage
|   |   |-- fighter_lookup.py         # Fighter lookup and feature assembly
|   |   |-- odds_client.py            # UFC odds client
|   |   |-- historical_backfill.py    # Historical odds backfill (7/3/1 day offsets)
|   |   |-- line_tracker.py           # UFC line movement snapshots
|   |   |-- line_movement.py          # Line movement analysis (sharp/steam detection)
|   |   |-- rankings_scraper.py       # UFC.com/ESPN rankings (snapshot-backed)
|   |   |-- method_odds.py            # Method-of-victory odds from The Odds API
|   |   |-- live_monitor.py           # Upcoming event monitoring
|   |   |-- prefight_signals.py       # Pre-fight signal detection
|   |   |-- name_utils.py             # Fighter name normalization
|   |   |-- kaggle_loader.py          # Kaggle dataset loader
|   |   |-- tennis_data.py            # ATP/WTA data ingestion and normalization
|   |   |-- tennis_odds.py            # Live tennis bookmaker discovery
|   |   `-- tennis_bookmaker_audit.py # Tennis bookmaker coverage audit
|   |-- features/
|   |   |-- build_features.py         # UFC feature engineering (EWM rolling, Elo, etc.)
|   |   |-- experimental_features.py  # Experimental feature pipeline
|   |   `-- tennis_features.py        # Tennis feature engineering
|   |-- model/
|   |   |-- train.py                  # UFC model training (XGBoost + logistic + no-odds)
|   |   |-- training_spec.py          # Training spec system and live feature contract
|   |   |-- evaluate.py               # UFC evaluation and calibration
|   |   |-- predict.py                # UFC prediction pipeline
|   |   |-- feature_selection.py      # SHAP/permutation feature selection
|   |   |-- compare.py                # Model comparison utilities
|   |   |-- hyperparam_search.py      # Hyperparameter search tooling (Optuna)
|   |   |-- train_experimental.py     # Experimental training pipelines
|   |   `-- tennis_model.py           # Tennis baseline training and inference
|   |-- strategy/
|   |   |-- backtest.py               # Backtesting engine
|   |   |-- bankroll.py               # Bankroll management and Kelly sizing
|   |   |-- duo_trader.py             # Duo-trader system (S + C coordination)
|   |   |-- value.py                  # Value detection and edge calculation
|   |   |-- model_variants.py         # Variant factory (rematch, Elo momentum, SoS, etc.)
|   |   |-- model_lab.py              # Model experimentation lab
|   |   |-- lab_stats.py              # Lab statistics and reporting
|   |   `-- triple_trader_backtest.py # Experimental triple-trader variant backtest
|   |-- polymarket/
|   |   |-- client.py                 # CLOB and Gamma API wrapper
|   |   |-- executor.py               # Order execution and limit-order handling
|   |   |-- markets.py                # UFC market discovery
|   |   |-- tennis_markets.py         # Tennis market discovery and matching
|   |   |-- monitor.py                # Position monitoring
|   |   `-- tracker.py                # Bet ledger and settlement
|   `-- web/
|       |-- app.py                    # Flask routes and API endpoints
|       |-- serve.py                  # Production entrypoint with live loops
|       `-- templates/                # Dashboard templates
|-- tests/                            # 21 test suites (schema contract, backtest, executor, web API, tennis, etc.)
|-- data/
|   |-- raw/                          # Raw scraped data, market snapshots, line history, rankings, method odds
|   |-- processed/                    # Processed UFC and tennis datasets
|   `-- logs/                         # Runtime logs, ledgers, plots, dashboard cache
|-- models/                           # Trained model artifacts
|-- entrypoint.sh                     # Docker entrypoint script
|-- requirements.txt
|-- Dockerfile
`-- railway.toml                      # Railway deployment config
```

## Configuration

All major strategy parameters live in `src/config.py`. The table below covers the most important UFC execution knobs; the full file also includes tennis settings and path constants.

| Parameter | Default | Description |
|---|---|---|
| `BLEND_WEIGHT` | 0.30 | Base model weight in the model-market blend |
| `MIN_EDGE_THRESHOLD` | 2% | Minimum blended edge to place a live UFC bet |
| `NEAR_MISS_MIN_EDGE` | 1% | Lower edge bound for near-miss limit-order candidates |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly sizing for value bets |
| `MAX_BET_FRACTION` | 4% | Max bankroll risked per value bet |
| `STOP_LOSS_FRACTION` | 60% | Stop trading after this drawdown |
| `MIN_FIGHTER_FIGHTS` | 3 | Minimum UFC fights required for both fighters |
| `TIME_DECAY_HALF_LIFE_DAYS` | 730 | Half-life for time-decay sample weights |
| `MODEL_RETRAIN_MONTHS` | 1 | Auto-retrain interval before predict/live |
| `MIN_BOOK_LIQUIDITY` | $50 | Minimum orderbook depth to consider a bet |
| `MAX_SLIPPAGE` | 3% | Maximum tolerated slippage |
| `MAX_BET_VS_BOOK_RATIO` | 25% | Max share of visible book liquidity to take |
| `LIMIT_BID_TTL_HOURS` | 24 | Cancel resting limit bids after this many hours |
| `LIMIT_BID_PRE_EVENT_HOURS` | 1 | Cancel limit bids shortly before the fight starts |
| `LIMIT_REPRICE_TICK_THRESHOLD` | 2 | Minimum tick gap before repricing an open limit bid |
| `LIMIT_REPRICE_MIN_AGE_MINUTES` | 30 | Minimum resting age before repricing upward |
| `LIMIT_REPRICE_MAX_UPDATES` | 2 | Max upward reprices per market/fighter |
| `INJURY_MOVE_THRESHOLD` | 15% | Probability shift that triggers injury review |
| `INJURY_PRICE_FLOOR` | $0.05 | Price floor that suggests the fight may be off |
| `INJURY_BLOCK_BETS` | true | Block bets when injury/cancellation signals fire |
| `ODDS_NOISE_STD` | 4% | Noise added to odds-derived training features |
| `REQUIRE_MODEL_AGREEMENT` | true | Require both UFC models to agree on direction |
| `MODEL_AGREEMENT_MIN_EDGE` | 1% | Minimum no-odds edge needed for agreement |
| `MIN_MODEL_PROB` | 40% | Skip bets below this blended probability |
| `MAX_DECIMAL_ODDS` | 3.0 | Skip longer prices than this decimal threshold |
| `EDGE_SCALING_BASE` | 2% | Base edge required at even money |
| `EDGE_SCALING_RATE` | 2% | Extra edge required as odds get longer |
| `BLEND_WEIGHT_MIN` | 0.15 | Dynamic blend floor for low-confidence picks |
| `BLEND_WEIGHT_MAX` | 0.50 | Dynamic blend ceiling for high-confidence picks |
| `BLEND_CONFIDENCE_THRESHOLD` | 0.65 | Confidence level that starts increasing blend weight |
| `BLEND_AGREEMENT_BOOST` | 0.10 | Extra blend weight when the no-odds model agrees strongly |
| `LINE_MOVEMENT_FILTER` | true | Enable line movement filter |
| `LINE_AGAINST_EXTRA_EDGE` | 2% | Extra edge required when line moves against bet |
| `LINE_SHARP_BLOCK` | true | Block bets where sharp/steam move disagrees |
| `CONVICTION_MIN_MODEL_PROB` | 65% | Confidence floor for Trader C |
| `CONVICTION_MIN_NO_ODDS_PROB` | 50% | No-odds model floor for Trader C |
| `CONVICTION_BET_FRACTION` | 5% | Base bankroll fraction for conviction bets |
| `CONVICTION_CONFIDENCE_BONUS` | 1% | Extra sizing per confidence step above threshold |
| `CONVICTION_MAX_BET_FRACTION` | 8% | Hard cap per conviction bet |
| `EWM_HALFLIFE` | 3 | Exponential weighted mean halflife (in fights) |
| `ROLLING_WINDOW` | 5 | Number of recent fights for rolling averages |

## Web Dashboard

### Local dashboard

Run the local dashboard with:

```bash
python -m src.bot web
```

The local dashboard runs on port 5050 by default. Use `--port` to change it locally:

```bash
python -m src.bot web --port 8080
```

It includes:

- Wallet balance and portfolio value
- Live P&L summary, bet history, and portfolio chart
- Per-trader breakdown and trader race charts
- Upcoming UFC events from monitoring snapshots
- Fight predictions with detailed breakdowns
- Injury/cancellation alerts and line-movement analysis
- Open limit-order tracking
- Filter funnel analysis that shows where fights get rejected
- Recent bot activity and significant-action views
- Geoblock and proxy diagnostics

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/summary` | GET | Portfolio summary |
| `/api/bets` | GET | All ledger bets |
| `/api/pnl-history` | GET | P&L history for charting |
| `/api/positions` | GET | Current open positions |
| `/api/trade-history` | GET | Completed trade history |
| `/api/balance` | GET | Wallet USDC balance and equity |
| `/api/geoblock-status` | GET | Current Polymarket geoblock/proxy status |
| `/api/predictions` | GET | Cached fight predictions |
| `/api/predictions-detail` | GET | Detailed prediction breakdown |
| `/api/trader-breakdown` | GET | Per-trader P&L, win rate, ROI |
| `/api/trader-race` | GET | Trader performance comparison |
| `/api/upcoming-events` | GET | Upcoming UFC events |
| `/api/bot-activity` | GET | Recent bot log activity |
| `/api/bot-activity-snapshot` | GET | Activity snapshot with metadata |
| `/api/significant-actions` | GET | Notable bot actions |
| `/api/open-limit-orders` | GET | Open limit orders and status |
| `/api/injury-alerts` | GET | Injury/cancellation alerts |
| `/api/filter-funnel` | GET | Filter pipeline analysis |
| `/api/line-movements` | GET | Line movement data |
| `/api/refresh-prices` | POST | Refresh open-position prices |
| `/api/settle-auto` | POST | Auto-settle resolved bets |
| `/api/settle/<bet_id>/<result>` | POST | Manually settle a bet |

### Production entrypoint

For always-on deployment, use the production entrypoint:

```bash
python -m src.web.serve
```

This is the entrypoint used by Railway and Docker. Unlike `python -m src.bot web`, it also starts:

- the background live betting loop
- the background monitor and line-tracking loop
- delayed CLOB initialization for live price and account data

Relevant production environment variables:

- `PORT` — Web server port (default `5050`)
- `BET_INTERVAL_MINUTES` — Live betting loop interval (default `10`)
- `MIN_EDGE` — Minimum edge override for production loop (default `0.02`)
- `MONITOR_INTERVAL_HOURS` — Background monitor interval (default `6`)
- `CLOB_PROXY_URL` — Optional HTTP proxy for CLOB traffic and geoblock diagnostics

## Deployment

The bot can be deployed as a Docker container on [Railway](https://railway.app) or any other container platform:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env ufc-betting-bot
```

The container entrypoint runs `python -m src.web.serve`.

## Disclaimer

This project is for educational and research purposes. Sports betting involves risk. Never bet more than you can afford to lose. Past model performance does not guarantee future results.
