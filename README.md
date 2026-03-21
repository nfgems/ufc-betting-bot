# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot with experimental ATP/WTA discovery and dry-run tooling. The repo covers data collection, live-compatible feature engineering, model training and evaluation, walk-forward backtesting, live prediction, and a Flask dashboard.

## Status As Of 2026-03-20

- The UFC feature system defines a pool of 150 live-compatible features across 18 families (differentials, individual rolling/Elo, encoded categoricals, physical attributes, finish method rates, odds-derived, event context, career records, cage rust/layoff, weight class moves, style matchup, experimental, rematch/H2H, Elo momentum, strength of schedule, line movement, rankings, method odds). The current production model uses 138 of these.
- The default `python -m src.bot train` flow uses training spec `full_live_contract_v2` (144 features). The promoted production artifact is `v5_fullfit_retrain` (138 features), recorded in [models/current_production_model.json](models/current_production_model.json). It was retrained after consolidating all historical odds sources for full `a_implied_prob` coverage.
- Tennis support is discovery, training, prediction, and dry-run only. Real-money tennis execution is not implemented.

## Main Components

- `src/data/`: scraping, fallbacks, odds ingestion, rankings, line tracking, live monitoring, and tennis data loaders
- `src/features/`: UFC and tennis feature builders
- `src/model/`: training specs, training, evaluation, prediction, and provenance tooling
- `src/strategy/`: backtests, value logic, duo-trader execution, and model selection utilities
- `src/polymarket/`: market lookup, CLOB client, execution, positions, and ledgers
- `src/web/`: Flask dashboard and hosted runtime entrypoint
- `models/`: canonical alias models, candidate artifacts, and promotion manifests
- `tests/`: regression and runtime coverage

## Prerequisites

- Python 3.11 or newer
- `ODDS_API_KEY` for most live odds workflows
- `POLYMARKET_PRIVATE_KEY` only if you want real-money Polymarket trading
- `BETSAPI_TOKEN` only for BetsAPI-backed tennis bookmaker workflows

## Setup

```bash
git clone https://github.com/nfgems/ufc-betting-bot.git
cd ufc-betting-bot
python -m venv .venv
```

Install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

If `python` is not on your Windows PATH, use `py -3.11 -m venv .venv` to create the virtual environment.

Create `.env` from `.env.example` and fill in only the variables your workflow needs:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Environment Variables

| Variable | Used for | Notes |
|---|---|---|
| `ODDS_API_KEY` | UFC live odds, backfills, prediction, and live workflows | Required for most non-offline UFC commands |
| `BETSAPI_TOKEN` | Tennis bookmaker audit and ingestion | Optional |
| `POLYMARKET_PRIVATE_KEY` | Trading and account access | Required for real-money trading |
| `POLYMARKET_FUNDER_ADDRESS` | Proxy wallet override | Optional; runtime can attempt auto-discovery |
| `CLOB_PROXY_URL` | Proxying CLOB traffic | Optional; surfaced by geoblock diagnostics |
| `POLYMARKET_AUTO_REDEEM` | Auto-claiming resolved winnings | Optional; set to `1` to redeem winnings from the background monitor only |
| `POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS` | Minimum gap between background auto-redeem checks | Optional; defaults to `6` hours |
| `POLYMARKET_AUTO_REDEEM_PENDING_TTL_HOURS` | How long to trust a missing relayer tx before clearing the pending lock | Optional; defaults to `24` hours |
| `POLYMARKET_RELAYER_URL` | Polymarket relayer base URL | Optional; defaults to `https://relayer-v2.polymarket.com` |
| `POLYMARKET_BUILDER_API_KEY` / `POLYMARKET_BUILDER_SECRET` / `POLYMARKET_BUILDER_PASSPHRASE` | Builder-authenticated relayer submissions | Optional; one supported auth mode for redeeming |
| `POLYMARKET_RELAYER_API_KEY` / `POLYMARKET_RELAYER_API_KEY_ADDRESS` | Direct relayer API key auth | Optional; alternative auth mode for redeeming |
| `WEB_DASHBOARD_TOKEN` | Dashboard auth on public binds | Required for hosted mutations; protected reads also use it on public binds |
| `LIVE_TRADING_MODE` | Hosted trading mode | `off`, `dry-run`, or `real` |
| `LIVE_MODEL` | Hosted model alias or explicit artifact path | Defaults to `xgboost` |
| `LIVE_TRADING_ARMED` | Real-trading arming switch | Must be `1` for `real` mode |
| `LIVE_TRADING_CONFIRMATION` | Real-trading confirmation string | Must equal `REAL_TRADING_ENABLED` for `real` mode |
| `PORT` | Web server port | Optional; defaults to `5050` |
| `WEB_HOST` | Web server bind address | Optional; defaults to `0.0.0.0` for hosted entrypoint |
| `MONITOR_INTERVAL_HOURS` | Background monitor loop interval | Optional; defaults to `6` |
| `BET_INTERVAL_MINUTES` | Hosted betting loop interval | Optional; defaults to `10` |
| `MIN_EDGE` | Edge threshold override for hosted trading | Optional; uses config default |
| `POLYMARKET_CHAIN_ID` | Polygon chain ID | Optional; defaults to `137` |
| `RAILWAY_VOLUME_MOUNT_PATH` | Railway persistent storage mount | Optional; used by Railway deployments for data/model/log persistence |
| `BETSAPI_REQUEST_MIN_INTERVAL_SECONDS` | BetsAPI rate-limit floor | Optional |
| `BETSAPI_429_RETRY_MIN_SECONDS` | BetsAPI 429-retry backoff floor | Optional |

## CLI Overview

All commands run from the project root with `python -m src.bot ...`.

### UFC workflow

```bash
# Refresh raw UFC data
python -m src.bot scrape

# Train using the default CLI training spec (currently full_live_contract_v2)
python -m src.bot train

# Train a specific contract explicitly
python -m src.bot train --spec full_live_contract_v5_fullfit

# Keep alternate artifacts separate instead of overwriting canonical paths
python -m src.bot train --spec full_live_contract_v5_fullfit --output-subdir candidates/full_live_contract_v5_fullfit

# Evaluate saved models against data/processed/test_set.csv
python -m src.bot evaluate

# Static or walk-forward backtesting
python -m src.bot backtest
python -m src.bot walkforward
python -m src.bot backtest-compare --walkforward

# Sensitivity analysis
python -m src.bot sensitivity

# Backfill historical odds from The Odds API
python -m src.bot backfill-odds
python -m src.bot backfill-odds --offsets 7,3,1 --fresh

# Live prediction and trading
python -m src.bot predict
python -m src.bot live --dry-run
python -m src.bot live --real

# Monitoring and operations
python -m src.bot monitor
python -m src.bot track-lines
python -m src.bot signals
python -m src.bot positions
python -m src.bot dashboard
python -m src.bot settle --auto
python -m src.bot redeem
```

Notes:

- `live --real` is blocked unless `LIVE_TRADING_ARMED=1` and `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`.
- `predict` and `live` load the canonical alias models such as `models/xgboost_model.pkl`.
- The promoted alias targets are recorded in [models/current_production_model.json](models/current_production_model.json).

### Tennis commands

```bash
python -m src.bot tennis-discover
python -m src.bot tennis-train
python -m src.bot tennis-bookmaker-audit
python -m src.bot tennis-predict
python -m src.bot tennis-live
```

Tennis trading is dry-run only.

## Training Specs And Model State

The repo uses a spec-driven training system in [src/model/training_spec.py](src/model/training_spec.py). The important distinction is:

- Default training flow: `full_live_contract_v2`
- Current promoted production artifact: `v5_fullfit_retrain` (spec `full_live_contract_v5_fullfit`)
- Canonical live aliases: `xgboost`, `xgboost_no_odds`, and `logistic`

If you are reproducing the currently promoted production line, use the manifest and spec files under [models/](models/) rather than assuming the default `train` command matches the promoted artifact.

## Web Dashboard

Local dashboard only:

```bash
python -m src.bot web
python -m src.bot web --port 8080
python -m src.bot web --offline
```

Hosted or always-on entrypoint:

```bash
python -m src.web.serve
```

Behavior:

- `python -m src.bot web` starts only the Flask dashboard.
- `python -m src.web.serve` starts the dashboard plus the background monitor loop, delayed CLOB initialization, and the hosted betting loop when `LIVE_TRADING_MODE` is `dry-run` or `real`.
- The hosted entrypoint binds `0.0.0.0` by default so Railway and Docker can reach it; override with `WEB_HOST` only if you intentionally need a different bind target.
- Readiness is exposed at `/healthz` and `/readyz`.
- Hosted startup fails closed for trading if required env vars, model artifacts, or writable ledger and log paths are missing.

Selected API routes:

- `/healthz`, `/readyz` — health and readiness probes
- `/api/summary` — dashboard overview
- `/api/predictions`, `/api/predictions-detail` — model predictions
- `/api/upcoming-events` — upcoming UFC events
- `/api/positions`, `/api/open-limit-orders` — Polymarket positions and orders
- `/api/balance` — wallet balance
- `/api/bets`, `/api/trade-history` — bet and trade history
- `/api/pnl-history` — P&L over time
- `/api/bot-activity`, `/api/significant-actions` — bot activity and notable actions
- `/api/trader-race`, `/api/trader-breakdown` — trader comparison metrics
- `/api/injury-alerts` — injury detection
- `/api/line-movements` — sharp money and line movement tracking
- `/api/filter-funnel` — prediction filter diagnostics
- `/api/geoblock-status` — geo-restriction diagnostics
- `/api/refresh-prices` (POST), `/api/settle-auto` (POST), `/api/redeem-auto` (POST) — operational actions

See [src/web/app.py](src/web/app.py) for the full route list.

## Deployment

Docker and Railway use the hosted entrypoint:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env -p 5050:5050 ufc-betting-bot
```

Safe hosted default:

```dotenv
LIVE_TRADING_MODE=off
```

Paper-trading hosted deploy:

```dotenv
LIVE_TRADING_MODE=dry-run
WEB_DASHBOARD_TOKEN=change_me
```

Real-money hosted deploy:

```dotenv
LIVE_TRADING_MODE=real
LIVE_TRADING_ARMED=1
LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED
WEB_DASHBOARD_TOKEN=change_me
```

For production operations and rollback details, see [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md).

## Disclaimer

This project is for research and education. Sports betting involves real financial risk. Never bet more than you can afford to lose, and do not treat historical model performance as a guarantee of future results.
