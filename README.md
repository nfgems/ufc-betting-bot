# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot with experimental ATP/WTA discovery and dry-run tooling. The repo covers data collection, live-compatible feature engineering, model training and evaluation, walk-forward backtesting, live prediction, and a Flask dashboard.

## Status As Of 2026-03-19

The previous README had drifted. The current repo state is:

- The UFC feature system defines 150 live-compatible features across 13 families.
- The default `python -m src.bot train` flow currently uses training spec `full_live_contract_v2` with 144 features.
- The currently promoted production artifact bundled under `models/` is `full_live_contract_v5_fullfit` with 138 features, recorded in [models/current_production_model.json](models/current_production_model.json).
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
| `WEB_DASHBOARD_TOKEN` | Dashboard auth on public binds | Required for hosted mutations; protected reads also use it on public binds |
| `LIVE_TRADING_MODE` | Hosted trading mode | `off`, `dry-run`, or `real` |
| `LIVE_MODEL` | Hosted model alias or explicit artifact path | Defaults to `xgboost` |
| `LIVE_TRADING_ARMED` | Real-trading arming switch | Must be `1` for `real` mode |
| `LIVE_TRADING_CONFIRMATION` | Real-trading confirmation string | Must equal `REAL_TRADING_ENABLED` for `real` mode |

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
- Current promoted production artifact: `full_live_contract_v5_fullfit`
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
- Readiness is exposed at `/healthz` and `/readyz`.
- Hosted startup fails closed for trading if required env vars, model artifacts, or writable ledger and log paths are missing.

Selected API routes:

- `/api/summary`
- `/api/predictions`
- `/api/predictions-detail`
- `/api/upcoming-events`
- `/api/open-limit-orders`
- `/api/balance`
- `/api/positions`
- `/api/bot-activity`
- `/api/geoblock-status`

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
