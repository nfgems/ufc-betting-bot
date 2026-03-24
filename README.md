# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot with experimental ATP/WTA data, modeling, and execution tooling. The repo covers data collection, live-compatible feature engineering, model training and evaluation, walk-forward backtesting, live prediction, and a Flask dashboard.

## Status As Of 2026-03-23

- The UFC feature system supports up to 202 live-compatible features across 20+ families. The active production model spec is `full_live_contract_v6_tuned`.
- On Railway, the runtime source of truth is the active production bundle manifest. The hosted service uses image-bundled model aliases plus the canonical `data/processed/fights_cleaned.csv` and `data/processed/features.csv` snapshot, with bundle validation at startup.
- The default `python -m src.bot train` flow uses training spec `full_live_contract_v6` (202 features). Candidate artifacts under `models/candidates/` and `data/processed/candidates/` are offline-only unless explicitly promoted.
- `data/raw/ufc-master.csv` remains a legacy training input for rebuild/training utilities. It is not the hosted inference source of truth.
- Tennis support covers discovery, training, prediction, dry-run execution, and an experimental shared-wallet execution path. That tennis path is disabled by default behind `TENNIS_TRADER_ENABLED` and is not a promoted production line. An experimental LLM operator gate is available for both UFC and tennis decision pipelines.
- Official ATP/WTA player-profile enrichment is available as a separate cached pipeline. It fills only missing static fields such as birth date-derived age, handedness, and height from official sources; it does not fabricate or backfill historical rankings from current profile pages.

## Main Components

- `src/data/`: scraping, fallbacks, odds ingestion, rankings, line tracking, live monitoring, tennis data loaders, player profiles, rankings history, and pre-UFC career scraping
- `src/features/`: UFC and tennis feature builders (including experimental features)
- `src/model/`: training specs, training, evaluation, prediction, feature provenance tooling, and model variant management
- `src/strategy/`: backtests, value logic, duo-trader execution, model selection utilities, LLM operator gates, and tennis decision logic
- `src/polymarket/`: market lookup, CLOB client, execution, positions, and ledgers
- `src/web/`: Flask dashboard, hosted runtime entrypoint, and operator UI
- `models/`: canonical alias models, candidate artifacts, and promotion manifests
- `scripts/`: one-off data collection, odds scraping, and analysis utilities
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
| `POLYMARKET_AUTO_REDEEM` | Auto-claiming resolved winnings | Optional; set to `1` to enable |
| `POLYMARKET_RELAYER_URL` | Polymarket relayer base URL | Optional; defaults to `https://relayer-v2.polymarket.com` |
| `POLYMARKET_BUILDER_API_KEY` / `POLYMARKET_BUILDER_SECRET` / `POLYMARKET_BUILDER_PASSPHRASE` | Builder-authenticated relayer submissions | Optional; one supported auth mode for redeeming |
| `POLYMARKET_RELAYER_API_KEY` / `POLYMARKET_RELAYER_API_KEY_ADDRESS` | Direct relayer API key auth | Optional; alternative auth mode for redeeming |
| `WEB_DASHBOARD_TOKEN` | Dashboard auth on public binds | Required for hosted mutations; protected reads also use it on public binds |
| `LIVE_TRADING_MODE` | Hosted trading mode | `off`, `dry-run`, or `real` |
| `LIVE_MODEL` | Hosted model alias or explicit artifact path | Defaults to `xgboost` |
| `LIVE_TRADING_ARMED` | Real-trading arming switch | Must be `1` for `real` mode |
| `LIVE_TRADING_CONFIRMATION` | Real-trading confirmation string | Must equal `REAL_TRADING_ENABLED` for `real` mode |
| `TENNIS_TRADER_ENABLED` | Enable the experimental tennis trader inside shared-wallet portfolio runs | Optional; defaults to `0` and should stay off unless you are intentionally exercising the tennis path |
| `TENNIS_PORTFOLIO_SHARE` | Share of wallet equity and cash reserved for tennis when the experimental trader is enabled | Optional; defaults to `0.25` |
| `TENNIS_TRADING_ARMED` | Tennis-specific real-execution arming switch | Required before any experimental non-dry-run tennis execution is allowed |
| `TENNIS_TRADING_CONFIRMATION` | Tennis-specific execution confirmation string | Must equal `EXPERIMENTAL_TENNIS_TRADING_ENABLED` before any experimental non-dry-run tennis execution is allowed |
| `TENNIS_LLM_VETO_ENABLED` | Enable the Gemini-based tennis veto operator | Optional; defaults to `0` |
| `TENNIS_LLM_VETO_FAIL_CLOSED` | Auto-skip tennis candidates if the veto layer is enabled but cannot complete | Optional; defaults to `1` |
| `TENNIS_LLM_VETO_MODEL` | Gemini model name used by the tennis veto operator | Optional; defaults to `gemini-2.5-flash` |
| `GEMINI_API_KEY` | Gemini API access for the tennis veto operator | Required only when `TENNIS_LLM_VETO_ENABLED=1` |
| `PORT` | Web server port | Optional; defaults to `5050` |
| `WEB_HOST` | Web server bind address | Optional; defaults to `0.0.0.0` for hosted entrypoint |
| `MONITOR_INTERVAL_HOURS` | Background monitor loop interval | Optional; defaults to `6` |
| `BET_INTERVAL_MINUTES` | Hosted betting loop interval | Optional; defaults to `10` |
| `MIN_EDGE` | Edge threshold override for hosted trading | Optional; uses config default |
| `POLYMARKET_CHAIN_ID` | Polygon chain ID | Optional; defaults to `137` |
| `RAILWAY_VOLUME_MOUNT_PATH` | Railway persistent storage mount | Optional; used by Railway deployments for data/model/log persistence |
| `UFC_DATA_DIR` | Override data directory path | Optional; defaults to `data/` under project root |
| `UFC_MODELS_DIR` | Override models directory path | Optional; defaults to `models/` under project root |
| `UFC_LOGS_DIR` | Override logs directory path | Optional; defaults to `logs/` under project root |
| `UFC_REFRESH_ENABLED` | Enable hosted UFC refresh loop | Optional; `1` runs scheduled UFC refreshes inside the always-on hosted service |
| `UFC_REFRESH_INTERVAL_HOURS` | Hosted UFC refresh cadence | Optional; defaults to `168` hours |
| `UFC_REFRESH_INITIAL_DELAY_MINUTES` | Delay first hosted UFC refresh after boot | Optional; defaults to `30` minutes |
| `UFC_REFRESH_LIMIT_FIGHTERS` | Debug cap for hosted UFC refresh | Optional; leave blank in production |
| `UFC_REFRESH_MIN_*` | Coverage-drop alert floors for hosted refresh | Optional; see `.env.example` for the full list |
| `BETSAPI_REQUEST_MIN_INTERVAL_SECONDS` | BetsAPI rate-limit floor | Optional |
| `BETSAPI_429_RETRY_MIN_SECONDS` | BetsAPI 429-retry backoff floor | Optional |

Polymarket client note: the pinned `py_clob_client` contract used here must expose `derive_api_key()` and `create_api_key()`. The legacy `create_or_derive_api_creds()` helper is no longer the runtime path.

## CLI Overview

All commands run from the project root with `python -m src.bot ...`.

### UFC workflow

```bash
# Refresh raw UFC data
python -m src.bot scrape

# Train using the default CLI training spec (currently full_live_contract_v6)
python -m src.bot train

# Train a specific contract explicitly
python -m src.bot train --spec full_live_contract_v6

# Keep alternate artifacts separate instead of overwriting canonical paths
python -m src.bot train --spec full_live_contract_v6 --output-subdir candidates/v6_eval

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
python -m src.bot ufc-refresh-scheduled
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
python -m src.bot tennis-predict
python -m src.bot tennis-live
python -m src.bot tennis-player-profiles
python -m src.bot tennis-refresh-daily
python -m src.bot tennis-rankings-history
python -m src.bot tennis-lockbox-eval
python -m src.bot tennis-bookmaker-audit
```

Tennis notes:

- `tennis-live` defaults to dry-run.
- `tennis-live --no-dry-run` is blocked unless `TENNIS_TRADING_ARMED=1` and `TENNIS_TRADING_CONFIRMATION=EXPERIMENTAL_TENNIS_TRADING_ENABLED`.
- Shared-wallet portfolio runs include tennis only when `TENNIS_TRADER_ENABLED=1`.
- Live tennis order placement inside shared-wallet portfolio runs also requires `TENNIS_TRADING_ARMED=1` and `TENNIS_TRADING_CONFIRMATION=EXPERIMENTAL_TENNIS_TRADING_ENABLED`.
- The promoted production manifest under `models/current_production_model.json` is UFC-only.

## Training Specs And Model State

The repo uses a spec-driven training system in [src/model/training_spec.py](src/model/training_spec.py). Available specs:

| Spec | Features | Notes |
|------|----------|-------|
| `full_live_contract_v2` | 132 | Legacy default |
| `full_live_contract_v5_fullfit` | 126 | Current promoted production spec |
| `full_live_contract_v6` (default) | 202 | Current default; expanded feature set with strike/position distributions, defensive quality, opponent strength |
| `full_live_contract_v6_tuned` | 202 | Optuna-tuned hyperparameters |
| `full_live_contract_v6_fullfit` | 202 | Full-fit variant for promotion |

Current promoted production artifact: `v5_fullfit_retrain` (spec `full_live_contract_v5_fullfit`, 126 features). Canonical live aliases: `xgboost`, `xgboost_no_odds`, and `logistic`.

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
- `/api/runtime-status` — hosted runtime component status
- `/api/closed-positions` — resolved Polymarket positions
- `/api/bot-activity-snapshot` — activity snapshot
- `/bet-history` — bet history page
- `/operator`, `/api/operator-decisions` — LLM operator interface and decisions

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

### Railway UFC Refresh

The repo includes a full UFC refresh command:

```bash
python -m src.bot ufc-refresh-scheduled
```

This refreshes the official active roster, backfills active-roster UFCStats data, rebuilds processed UFC artifacts, and writes a profile audit snapshot.

For Railway, the important constraint is that persistent volumes are attached per service. If your always-on web service owns the UFC data volume, a second cron service will not update that same on-disk dataset. The practical Railway setup is to enable the hosted UFC refresh loop inside the existing web service so it runs against the same mounted volume.

Recommended hosted settings:

```dotenv
UFC_REFRESH_ENABLED=1
UFC_REFRESH_INTERVAL_HOURS=168
UFC_REFRESH_INITIAL_DELAY_MINUTES=30
```

Notes:

- `168` hours means once per week. Adjust if you want a tighter cadence.
- Leave `UFC_REFRESH_LIMIT_FIGHTERS` blank in production. It exists only for smoke testing.
- The hosted refresh loop writes through the same guarded atomic CSV paths as the manual refresh command, so empty scrapes do not replace good artifacts with blank files.
- Refresh failures are reported immediately in the hosted runtime status as a degraded `ufc_refresh_loop` component.
- Coverage-drop alerts are optional. Set one or more `UFC_REFRESH_MIN_*` env vars if you want the hosted refresh loop to mark itself degraded when audited coverage falls below your chosen floor.

## Disclaimer

This project is for research and education. Sports betting involves real financial risk. Never bet more than you can afford to lose, and do not treat historical model performance as a guarantee of future results.
