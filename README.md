# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot, with data collection, live-compatible feature engineering, model evaluation, backtesting, and a Flask dashboard.

## Current release — October 16, 2026

This checkout preserves the source baseline from the deployed Railway image:

```text
ghcr.io/nfgems/ufc-betting-bot@sha256:6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94
```

Railway service `ufc-bot` in project `ufc-betting-bot`, environment `production`, was verified online on October 16 with deployment `c48776b2-475a-45d3-bc85-1faca04d4c6c` and its persistent volume at `/app/logs`.

The October 17 account-read diagnostics fix is saved as an undeployed source change in [source-change-manifest.json](source-change-manifest.json). Retry and trading policies are unchanged.

Read [Current release](docs/CURRENT_RELEASE.md) for reproduction and preservation details, and [release-source-manifest.json](release-source-manifest.json) for the original deployed source hashes. The [canonical living handoff](docs/PRACTICAL_MODEL_DATA_RESET_LIVING_HANDOFF.md) records dated deployment evidence and remaining work; its newest entries come first.

- Production starts `/bin/bash /app/refresh-entrypoint.sh`, which launches `src.combined_service` under the application account. One supervisor owns the refresh worker and one separate UFC trading consumer.
- The combined service reads an accepted data/prediction publication and verifies its model and data hashes. Models, accepted baseline data, seed, and one required provenance record are inherited from the pinned private image.
- The October 11 activation enabled guarded real UFC trading. Subsequent dashboard and market-coverage releases preserved that configuration. BTC profiles and automatic model training/promotion remain disabled.
- The October 16 coverage fix displays observed Polymarket quotes for the full published card, including saved-only predictions without current bookmaker odds. Missing bookmaker inputs stay missing; saved timestamps and execution eligibility remain distinct.
- Dashboard probability gaps display as percentages, with explanatory help text. Current, saved, stale, and unavailable observations are labeled separately.
- Deployment and readiness evidence does not establish future order fills, settlement, or profitability. The handoff retains the limits of each verification.

A source commit or push does not authorize a Railway deployment. The live service is pinned to its image; source publication and runtime changes are separate operations.

## Runtime layout

| Component | Purpose |
|---|---|
| [refresh-entrypoint.sh](refresh-entrypoint.sh) | Validate controls and persistent paths, install the retained seed, then start the combined service |
| [src/combined_service.py](src/combined_service.py) | Own the web server, refresh calendar worker, and trading child process |
| [src/data/refresh_service.py](src/data/refresh_service.py) | Recover pending work and publish a validated data/prediction pair |
| [src/data/incremental_refresh.py](src/data/incremental_refresh.py) | Produce verified additions while preserving the accepted historical baseline |
| [src/trading_worker.py](src/trading_worker.py) / [src/refresh_live.py](src/refresh_live.py) | Consume the published snapshot, obtain current market observations, and run guarded S/C/M decisions |
| [src/web/refresh_dashboard.py](src/web/refresh_dashboard.py) | Connect the public dashboard to current saved reports and read-only account observations |
| [src/features/](src/features/) / [src/model/](src/model/) | Feature contracts, training, evaluation, prediction, and orientation handling |
| [src/strategy/](src/strategy/) / [src/polymarket/](src/polymarket/) | Selection, sizing, execution, accounting, and standalone crypto tools |
| [src/bot.py](src/bot.py) | Standalone CLI commands for research, data operations, and legacy workflows |
| [tests/](tests/) | Tests for the saved release |

`legacy-entrypoint.sh` and `src.web.serve` remain in the source tree, but they are not the current production startup path. `APP_ROLE` does not select a role in the current refresh entrypoint.

## Reproduce and verify the saved source

Requirements:

- Git and Python 3.11 or newer for source checks and local development.
- Docker and authenticated access to the private GHCR base image for the saved runtime.
- Separately provisioned credentials and persistent state for operational use.

```bash
git clone https://github.com/nfgems/ufc-betting-bot.git
cd ufc-betting-bot
python tools/verify_release_source.py
python scripts/check_tracked_artifact_integrity.py
docker build --network none -t ufc-saved-source:test .
```

Authenticate to GHCR through your normal credential mechanism before building. Docker must be able to obtain the pinned base image; `--network none` disables networking for build steps, not registry access.

The source verifier checks 285 repository files. The image verifier also checks the provenance record inherited from the base, for 286 runtime files. Preserve raw source/config bytes: line-ending conversion can invalidate these hashes. The Dockerfile copies versioned source over the pinned image and verifies it; it does not copy local models, credentials, betting ledgers, or datasets.

A clone plus `pip install` is not a complete reproduction of the accepted production model/data payload. Historical metadata under [models/](models/) is retained for reference; `models/current_production_model.json` describes the older August 4 bundle and is not the current combined service's accepted-model authority.

## Local development and CLI

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Keep `.env` local. [.env.example](.env.example) and [src/config.py](src/config.py) include settings for both current and legacy tools; they are not a ready-to-deploy configuration for the combined service. Local prediction, training, and backtesting also need their corresponding data/model inputs.

Commands run from the repository root:

```bash
python -m src.bot --help
python -m src.bot train --help
python -m src.bot backtest --help
python -m src.bot predict --help
python -m src.bot web --offline
```

The CLI also provides `scrape`, `evaluate`, `walkforward`, `backtest-compare`, `backfill-odds`, `monitor`, `track-lines`, `positions`, and `line-history-archive`. These are separate operational or research tools, not steps required to start the accepted release.

- `python -m src.bot web` starts the standalone legacy dashboard. It does not reproduce the combined refresh/trading service.
- Standalone `live --dry-run` and `live --real` remain available, but the current hosted consumer runs through `src.trading_worker` and `src.refresh_live`.
- `ufc-refresh-scheduled` is the older guarded roster/rebuild command. It is not the current incremental service's Tuesday scheduler; do not run it alongside the production writer.
- Training uses named contracts in [src/model/training_spec.py](src/model/training_spec.py). Use explicit inputs/specs and isolated output paths for experiments. Local aliases or a training command do not promote a model into the combined service.
- The accepted runtime retains primary, no-odds, and logistic models with exact hashes in `/app/release_payload/selected-runtime-manifest.json`. The selected source manifest and final published snapshot bind their corresponding data and predictions.
- Only features available at live inference belong in a training contract. Preserve real observed values and honest NaNs; do not invent defaults or bypass training odds noise. Training and inference retain their A/B orientation handling.

The separate `btc5m`, `btc5m-paper`, and `btc5m-opportunity` commands remain available for crypto work. They are not started by the combined UFC service. Setting `BTC5M_LIVE_PROFILES` to a nonempty value makes combined startup fail.

## Combined service configuration

These are the principal settings for the current container entrypoint. Advanced source, market, and execution settings remain in [src/config.py](src/config.py).

| Setting | Current behavior |
|---|---|
| `PORT` | Container entrypoint defaults to `8080`; Railway supplies its port |
| `WEB_DASHBOARD_TOKEN` | Required at startup in both supported modes; protects private APIs and controls |
| `LIVE_TRADING_MODE` | Exactly `off` or `real`; `dry-run` is rejected by combined startup |
| `LIVE_MODEL` | Must be explicitly selected for real startup and match the published accepted primary model |
| `LIVE_TRADING_ARMED` | Must be `1` for real trading |
| `LIVE_TRADING_CONFIRMATION` | Must be `REAL_TRADING_ENABLED` for real trading |
| `ODDS_API_KEY` | Current bookmaker observations and live odds workflows |
| `POLYMARKET_PRIVATE_KEY` / `POLYMARKET_FUNDER_ADDRESS` | Authenticated account/trading configuration; account ownership and cash/order reads must pass |
| `UFC_INCREMENTAL_REFRESH_ROOT` | Defaults to `/app/logs/incremental_refresh`; entrypoint confines it to that namespace |
| `UFC_REFRESH_TRADING_STATE_DIR` | Defaults to `/app/logs/refresh_trading_paper` in off mode or `/app/logs/refresh_trading_live` in real mode; must stay inside the matching namespace |
| `UFC_LOGS_DIR` | Real consumer uses the original accounting root, `/app/logs` for this deployment |
| `BTC5M_LIVE_PROFILES` | Must remain empty |
| `UFC_MONTHLY_CANDIDATE_ENABLED` | Must remain disabled |
| `UFC_V7_PROMOTION_MONTH_UTC` | Must remain empty |
| `MAX_BET_HOURS_BEFORE_EVENT` | Shared S/C/M entry window, default 48 hours before the fight |

`LIVE_TRADING_MODE=off` disables real orders, while refresh, market/account observations, and simulated decisions can still run. Paper data and logs use a separate namespace. It is not an offline or no-network mode.

Real mode additionally requires valid model artifacts, writable accounting paths, verified account/ledger ownership, the existing arming checks, and successful runtime safety reads. Arming values alone are insufficient. The trading consumer normally waits 300 seconds between completed cycles.

### Local container example with real orders disabled

Use an intentionally configured local environment file with at least:

```dotenv
LIVE_TRADING_MODE=off
WEB_DASHBOARD_TOKEN=replace_with_a_private_token
BTC5M_LIVE_PROFILES=
UFC_MONTHLY_CANDIDATE_ENABLED=0
UFC_V7_PROMOTION_MONTH_UTC=
```

Provision the credentials needed for the observations you intend to run, then use a separate local Docker volume:

```bash
docker run --rm --env-file .env -e LIVE_TRADING_MODE=off -e PORT=8080 -p 127.0.0.1:8080:8080 --mount type=volume,src=ufc-local-state,dst=/app/logs ufc-saved-source:test
```

This starts workers and may request source/market data. For source validation without startup, use the verification commands above and the isolated checks in [.github/workflows/ci.yml](.github/workflows/ci.yml). Keep local experiments separate from the live Railway volume.

## Refresh and persistent state

The current service schedules data updates for **Tuesday at 11:00 a.m. America/New_York**, with **Wednesday at 11:00 a.m.** available only after a failed or missed Tuesday. Daylight-saving changes follow the named timezone. A successful week's update is not repeated.

The service recovers an existing pending job before considering new calendar work. Original intent, claims, consumed identifiers, and completion records survive process restarts. Failures retain the previous accepted publication; a new dataset becomes served only when its matching prediction report and original model hashes validate.

Refresh preserves the accepted historical baseline and produces verified new additions. It does not retrain or automatically promote models. The older `UFC_REFRESH_ENABLED`, `UFC_REFRESH_INTERVAL_HOURS`, and `UFC_REFRESH_INITIAL_DELAY_MINUTES` settings belong to the legacy loop and do not configure this calendar worker.

| Path | Contents |
|---|---|
| `/app/release_payload` | Accepted baseline, selected source/runtime manifests, and model/data references inherited from the image |
| `/app/refresh_seed` | Retained startup seed; installed through the checked seed installer |
| `/app/logs/incremental_refresh` | Durable calendar, pending jobs, data generations, prediction reports, and final publication |
| `/app/logs/refresh_trading_live` | Real consumer reports, heartbeat, and pause control |
| `/app/logs/refresh_trading_paper` | Separate simulated consumer state, data, and logs |
| `/app/logs` | Existing production accounting and persistent logs |

Code rollback must retain newer valid volume records and accounting. Do not restore old ledgers, erase pending work, or replace the active entrypoint with the legacy startup.

Line-history archive/list/restore utilities remain available through `python -m src.bot line-history-archive`. See [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md) for historical operational details, using the current release and newest handoff entries when older startup instructions conflict.

## Dashboard and trading behavior

The deployed dashboard is public at [ufc-bot-production.up.railway.app](https://ufc-bot-production.up.railway.app/). Public pages include `/`, `/ufc`, `/predictions`, `/activity`, `/bet-history`, and `/execution-breakdown`.

Public read APIs expose dashboard summaries, predictions, account views, activity, and execution explanations with sensitive identifiers redacted. See [src/web/refresh_dashboard.py](src/web/refresh_dashboard.py) for the installed read-route allowlist. Legacy settlement, redemption, and order-mutation routes are not registered by this dashboard adapter.

| Endpoint | Access and purpose |
|---|---|
| `GET /healthz` | Public process liveness and trading-mode summary |
| `GET /readyz` | Public accepted-publication and worker readiness; returns 503 when unavailable |
| `GET /api/dashboard-status` | Public dashboard data availability/freshness |
| `GET /api/refresh-status` | Authenticated refresh state and served/pending generation details |
| `GET /api/refresh-predictions` | Authenticated published prediction report |
| `GET /api/trading-status` | Authenticated consumer status |
| `GET /api/trading-predictions` | Authenticated latest consumer report |
| `POST /api/trading-control` | Authenticated pause/resume control with `{"paused": true}` or `{"paused": false}` |

Private APIs accept `Authorization: Bearer <token>`, `X-Dashboard-Token`, or an authenticated session. Public dashboard access does not remove the startup token requirement.

The UFC portfolio uses Single (S), Conviction (C), and Model Tracker (M), with separate accounting. Retired traders remain visible where needed for historical reconciliation. Execution retains the shared 48-hour entry window, current event-time checks, liquidity and cash requirements, duplicate-exposure controls, and data-quality checks.

Predictions or observed quotes alone do not imply trade eligibility. Saved-only predictions retain their timestamps and missing current bookmaker inputs; the full-card quote display does not expand the execution candidate set. The execution-breakdown page reports the saved S/C/M decisions associated with the accepted cycle.

## Checks and archived research

Before pushing source changes:

```bash
python tools/verify_release_source.py
python scripts/check_tracked_artifact_integrity.py
python -m compileall src tests
```

The release's full test suite runs inside the pinned image with networking and orders disabled, without application startup. See [.github/workflows/ci.yml](.github/workflows/ci.yml) for the exact image-based pytest and dashboard-script checks. Running `pytest -q tests/` directly requires a compatible environment and the relevant private runtime assets.

Earlier source/tests are preserved on `archive/main-before-production-sync-20261016`; pending research is on `archive/research-before-production-sync-20261016`. They are separate versions, not additions to the current release's test/model tree. The reconciliation backup is in the sibling `../ufc-git-reconciliation-20261016/research-backup` directory.

Older March scratch material remains in the private `nfgems/ufc-betting-bot-worktree-archive-20260323` repository. Ignored datasets, original research evidence, and consumed authorization records remain at their original local paths. A source archive does not recreate authorization to rerun a closed experiment.

Never commit `.env`, credentials, betting ledgers, or model pickle artifacts.

## Disclaimer

Sports betting involves real financial risk. Never bet more than you can afford to lose, and do not treat historical model performance as a guarantee of future results.
