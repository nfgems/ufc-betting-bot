# Production Runbook

## Startup Modes

- `LIVE_TRADING_MODE=off`
  Safe default. The web app, monitor loop, and CLOB init can still start, but the betting loop stays disabled.
- `LIVE_TRADING_MODE=dry-run`
  Starts the hosted betting loop in paper mode. Real orders are never submitted.
- `LIVE_TRADING_MODE=real`
  Starts the hosted betting loop in real-money mode only if all readiness checks pass.

## Required Environment Variables

Base hosted deploy:

- `ODDS_API_KEY`
- `WEB_DASHBOARD_TOKEN` on any public bind
- `LIVE_TRADING_MODE`
- `LIVE_MODEL` if you are not using the default `xgboost`

Real-money deploy:

- `POLYMARKET_PRIVATE_KEY`
- `LIVE_TRADING_ARMED=1`
- `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`
- `POLYMARKET_FUNDER_ADDRESS` is recommended; if omitted, the bot falls back to proxy-wallet auto-discovery

Optional:

- `CLOB_PROXY_URL`
- `MIN_EDGE`
- `BET_INTERVAL_MINUTES`
- `MONITOR_INTERVAL_HOURS`

## Current UFC Model

- Current promoted live alias: `xgboost`
- Current promoted production bundle: `models/current_production_model.json`
- As of `2026-03-19`, the `xgboost` and `xgboost_no_odds` aliases point to the V5 full-data production refit derived from the frozen `2014-2026` UFC dataset.
- Leave `LIVE_MODEL` unset to use the promoted alias, or set it explicitly only when testing an alternate artifact.

## Readiness Checks

`src.web.serve` fails closed for trading when any required startup check fails. Current checks cover:

- live mode validity
- required secrets
- required model artifacts
- writable log, prediction-cache, and ledger paths
- dashboard mutation token on public binds
- explicit real-trading arming and confirmation env vars

## Health Endpoints

- `GET /healthz`
  Process is up. Always returns `200` if the web server is serving.
- `GET /readyz`
  Hosted runtime status. Returns `200` only when the configured startup mode is ready; otherwise returns `503` with the blocking errors.

## Kill Switch

1. Set `LIVE_TRADING_MODE=off`.
2. Redeploy or restart the service.
3. Verify `GET /readyz` reports `effective_live_mode=off`.

Emergency fallback:

1. Remove `LIVE_TRADING_ARMED` or change `LIVE_TRADING_CONFIRMATION`.
2. Redeploy or restart the service.
3. Confirm the betting loop does not restart.

## Rollback

1. Redeploy the previous known-good image or commit.
2. Keep `LIVE_TRADING_MODE=off` during rollback verification.
3. Confirm `/healthz` is green and `/readyz` reflects the expected disabled state.
4. Re-arm only after startup checks are clean again.
5. If the rollback is model-only, restore the prior alias targets from `models/backups/20260319_v2_pre_v5_promotion`.

## First-Live Checklist

1. Confirm the promoted model artifacts exist under `models/`.
2. Confirm `ODDS_API_KEY`, `POLYMARKET_PRIVATE_KEY`, and `WEB_DASHBOARD_TOKEN` are present in the deploy environment.
3. Start with `LIVE_TRADING_MODE=dry-run` and verify `/readyz` returns `200`.
4. Verify the dashboard loads and recent activity is visible.
5. Verify ledgers and `data/logs/bot.log` are writable in the deployed container/volume.
6. Verify any proxy/geoblock requirements through `/api/geoblock-status`.
7. Change to `LIVE_TRADING_MODE=real`, set `LIVE_TRADING_ARMED=1`, and set `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`.
8. Redeploy and confirm `/readyz` returns `200` with `effective_live_mode=real`.
