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
- `POLYMARKET_CLOB_URL` (defaults to `https://clob.polymarket.com`; override only for intentional CLOB endpoint smoke tests)
- `POLYMARKET_BUILDER_CODE` for V2 builder attribution
- `POLYMARKET_RELAYER_URL`
- `POLYMARKET_RELAYER_API_KEY`
- `POLYMARKET_RELAYER_API_KEY_ADDRESS`
- `MIN_EDGE`
- `BET_INTERVAL_MINUTES`
- `MONITOR_INTERVAL_HOURS`
- `GEMINI_TRACKER_CONFIDENCE_CAP`
- `APP_ROLE` (defaults to `web`; use `ufc-refresh-scheduled` only for a one-shot refresh service)

Removed for CLOB V2:

- `POLYMARKET_BUILDER_API_KEY`
- `POLYMARKET_BUILDER_SECRET`
- `POLYMARKET_BUILDER_PASSPHRASE`

## Current UFC Model

- Current promoted live alias: `xgboost`
- Current promoted production bundle: `models/current_production_model.json`
- As of `2026-06-05`, the `xgboost`, `xgboost_no_odds`, and `logistic` aliases point to the V6 full-fit production bundle (`full_live_contract_v6_fullfit`, 202 features).
- The local manifest reports `bundle_id=ufc-production-20260529-full_live_contract_v6_fullfit`, `built_at=2026-05-30T00:32:22.296122+00:00`, and `snapshot_max_event_date=2026-05-29`.
- Railway `/readyz` and startup logs report the active production bundle loaded from the mounted runtime manifest.
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
5. If the rollback is model-only, restore the prior production alias targets from `models/backups/pre_new_model_promotion_20260529_202737`. The older `models/backups/20260326_v6_tuned_pre_fullfit_promotion` directory is for rolling back past the March 2026 V6 full-fit promotion.

## First-Live Checklist

1. Confirm the promoted model artifacts exist under `models/`.
2. Confirm `ODDS_API_KEY`, `POLYMARKET_PRIVATE_KEY`, and `WEB_DASHBOARD_TOKEN` are present in the deploy environment.
3. Confirm CLOB balance and allowance with `ClobClientWrapper().get_balance_allowance()`.
4. For V2 live trading, confirm the proxy wallet has usable pUSD collateral. If it only has USDC.e, wrap into pUSD before arming.
5. Start with `LIVE_TRADING_MODE=dry-run` and verify `/readyz` returns `200`.
6. Verify the dashboard loads and recent activity is visible.
7. Verify ledgers and `bot.log` are writable in the active `LOGS_DIR` (on Railway this should follow `RAILWAY_VOLUME_MOUNT_PATH` unless `UFC_LOGS_DIR` overrides it).
8. Verify any proxy/geoblock requirements through `/api/geoblock-status`.
9. Change to `LIVE_TRADING_MODE=real`, set `LIVE_TRADING_ARMED=1`, and set `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`.
10. Redeploy and confirm `/readyz` returns `200` with `effective_live_mode=real`.

## CLOB V2 Runtime

Since `2026-05-28`, this repo runs on the migrated CLOB V2 stack with `py-clob-client-v2==1.0.0`. The previous April 28, 2026 cutover checklist is historical; normal production operations should treat V2 as the baseline.

Before re-arming real trading after any CLOB/client change:

1. Keep `LIVE_TRADING_MODE=off` during deploy verification.
2. Verify `/healthz` and `/readyz`.
3. Verify `get_balance_allowance()`, `get_cash_balance_details()`, `get_open_orders()`, and `PositionMonitor().compute_pnl()`.
4. Confirm the proxy wallet has usable pUSD collateral and allowance.
5. If a write-path smoke is approved, place and cancel one tiny limit order only after verifying balance, allowance, market `tick_size`, and `neg_risk`.
6. Re-arm only after the read checks and any approved write smoke are clean.

Rollback posture:

- Do not use a V1 client rollback as recovery after April 28, 2026.
- Disable trading first with `LIVE_TRADING_MODE=off`.
- Hot-fix forward on the migrated V2 stack, redeploy, smoke test, then re-arm.
