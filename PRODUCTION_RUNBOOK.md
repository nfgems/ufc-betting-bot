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
- `POLYMARKET_CLOB_URL` (defaults to `https://clob.polymarket.com`; use `https://clob-v2.polymarket.com` for pre-cutover V2 smoke tests)
- `POLYMARKET_BUILDER_CODE` for V2 builder attribution
- `POLYMARKET_RELAYER_URL`
- `POLYMARKET_RELAYER_API_KEY`
- `POLYMARKET_RELAYER_API_KEY_ADDRESS`
- `MIN_EDGE`
- `BET_INTERVAL_MINUTES`
- `MONITOR_INTERVAL_HOURS`

Removed for CLOB V2:

- `POLYMARKET_BUILDER_API_KEY`
- `POLYMARKET_BUILDER_SECRET`
- `POLYMARKET_BUILDER_PASSPHRASE`

## Current UFC Model

- Current promoted live alias: `xgboost`
- Current promoted production bundle: `models/current_production_model.json`
- As of `2026-03-25`, the `xgboost` and `xgboost_no_odds` aliases point to the V6 tuned production bundle (`full_live_contract_v6_tuned`, 202 features).
- Railway `/readyz` and startup logs report `bundle_id=ufc-production-20260323-full_live_contract_v6_tuned` for the hosted service.
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
3. Confirm CLOB balance and allowance with `ClobClientWrapper().get_balance_allowance()`.
4. For V2 live trading, confirm the proxy wallet has usable pUSD collateral. If it only has USDC.e, wrap into pUSD before arming.
5. Start with `LIVE_TRADING_MODE=dry-run` and verify `/readyz` returns `200`.
6. Verify the dashboard loads and recent activity is visible.
7. Verify ledgers and `bot.log` are writable in the active `LOGS_DIR` (on Railway this should follow `RAILWAY_VOLUME_MOUNT_PATH` unless `UFC_LOGS_DIR` overrides it).
8. Verify any proxy/geoblock requirements through `/api/geoblock-status`.
9. Change to `LIVE_TRADING_MODE=real`, set `LIVE_TRADING_ARMED=1`, and set `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`.
10. Redeploy and confirm `/readyz` returns `200` with `effective_live_mode=real`.

## CLOB V2 Cutover

Polymarket CLOB V2 is scheduled for April 28, 2026 at about 11:00 UTC, with about one hour of downtime. Open orders are wiped during the cutover. The migrated code uses `py-clob-client-v2==1.0.0`, supports runtime CLOB version detection, and keeps `POLYMARKET_CLOB_URL` overridable for pre-cutover V2 smoke tests.

Pre-cutover deploy posture:

1. Before any deploy, set `LIVE_TRADING_MODE=off` and redeploy/restart.
2. Verify `/readyz` reports `effective_live_mode=off`.
3. Deploy the migrated code while trading remains disabled.
4. Smoke read paths against `https://clob-v2.polymarket.com` by setting `POLYMARKET_CLOB_URL` locally, not in production unless intentionally testing that target.
5. If an operator approves a write-path smoke, place and cancel one tiny limit order only after verifying balance, allowance, market `tick_size`, and `neg_risk`.

Cutover-day checks:

1. Keep trading disabled before the maintenance window.
2. Snapshot open orders, positions, and cash.
3. Cancel open orders before the window if any remain.
4. Watch `https://status.polymarket.com`, Polymarket announcements, `https://clob.polymarket.com/version`, and `https://clob.polymarket.com/ok`.
5. After V2 is live, verify `/healthz`, `/readyz`, `get_balance_allowance()`, `get_cash_balance_details()`, `get_open_orders()`, and `PositionMonitor().compute_pnl()`.
6. Re-arm only after pUSD collateral and allowance are confirmed.

Post-cutover rollback posture:

- Do not use a V1 client rollback as recovery after April 28, 2026.
- Disable trading first with `LIVE_TRADING_MODE=off`.
- Hot-fix forward on the migrated V2 stack, redeploy, smoke test, then re-arm.
