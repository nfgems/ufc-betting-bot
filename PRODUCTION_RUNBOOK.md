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
- `POLYMARKET_FUNDER_ADDRESS`
- `LIVE_TRADING_ARMED=1`
- `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`

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
- `APP_ROLE` (defaults to `web`; use `ufc-refresh-scheduled` only for a one-shot refresh service)
- `UFC_REFRESH_INTERVAL_HOURS` (hosted refresh cadence; capped at 24 hours so completed-card results are pulled before live betting windows)

Removed for CLOB V2:

- `POLYMARKET_BUILDER_API_KEY`
- `POLYMARKET_BUILDER_SECRET`
- `POLYMARKET_BUILDER_PASSPHRASE`

## Current UFC Model

- Current promoted live alias: `xgboost`
- Current promoted production bundle: `models/current_production_model.json`
- The current weights were built on `2026-07-22` as a scheduled same-spec refit of the durability contract originally selected and promoted on `2026-06-11`. The `xgboost` and `logistic` aliases use `full_live_contract_v6_durability_fullfit` (211 features), and `xgboost_no_odds` uses the matching `full_live_contract_v6_durability_fullfit_no_odds` artifact.
- The local manifest reports `bundle_id=ufc-production-20260718-full_live_contract_v6_durability_fullfit`, `built_at=2026-07-22T04:47:57.599918+00:00`, and `snapshot_max_event_date=2026-07-18`.
- Railway `/readyz` is the hosted source of truth and reports the active production bundle loaded from the mounted runtime manifest. As verified on `2026-07-23`, it reports the same bundle, embedded model contracts, and processed snapshot.
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
5. To roll back the current July refit while staying on the durability contract, restore the production model aliases and training-spec artifact from `models/backups/pre_refit_20260711` (the original June 11 durability artifacts). Do not copy that directory's manifest unchanged while retaining the current processed snapshot: its hashes describe the June 6 snapshot. Use it only as source metadata, then run `python scripts/reconcile_production_bundle_manifest.py --source-manifest models/backups/pre_refit_20260711/current_production_model.json --processed-dir data/processed` to write a manifest for the artifacts and processed snapshot actually being served.
6. To roll back past the durability promotion to the May 29 V6 full-fit model, restore `models/backups/pre_new_model_promotion_20260611_durability`; if the processed snapshot must roll back too, restore `data/processed/backup_pre_durability_20260611`, then reconcile the production bundle manifest against that chosen snapshot.
7. The older `models/backups/pre_new_model_promotion_20260529_202737` directory is for rolling back past the May 29 V6 full-fit promotion.

## Line-History Archive Operations

Expired odds and Polymarket line-history CSVs are compressed and copied to the configured private object-storage bucket before their live-volume copies are removed. The operator commands below only list and restore objects; they never delete bucket objects.

List the first 100 objects:

```bash
python -m src.bot line-history-archive list
```

Narrow the listing to a category and month:

```bash
python -m src.bot line-history-archive list --category odds --year 2026 --month 1
```

Use `--json` for machine-readable output. If the result has a `next_cursor`, pass that opaque value back exactly as printed:

```bash
python -m src.bot line-history-archive list --category odds --limit 100 --cursor '<next_cursor>'
```

Restore an exact key returned by the listing:

```bash
python -m src.bot line-history-archive restore 'ufc/line-history/v1/odds/2026/01/odds_20260101_120000.csv.gz'
```

The safe default destination is `DATA_DIR/restored_line_history`, which is persistent on the hosted volume but deliberately outside `DATA_DIR/raw/line_history`. The live bot does not automatically consume restored files. Use `--output-dir` only when a different isolated working directory is intentional. Existing files are never overwritten unless `--force` is supplied; even with `--force`, replacement happens only after gzip, size, and checksum validation succeeds.

For Railway production, first open a shell on the existing `ufc-bot` service so the command uses its private bucket variables and mounted data volume:

```bash
railway ssh --service ufc-bot --environment production
python -m src.bot line-history-archive list --category odds --json
```

Restoring leaves the compressed bucket copy untouched. There is intentionally no archive-delete CLI command.

## First-Live Checklist

1. Confirm the promoted model artifacts exist under `models/`.
2. Confirm `ODDS_API_KEY`, `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_FUNDER_ADDRESS`, and `WEB_DASHBOARD_TOKEN` are present in the deploy environment.
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
