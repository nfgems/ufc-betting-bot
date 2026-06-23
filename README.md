# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot. The repo covers UFC data collection, live-compatible feature engineering, model training and evaluation, walk-forward backtesting, live prediction, and a Flask dashboard.

## Status As Of 2026-06-23

- The active production model spec is `full_live_contract_v6_durability_fullfit` (211 live-compatible features across 20+ families — the V6 contract plus 9 loss-method/durability decomposition features). The repo also includes an offline `full_live_contract_v7` evaluation candidate at 223 features, but it is not the promoted runtime bundle.
- On Railway, the runtime source of truth is the active production bundle manifest bootstrapped under the mounted data volume. The hosted service uses image-bundled model aliases from `/app/models` plus the canonical `data/processed/fights_cleaned.csv` and `data/processed/features.csv` snapshot, with bundle validation at startup. The entrypoint intentionally ignores legacy hosted overrides for `UFC_MODELS_DIR` and `UFC_PRODUCTION_BUNDLE_MANIFEST` so Railway does not accidentally load model artifacts from a stale volume. The canonical snapshot is rolled forward in-place by the hosted UFC refresh loop; the manifest's `snapshot_max_event_date` reflects the promotion-time snapshot, not live coverage.
- The default `python -m src.bot train` flow resolves the active production bundle spec; currently that is `full_live_contract_v6_durability_fullfit` (211 features). Candidate artifacts under `models/candidates/` and `data/processed/candidates/` are offline-only unless explicitly promoted.
- `data/raw/ufc-master.csv` remains a legacy training input for rebuild/training utilities. It is not the hosted inference source of truth.
- Upcoming-card context uses UFC.com as the primary live schedule and fight-card source because UFCStats can lag or omit scheduled fights. UFCStats remains the fallback upcoming-card source and the historical/stat backfill source.
- The repository is UFC-only. The tennis pipeline was removed after internal evaluation showed no marginal value over market odds.
- The live trading loop runs a four-trader race: Single (S, blended model value bets), Conviction (C, high-conviction unblended), Model Tracker (M, flat-bet tracker on model predictions), and Gemini Tracker (G, flat-bet tracker on Gemini picks). Each trader has its own bankroll, ledger, and execution path. All four traders share the 48-hour pre-event bet window governed by `MAX_BET_HOURS_BEFORE_EVENT`. Resting limit bids are pulled 2h before the fight starts (`LIMIT_BID_PRE_EVENT_HOURS`), no new resting limit bids are placed inside that 2h window, marketable orders inside that window must have enough best-ask liquidity to avoid a resting remainder, and no new bets are placed within the final 1h before start (`LIVE_TRADE_START_BUFFER`).
- Line-movement and near-zero market-price signals are surfaced as advisory market alerts only. They do not hard-block fights; trade eligibility still comes from the shared 48-hour betting window plus the normal value, edge, liquidity, and live-trading arming checks.
- Live predictions are incrementally cached to disk and synced to the dashboard, so predictions survive restarts and the dashboard reflects the latest state without a full re-run. The dashboard also reconciles its bet/PnL history against Polymarket activity so historical totals are preserved across restarts.
- The dashboard exposes a unified Gemini reasoning feed at `/reasoning` (backed by `/api/gemini-reasoning`) that combines the LLM operator's gate decisions and the Gemini Tracker's picks into one view, grouped by fight week (with fight-date and logged-date views) and labeled with each fight's current status.
- Each live or dry-run betting cycle writes a structured execution decision audit (`execution_decision_audit.jsonl` plus a `_latest.json` snapshot under the logs dir) recording, per fight, why each of the four traders (S/C/M/G) bet or skipped — bet-window and market-match filters, value/conviction gate reasons, LLM-operator block/pass, tracker decisions, and executor-level skips (liquidity, taker-fee net edge, limit-bid window, duplicate position, insufficient cash, min order size) down to the final placed/dry-run/failed order result. The operator views this on the `/execution-breakdown` dashboard page (backed by `/api/execution-breakdown`).
- The current production bundle was promoted on 2026-06-11 (bundle `ufc-production-20260606-full_live_contract_v6_durability_fullfit`, spec `full_live_contract_v6_durability_fullfit`) as a full-fit refit on corrected 2014–2026 data with a processed snapshot through 2026-06-06. It extends the prior V6 contract with 9 loss-method/durability decomposition features (KO/submission loss rates and a recent-KO-loss flag, in a/b/diff form) for 211 features total, on top of the A/B orientation parity (mirror-augmented training plus symmetric inference, which removed the historical positional bias where the training slot A was the winner far more often than chance), no-vig odds normalization, and invalid-moneyline filtering carried forward from the prior 2026-05-29 refit.
- `WARNING`/`ERROR`/`CRITICAL` log events are mirrored to a durable `alerts.jsonl` sidecar (independent of `bot.log`'s INFO volume) and surfaced through `/api/bot-alerts` in a pinned alerts panel on the Activity page, so they stay visible for a retention window (`ACTIVITY_ALERT_RETENTION_HOURS`, default 72h) instead of scrolling out of the recent-log feed.
- Before live trading, the runtime enforces a bundle-freshness guard: `predict` logs a warning and `live --real` warns outside the betting window, then blocks once an in-window fight would trade, when the promoted model is older than one month or the processed snapshot is missing a known completed UFC card before the active card. If the completed-card schedule fetch fails, the guard reuses the last successful completed-card set for up to an hour; beyond that it degrades to an advisory-only 7-day age check that warns but never blocks, because the age heuristic cannot distinguish a stale snapshot from a long inter-card gap (a Saturday-to-next-Sunday gap is 8 days). Adjacent one-day source-date offsets are treated as covered because UFC.com/Odds API can label late US cards by the UTC rollover date while UFCStats keeps the local card date.
- A BTC 5-minute Polymarket up/down momentum runner ships alongside the UFC pipeline as a separate strategy. It is paper/dry-run by default and stays dormant on the hosted service until profiles are promoted via `BTC5M_LIVE_PROFILES` (each promoted profile keeps its own ledger under `BTC5M_LIVE_LEDGER_DIR`). The loop shares the same `LIVE_TRADING_MODE` switch as the UFC loop — it paper-trades on `dry-run` and trades real money only on `real` with the same two-key arming plus `POLYMARKET_PRIVATE_KEY`. Operator entry points are the `btc5m`, `btc5m-paper`, and `btc5m-opportunity` CLI commands, and live state is surfaced on the `/btc5m` dashboard page (backed by `/api/btc5m/live`).

## Archive Note

On 2026-03-23, leftover scratch artifacts were intentionally moved out of the main repo into the separate private archive repo `nfgems/ufc-betting-bot-worktree-archive-20260323`.

This archive contains handoff notes, HTML captures, temp outputs, and some offline UFC experiment artifacts that were cluttering the main worktree. These files are not part of the promoted production runtime.

If an older offline-only artifact seems to be missing from this repo, check that private archive repo first before assuming it was deleted permanently.

`.env` and other local secret-bearing files were intentionally excluded from that archive and must remain local-only.

## Main Components

- `src/data/`: scraping, fallbacks, odds ingestion, rankings, line tracking, live monitoring, fighter profiles, rankings history, and pre-UFC career scraping. UFCStats scraping goes through a shared HTTP client (`src/data/ufcstats_http.py`) that solves their browser-check challenge
- `src/features/`: UFC feature builders (including experimental features)
- `src/model/`: training specs, training, evaluation, prediction, A/B orientation parity (`src/model/orientation.py`), feature provenance tooling, and model variant management
- `src/strategy/`: backtests, value logic, four-trader race (S/C/M/G), bankroll management, model selection utilities, LLM operator gates, and the per-cycle execution decision audit (`src/strategy/execution_audit.py`)
- `src/polymarket/`: market lookup, CLOB client, execution, positions, ledgers, and the BTC 5-minute momentum runner (`btc_5m.py`) with its forward profile opportunity harness (`btc5m_opportunity.py`) and shadow exit models (`btc5m_exit.py`)
- `src/web/`: Flask dashboard, hosted runtime entrypoint, operator UI, and the durable activity alert store (`src/web/alert_store.py`)
- `models/`: canonical alias models, candidate artifacts, and promotion manifests
- `scripts/`: one-off data collection, odds scraping, and analysis utilities
- `tests/`: regression and runtime coverage

## Prerequisites

- Python 3.11 or newer
- `ODDS_API_KEY` for most live odds workflows
- `POLYMARKET_PRIVATE_KEY` only if you want real-money Polymarket trading
- `BETSAPI_TOKEN` only for BetsAPI-backed MMA workflows

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
| `BETSAPI_TOKEN` | BetsAPI MMA odds workflows | Optional |
| `POLYMARKET_PRIVATE_KEY` | Trading and account access | Required for real-money trading |
| `POLYMARKET_FUNDER_ADDRESS` | Proxy wallet override | Optional; runtime can attempt auto-discovery |
| `POLYMARKET_CLOB_URL` | Polymarket CLOB API base URL | Optional; defaults to `https://clob.polymarket.com` |
| `CLOB_PROXY_URL` | Proxying CLOB traffic | Optional; surfaced by geoblock diagnostics |
| `POLYMARKET_GEOBLOCK_TIMEOUT_SECONDS` | Timeout (seconds) for the Polymarket geoblock check request | Optional; defaults to `4.0`, floored at `0.5` |
| `POLYMARKET_BUILDER_CODE` | Polymarket builder attribution code for order submissions | Optional |
| `POLYMARKET_AUTO_REDEEM` | Auto-claiming resolved winnings | Optional; set to `1` to enable |
| `POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS` | Auto-redeem cooldown window | Optional; defaults to `6` hours |
| `POLYMARKET_AUTO_REDEEM_PENDING_TTL_HOURS` | Pending auto-redeem transaction TTL | Optional; defaults to `24` hours |
| `POLYMARKET_RELAYER_URL` | Polymarket relayer base URL | Optional; defaults to `https://relayer-v2.polymarket.com` |
| `POLYMARKET_RELAYER_API_KEY` / `POLYMARKET_RELAYER_API_KEY_ADDRESS` | Relayer API key auth for redeeming resolved positions | Optional; required by `redeem` and hosted auto-redeem |
| `WEB_DASHBOARD_TOKEN` | Dashboard mutation auth on public binds | Read endpoints remain public. On public binds, hosted startup warns if this is missing in `dry-run` and fails closed in `real` |
| `LIVE_TRADING_MODE` | Hosted trading mode | `off`, `dry-run`, or `real` |
| `BTC5M_LIVE_PROFILES` | Hosted BTC 5m promoted profiles | Optional; blank by default, which keeps the hosted BTC 5m loop dormant. Comma-separate profile names to run them always-on |
| `BTC5M_LIVE_LEDGER_DIR` | Hosted BTC 5m promoted-profile ledgers | Optional; defaults to `data/logs/btc5m_live` |
| `BTC5M_PRICE_SOURCE` | BTC reference price feed for the hosted BTC 5m loop and CLI runner | Optional; `binance` (default) or `coinbase`. Binance data endpoints can be geoblocked from hosted/US egress; set to `coinbase` if the hosted price feed fails (there is no automatic fallback) |
| `LIVE_MODEL` | Hosted model alias or explicit artifact path | Defaults to `xgboost` |
| `UFC_PRODUCTION_BUNDLE_MANIFEST` | Production bundle manifest path | Advanced local override; defaults to `models/current_production_model.json` locally. The Docker/Railway entrypoint sets this to the mounted runtime manifest and ignores legacy hosted overrides |
| `LIVE_TRADING_ARMED` | Real-trading arming switch | Must be `1` for `real` mode |
| `LIVE_TRADING_CONFIRMATION` | Real-trading confirmation string | Must equal `REAL_TRADING_ENABLED` for `real` mode |
| `GEMINI_API_KEY` | Gemini API access for the UFC LLM operator | Optional; only needed when using operator synthesis |
| `GEMINI_OPERATOR_MODEL` | Gemini model override for the operator | Optional; defaults to `gemini-3.1-pro-preview` |
| `GEMINI_OPERATOR_FALLBACK_MODELS` | Comma-separated fallback Gemini models | Optional; defaults to `gemini-3.5-flash,gemini-3-flash-preview,gemini-2.5-pro,gemini-2.5-flash`. Retired Gemini models (e.g. `gemini-3-pro-preview`, `gemini-2.0-flash`) are skipped with a warning even if configured |
| `GEMINI_OPERATOR_TIMEOUT_MS` / `GEMINI_OPERATOR_RESEARCH_TIMEOUT_MS` / `GEMINI_OPERATOR_SYNTHESIS_TIMEOUT_MS` | Gemini operator request timeouts | Optional; defaults are tuned separately for research and synthesis |
| `GEMINI_OPERATOR_PRIMARY_MODEL_RETRIES` / `GEMINI_OPERATOR_FALLBACK_RETRIES_PER_MODEL` | Gemini operator retry counts | Optional; defaults to `5` primary attempts and `2` per fallback model |
| `GEMINI_OPERATOR_RETRY_INITIAL_DELAY_SECONDS` / `GEMINI_OPERATOR_RETRY_MAX_DELAY_SECONDS` / `GEMINI_OPERATOR_RETRY_JITTER_SECONDS` | Gemini operator retry backoff controls | Optional |
| `GEMINI_OPERATOR_OVERLOAD_FAILURE_THRESHOLD` / `GEMINI_OPERATOR_OVERLOAD_COOLDOWN_SECONDS` | Gemini transient-failure circuit breaker | Optional |
| `GEMINI_RESEARCH_CACHE_TTL_SECONDS` | Gemini grounded-research cache TTL | Optional; defaults to `900` seconds |
| `GEMINI_TRACKER_CONFIDENCE_CAP` | Gemini Tracker confidence display and ledger cap | Optional; defaults to `0.85` and is clamped between `0.5` and `1.0`. Gemini Tracker bets use market-neutral probability and store Gemini confidence separately |
| `LLM_OPERATOR_ENABLED` | Enable or disable the UFC LLM operator gate | Optional; defaults to `1` |
| `LLM_OPERATOR_MODE` | Operator behavior mode | Optional; `gate` blocks bets, `advisory` only annotates |
| `LLM_OPERATOR_CACHE_TTL` | Operator decision cache TTL in seconds | Optional; defaults to `0` |
| `LLM_OPERATOR_FAILURE_CACHE_TTL_SECONDS` | Operator failure-cache TTL | Optional; defaults to `1800` seconds |
| `LLM_OPERATOR_POST_EVENT_RETENTION_HOURS` | How long post-event operator cache entries are retained | Optional; defaults to `48` hours |
| `LLM_OPERATOR_LOCK_TIMEOUT_SECONDS` | Operator process-lock acquisition timeout | Optional; defaults to `20` seconds |
| `LLM_OPERATOR_LOCK_STALE_SECONDS` | Operator stale-lock age before takeover | Optional; defaults to `300` seconds |
| `PORT` | Web server port | Optional; defaults to `5050` |
| `WEB_HOST` | Web server bind address | Optional; defaults to `0.0.0.0` for hosted entrypoint |
| `DASHBOARD_EVENT_TIMEZONE` | Dashboard event-time display timezone | Optional; defaults to `America/New_York` |
| `MONITOR_INTERVAL_HOURS` | Background monitor loop interval | Optional; defaults to `6` |
| `BET_INTERVAL_MINUTES` | Hosted betting loop interval | Optional; defaults to `10` |
| `MIN_EDGE` | Edge threshold override for hosted trading | Optional; uses config default |
| `APP_ROLE` | Docker/Railway entrypoint role | Optional; defaults to `web`. `ufc-refresh-scheduled` runs the scheduled UFC refresh command once |
| `MAX_BET_HOURS_BEFORE_EVENT` | Shared pre-event bet window for all traders (S/C/M/G) | Optional; defaults to `48` hours. Bets outside this window are skipped |
| `TRACKER_MIN_HOURS_BEFORE_EVENT` | Deprecated tracker-only entry window | Optional; retained for backward compat. Trackers now follow `MAX_BET_HOURS_BEFORE_EVENT` |
| `POLYMARKET_CHAIN_ID` | Polygon chain ID | Optional; defaults to `137` |
| `RAILWAY_VOLUME_MOUNT_PATH` | Railway persistent storage mount | Optional; used by Railway deployments for data/model/log persistence |
| `UFC_DATA_DIR` | Override data directory path | Optional; defaults to `data/` under project root |
| `UFC_MODELS_DIR` | Models directory path | Advanced local override; defaults to `models/` under project root. The Docker/Railway entrypoint forces `/app/models` and ignores legacy hosted overrides |
| `UFC_LOGS_DIR` | Override logs directory path | Optional; defaults to `data/logs` locally. In hosted runtime, this may resolve directly to `RAILWAY_VOLUME_MOUNT_PATH` when set |
| `UFC_REFRESH_ENABLED` | Enable hosted UFC refresh loop | Optional; `1` runs scheduled UFC refreshes inside the always-on hosted service |
| `UFC_REFRESH_INTERVAL_HOURS` | Hosted UFC refresh cadence | Optional; defaults to `24` hours and is capped at `24` hours for freshness safety |
| `UFC_REFRESH_INITIAL_DELAY_MINUTES` | Delay first hosted UFC refresh after boot | Optional; defaults to `30` minutes |
| `UFC_REFRESH_LIMIT_FIGHTERS` | Debug cap for hosted UFC refresh | Optional; leave blank in production |
| `UFC_REFRESH_NEW_FIGHTER_ALERT_GRACE_DAYS` | Exclude brand-new roster additions from new-fighter coverage floors | Optional; defaults to `7` days |
| `UFC_REFRESH_PROFILE_SUPPLEMENT_*` | Optional new-fighter profile supplement pass during scheduled refresh | Advanced controls: `..._ENABLED`, `..._LIMIT`, and `..._SOURCES` |
| `UFC_REFRESH_MIN_*` | Coverage-drop alert floors for hosted refresh | Optional; see `.env.example` for the full list |
| `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS` / `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT` | Escalation thresholds for cached active-roster rows retained because they disappeared from the latest UFC.com live sync | Optional; defaults to `50` rows and `5.0%`. Normal small live-sync omissions are logged as notes; larger omissions degrade the hosted refresh component |
| `TAPOLOGY_PROXY_URL` | HTTP/HTTPS proxy for Tapology fallback paths | Optional. The Docker/Railway image also enables a headed Chromium/Xvfb browser fallback for Tapology when normal HTTP hits Cloudflare. A proxy is no longer required for the hosted Tapology path, but this remains available as an override and is passed to Chromium when possible |
| `TAPOLOGY_BROWSER_FALLBACK_ENABLED` | Enable hosted headed-browser Tapology recovery | Optional; Docker defaults this to `1`. When enabled and Chromium/Xvfb are present, Tapology search/profile/ranking pages that fail through normal HTTP are retried through a headed browser session and cached for the process. Advanced tuning (optional, safe defaults): `TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS` (`20`), `TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS` (`20`), and `TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS` (`3`) control page/ready timeouts and inter-request pacing; `TAPOLOGY_BROWSER_BINARY` / `TAPOLOGY_CHROMEDRIVER_BINARY` / `TAPOLOGY_XVFB_BINARY` override the Chromium/chromedriver/Xvfb paths (already preset in the Docker image) |
| `BRAVE_SEARCH_API_KEY` | Official Brave Search API token for site-search profile recovery | Optional. When unset, the bot skips Brave consumer-page scraping by default to avoid Railway egress 429s |
| `BRAVE_SEARCH_API_URL` | Override for the Brave Search API endpoint | Optional; defaults to `https://api.search.brave.com/res/v1/web/search`. Advanced/rarely needed |
| `BRAVE_SEARCH_HTML_FALLBACK_ENABLED` | Legacy Brave consumer HTML search fallback | Optional; defaults to `0`. Enable only for local/manual debugging |
| `BRAVE_SEARCH_TIMEOUT_SECONDS` | Brave site-search timeout | Optional; defaults to `12` seconds |
| `FIGHTDX_REQUEST_TIMEOUT_SECONDS` | Per-request timeout for FightDX fighter-profile fetches | Optional; defaults to `8` seconds (floored at `1.0`) |
| `FIGHTDX_FAILURE_COOLDOWN_SECONDS` | Cooldown after a FightDX fetch failure before that source is retried (prevents repeated slow-timeout amplification) | Optional; defaults to `180` seconds (floored at `0.0`) |
| `BETSAPI_REQUEST_MIN_INTERVAL_SECONDS` | BetsAPI rate-limit floor | Optional |
| `BETSAPI_429_RETRY_MIN_SECONDS` | BetsAPI 429-retry backoff floor | Optional |
| `METHOD_ODDS_BFO_REQUEST_TIMEOUT_SECONDS` | Per-request timeout for Best Fight Odds method-odds fetches | Optional; defaults to `12` seconds (floored at `1.0`) |
| `METHOD_ODDS_BFO_MAX_RETRIES` | Retry count for failed BFO method-odds fetches | Optional; defaults to `1` (floored at `0`) |
| `METHOD_ODDS_BFO_RETRY_BACKOFF_SECONDS` | Backoff between BFO method-odds retries | Optional; defaults to `2` seconds (floored at `0.0`) |
| `METHOD_ODDS_BFO_FAILURE_BUDGET` | Consecutive-failure budget per snapshot before the BFO path short-circuits | Optional; defaults to `2` (floored at `1`) |
| `ACTIVITY_ALERT_RETENTION_HOURS` | Durable Activity-dashboard alert retention window | Optional; defaults to `72` hours (clamped to a 1-hour minimum). `WARNING`/`ERROR`/`CRITICAL` logs are mirrored to a dedicated `alerts.jsonl` so they stay visible in the Activity view beyond the recent-log window |

Polymarket client note: the pinned `py_clob_client` contract used here must expose `derive_api_key()` and `create_api_key()`. The legacy `create_or_derive_api_creds()` helper is no longer the runtime path.

## CLI Overview

All commands run from the project root with `python -m src.bot ...`.

### UFC workflow

```bash
# Refresh raw UFC data
python -m src.bot scrape

# Train using the default CLI training spec (currently the promoted production spec)
python -m src.bot train

# Train a specific contract explicitly
python -m src.bot train --spec full_live_contract_v6_tuned

# Keep alternate artifacts separate instead of overwriting canonical paths
python -m src.bot train --spec full_live_contract_v6_tuned --output-subdir candidates/v6_eval

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
- `predict` warns, and `live --real` is blocked, when the runtime bundle is stale — i.e. the promoted model is older than one month (`MODEL_RETRAIN_MONTHS`) or the processed snapshot is older than 7 days (`LIVE_PROCESSED_REFRESH_MAX_AGE_DAYS`). Refresh data and refit before live trading if you hit this guard.
- `backtest` defaults to `--execution-mode realistic` (models realistic fills and slippage); `walkforward` still defaults to `legacy`. Pass `--execution-mode` to override either.
- `predict` and `live` load the canonical alias models such as `models/xgboost_model.pkl` by default. Override with the `--model` CLI flag or the `LIVE_MODEL` env var (alias name or explicit artifact path).
- The promoted alias targets are recorded in [models/current_production_model.json](models/current_production_model.json).

### BTC 5m Polymarket runner

The BTC 5-minute up/down runner is a separate strategy from the UFC pipeline. It is dry-run by default; `--real` requires the same Polymarket real-money arming as UFC live trading.

```bash
# Single-market momentum runner (dry-run by default)
python -m src.bot btc5m --once
python -m src.bot btc5m --profile conservative --poll-seconds 10
python -m src.bot btc5m --real   # real money; requires Polymarket arming env vars

# Compare strategy profiles side by side in paper mode
python -m src.bot btc5m-paper --profiles late_capture,conservative --once
python -m src.bot btc5m-paper --settle-only

# Forward profile opportunity harness (writes JSON + Markdown report)
python -m src.bot btc5m-opportunity --profiles all --target-markets 500
```

Notes:

- `btc5m --real` is blocked unless the Polymarket real-money arming env vars are set (same arming model as `live --real`). `btc5m-paper` and `btc5m-opportunity` are always simulated.
- Profiles are `conservative` (default), `balanced`, `late`, `confidence`, `cheap_side`, `late_capture`, and `aggressive`. `btc5m-opportunity --profiles all` runs every profile.
- The hosted always-on version of this loop is configured separately — see the Deployment section and `BTC5M_LIVE_PROFILES`.

## Training Specs And Model State

The repo uses a spec-driven training system in [src/model/training_spec.py](src/model/training_spec.py). Common named specs:

| Spec | Features | Notes |
|------|----------|-------|
| `full_live_contract_v2` | 132 | Legacy default |
| `full_live_contract_v5_fullfit` | 126 | Prior promoted production spec |
| `full_live_contract_v6` | 202 | Base V6 contract with expanded feature set |
| `full_live_contract_v6_tuned` | 202 | Optuna-tuned V6 contract; prior promoted spec (2026-03-23), now superseded by `_fullfit` |
| `full_live_contract_v6_fullfit` | 202 | Prior promoted production spec (full-fit refit of the tuned V6 winner; the 2026-05-29 refit added A/B orientation parity and refreshed data); superseded 2026-06-11 by `_durability_fullfit` |
| `full_live_contract_v6_durability_fullfit` | 211 | Current promoted production spec: the V6 full-fit contract plus 9 loss-method/durability decomposition features (KO/submission loss rates and a recent-KO-loss flag); promoted 2026-06-11 |
| `full_live_contract_v7` | 223 | Offline evaluation candidate: V6 plus amateur-career summary features |

Legacy named specs such as `full_live_contract_v1`, `full_live_contract_v3`, `full_live_contract_v4`, `full_live_contract_v4_138`, and `full_live_contract_v4_144` are still resolvable through `resolve_named_training_spec()`, but they are not part of the current production line.

Current promoted production artifact: bundle `ufc-production-20260606-full_live_contract_v6_durability_fullfit` (spec `full_live_contract_v6_durability_fullfit`, 211 features), promoted 2026-06-11 from corrected 2014–2026 data with a processed snapshot through 2026-06-06. Canonical live aliases: `xgboost`, `xgboost_no_odds`, and `logistic` (the `xgboost_no_odds` variant uses the matching `full_live_contract_v6_durability_fullfit_no_odds` spec, which drops the odds features rather than the durability features).

**A/B orientation parity:** training applies automatic A/B mirror augmentation — each observed fight is also added with the two fighters' sides swapped — together with orientation-aware cross-validation, and live prediction symmetrizes by averaging the forward and A/B-swapped predictions. This keeps live inference (alphabetical fighter ordering) consistent with the training distribution and removes the historical positional bias where the training slot A was the winner far more often than chance. Implied-odds probabilities are also no-vig normalized, and invalid moneyline rows are dropped before training (including duplicated heavy-favorite rows where both fighters share the same low price; legitimate equal pick'em prices are retained). The current promoted spec (`full_live_contract_v6_durability_fullfit`, 211 features) layers a loss-method durability feature family on top of this A/B-parity V6 contract — a/b/diff variants of `loss_ko_rate`, `loss_sub_rate`, and `recent_ko_loss`, with NaN-honest denominators for fighters lacking the relevant loss history. See [src/model/orientation.py](src/model/orientation.py).

If you are reproducing the currently promoted production line, use the manifest and spec files under [models/](models/). The production manifest is the source of truth for the active aliases and processed snapshot metadata. For how the current durability production model was selected and promoted, see [docs/DURABILITY_PROMOTION_RUNBOOK.md](docs/DURABILITY_PROMOTION_RUNBOOK.md); the supporting experiment results and model-improvement analysis are in [docs/EXPERIMENT_RESULTS_2026-06-10.md](docs/EXPERIMENT_RESULTS_2026-06-10.md) and [docs/MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md](docs/MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md).

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
- `/api/open-bets-enriched`, `/api/profile-bets` — enriched open-bet and per-profile bet views
- `/api/pnl-history` — P&L over time
- `/api/bot-activity`, `/api/significant-actions` — bot activity and notable actions
- `/api/bot-alerts` — durable `WARNING`/`ERROR`/`CRITICAL` alerts for the retention window (powers the Activity page's pinned alerts panel)
- `/api/trader-race`, `/api/trader-breakdown` — trader comparison metrics
- `/api/injury-alerts` — advisory market alerts for unusual line movement or near-zero prices
- `/api/filter-funnel` — prediction filter diagnostics
- `/api/geoblock-status` — geo-restriction diagnostics
- `/api/refresh-prices` (POST), `/api/settle-auto` (POST), `/api/redeem-auto` (POST), `/api/reconcile-limit-orders` (POST) — operational actions
- `/api/runtime-status` — hosted runtime component status
- `/api/closed-positions` — resolved Polymarket positions
- `/api/bot-activity-snapshot` — activity snapshot
- `/ufc`, `/predictions`, `/activity`, `/bet-history`, `/btc5m`, `/reasoning`, `/execution-breakdown` — dashboard pages (`/btc5m` is the read-only BTC 5m up/down monitor)
- `/api/tracker-decisions` — tracker trader decision log
- `/operator`, `/api/operator-decisions` — LLM operator interface and decisions
- `/api/gemini-reasoning` — unified Gemini reasoning feed merging LLM operator gate decisions and Gemini Tracker picks (supports `?source=all|operator|tracker` and `?limit=`); powers the `/reasoning` page
- `/api/execution-breakdown` — structured per-cycle, per-fight, per-trader (S/C/M/G) execution decision audit (returns the latest cycle by default; supports `?history=1`, `?cycle_id=`, and `?limit=`); powers the `/execution-breakdown` page
- `/api/btc5m/live` — read-only BTC 5m monitor snapshot (per-profile ledger stats, latest and recent signals, ledger freshness, and alerts); powers the `/btc5m` page

See [src/web/app.py](src/web/app.py) for the full route list.

## Deployment

Docker and Railway use the hosted entrypoint:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env -p 5050:5050 ufc-betting-bot
```

The Docker/Railway entrypoint defaults to `APP_ROLE=web`, starts `python -m src.web.serve`, and bootstraps the runtime production-bundle manifest into the mounted data volume before startup. For hosted web services, leave `UFC_MODELS_DIR` and `UFC_PRODUCTION_BUNDLE_MANIFEST` unset unless you are intentionally changing the entrypoint behavior in code.

Safe hosted default:

```dotenv
LIVE_TRADING_MODE=off
```

Paper-trading hosted deploy:

```dotenv
LIVE_TRADING_MODE=dry-run
WEB_DASHBOARD_TOKEN=change_me
```

BTC 5m always-on paper deploy, only after choosing promoted profiles:

```dotenv
LIVE_TRADING_MODE=dry-run
BTC5M_LIVE_PROFILES=late_capture
WEB_DASHBOARD_TOKEN=change_me
```

Leave `BTC5M_LIVE_PROFILES` blank to keep the hosted BTC 5m loop dormant. Multiple promoted profiles can run concurrently with separate ledgers under `BTC5M_LIVE_LEDGER_DIR`.

The BTC 5m loop shares the same `LIVE_TRADING_MODE` switch as the UFC loop: it paper-trades when `LIVE_TRADING_MODE=dry-run` and trades real money only when `LIVE_TRADING_MODE=real` (with full arming). With `LIVE_TRADING_MODE=off` the loop stays dormant even if `BTC5M_LIVE_PROFILES` is set — there is no BTC-specific mode env, so a promoted profile needs both `BTC5M_LIVE_PROFILES` and a non-`off` `LIVE_TRADING_MODE` to come live. Profile names are validated against the promoted-profile set; an unknown or misspelled entry fails closed and disables the entire BTC loop (surfaced on the `btc5m_loop` runtime component). Use `BTC5M_LIVE_PROFILES=all` to run every configured profile rather than guessing names.

Real-money hosted deploy:

```dotenv
LIVE_TRADING_MODE=real
LIVE_TRADING_ARMED=1
LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED
WEB_DASHBOARD_TOKEN=change_me
```

For BTC 5m real-money hosting, also set `BTC5M_LIVE_PROFILES` and `POLYMARKET_PRIVATE_KEY`. Missing any real-money arming env blocks the BTC loop from starting.

On public binds, `WEB_DASHBOARD_TOKEN` is recommended in `dry-run` so mutation routes stay protected. In `real` mode on a public bind, hosted startup requires it.

For production operations and rollback details, see [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md).

### Railway UFC Refresh

The repo includes a full UFC refresh command:

```bash
python -m src.bot ufc-refresh-scheduled
```

This refreshes the official active roster, backfills active-roster UFCStats data, rebuilds processed UFC artifacts, and writes a profile audit snapshot. Live upcoming-event discovery uses UFC.com first and falls back to UFCStats when UFC.com has no usable event rows.

For Railway, the important constraint is that persistent volumes are attached per service. If your always-on web service owns the UFC data volume, a second cron service will not update that same on-disk dataset. The practical Railway setup is to enable the hosted UFC refresh loop inside the existing web service so it runs against the same mounted volume.

Recommended hosted settings:

```dotenv
UFC_REFRESH_ENABLED=1
UFC_REFRESH_INTERVAL_HOURS=24
UFC_REFRESH_INITIAL_DELAY_MINUTES=30
```

Notes:

- Hosted refresh is capped at `24` hours so completed-card results are incorporated before the next betting window.
- Leave `UFC_REFRESH_LIMIT_FIGHTERS` blank in production. It exists only for smoke testing.
- The scheduled refresh also supports an optional profile-supplement pass for new active fighters. Use `UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED=0` to disable it, `UFC_REFRESH_PROFILE_SUPPLEMENT_LIMIT` to smoke-test it, and `UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES` to restrict sources. The Wikipedia source retries HTTP 429 responses up to four total attempts, honoring `Retry-After` when present and otherwise backing off from 10 seconds.
- The hosted refresh loop writes through the same guarded atomic CSV paths as the manual refresh command, so empty scrapes do not replace good artifacts with blank files.
- Real-money live trading checks processed-snapshot freshness against known completed UFC card dates when live UFC.com context is available. This avoids blocking late US cards solely because UTC has rolled into the next day, tolerates the common one-day UFC.com/Odds API versus UFCStats card-date offset, and avoids blocking valid bets just because two UFC cards are 8+ calendar days apart. Outside the configured betting window this is a warning; once a matching fight is in-window, the bot still fails closed when the snapshot is missing a completed intervening card.
- Refresh failures are reported immediately in the hosted runtime status as a degraded `ufc_refresh_loop` component.
- While a scheduled refresh is rebuilding the processed snapshot, if the bundle-freshness guard blocks a live cycle mid-refresh the betting loop reports a `degraded` "live trading paused while scheduled UFC refresh rebuilds the processed snapshot" status instead of an error, and resumes automatically once the refresh completes.
- Coverage-drop alerts are optional. Set one or more `UFC_REFRESH_MIN_*` env vars if you want the hosted refresh loop to mark itself degraded when audited coverage falls below your chosen floor.
- The official UFC roster live sync can retain previously tracked cached rows that disappear from a later UFC.com scrape. Small retained-row counts are reported as notes; unusually large counts degrade the refresh loop when they exceed `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS` or `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT`.

## Disclaimer

This project is for research and education. Sports betting involves real financial risk. Never bet more than you can afford to lose, and do not treat historical model performance as a guarantee of future results.
