# UFC Betting Bot

Machine-learning UFC fight prediction and Polymarket execution bot. The repo covers UFC data collection, live-compatible feature engineering, model training and evaluation, walk-forward backtesting, live prediction, and a Flask dashboard.

## Status As Of 2026-07-23

- The active production model spec is `full_live_contract_v6_durability_fullfit` (211 live-compatible features across 20+ families — the V6 contract plus 9 loss-method/durability decomposition features). The repo also includes an offline `full_live_contract_v7` evaluation candidate at 223 features, but it is not the promoted runtime bundle.
- On Railway, the runtime source of truth is the reconciled production bundle manifest under the mounted data volume. As verified from production on 2026-07-23, the hosted service is ready and is running `full_live_contract_v6_durability_fullfit` from runtime bundle `ufc-production-20260718-full_live_contract_v6_durability_fullfit`, built 2026-07-22 with its processed snapshot current through 2026-07-18. The repository manifest in [models/current_production_model.json](models/current_production_model.json) now records that same scheduled refit. Hosted refresh reconciliation preserves the embedded model contract while refreshing persistent processed artifacts and manifest metadata. The entrypoint intentionally ignores legacy hosted overrides for `UFC_MODELS_DIR` and `UFC_PRODUCTION_BUNDLE_MANIFEST` so Railway does not accidentally load model artifacts from a stale volume.
- The default `python -m src.bot train` flow resolves the active production bundle spec; currently that is `full_live_contract_v6_durability_fullfit` (211 features). Candidate artifacts under `models/candidates/` and `data/processed/candidates/` are offline-only unless explicitly promoted.
- `data/raw/ufc-master.csv` remains a legacy training input for rebuild/training utilities. It is not the hosted inference source of truth.
- Upcoming-card context uses UFC.com as the primary live schedule and fight-card source because UFCStats can lag or omit scheduled fights. UFCStats remains the fallback upcoming-card source and the historical/stat backfill source. If UFC.com marks a late US card completed while bookmaker fights on that same card still have future UTC commence times, the live collector recovers the completed card only when its fighter pairs match those active fights; this preserves the official local card date without weakening the missing-prior-card freshness guard.
- Official UFC active-roster requests retry timeouts, connection failures, and transient HTTP `500`/`502`/`503`/`504` responses up to three total attempts. ESPN fighter search/profile fallbacks likewise retry transient connection/decoding failures and HTTP `429`/`500`/`502`/`503`/`504` responses up to three total attempts.
- Live method-of-victory odds use Best Fight Odds because The Odds API's MMA method market keys return `422` and are intentionally not queried. A card with no published method props is reported as `unavailable`, not as a collection failure; only transient BFO failures retry. A live fallback snapshot must match the requested fight and event context, contain per-fight coverage, and be no older than `METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS` (default 48 hours); malformed timestamps fail stale. Runtime status distinguishes current, partial, unavailable, and fallback coverage, while missing props escalate to a warning only inside `METHOD_ODDS_EXPECTED_WINDOW_HOURS` (default 48 hours).
- Sherdog fallback collection recovers automatically from transient Cloudflare challenges: direct requests enter a 30-minute cooldown instead of remaining disabled for the process lifetime, then probe again. While the cooldown is active, profile discovery and immutable pre-UFC fight history can fall back to the newest Wayback Machine snapshot. Direct Sherdog access and the degraded-mode fallback were both verified from Railway production on 2026-07-08.
- The sports prediction pipeline is UFC-only; the tennis pipeline was removed after internal evaluation showed no marginal value over market odds. A separate crypto 5-minute Polymarket strategy also ships in this repository.
- The live trading loop runs a three-trader race: Single (S, blended model value bets), Conviction (C, high-conviction unblended), and Model Tracker (M, flat-bet tracker on model predictions). Each trader has its own bankroll, ledger, and execution path. All three traders share the 48-hour pre-event bet window governed by `MAX_BET_HOURS_BEFORE_EVENT`. Resting limit bids are pulled 2h before the fight starts (`LIMIT_BID_PRE_EVENT_HOURS`), no new resting limit bids are placed inside that 2h window, marketable orders inside that window must have enough best-ask liquidity to avoid a resting remainder, and no new bets are placed within the final 1h before start (`LIVE_TRADE_START_BUFFER`).
- UFC prediction and bet selection are fully model-driven; there is no LLM pass/block gate between an eligible S/C candidate and normal execution checks.
- The retired G ledger remains registered only for backward compatibility: settlement, exposure reconciliation, dashboard history, and startup cleanup can still see old G positions, but no code path creates a new G pick or bet. Confirmed-unfilled legacy G resting orders are cancelled on a best-effort basis; cleanup degradation is reported but cannot defer S/C execution.
- S and C have strict execution-policy priority over the test trackers. M is deferred until the final full-card S/C pass, lower-priority resting remainders are reconciled/cancelled before S/C sizing, and positions or orders attributable only to M or legacy G do not trigger S/C duplicate-wallet gates. Unknown, manual, or S/C-owned exposure remains fail-closed.
- Real execution fails closed when a precise event time is missing or unparseable, or when the bot cannot verify open CLOB orders, live wallet positions, or spendable wallet cash. In those states no new order is placed until authoritative exchange, balance, and timing data are available.
- Line-movement and near-zero market-price signals are surfaced as advisory market alerts only. They do not hard-block fights; trade eligibility still comes from the shared 48-hour betting window plus the normal value, edge, liquidity, and live-trading arming checks.
- Live predictions are incrementally cached to disk and synced to the dashboard, so predictions survive restarts and the dashboard reflects the latest state without a full re-run. Cache reuse is invalidated when material model, odds, method-odds, line-feature, event-context, or runtime inputs change. A per-fight data-quality gate keeps diagnostic predictions visible but marks them `trade_blocked` and withholds them from both paper and real execution when provenance is unsafe, UFCStats history is unavailable, lower-fidelity fallback data is disallowed, or an experienced fighter is missing too many critical features. Verified newcomers may retain honest native missing values. Blocked rows are retried after `LIVE_DATA_QUALITY_RETRY_SECONDS` instead of being reused for the full cache lifetime. The dashboard also reconciles its bet/PnL history against Polymarket activity so historical totals are preserved across restarts.
- The dashboard's Open Bets section intentionally shows every live wallet position, regardless of sport classification. Fighter-winner markets retain the fighter/opponent layout; Yes/No and Over/Under-style positions use the actual Polymarket question as the card title so manual or otherwise untracked positions do not appear as `Unknown` or `Yes vs No`.
- Each live or dry-run betting cycle writes a structured execution decision audit (`execution_decision_audit.jsonl` plus a `_latest.json` snapshot under the logs dir) recording, per fight, why each of the three model-driven traders (S/C/M) bet or skipped — bet-window and market-match filters, value/conviction gate reasons, tracker decisions, and executor-level skips (liquidity, taker-fee net edge, limit-bid window, duplicate position, insufficient cash, min order size) down to the final placed/dry-run/failed order result. The dashboard exposes this on `/execution-breakdown` (backed by `/api/execution-breakdown`).
- The current production weights were built on 2026-07-22 as a scheduled same-spec refit of the durability model selected and promoted on 2026-06-11. The refit used refreshed fight data through 2026-06-27 and ships with a processed snapshot through 2026-07-18; it changed neither the feature contract nor its hyperparameters. The 211-feature contract extends V6 with 9 loss-method/durability decomposition features (KO/submission loss rates and a recent-KO-loss flag, in a/b/diff form), on top of A/B orientation parity (mirror-augmented training plus symmetric inference), no-vig odds normalization, and invalid-moneyline filtering.
- `WARNING`/`ERROR`/`CRITICAL` log events are mirrored to a durable `alerts.jsonl` sidecar independent of `bot.log`'s INFO volume and surfaced through `/api/bot-alerts` in the Activity page's pinned alerts panel. Repeated observations are coalesced into incidents. Lifecycle-managed incidents remain active until their producer writes an explicit recovery event; recovered and unmanaged incidents follow `ACTIVITY_ALERT_RETENTION_HOURS` (default 72h).
- Persistent runtime growth is bounded: `bot.log` rotates, append-only audit histories are tail-compacted, rankings/method/card snapshots are pruned on configurable schedules, and expired line-history CSVs are compressed and copied to a private S3-compatible archive before their live-volume copies are removed. Railway production forces archive-required behavior; if the bucket is not configured or an upload fails, expired line-history files are preserved and an operational alert is raised instead of deleting data.
- Before live trading, the runtime enforces a bundle-freshness guard: `predict` logs a warning and `live --real` warns outside the betting window, then blocks once an in-window fight would trade, when the promoted model is older than one month or the processed snapshot is missing a known completed UFC card before the active card. If the completed-card schedule fetch fails, the guard reuses the last successful completed-card set for up to an hour; beyond that it degrades to an advisory-only 7-day age check that warns but never blocks, because the age heuristic cannot distinguish a stale snapshot from a long inter-card gap (a Saturday-to-next-Sunday gap is 8 days). Adjacent one-day source-date offsets are treated as covered, and an active late-US card recovered by fighter identity keeps its official local date even after UFC.com moves it into the completed list.
- A crypto 5-minute Polymarket up/down momentum runner ships alongside the UFC pipeline as a separate strategy. It defaults to BTC and also supports ETH and SOL via per-asset profiles (defined by `BTC5M_ALT_5M_ASSETS` in [src/config.py](src/config.py)). As verified from Railway production on 2026-07-23, `BTC5M_LIVE_PROFILES` is blank and the hosted crypto loop reports itself dormant; the persisted crypto emergency stop also remains active. The runner is paper/dry-run by default and stays dormant on the hosted service until profiles are deliberately configured (each configured profile keeps its own ledger under `BTC5M_LIVE_LEDGER_DIR`). The loop shares the same `LIVE_TRADING_MODE` switch as the UFC loop — it paper-trades on `dry-run` and trades real money only on `real` with the same two-key arming plus `POLYMARKET_PRIVATE_KEY`. Operator entry points are the `btc5m`, `btc5m-paper`, and `btc5m-opportunity` CLI commands; the dedicated `/btc5m` dashboard page and its monitor API were removed in 2026-07, so hosted loop state is surfaced only through `/api/runtime-status` components.

## Archive Note

On 2026-03-23, leftover scratch artifacts were intentionally moved out of the main repo into the separate private archive repo `nfgems/ufc-betting-bot-worktree-archive-20260323`.

This archive contains handoff notes, HTML captures, temp outputs, and some offline UFC experiment artifacts that were cluttering the main worktree. These files are not part of the promoted production runtime.

If an older offline-only artifact seems to be missing from this repo, check that private archive repo first before assuming it was deleted permanently.

`.env` and other local secret-bearing files were intentionally excluded from that archive and must remain local-only.

## Main Components

- `src/data/`: scraping, fallbacks, odds ingestion, rankings, line tracking and archive support, live monitoring, fighter profiles, rankings history, and pre-UFC career scraping. UFCStats scraping goes through a shared HTTP client (`src/data/ufcstats_http.py`) that solves their browser-check challenge
- `src/features/`: UFC feature builders (including experimental features)
- `src/model/`: training specs, training, evaluation, prediction, A/B orientation parity (`src/model/orientation.py`), feature provenance tooling, and model variant management
- `src/strategy/`: backtests, value logic, three-trader race (S/C/M), bankroll management, model selection utilities, and the per-cycle execution decision audit (`src/strategy/execution_audit.py`)
- `src/polymarket/`: market lookup, CLOB client, execution, positions, ledgers, and the crypto 5-minute up/down momentum runner (`btc_5m.py`, BTC by default with ETH/SOL profiles) with its forward profile opportunity harness (`btc5m_opportunity.py`) and shadow exit models (`btc5m_exit.py`)
- `src/web/`: Flask dashboard, hosted runtime entrypoint, and the durable activity alert store (`src/web/alert_store.py`)
- `src/storage_retention.py`: bot-log rotation, append-only history compaction, and safe snapshot-retention helpers for persistent hosted storage
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

Common operator-facing variables are listed below. See `.env.example` for deployable defaults and [src/config.py](src/config.py) for the complete advanced configuration surface.

| Variable | Used for | Notes |
|---|---|---|
| `ODDS_API_KEY` | UFC live odds, backfills, prediction, and live workflows | Required for most non-offline UFC commands |
| `BETSAPI_TOKEN` | BetsAPI MMA odds workflows | Optional |
| `POLYMARKET_PRIVATE_KEY` | Trading and account access | Required for real-money trading |
| `POLYMARKET_FUNDER_ADDRESS` | Proxy wallet override and hosted reconciliation wallet | The client can attempt auto-discovery, but the current Docker/Railway real-mode startup reconciliation requires this value explicitly |
| `POLYMARKET_CLOB_URL` | Polymarket CLOB API base URL | Optional; defaults to `https://clob.polymarket.com` |
| `CLOB_PROXY_URL` | Proxying CLOB traffic | Optional; surfaced by geoblock diagnostics |
| `POLYMARKET_GEOBLOCK_TIMEOUT_SECONDS` | Timeout (seconds) for the Polymarket geoblock check request | Optional; defaults to `4.0`, floored at `0.5` |
| `POLYMARKET_CLOB_*_TIMEOUT_SECONDS` | Explicit CLOB connect/read/write/pool phase timeouts | Optional; defaults to `3`/`10`/`5`/`2` seconds. Market-info, balance, and open-orders safety reads temporarily apply these values under the shared transport lock |
| `POLYMARKET_MARKET_INFO_MAX_ATTEMPTS` / `POLYMARKET_MARKET_INFO_RETRY_BACKOFF_SECONDS` | Bounded canonical CLOB market-metadata retry policy | Optional; defaults to `2` attempts and `0.5` seconds of exponential backoff |
| `POLYMARKET_MARKET_INFO_TOTAL_BUDGET_SECONDS` | Retry-admission deadline for canonical CLOB market metadata | Optional; defaults to `25` seconds. Transport-lock admission and new retries stop at the deadline; HTTP phase timeouts remain independently bounded |
| `POLYMARKET_BALANCE_MAX_ATTEMPTS` / `POLYMARKET_BALANCE_RETRY_BACKOFF_SECONDS` | Bounded authenticated cash-balance retry policy | Optional; defaults to `3` attempts and `0.5` seconds of exponential backoff; backoff must be finite and nonnegative |
| `POLYMARKET_BALANCE_TOTAL_BUDGET_SECONDS` | Retry-admission deadline for an authenticated cash-balance safety read | Optional; defaults to `25` seconds. After lazy client/credential initialization and request-parameter construction, transport resolution, lock admission, and new retries are admitted only within this soft deadline; HTTP phase timeouts remain independently bounded, so this is not a hard cancellation deadline |
| `POLYMARKET_OPEN_ORDERS_TOTAL_BUDGET_SECONDS` | Retry-admission deadline for a complete open-orders safety read | Optional; defaults to `25` seconds. After client initialization, transport-lock admission and new retries stop at the deadline; HTTP phase timeouts remain independently bounded, so this is not a hard cancellation deadline |
| `POLYMARKET_OPEN_ORDERS_MAX_ATTEMPTS` / `POLYMARKET_OPEN_ORDERS_RETRY_BACKOFF_SECONDS` | Bounded open-orders retry policy | Optional; defaults to `2` attempts and `0.5` seconds of exponential backoff |
| `POLYMARKET_BUILDER_CODE` | Polymarket builder attribution code for order submissions | Optional |
| `POLYMARKET_AUTO_REDEEM` | Auto-claiming resolved winnings | Optional; set to `1` to enable |
| `POLYMARKET_AUTO_REDEEM_COOLDOWN_HOURS` | Auto-redeem cooldown window | Optional; defaults to `6` hours |
| `POLYMARKET_AUTO_REDEEM_PENDING_TTL_HOURS` | Pending auto-redeem transaction TTL | Optional; defaults to `24` hours |
| `POLYMARKET_RELAYER_URL` | Polymarket relayer base URL | Optional; defaults to `https://relayer-v2.polymarket.com` |
| `POLYMARKET_RELAYER_API_KEY` / `POLYMARKET_RELAYER_API_KEY_ADDRESS` | Relayer API key auth for redeeming resolved positions | Optional; required by `redeem` and hosted auto-redeem |
| `WEB_DASHBOARD_TOKEN` | Dashboard auth on public binds | Mutation routes require it. Read routes remain reachable, but sensitive execution fields may be redacted without a valid token. Hosted startup warns if this is missing in `dry-run` and fails closed in `real` |
| `LIVE_TRADING_MODE` | Hosted trading mode | `off`, `dry-run`, or `real` |
| `BTC5M_LIVE_PROFILES` | Hosted BTC 5m configured profiles | Optional; blank by default, which keeps the hosted BTC 5m loop dormant. Comma-separate profile names to run them always-on |
| `BTC5M_LIVE_LEDGER_DIR` | Hosted BTC 5m per-profile ledgers | Optional; defaults to `data/logs/btc5m_live` |
| `BTC5M_REAL_TRADING_WINDOW_ENABLED` / `BTC5M_REAL_TRADING_WINDOW_TIMEZONE` / `BTC5M_REAL_TRADING_START_HOUR` / `BTC5M_REAL_TRADING_END_HOUR` | Real crypto 5m trading schedule | Optional; defaults to enabled, `America/New_York`, Monday `09:00` through Friday `17:00`. These only gate real crypto 5m CLOB submissions; dry-run, paper, and opportunity runs are unaffected |
| `BTC5M_PRICE_SOURCE` | Crypto reference price feed for the hosted 5m loop and CLI runner (BTC by default) | Optional; one of `binance` (default), `coinbase`, or `hyperliquid`. Binance data endpoints can be geoblocked from hosted/US egress; set to `coinbase` or `hyperliquid` if the hosted feed fails (this env var itself has no automatic fallback). Alt-asset profiles define their own price source and fallback chain |
| `LIVE_MODEL` | Hosted model alias or explicit artifact path | Defaults to `xgboost` |
| `UFC_PRODUCTION_BUNDLE_MANIFEST` | Production bundle manifest path | Advanced local override; defaults to `models/current_production_model.json` locally. The Docker/Railway entrypoint sets this to the mounted runtime manifest and ignores legacy hosted overrides |
| `LIVE_TRADING_ARMED` | Real-trading arming switch | Must be `1` for `real` mode |
| `LIVE_TRADING_CONFIRMATION` | Real-trading confirmation string | Must equal `REAL_TRADING_ENABLED` for `real` mode |
| `LIVE_DATA_QUALITY_BLOCK_FALLBACK` | Block lower-fidelity fighter fallback rows from execution | Optional; defaults to `1`. Predictions remain visible for diagnosis, but affected rows do not reach any paper or real trader |
| `LIVE_DATA_QUALITY_MAX_MISSING_CRITICAL` | Maximum missing critical UFC feature fields for an experienced fighter | Optional; defaults to `4` |
| `LIVE_DATA_QUALITY_RETRY_SECONDS` | Retry cadence for data-quality-blocked predictions | Optional; defaults to `3600` seconds and is floored at `60` |
| `PORT` | Web server port | Optional; defaults to `5050` |
| `WEB_HOST` | Web server bind address | Optional; defaults to `0.0.0.0` for hosted entrypoint |
| `DASHBOARD_EVENT_TIMEZONE` | Dashboard event-time display timezone | Optional; defaults to `America/New_York` |
| `MONITOR_INTERVAL_HOURS` | Background monitor loop interval | Optional; defaults to `6` |
| `BET_INTERVAL_MINUTES` | Hosted betting loop interval | Optional; defaults to `10` |
| `MIN_EDGE` | Edge threshold override for hosted trading | Optional; uses config default |
| `APP_ROLE` | Docker/Railway entrypoint role | Optional; defaults to `web`. `ufc-refresh-scheduled` runs the scheduled UFC refresh command once |
| `MAX_BET_HOURS_BEFORE_EVENT` | Shared pre-event bet window for all traders (S/C/M) | Optional; defaults to `48` hours. Bets outside this window are skipped |
| `TRACKER_MIN_HOURS_BEFORE_EVENT` | Deprecated tracker-only entry window | Optional; retained for backward compat. Trackers now follow `MAX_BET_HOURS_BEFORE_EVENT` |
| `POLYMARKET_CHAIN_ID` | Polygon chain ID | Optional; defaults to `137` |
| `RAILWAY_VOLUME_MOUNT_PATH` | Railway persistent storage mount | Optional; used by Railway deployments for data/model/log persistence |
| `UFC_DATA_DIR` | Override data directory path | Optional; defaults to `data/` under project root |
| `UFC_MODELS_DIR` | Models directory path | Advanced local override; defaults to `models/` under project root. The Docker/Railway entrypoint forces `/app/models` and ignores legacy hosted overrides |
| `UFC_LOGS_DIR` | Override logs directory path | Optional; defaults to `data/logs` locally. In hosted runtime, this may resolve directly to `RAILWAY_VOLUME_MOUNT_PATH` when set |
| `RUNTIME_LOG_MAX_BYTES` / `RUNTIME_LOG_BACKUP_COUNT` | Hosted `bot.log` rotation | Optional; defaults to `50 MiB` with `2` backups |
| `EXECUTION_AUDIT_MAX_BYTES` / `TRACKER_DECISION_LOG_MAX_BYTES` / `TRACKER_DECISION_READ_LIMIT` | Bound execution-audit and Model Tracker decision history | Optional; defaults to `100 MiB`, `50 MiB`, and `25,000` read rows respectively |
| `RANKINGS_SNAPSHOT_*` / `METHOD_ODDS_SNAPSHOT_*` / `CARD_SNAPSHOT_*` | Persistent snapshot retention and pruning | Optional; controls full-resolution windows, daily survivors, hard retention, file caps, and prune intervals |
| `LINE_HISTORY_RETENTION_DAYS` / `LINE_HISTORY_PRUNE_INTERVAL_SECONDS` | Live-volume odds and Polymarket line-history retention | Optional; defaults to `180` days and an hourly prune check |
| `LINE_HISTORY_ARCHIVE_REQUIRED` / `LINE_HISTORY_ARCHIVE_*` | Private S3-compatible line-history archive | Railway production forces archive-required behavior. Configure bucket, endpoint, region, credentials, prefix, and URL style through the specific `LINE_HISTORY_ARCHIVE_*` variables |
| `UFC_REFRESH_ENABLED` | Enable hosted UFC refresh loop | Optional; `1` runs scheduled UFC refreshes inside the always-on hosted service |
| `UFC_REFRESH_INTERVAL_HOURS` | Hosted UFC refresh cadence | Optional; defaults to `24` hours and is capped at `24` hours for freshness safety |
| `UFC_REFRESH_INITIAL_DELAY_MINUTES` | Delay first hosted UFC refresh after boot | Optional; defaults to `30` minutes |
| `UFC_REFRESH_LIMIT_FIGHTERS` | Debug cap for hosted UFC refresh | Optional; leave blank in production |
| `UFC_REFRESH_NEW_FIGHTER_ALERT_GRACE_DAYS` | Exclude brand-new roster additions from new-fighter coverage floors | Optional; defaults to `7` days |
| `UFC_REFRESH_PROFILE_SUPPLEMENT_*` | Optional new-fighter profile supplement pass during scheduled refresh | Advanced controls: `..._ENABLED`, `..._LIMIT`, and `..._SOURCES` |
| `UFC_REFRESH_MIN_*` | Coverage-drop alert floors for hosted refresh | Optional; see `.env.example` for the full list |
| `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS` / `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT` | Escalation thresholds for cached active-roster rows retained because they disappeared from the latest UFC.com live sync | Optional; defaults to `50` rows and `5.0%`. Normal small live-sync omissions are logged as notes; larger omissions degrade the hosted refresh component |
| `TAPOLOGY_PROXY_URL` | HTTP/HTTPS proxy for direct Tapology origin paths | Optional but recommended for stable hosted refresh egress. The weekly GitHub workflow retries classified egress blocks on a fresh alternate runner pool and automatically uses an Actions secret with this name when configured. Setting a proxy also permits direct Tapology runtime fetching on Railway and passes the proxy to Chromium when possible |
| `TAPOLOGY_RUNTIME_FETCH_ENABLED` | Permit direct Tapology origin/browser fetching on Railway | Optional; defaults to `0`. The reader-service path is attempted first on Railway and remains available when direct origin fetching is disabled. Set this to `1` only when direct hosted Tapology access is intentional |
| `TAPOLOGY_READER_BASE_URL` / `TAPOLOGY_READER_FALLBACK_ENABLED` | Reader-service Tapology fallback (preferred on Railway) | Optional; enabled by default (`TAPOLOGY_READER_FALLBACK_ENABLED=1`) using `TAPOLOGY_READER_BASE_URL` (defaults to `https://r.jina.ai/`). On Railway (detected via `RAILWAY_PROJECT_ID` / `RAILWAY_SERVICE_ID` / `RAILWAY_ENVIRONMENT`), Tapology profile and fight-history pages are attempted through the reader before the direct-origin gate; search can also use the reader and search-index discovery without enabling origin fetches. Set `TAPOLOGY_READER_FALLBACK_ENABLED=0` to disable. Advanced: `TAPOLOGY_READER_TIMEOUT_SECONDS` sets the request timeout and defaults to `45` seconds |
| `TAPOLOGY_READER_API_KEY` | Optional bearer token for the configured Tapology reader service | Optional. `JINA_API_KEY` is accepted as a fallback name. The weekly workflow reads Actions secrets with either name, never logs the token, and still supports anonymous reader access when neither is configured |
| `TAPOLOGY_READER_BLOCK_COOLDOWN_SECONDS` | Tapology reader circuit cooldown after a blocking response | Optional; defaults to `900` seconds. The first post-cooldown probe performs real I/O before declaring recovery |
| `TAPOLOGY_BROWSER_FALLBACK_ENABLED` | Enable headed-browser Tapology origin recovery | Optional; Docker defaults this to `1`. When direct runtime fetching is allowed and Chromium/Xvfb are present, Tapology origin pages that fail through normal HTTP can be retried through a headed browser session and cached for the process. Advanced tuning (optional, safe defaults): `TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS` (`20`), `TAPOLOGY_BROWSER_READY_TIMEOUT_SECONDS` (`20`), and `TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS` (`3`) control page/ready timeouts and inter-request pacing; `TAPOLOGY_BROWSER_BINARY` / `TAPOLOGY_CHROMEDRIVER_BINARY` / `TAPOLOGY_XVFB_BINARY` override the Chromium/chromedriver/Xvfb paths (already preset in the Docker image) |
| `SHERDOG_BLOCK_COOLDOWN_SECONDS` | Retry cooldown after a Sherdog Cloudflare challenge | Optional; defaults to `1800` seconds. After the cooldown, the next direct request probes Sherdog again so access can recover without a process restart |
| `SHERDOG_WAYBACK_FALLBACK_ENABLED` / `SHERDOG_WAYBACK_TIMEOUT_SECONDS` | Degraded-mode Sherdog profile and fight-history recovery | Optional; fallback defaults to enabled and uses the newest Wayback Machine snapshot only while direct Sherdog access is in its Cloudflare cooldown. The request timeout defaults to `45` seconds and is floored at `1` second |
| `BRAVE_SEARCH_API_KEY` | Official Brave Search API token for site-search profile recovery | Optional. When unset, the bot skips Brave consumer-page scraping by default to avoid Railway egress 429s |
| `BRAVE_SEARCH_API_URL` | Override for the Brave Search API endpoint | Optional; defaults to `https://api.search.brave.com/res/v1/web/search`. Advanced/rarely needed |
| `BRAVE_SEARCH_HTML_FALLBACK_ENABLED` | Legacy Brave consumer HTML search fallback | Optional; defaults to `0`. Enable only for local/manual debugging |
| `BRAVE_SEARCH_TIMEOUT_SECONDS` | Brave site-search timeout | Optional; defaults to `12` seconds |
| `FIGHTDX_REQUEST_TIMEOUT_SECONDS` | Per-request timeout for FightDX fighter-profile fetches | Optional; defaults to `8` seconds (floored at `1.0`) |
| `FIGHTDX_REQUEST_MAX_ATTEMPTS` | Bounded FightDX attempts for transient failures and challenge responses | Optional; defaults to `2` total attempts |
| `FIGHTDX_FAILURE_COOLDOWN_SECONDS` | Cooldown after a FightDX fetch failure before that source is retried (prevents repeated slow-timeout amplification) | Optional; defaults to `180` seconds (floored at `0.0`) |
| `BETSAPI_REQUEST_MIN_INTERVAL_SECONDS` | BetsAPI rate-limit floor | Optional |
| `BETSAPI_429_RETRY_MIN_SECONDS` | BetsAPI 429-retry backoff floor | Optional |
| `METHOD_ODDS_BFO_REQUEST_TIMEOUT_SECONDS` | Per-request timeout for Best Fight Odds method-odds fetches | Optional; defaults to `20` seconds (floored at `1.0`) |
| `METHOD_ODDS_BFO_MAX_RETRIES` | Retry count for transient BFO method-odds fetch failures | Optional; defaults to `3` (floored at `0`) |
| `METHOD_ODDS_BFO_RETRY_BACKOFF_SECONDS` | Backoff between BFO method-odds retries | Optional; defaults to `3` seconds (floored at `0.0`) |
| `METHOD_ODDS_BFO_FAILURE_BUDGET` | Consecutive-failure budget per snapshot before the BFO path short-circuits | Optional; defaults to `3` (floored at `1`) |
| `METHOD_ODDS_COLLECTION_MAX_ATTEMPTS` | Snapshot-level attempts after retryable source failures | Optional; defaults to `3` (floored at `1`). Expected no-props/unavailable responses do not retry |
| `METHOD_ODDS_EXPECTED_WINDOW_HOURS` | Near-event window in which missing method props become a warning | Optional; defaults to `48` hours (floored at `0.0`) |
| `METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS` | Maximum age for a live method-odds fallback snapshot | Optional; defaults to `48` hours. Malformed timestamps and older snapshots fail stale |
| `LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS` | Shared successful UFC event-page cache lifetime | Optional; defaults to `540` seconds so overlapping consumers share one scan while the next 10-minute betting cycle refreshes |
| `ACTIVITY_ALERT_RETENTION_HOURS` | Recovered and unmanaged Activity-alert history window | Optional; defaults to `72` hours (clamped to a 1-hour minimum). Lifecycle-managed active incidents remain visible until explicit recovery |

In real-money UFC mode, an exhausted or invalid open-orders read defers the
entire betting cycle with zero new orders; it is never interpreted as an empty
order list. The dashboard uses one 7-second read attempt inside an 8-second
outer wait and returns HTTP 503 when live order state remains unavailable.

Polymarket client note: the pinned `py_clob_client` contract used here must expose `derive_api_key()` and `create_api_key()`. The legacy `create_or_derive_api_creds()` helper is no longer the runtime path.

## CLI Overview

All commands run from the project root with `python -m src.bot ...`.

### UFC workflow

```bash
# Scrape historical UFCStats fighter and fight data
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
python -m src.bot backtest --static
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

# Inspect or safely restore archived line-history snapshots
python -m src.bot line-history-archive list
python -m src.bot line-history-archive restore '<exact-object-key>'
```

Notes:

- `scrape` is the historical UFCStats scrape. Use `ufc-refresh-scheduled` for the active-roster sync, UFCStats backfill, processed-data rebuild, and profile audit.
- `live --real` requires `ODDS_API_KEY`, `POLYMARKET_PRIVATE_KEY`, the primary and no-odds model artifacts, writable log/cache/ledger paths, `LIVE_TRADING_ARMED=1`, and `LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED`. The two arming values are necessary but not sufficient.
- `predict` logs runtime-bundle freshness warnings. `live --real` warns outside the shared betting window and blocks once an in-window fight could trade if the promoted model is older than one month (`MODEL_RETRAIN_MONTHS`) or the processed snapshot is missing a known completed UFC card before the active card. If completed-card discovery is unavailable beyond its one-hour cache, the 7-day snapshot-age heuristic (`LIVE_PROCESSED_REFRESH_MAX_AGE_DAYS`) is advisory only because it cannot distinguish stale data from a long gap between UFC cards.
- `predict` and `live` may still display a diagnostic prediction that is marked `trade_blocked`. Data-quality-blocked rows are removed before all three traders run; if every row is blocked, the cycle is degraded and confirmed unfilled resting orders are maintained or cancelled without placing replacements.
- `backtest` defaults to `--execution-mode realistic` (models realistic fills and slippage); `walkforward` still defaults to `legacy`. Pass `--execution-mode` to override either.
- CLI `predict` and `live` load the canonical `xgboost` alias by default and use `--model` for an alias or explicit artifact override. `LIVE_MODEL` configures the hosted `python -m src.web.serve` entrypoint; it does not override the CLI parser default.
- The repository promotion aliases are recorded in [models/current_production_model.json](models/current_production_model.json); hosted runtime paths and reconciled snapshot metadata come from the mounted runtime manifest exposed in `/readyz`.
- `line-history-archive` only lists and restores exact bucket objects; it has no delete operation. Restores default to `data/restored_line_history`, outside the live input tree. See [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md) for filtering, pagination, validation, and Railway-shell examples.

### Crypto 5m Polymarket runner

The crypto 5-minute up/down runner is a separate strategy from the UFC pipeline (BTC by default, with ETH/SOL asset profiles). It is dry-run by default; `--real` requires the same Polymarket real-money arming as UFC live trading.

```bash
# Single-profile momentum runner (continuous unless --once; dry-run by default)
python -m src.bot btc5m --once
python -m src.bot btc5m --profile conservative --poll-seconds 1
python -m src.bot btc5m --real   # real money; requires Polymarket arming env vars

# Compare strategy profiles side by side in paper mode
python -m src.bot btc5m-paper --profiles late_capture,conservative --once
python -m src.bot btc5m-paper --settle-only

# Forward profile opportunity harness (writes JSON + Markdown report)
python -m src.bot btc5m-opportunity --profiles all --target-markets 500
```

Notes:

- `btc5m --real` is blocked unless the Polymarket real-money arming env vars are set (same arming model as `live --real`). `btc5m-paper` and `btc5m-opportunity` are always simulated.
- Hosted crypto 5m profile loops default to `BTC5M_POLL_SECONDS=1`. HTTP `418/429/451` upstream responses are reported with endpoint/status metadata in runtime status and the Activity dashboard.
- The available risk profiles are defined in `BTC5M_PROFILES` in [src/config.py](src/config.py) — `conservative` is the default, alongside a large family of `late_capture_*` and `cheap_below*` tuning variants plus per-asset alt-coin profiles (e.g. `eth_late_capture_gap005`) generated from `BTC5M_ALT_5M_ASSETS`. `btc5m-opportunity --profiles all` runs every asset and variant.
- At the 2026-07-23 Railway verification recorded in the Status section, production had no configured live crypto profiles. The previously used BTC `late_capture_gap005*` / `late_capture_gap0025*` profiles remain available in code with a `$50` target trade notional, a `$55` max notional per trade, and a `$200` daily loss limit per profile, but they are not active merely because they exist in `BTC5M_PROFILES`.
- ETH/SOL profiles remain available for paper/opportunity evaluation and use Binance first, Coinbase as direct backup, and Hyperliquid last; they were not active at that production verification.
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
| `full_live_contract_v6_durability_fullfit` | 211 | Current promoted production contract: the V6 full-fit contract plus 9 loss-method/durability decomposition features (KO/submission loss rates and a recent-KO-loss flag); selected/promoted 2026-06-11 and refit without contract changes 2026-07-22 |
| `full_live_contract_v7` | 223 | Offline evaluation candidate: V6 plus amateur-career summary features |

Legacy named specs such as `full_live_contract_v1`, `full_live_contract_v3`, `full_live_contract_v4`, `full_live_contract_v4_138`, and `full_live_contract_v4_144` are still resolvable through `resolve_named_training_spec()`, but they are not part of the current production line.

Current repository production artifact: bundle `ufc-production-20260718-full_live_contract_v6_durability_fullfit` (spec `full_live_contract_v6_durability_fullfit`, 211 features), built 2026-07-22 as a scheduled same-spec refit using refreshed fight data through 2026-06-27 and a bundled processed snapshot through 2026-07-18. It retains the contract selected and promoted on 2026-06-11; no feature or hyperparameter changes were made in the July refit. Railway currently runs this embedded model contract from the reconciled runtime bundle described in the Status section. Canonical live aliases are `xgboost`, `xgboost_no_odds`, and `logistic` (the `xgboost_no_odds` variant uses the matching `full_live_contract_v6_durability_fullfit_no_odds` spec, which drops the odds features rather than the durability features).

**A/B orientation parity:** training applies automatic A/B mirror augmentation — each observed fight is also added with the two fighters' sides swapped — together with orientation-aware cross-validation, and live prediction symmetrizes by averaging the forward and A/B-swapped predictions. This keeps live inference (alphabetical fighter ordering) consistent with the training distribution and removes the historical positional bias where the training slot A was the winner far more often than chance. Implied-odds probabilities are also no-vig normalized, and invalid moneyline rows are dropped before training (including duplicated heavy-favorite rows where both fighters share the same low price; legitimate equal pick'em prices are retained). The current promoted spec (`full_live_contract_v6_durability_fullfit`, 211 features) layers a loss-method durability feature family on top of this A/B-parity V6 contract — a/b/diff variants of `loss_ko_rate`, `loss_sub_rate`, and `recent_ko_loss`, with NaN-honest denominators for fighters lacking the relevant loss history. See [src/model/orientation.py](src/model/orientation.py).

If you are reproducing the promoted model line, use the manifest and spec files under [models/](models/). The repository manifest records the current scheduled refit and retains the original June promotion under its `prior_promotion` metadata; on Railway, the mounted reconciled manifest is the source of truth for active paths and hosted processed-snapshot metadata. For how the durability contract was selected and originally promoted, see [docs/DURABILITY_PROMOTION_RUNBOOK.md](docs/DURABILITY_PROMOTION_RUNBOOK.md); the supporting experiment results and model-improvement analysis are in [docs/EXPERIMENT_RESULTS_2026-06-10.md](docs/EXPERIMENT_RESULTS_2026-06-10.md) and [docs/MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md](docs/MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md).

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
- Hosted startup fails closed for trading if required env vars, model artifacts, or writable ledger and log paths are missing. The container entrypoint also aborts on persistent-roster sanitation failure and, in real mode, on failed wallet-position reconciliation.
- Predictions that fail the live data-quality gate remain visible with `trade_blocked` diagnostics but are excluded from trader candidate selection; the filter funnel reports them at the `Data Quality` stop.
- On public binds, mutation routes require `WEB_DASHBOARD_TOKEN`; selected execution read responses redact sensitive fields unless the same token is supplied.
- The Activity alerts panel separates active and recovered incidents. Lifecycle-managed incidents stay active until an explicit recovery event arrives; quiet time alone does not resolve them.
- Open Bets is wallet-wide rather than sport-filtered: it includes bot-tracked and manual/untracked positions from any Polymarket category, while the other dashboard sport filters continue to classify their own views normally.

Selected API routes:

- `/healthz`, `/readyz` — health and readiness probes
- `/api/summary` — dashboard overview
- `/api/predictions`, `/api/predictions-detail` — model predictions
- `/api/upcoming-events` — upcoming UFC events
- `/api/positions`, `/api/open-limit-orders` — Polymarket positions and orders
- `/api/balance` — wallet balance
- `/api/bets`, `/api/trade-history` — bet and trade history
- `/api/open-bets-enriched`, `/api/profile-bets` — wallet-wide enriched open positions and per-profile bet views
- `/api/pnl-history` — P&L over time
- `/api/bot-activity`, `/api/significant-actions` — bot activity and notable actions
- `/api/bot-alerts` — coalesced active and recovered `WARNING`/`ERROR`/`CRITICAL` incidents (powers the Activity page's pinned alerts panel)
- `/api/trader-race`, `/api/trader-breakdown` — trader comparison metrics
- `/api/injury-alerts` — advisory market alerts for unusual line movement or near-zero prices
- `/api/filter-funnel` — prediction filter diagnostics
- `/api/geoblock-status` — geo-restriction diagnostics
- `/api/refresh-prices` (POST), `/api/settle-auto` (POST), `/api/settle/<bet_id>/<result>` (POST, manual single-bet settle), `/api/redeem-auto` (POST), `/api/reconcile-limit-orders` (POST) — operational actions
- `/api/runtime-status` — hosted runtime component status
- `/api/closed-positions` — resolved Polymarket positions
- `/api/bot-activity-snapshot` — activity snapshot
- `/ufc`, `/predictions`, `/activity`, `/bet-history`, `/execution-breakdown` — dashboard pages
- `/api/tracker-decisions` — Model Tracker (M) decision log
- `/api/execution-breakdown` — structured per-cycle, per-fight, per-trader (S/C/M) execution decision audit (returns the latest cycle by default; supports `?history=1`, `?cycle_id=`, `?limit=`, and `?offset=` pagination); powers the `/execution-breakdown` page

See [src/web/app.py](src/web/app.py) for the full route list.

## Deployment

Docker and Railway use the hosted entrypoint:

```bash
docker build -t ufc-betting-bot .
docker run --env-file .env -p 5050:5050 ufc-betting-bot
```

The Docker/Railway entrypoint defaults to `APP_ROLE=web`, starts `python -m src.web.serve`, and bootstraps the runtime production-bundle manifest into the mounted data volume before startup. Persistent migrations use verified atomic copies, a valid volume active roster remains authoritative, and startup sanitizes that roster before serving; invalid persisted state may recover only from a validated image fallback. In real mode, the entrypoint also reconciles Polymarket positions and refuses to start if reconciliation fails. For hosted web services, leave `UFC_MODELS_DIR` and `UFC_PRODUCTION_BUNDLE_MANIFEST` unset unless you are intentionally changing the entrypoint behavior in code.

### Persistent storage and line-history archive

Hosted logs, execution audits, rankings snapshots, method-odds snapshots, card snapshots, and line histories all have explicit size or retention bounds. Odds and Polymarket line-history CSVs default to 180 days on the live volume.

Railway production forces line-history archiving before deletion. Configure the `LINE_HISTORY_ARCHIVE_*` variables from a private S3-compatible bucket. If the bucket is missing, source validation fails, or an upload fails, the bot preserves the expired live-volume file and raises an operational alert; it does not silently delete the only copy.

Operators can list or safely restore exact archived objects without deleting the bucket copy:

```bash
python -m src.bot line-history-archive list --category odds --json
python -m src.bot line-history-archive restore '<exact-object-key>'
```

Restores default outside the live line-history tree and are validated before publication. See [PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md) for pagination, destination overrides, and Railway-shell examples.

Safe hosted default:

```dotenv
LIVE_TRADING_MODE=off
```

Paper-trading hosted deploy:

```dotenv
LIVE_TRADING_MODE=dry-run
ODDS_API_KEY=replace_me
WEB_DASHBOARD_TOKEN=change_me
```

Crypto 5m always-on paper deploy example, only after deliberately choosing profiles:

```dotenv
LIVE_TRADING_MODE=dry-run
ODDS_API_KEY=replace_me
BTC5M_LIVE_PROFILES=late_capture_gap005
WEB_DASHBOARD_TOKEN=change_me
```

The profile above is an example, not the current Railway configuration or a strategy recommendation. Leave `BTC5M_LIVE_PROFILES` blank to keep the hosted 5m loop dormant. Multiple configured profiles can run concurrently with separate ledgers under `BTC5M_LIVE_LEDGER_DIR`.

The crypto 5m loop shares the same `LIVE_TRADING_MODE` switch as the UFC loop: it paper-trades when `LIVE_TRADING_MODE=dry-run` and trades real money only when `LIVE_TRADING_MODE=real` (with full arming). With `LIVE_TRADING_MODE=off` the loop stays dormant even if `BTC5M_LIVE_PROFILES` is set — there is no separate 5m mode env, so a configured profile needs both `BTC5M_LIVE_PROFILES` and a non-`off` `LIVE_TRADING_MODE` to come live. Profile names are validated against `BTC5M_PROFILES`; an unknown or misspelled entry fails closed and disables the entire 5m loop (surfaced on the `btc5m_loop` runtime component). Avoid `BTC5M_LIVE_PROFILES=all` in production unless you intentionally want every configured BTC, ETH, and SOL profile live.

Real-money hosted deploy:

```dotenv
LIVE_TRADING_MODE=real
ODDS_API_KEY=replace_me
POLYMARKET_PRIVATE_KEY=replace_me
POLYMARKET_FUNDER_ADDRESS=0x...
LIVE_TRADING_ARMED=1
LIVE_TRADING_CONFIRMATION=REAL_TRADING_ENABLED
WEB_DASHBOARD_TOKEN=change_me
```

For crypto 5m real-money hosting, also set `BTC5M_LIVE_PROFILES`. Missing the private key or any real-money arming env blocks the 5m loop from starting.

Real crypto 5m orders are schedule-gated by default: they can only submit between Monday `09:00` and Friday `17:00` in `America/New_York`. Railway does not need new variables for that default because the code falls back to it when the `BTC5M_REAL_TRADING_*` env vars are absent; set those vars only if you intentionally want to override or disable the window.

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
- The scheduled refresh also supports an optional profile-supplement pass for new active fighters. Use `UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED=0` to disable it, `UFC_REFRESH_PROFILE_SUPPLEMENT_LIMIT` to smoke-test it, and `UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES` to restrict sources. By default it tries faster structured sources first (`espn`, `fightdx`, `martialbot`), then Tapology/Sherdog, with Wikipedia last because that source retries HTTP 429 responses up to four total attempts, honoring `Retry-After` when present and otherwise backing off from 10 seconds.
- Official UFC active-roster page fetches retry transient server failures (`500`, `502`, `503`, `504`) as well as timeouts and connection failures up to three total attempts. ESPN profile/search fallbacks use the same three-attempt ceiling for transient failures and additionally retry `429` responses and JSON decoding/empty-response failures.
- On Railway, Tapology's reader-service path can run while direct Tapology origin access remains disabled. Blocking reader failures open a timed circuit (900 seconds by default); the first post-cooldown probe must perform real I/O before recovery is reported. FightDX retries bounded transient/challenge failures before entering its own cooldown. Sherdog uses direct access normally, enters a configurable cooldown on Cloudflare challenges, and can use Wayback snapshots during that cooldown.
- An official active-roster scan is considered complete only with explicit pagination completion and successful parsing of every selected fighter card. Parser drift, incomplete pagination, or a suspicious live shrink marks the scan incomplete, retains cached rows missing from that scan, and freezes their missing-row counters.
- A fighter missing from a complete live scan is retained temporarily and expires only after three complete misses. Verified inactive fighters are not resurrected; unknown, blank, or untrusted inactive identities are quarantined from coverage, and strong URL-identity conflicts prevent unsafe same-name deletion.
- The hosted refresh loop writes through the same guarded atomic CSV paths as the manual refresh command, so empty or incomplete scrapes do not replace good artifacts with blank files. Container startup separately sanitizes the persisted roster atomically and refuses an empty or invalid result.
- Real-money live trading checks processed-snapshot freshness against known completed UFC card dates when live UFC.com context is available. This avoids blocking late US cards solely because UTC has rolled into the next day, tolerates the common one-day UFC.com/Odds API versus UFCStats card-date offset, and recovers a just-completed UFC.com card only when its fighter pairs match still-active bookmaker fights. Outside the configured betting window this is a warning; once a matching fight is in-window, the bot still fails closed when the snapshot is missing a distinct completed intervening card.
- Refresh failures are reported immediately in the hosted runtime status as a degraded `ufc_refresh_loop` component.
- While a scheduled refresh is rebuilding the processed snapshot, if the bundle-freshness guard blocks a live cycle mid-refresh the betting loop reports a `degraded` "live trading paused while scheduled UFC refresh rebuilds the processed snapshot" status instead of an error, and resumes automatically once the refresh completes.
- Coverage-drop alerts are optional. Set one or more `UFC_REFRESH_MIN_*` env vars if you want the hosted refresh loop to mark itself degraded when audited coverage falls below your chosen floor.
- Retained rows missing from a complete UFC.com sync are reported as notes while their miss counters are below the expiry threshold; unusually large retained sets degrade the refresh loop when they exceed `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS` or `UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_PCT`.

## Disclaimer

This project is for research and education. Sports betting involves real financial risk. Never bet more than you can afford to lose, and do not treat historical model performance as a guarantee of future results.
