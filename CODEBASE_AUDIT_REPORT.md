# UFC Betting Bot — Comprehensive Codebase Audit Report

**Date:** 2026-03-17
**Auditor:** Claude Opus 4.6 (1M context, automated full-codebase review)
**Scope:** Every file in the repository — 100+ source files, 35 test files, all config/deployment artifacts, scripts, templates, data pipeline code
**Method:** Five parallel audit agents each read every file in their assigned domain (data layer, model layer, strategy layer, polymarket execution, tests/config/infra). Findings were cross-referenced, verified, and deduplicated.

---

## 1. Executive Summary

This is a sophisticated, production-grade betting system with ~60,000 lines of Python across UFC and tennis prediction pipelines, Polymarket execution, and a web dashboard. The architecture shows genuine engineering rigor — clear separation between data ingestion, feature engineering, model training, strategy evaluation, and trade execution. Quality gates (selection gate, promotion gate, preflight audit) and walk-forward backtesting demonstrate serious methodology.

**However, this audit uncovered issues that could directly lose money:**

1. **The `get_cash_balance` method uses a heuristic that misclassifies real balances** — a $1,500 balance is divided by 1,000,000, returning $0.0015. This feeds directly into Kelly bet sizing.
2. **No `.dockerignore` exists** — `COPY . .` in the Dockerfile bakes `.env` (including the Polymarket private key) into every Docker image layer if built locally.
3. **The web dashboard has zero authentication** — anyone who can reach the port can settle bets, manipulate P&L records, and trigger API calls.
4. **No transaction journaling exists** — if the process crashes between placing an order on Polymarket and recording it in the ledger, real money is deployed but untracked, enabling duplicate bets on restart.
5. **The `_name_match` function in the executor has a substring bug** — "Ian" matches "Brian", potentially placing bets on the wrong fighter.
6. **The conviction trader has no positive-EV gate** — it can place bets with negative expected value against market odds purely based on model agreement.

The codebase is not unsafe to run, but the combination of real-money execution, missing authentication, absent transaction journaling, and several data-integrity issues means the risk surface is larger than it should be.

---

## 2. Critical Issues

Issues that could lose money, corrupt data, or expose secrets.

### CRIT-01: `get_cash_balance` uses a heuristic that misclassifies real balances
**File:** [client.py:427-444](src/polymarket/client.py#L427-L444)

```python
balance = float(balance_str) / 1e6 if float(balance_str) > 1000 else float(balance_str)
```

This assumes values >1000 are in atomic USDC units (6 decimals). A real balance of $1,500 USDC would be divided by 1,000,000, returning $0.0015. Conversely, 800 atomic units ($0.0008) would be treated as $800. Since `BankrollManager` uses this to size bets via Kelly criterion, the system could bet 100x too much or too little.

**Fix:** Query the USDC contract's `decimals()` function, or use a known constant (USDC on Polygon has 6 decimals). Never use a value-based heuristic.

---

### CRIT-02: No `.dockerignore` — secrets baked into container images
**File:** [Dockerfile](Dockerfile), line 10

```dockerfile
COPY . .
```

The `.env` file is correctly excluded from git via `.gitignore`. However, there is no `.dockerignore` file. If you build the Docker image locally (where `.env` exists on disk), `COPY . .` copies `.env` — including the Polymarket private key — into every layer of the container image. Anyone with access to the image can extract the key.

**Fix:** Create a `.dockerignore`:
```
.env
*.env
.git/
__pycache__/
data/
logs/
models/
```

---

### CRIT-03: Container runs as root
**File:** [Dockerfile](Dockerfile)

The Dockerfile never creates a non-root user. The Flask web server and all financial transaction handlers run as root. If any dependency vulnerability is exploited, the attacker has root access.

**Fix:** Add `RUN useradd -m appuser && USER appuser` to the Dockerfile.

---

### CRIT-04: No transaction journaling — orders can be placed but not recorded
**File:** [executor.py:1498-1582](src/polymarket/executor.py#L1498-L1582)

The critical path is: (1) sign and post order to CLOB, (2) debit bankroll, (3) record in ledger. If the process crashes between steps 1 and 3, real money is deployed on Polymarket but the bot has no record. On restart, the duplicate check sees no existing position and places another order. The `order_log` is in-memory only (line 266) — a crash loses it.

**Fix:** Write a "pending" record to disk before posting the order. After CLOB confirms, update to "confirmed". On startup, scan for pending records and reconcile against CLOB open orders.

---

### CRIT-05: Web dashboard has zero authentication
**File:** [app.py](src/web/app.py)

Every route is publicly accessible. Mutation endpoints include:
- `POST /api/settle-auto` (line 296) — auto-settles bets
- `POST /api/settle/<bet_id>/<result>` (line 307) — manually settles as won/lost
- `POST /api/refresh-prices` (line 245) — triggers API calls

The `railway.toml` healthcheck confirms the app is publicly reachable. Anyone who finds the URL can manipulate the betting ledger.

**Fix:** Add authentication (at minimum HTTP basic auth or a shared secret header) to all mutation endpoints.

---

### CRIT-06: `_name_match` in executor has substring false positive bug
**File:** [executor.py:2000-2009](src/polymarket/executor.py#L2000-L2009)

```python
parts1 = clean1.split()
parts2 = clean2.split()
if parts1[-1] == parts2[-1]:  # Same last name
    if parts1[0] in parts2[0] or parts2[0] in parts1[0]:  # First name containment
        return True
```

`parts1[0] in parts2[0]` is a **substring** check. "Ian" matches "Brian" (because `"Ian" in "Brian"` is True). This could cause predictions to be matched to the wrong fighter's Polymarket market, placing real-money bets on the wrong person.

**Fix:** Replace substring check with a more robust comparison — require Levenshtein distance <= 2, or check that one name starts with the other, or use `name_utils.same_person_name()`.

---

### CRIT-07: Conviction trader has no positive-EV gate
**File:** [value.py:507-614](src/strategy/value.py#L507-L614)

`find_conviction_bets()` selects bets purely on model agreement (model_prob >= 65%, no_odds_prob >= 50%). There is no check that expected value against market odds is positive. A fighter at model_prob=0.65 with market_prob=0.70 has **negative edge** yet passes all conviction gates. The `conviction_ev_check` flag exists in `VariantConfig` but is never checked in the live code path.

**Fix:** Add `if edge <= 0: continue` or check `conviction_ev_check` in `find_conviction_bets()`.

---

### CRIT-08: Look-ahead bias in scraped-data feature path
**File:** [fighter_lookup.py:454-465](src/data/fighter_lookup.py#L454-L465)

```python
def _parse_dob_to_age(dob_str: str) -> float:
    age = (datetime.now() - dob).days / 365.25
```

**File:** [fighter_lookup.py:1158-1163](src/data/fighter_lookup.py#L1158-L1163)

```python
days = (datetime.now() - last_fight["event_date"]).days
features["days_since_last_fight"] = max(days, 0)
```

Both functions compute features relative to `datetime.now()` instead of the fight's event date. During backtesting, a fighter who last fought in 2020 shows 2000+ days layoff when the actual layoff at fight time was ~365 days. The processed-history path handles this correctly via `as_of_date`, but the scraped-data path does not.

**Fix:** Accept and use an `as_of_date` parameter in both functions.

---

### CRIT-09: Backfill scripts overwrite production data with no backup
**File:** [backfill_odds_api.py:169](scripts/backfill_odds_api.py#L169)
**File:** [incremental_backfill.py:728](scripts/incremental_backfill.py#L728)
**File:** [merge_kaggle_data.py:306](scripts/merge_kaggle_data.py#L306)

All three scripts directly overwrite `ufc-master.csv` (the canonical training dataset) without creating a backup first. A single malformed API response could corrupt the entire training corpus.

**Fix:** Write to a temp file first, validate the output, then atomically rename. Keep the previous version as `.bak`.

---

### CRIT-10: Duplicate definitions silently overwrite critical constants
**File:** [build_features.py](src/features/build_features.py)

- Line 852: `ODDS_FEATURE_NAMES` defined as a `set`
- Line 1053: `ODDS_FEATURE_NAMES` **redefined** as a `tuple` (overwrites the set)
- Line 860: `MARKET_DERIVED_FEATURE_NAMES` defined as `ODDS_FEATURE_NAMES | {line features}`
- Line 1054: `MARKET_DERIVED_FEATURE_NAMES` **redefined** as `set(MARKET_DERIVED_DENYLIST)`, dropping line movement features
- Line 870: `exclude_market_derived_features` defined
- Line 1075: `exclude_market_derived_features` **redefined** with different behavior
- Line 875: `get_feature_columns_no_odds` defined with 1 parameter
- Line 1136: `get_feature_columns_no_odds` **redefined** with 2 parameters

Python silently uses the last definition. The intermediate definitions (which set up `MARKET_DERIVED_FEATURE_NAMES` with line movement features) are dead code. Consumers see different behavior depending on whether they access the variable before or after the redefinition.

**Fix:** Remove the duplicate definitions. Keep only the final version of each, and update the intermediate code to match.

---

### CRIT-11: Bankroll can go negative — no floor enforcement
**File:** [bankroll.py:199](src/strategy/bankroll.py#L199)

```python
self.bankroll -= amount
```

`place_bet()` deducts the bet amount without verifying sufficient funds. Additionally, `place_bet()` does not check `is_stopped` — a stopped bankroll can still have bets placed on it if the caller uses `place_bet()` directly instead of going through `kelly_bet_size()`. External callers (like `duo_trader.py` with `override_bet_size`) bypass the Kelly function entirely.

**Fix:** Add `if self.bankroll < amount: raise InsufficientFundsError` and `if self.is_stopped: return {}` in `place_bet()`.

---

### CRIT-12: Bankroll debited before confirming order fill
**File:** [executor.py:1562-1569](src/polymarket/executor.py#L1562-L1569)

For market orders (Fill-or-Kill), `order_info["status"] in ("placed", "dry_run")` triggers `self.bankroll.place_bet()`. But "placed" only means the CLOB API accepted the request — it does NOT confirm the order was filled. If the FOK order is rejected server-side (insufficient liquidity), the bankroll is already debited with no refund mechanism. Same issue at lines 1776-1783 for near-miss limits.

**Fix:** For market orders, verify fill status from CLOB response before debiting bankroll.

---

### CRIT-13: `test_bet.py` places real $1 bets on production Polymarket
**File:** [test_bet.py](test_bet.py)

This script at the repo root uses production credentials to place real orders. It has no `--dry-run` flag, no confirmation prompt, and its filename looks like a pytest test. Running `pytest` from the repo root would collect this file.

**Fix:** Rename to `scripts/manual_test_bet.py`, add a confirmation prompt, or add `if __name__ != "__main__": sys.exit()` guard.

---

### CRIT-14: No maximum position size or total exposure limit
**Files:** [executor.py](src/polymarket/executor.py), [bankroll.py](src/strategy/bankroll.py)

There is no hard cap on total deployed capital. If the model produces many value bets simultaneously (or a model bug produces `blended_prob = 0.99` on every fight), the bot could exhaust the entire bankroll in one cycle. `MAX_BET_FRACTION` (4%) applies per bet, but there is no per-event or total-exposure cap.

**Fix:** Add a `MAX_EVENT_EXPOSURE` config (e.g., 20% of bankroll per event) and a `MAX_TOTAL_EXPOSURE` (e.g., 60% of bankroll across all open bets).

---

### CRIT-15: Model .pkl files tracked in Git
**Files:** `models/logistic_model.pkl`, `models/tennis/surface_elo.pkl`, `models/candidates/full_live_contract_v3/*.pkl`

Pickle files are tracked in git (confirmed via `git ls-files`). These are binary blobs that inflate repo size and cannot be meaningfully diffed. Worse, `.pkl` files are a known deserialization attack vector — anyone with write access to the repo could replace a model file with a malicious payload that executes arbitrary code on `joblib.load()`.

The `.gitignore` ignores `data/models/` but the top-level `models/` directory is NOT ignored.

**Fix:** Add `models/` and `*.pkl` to `.gitignore`. Store models in a separate artifact store (S3, Railway volume).

---

## 3. Major Issues

Significant bugs, design flaws, or reliability risks.

### MAJ-01: Blend-B probability asymmetry in production
**File:** [value.py:341-343](src/strategy/value.py#L341-L343)

```python
blend_a = blend_probability(model_a, market_a, dyn_weight)
blend_b = 1.0 - blend_a
```

The dynamic blend weight is computed only for side A based on A's model confidence and no-odds agreement. Side B's blend is simply `1 - blend_a`, meaning fighter B's own model confidence signals are completely ignored when computing B's blended probability. The fix (`use_independent_blend_b`) exists as an opt-in variant but is **not the production default**.

**Impact:** Systematic mis-pricing on the B-side of every bet. If the model is more confident about fighter B, that confidence is discarded.

---

### MAJ-02: Double counting of open bets in bankroll sync
**File:** [bankroll.py:40-92](src/strategy/bankroll.py#L40-L92)

When `auto_detect_balance=True`, the constructor sets `initial_bankroll = live_balance` (the Polymarket CLOB cash balance, which already excludes cash committed to open orders). Then `_sync_from_ledger()` computes:
```python
self.bankroll = self.initial_bankroll + realized_pnl - total_wagered
```
The `live_balance` already reflects realized P&L (profits are in the balance, losses are gone) and already excludes open bet cash (that money is committed on the CLOB). Adding `realized_pnl` and subtracting `total_wagered` double-counts both, making the tracked bankroll lower than reality.

**Fix:** Either skip `_sync_from_ledger()` when using live balance, or use the initial deposit amount (not live balance) as the starting point for ledger replay.

---

### MAJ-03: No circuit breaker for consecutive API failures
**Files:** [executor.py](src/polymarket/executor.py), [client.py](src/polymarket/client.py)

If the Polymarket API is returning errors, the bot keeps attempting to place orders, logging warnings, and potentially corrupting internal state (bankroll debited for "placed" orders that actually failed upstream). There is no mechanism to halt trading after N consecutive failures.

**Fix:** Track consecutive failures; after 3-5 in a row, set a circuit breaker that blocks all trading until manual reset or a successful health check.

---

### MAJ-04: Thread safety issues with shared mutable state
**File:** [app.py:38-44](src/web/app.py#L38-L44)
**File:** [serve.py:169](src/web/serve.py#L169)

The `_clob_client` global is written by a background thread and read by Flask request handlers without synchronization. `set_clob_client` writes both `_clob_client` and `_position_monitor` without holding `_monitor_lock`, while some readers use the lock and others don't.

---

### MAJ-05: Global config mutation during variant training is not thread-safe
**File:** [model_variants.py:171-175](src/strategy/model_variants.py#L171-L175)

```python
cfg.TIME_DECAY_HALF_LIFE_DAYS = variant.time_decay_half_life
sample_weights = _compute_sample_weights(train_df)
cfg.TIME_DECAY_HALF_LIFE_DAYS = original_half_life
```

This mutates global config state. Combined with `ProcessPoolExecutor` in `run_evaluation.py`, concurrent variant training could see each other's config mutations. Same issue with monkey-patching `_compute_rolling_stats` in `build_features_ewm` (lines 256-288).

**Fix:** Pass `time_decay_half_life` as a parameter instead of mutating global state. Use dependency injection for rolling stats functions.

---

### MAJ-06: Flask debug mode enables remote code execution
**File:** [app.py:1503](src/web/app.py#L1503)

```python
app.run(host="0.0.0.0", port=port, debug=debug)
```

The `--debug` flag enables Werkzeug's interactive debugger, which allows arbitrary Python code execution from any browser. Combined with `host="0.0.0.0"` (all interfaces), this is an RCE risk.

**Fix:** When debug is enabled, bind to `127.0.0.1` only.

---

### MAJ-07: Dependencies use minimum version pins, no lockfile
**File:** [requirements.txt](requirements.txt)

All 17 dependencies use `>=` pins (e.g., `flask>=3.0`, `xgboost>=2.0`). No lockfile exists. XGBoost model serialization format can change across major versions, making saved `.pkl` models unloadable. `py-clob-client` (the Polymarket order client) API changes could break trading.

**Fix:** Generate `requirements.lock` via `pip freeze` and use it for production builds.

---

### MAJ-08: `entrypoint.sh` truncates logs on every container restart
**File:** [entrypoint.sh:33](entrypoint.sh#L33)

```bash
> /app/data/logs/bot.log
```

With `restartPolicyType = "ALWAYS"` in `railway.toml`, every crash wipes the log — destroying evidence of what caused the crash.

---

### MAJ-09: `_acquire_file_lock` spins indefinitely with no timeout
**File:** [tracker.py:52-66](src/polymarket/tracker.py#L52-L66)

On Windows, if a lock file is held by a crashed process, this spins forever:
```python
while True:
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    except OSError:
        time.sleep(0.05)
```

**Fix:** Add a timeout (e.g., 30 seconds) and raise a `TimeoutError`.

---

### MAJ-10: Corrupt ledger silently resets to empty, losing all bet history
**File:** [tracker.py:226-228](src/polymarket/tracker.py#L226-L228)

```python
logger.warning("Corrupt ledger file, starting fresh")
```

If the JSON file is corrupted (e.g., partial write during crash — see CRIT-04), ALL historical bet data is silently lost. No backup is created before overwriting.

**Fix:** Copy the corrupt file to `.bak` before resetting. Add a startup warning if the ledger was reset.

---

### MAJ-11: Combinatorial sweep produces 2,916+ configs — classic overfitting
**File:** [duo_trader_sweep.py:651-682](src/strategy/duo_trader_sweep.py#L651-L682)

The nested sweep loops produce 2,916 configurations per variant. There is no multiple-comparison correction, no holdout set reserved from the sweep, and the "best" config is selected by sorting on ROI. This is a textbook data-snooping/p-hacking risk.

**Fix:** Reserve a holdout period for final validation, apply Bonferroni or similar correction, or reduce the parameter grid.

---

### MAJ-12: Women's divisions collapsed into men's divisions in rankings
**File:** [rankings_scraper.py:45-59](src/data/rankings_scraper.py#L45-L59)

```python
_WC_ALIASES = {
    "women's flyweight": "flyweight",
    "women's bantamweight": "bantamweight",
    ...
}
```

Women's and men's divisions are merged. A fighter ranked #5 in Women's Bantamweight is indistinguishable from #5 in Men's Bantamweight. Rankings will be incorrect when used as features.

---

### MAJ-13: SHAP feature selection uses wrong hyperparameters and leaks
**File:** [feature_selection.py:58-70](src/model/feature_selection.py#L58-L70)
**File:** [train.py:384-392](src/model/train.py#L384-L392)

Two issues: (1) `shap_feature_ranking` trains a separate XGBoost with `n_estimators=200, max_depth=5, lr=0.05`, which differs from production (`n_estimators=135, max_depth=7, lr=0.0124`). Features selected as important by this proxy model may not match the production model. (2) SHAP selection sees training targets before feature selection, then the selected features train on the same data — this is feature selection bias.

**Fix:** Use production hyperparameters for SHAP, and move feature selection inside the CV loop.

---

### MAJ-14: Calibration CV uses `random_5fold` in baselines (temporal leakage)
**File:** [compare.py:199](src/model/compare.py#L199), [train_experimental.py:344](src/model/train_experimental.py#L344)

Random 5-fold CV for calibration leaks future data into the calibration step (a future fold calibrates past predictions). The baseline in model comparison uses `random_5fold` while production uses `timeseries_5fold`, giving the baseline an unfair advantage and potentially causing better models to be rejected.

---

### MAJ-15: `_on_chain_usdc_balance` queries wrong USDC contract
**File:** [client.py:446-461](src/polymarket/client.py#L446-L461)

The fallback balance check queries USDC.e (bridged) at `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`. Polymarket uses native USDC at `0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359`. This fallback always returns $0.

---

### MAJ-16: No input validation on `side` parameter in order placement
**File:** [client.py:293-341](src/polymarket/client.py#L293-L341)

```python
order_side = BUY if side.upper() == "BUY" else SELL
```

Any typo like `"BU"` or `""` silently resolves to SELL, placing an order on the wrong side with real money.

**Fix:** Validate against an enum: `if side.upper() not in ("BUY", "SELL"): raise ValueError`.

---

### MAJ-17: Stale dry-run limit bids accumulate forever, blocking future dry runs
**File:** [executor.py:1845-1846](src/polymarket/executor.py#L1845-L1846)

`cancel_stale_limit_bids` explicitly skips `dry_run` bets. But `add_bet` creates open ledger entries for dry runs. These are never cleaned up, and the duplicate check at line 1342 prevents new dry-run bets on markets that had prior dry-run bets.

---

### MAJ-18: `max_drawdown` formula in `BankrollManager.get_stats()` is incorrect
**File:** [bankroll.py:275](src/strategy/bankroll.py#L275)

```python
"max_drawdown": 1 - (min(b["bankroll_before"] for b in settled) / self.peak_bankroll)
```

This computes `1 - (global_min / global_peak)`, but the minimum could occur before the peak. True max drawdown is the largest peak-to-trough decline, requiring sequential tracking.

---

### MAJ-19: Fuzzy fighter matching in fallback scrapers can return wrong fighter
**File:** [fallback_scrapers.py:163-178](src/data/fallback_scrapers.py#L163-L178)

The Sherdog search scoring allows matching on last name alone (`score >= 5`). If two fighters share a surname, this returns the wrong fighter's stats, which feeds into model features.

---

### MAJ-20: Proxy URL credentials potentially logged
**File:** [client.py:111](src/polymarket/client.py#L111)

```python
logger.info(f"CLOB proxy enabled: {clob_proxy.split('@')[-1]}")
```

If the proxy URL contains credentials in a format without `@`, the entire URL (including credentials) is logged.

---

### MAJ-21: PNL history file and line history grow unboundedly
**File:** [tracker.py:531](src/polymarket/tracker.py#L531) — `pnl_history.jsonl` appended every 30s with no rotation.
**File:** [line_tracker.py:316-370](src/data/line_tracker.py#L316-L370) — `load_line_history` reads ALL CSV snapshots on every call.

Both degrade performance over weeks of operation.

---

## 4. Minor Issues

Code quality, maintainability, and minor correctness concerns.

### MIN-01: HTTP URLs for UFCStats (MITM risk for training data)
**File:** [config.py:33-36](src/config.py#L33-L36) — All UFCStats URLs use `http://` instead of `https://`.

### MIN-02: API keys in query parameters appear in error tracebacks
**Files:** [odds_client.py:41](src/data/odds_client.py#L41), [method_odds.py:668](src/data/method_odds.py#L668), [betsapi_mma.py:318](src/data/betsapi_mma.py#L318) — API keys passed as query params are visible in any logged URL or exception traceback.

### MIN-03: Critical API keys default to empty string silently
**File:** [config.py:45-57](src/config.py#L45-L57) — `ODDS_API_KEY`, `BETSAPI_TOKEN`, `POLYMARKET_PRIVATE_KEY` all default to `""` with no startup warning. This leads to confusing runtime errors instead of clear "missing API key" errors.

### MIN-04: Multiple bare `except Exception: pass` patterns
**Files:** [bot.py:665-666](src/bot.py#L665-L666) (injury detection), [app.py:139-140](src/web/app.py#L139-L140) (log reading), [app.py:340-347](src/web/app.py#L340-L347) (balance fetch), [bankroll.py:59-63](src/strategy/bankroll.py#L59-L63) (ledger sync) — Exceptions swallowed silently, hiding bugs.

### MIN-05: Unbounded in-memory caches
**Files:** [fighter_lookup.py:40-43](src/data/fighter_lookup.py#L40-L43) (4 caches), [fallback_scrapers.py:35](src/data/fallback_scrapers.py#L35), [method_odds.py:55](src/data/method_odds.py#L55) — Module-level caches grow indefinitely in long-running processes.

### MIN-06: Google scraping for weigh-in data always fails
**File:** [live_monitor.py:297-313](src/data/live_monitor.py#L297-L313) — Queries Google directly, which blocks automated requests. This code path never produces data.

### MIN-07: Tennis player name matching too loose (first-initial fallback)
**File:** [tennis_bookmaker_audit.py:86-107](src/data/tennis_bookmaker_audit.py#L86-L107) — Final fallback matches any two players sharing the same surname and first initial. "Novak Djokovic" would match "Natalija Djokovic".

### MIN-08: No CSRF protection on mutation endpoints
**File:** [app.py:245,296,307](src/web/app.py#L245) — POST endpoints have no CSRF tokens or custom header requirements.

### MIN-09: Timestamp naive/UTC mismatch across modules
**Files:** [line_tracker.py:90](src/data/line_tracker.py#L90), [monitor.py:256](src/polymarket/monitor.py#L256), [tracker.py:302](src/polymarket/tracker.py#L302) — Mix of `datetime.now()` (naive local time) and UTC. Causes incorrect TTL calculations if local timezone is not UTC.

### MIN-10: Missed weight detection uses substring matching
**File:** [live_monitor.py:388-392](src/data/live_monitor.py#L388-L392) — `if fighter in fa` matches "Lee" in "O'Malley". Should use `name_utils.same_person_name`.

### MIN-11: Hardcoded referee tendency scores are stale and unvalidated
**File:** [live_monitor.py:418-440](src/data/live_monitor.py#L418-L440) — Arbitrary floats with no data backing. Includes inactive referees (Yamasaki, Mazzagatti). These scores influence real betting decisions.

### MIN-12: `_parse_pct` treats 1.01 as a percentage value
**File:** [kaggle_loader.py:75-86](src/data/kaggle_loader.py#L75-L86) — Values <= 1.0 are multiplied by 100; values > 1.0 are kept as-is. The boundary is ambiguous for values near 1.0.

### MIN-13: Experimental features always included despite "experimental" label
**File:** [build_features.py:642](src/features/build_features.py#L642) — `add_experimental_features()` is called unconditionally in `build_features()`.

### MIN-14: Stance encoding uses `-1` for unknown (bad for tree models)
**File:** [build_features.py:497-498](src/features/build_features.py#L497-L498) — Unknown stances get `-1`, which XGBoost treats as a meaningful numeric value rather than missing.

### MIN-15: `days_since_last_fight` default of 365 for first fights is arbitrary
**File:** [build_features.py:184](src/features/build_features.py#L184) — Exactly at the boundary of `cage_rust` triggering.

### MIN-16: Model .pkl files have no version metadata
**Files:** [train.py:439-441](src/model/train.py#L439-L441), [tennis_model.py:498](src/model/tennis_model.py#L498) — No version of scikit-learn/XGBoost stored with models. Version changes can cause silent behavior changes on load.

### MIN-17: `bet_id` derived from list length creates potential ID collisions
**File:** [tracker.py:302](src/polymarket/tracker.py#L302) — If bets are manually deleted or the ledger rebuilt, IDs can collide. Merged-ledger ID remapping (line 489) means a bet's ID changes depending on which ledgers exist.

### MIN-18: Promotion gate can be gamed by increasing bet volume
**File:** [promotion_gate.py:423-438](src/strategy/promotion_gate.py#L423-L438) — Passes if either ROI **or** total profit exceeds baseline.

### MIN-19: `.env.example` missing `BETSAPI_TOKEN`
**File:** [.env.example](.env.example) — Developers won't know this variable exists.

### MIN-20: Race condition in CSV/JSON file writes across the data layer
**Files:** [historical_backfill.py:328](src/data/historical_backfill.py#L328), [line_tracker.py:124-125](src/data/line_tracker.py#L124-L125), [monitor.py:270-277](src/polymarket/monitor.py#L270-L277) — No file locking or atomic write pattern. Concurrent processes can corrupt files.

### MIN-21: Three inconsistent name normalization functions
**Files:** [name_utils.py](src/data/name_utils.py), [ufc_refresh.py:844-847](src/data/ufc_refresh.py#L844-L847), [betsapi_mma.py](src/data/betsapi_mma.py) — Three different normalization approaches (NFKD + `\w` filter, casefold only, NFKD + `[a-z0-9]` filter). If a fighter name is normalized with one function for key generation and another for lookup, the match fails silently. Accented names like "Jiri Prochazka" will not match across normalizers.

### MIN-22: Prediction-time feature completeness not validated
**File:** [predict.py:17-24](src/model/predict.py#L17-L24) — `_ordered_feature_frame` silently fills missing columns with NaN. Combined with native_nan imputation, predictions are generated even when critical features (like `a_elo` or `a_implied_prob`) are completely absent. No warning is logged.

### MIN-23: Odds noise uses a fixed seed, not true regularization
**File:** [train.py:147](src/model/train.py#L147) — `_add_odds_noise` defaults to `RandomState(42)`. Every training run applies identical noise, making it a deterministic feature transformation rather than true regularization. The noise std (0.04) is unvalidated.

### MIN-24: Geoblock detection logs but doesn't block order execution
**File:** [client.py:230-250](src/polymarket/client.py#L230-L250) — `_log_geoblock_status` only logs a warning when `blocked=True`. Orders proceed regardless.

### MIN-25: `_passes_underdog_filters` bypasses quality filters
**File:** [value.py:275-277](src/strategy/value.py#L275-L277) — This lambda passes only 4 args to `_passes_filters`, skipping no-odds agreement, fighter experience, and line movement filters.

---

## 5. Nitpicks

Style, naming, and small improvements.

### NIT-01: `_clean_text` duplicated in three files
[fallback_scrapers.py:50](src/data/fallback_scrapers.py#L50), [fighter_lookup.py:133](src/data/fighter_lookup.py#L133), [scraper.py:39](src/data/scraper.py#L39).

### NIT-02: `_safe_float` has four different implementations
[fallback_scrapers.py:54](src/data/fallback_scrapers.py#L54), [fighter_lookup.py:405](src/data/fighter_lookup.py#L405), [kaggle_loader.py:89](src/data/kaggle_loader.py#L89), [betsapi_mma.py:100](src/data/betsapi_mma.py#L100) — Subtly different behaviors (some return `None`, others `np.nan`; some handle `"N/A"`, some don't).

### NIT-03: f-string logging defeats lazy evaluation
Nearly all files use `logger.info(f"...")` instead of `logger.info("...", arg)`. The f-string is evaluated even when the log level is disabled.

### NIT-04: Importing private functions across modules
[bot.py:605](src/bot.py#L605), [app.py:24](src/web/app.py#L24), [executor.py:38](src/polymarket/executor.py#L38) — Creates coupling to implementation details.

### NIT-05: `.gitattributes` only covers `.sh` and `Dockerfile`
Python, HTML, CSV, and JSON files can still get CRLF on Windows, causing diff noise.

### NIT-06: `_parse_json_field` duplicated in `markets.py` and `tennis_markets.py`
Identical utility function should be shared.

### NIT-07: Tapology scraper is dead code
[fallback_scrapers.py:347-368](src/data/fallback_scrapers.py#L347-L368) — Three functions that always return `None` or `[]`.

### NIT-08: `scrape_all_fighter_urls` returns `list(set(...))` — non-deterministic order
[scraper.py:211](src/data/scraper.py#L211).

### NIT-09: Inconsistent predicted_winner threshold (UFC `> 0.5`, tennis `>= 0.5`)
[predict.py:149](src/model/predict.py#L149) vs [tennis_model.py:208](src/model/tennis_model.py#L208).

### NIT-10: `POLYMARKET_CHAIN_ID` hardcoded to 137 (Polygon mainnet), no testnet option
[config.py:58](src/config.py#L58).

### NIT-11: `use_label_encoder=False` deprecated in newer XGBoost
Multiple files still pass this parameter, generating warnings in XGBoost >= 1.6.

### NIT-12: `_method_group` doesn't handle DQ or Overturned results
[build_features.py:275-285](src/features/build_features.py#L275-L285) — Win method rates won't sum to 1.0 for fighters with DQ wins.

### NIT-13: `_FRESH_DATA_CUTOFF` in promotion_gate.py is hardcoded to a specific date
[promotion_gate.py:61](src/strategy/promotion_gate.py#L61) — `_FRESH_DATA_CUTOFF = "2024-12-14"` grows stale over time.

### NIT-14: `positions_cache` in monitor.py is initialized but never used
[monitor.py:42](src/polymarket/monitor.py#L42) — Dead code.

---

## 6. Positive Observations

### Quality gates are genuinely rigorous
The selection gate, promotion gate, and preflight audit form a meaningful barrier to deploying bad models. Walk-forward backtesting with time-series splits shows proper methodology.

### Excellent concurrency testing
`test_executor_duplicates.py` uses real threading with slow mock CLOB clients to test race conditions. Cross-ledger duplicate prevention is well-tested.

### Good XSS hygiene
The web dashboard consistently uses an `esc()` function for user-facing data via DOM `textContent` escaping. No Jinja SSTI risk since no user-supplied data is passed to template context.

### Feature provenance tracking
The `feature_provenance.py` module with JSON audit trails is a strong practice for ML systems. The spec-based training pipeline (`training_spec.py`) provides reproducibility.

### Ledger mutation isolation
Tests verify that returned bet dicts are detached copies — mutations don't affect the ledger.

### Tennis and UFC models are properly separated
No cross-contamination between the two model domains. Each has its own feature builder, model type, and evaluation pipeline.

### Robust name matching utilities
`name_utils.py` handles a wide range of name normalization cases (diacritics, suffixes, abbreviations).

### Safety guard against live tennis trading
`test_bot_tennis_live.py` verifies that real-money tennis trading is explicitly blocked.

### Comprehensive test suite
40+ test files with meaningful assertions. No vacuous tests found. Proper use of mocking, `tmp_path`, and `pytest.approx`. Tests cover edge cases including concurrent execution, stale cache, partial fills, and cross-ledger coordination.

### Atomic ledger writes
`tracker.py` uses `tempfile.mkstemp` + `os.replace` for ledger saves — properly atomic on most filesystems. (Though Windows `os.replace` has caveats — see MAJ-09.)

---

## 7. Recommended Priority Actions

Ranked by risk (money loss, data corruption, security exposure):

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| **P0** | CRIT-01 | Fix `get_cash_balance` heuristic — this can cause 1000x over/under-betting | 1 hour |
| **P0** | CRIT-02 | Create `.dockerignore` to prevent `.env` from leaking into images | 15 min |
| **P0** | CRIT-04 | Add write-ahead log for order placement — prevent lost/duplicate orders | 4 hours |
| **P0** | CRIT-05 | Add authentication to web dashboard mutation endpoints | 2 hours |
| **P0** | CRIT-06 | Fix `_name_match` substring bug — can bet on wrong fighter | 30 min |
| **P0** | CRIT-07 | Add positive-EV check to conviction trader | 30 min |
| **P0** | CRIT-11 | Add bankroll floor + stop-loss check in `place_bet()` | 30 min |
| **P0** | CRIT-14 | Add max event and total exposure limits | 1 hour |
| **P1** | MAJ-01 | Enable `use_independent_blend_b` as production default | 30 min |
| **P1** | CRIT-12 | Verify order fill before debiting bankroll | 2 hours |
| **P1** | MAJ-16 | Validate `side` parameter against enum | 15 min |
| **P1** | MIN-24 | Make geoblock detection actually block orders | 30 min |
| **P1** | CRIT-10 | Clean up duplicate definitions in `build_features.py` | 1 hour |
| **P1** | CRIT-08 | Add `as_of_date` to scraped-data feature functions | 2 hours |
| **P1** | CRIT-09 | Add backup-before-write to backfill scripts | 1 hour |
| **P1** | MAJ-03 | Add circuit breaker for consecutive API failures | 2 hours |
| **P2** | MAJ-07 | Generate requirements lockfile | 15 min |
| **P2** | MAJ-09 | Add timeout to file lock acquisition | 30 min |
| **P2** | MAJ-02 | Fix double-counting of open bets in bankroll init | 1 hour |
| **P2** | MAJ-11 | Add holdout validation to sweep, reduce parameter grid | 2 hours |
| **P2** | MAJ-13 | Fix SHAP feature selection: use production hyperparams, move inside CV | 2 hours |
| **P2** | CRIT-15 | Remove .pkl from git tracking, add to .gitignore | 30 min |
| **P3** | MAJ-04/05 | Fix thread safety issues (global config, monkey-patching) | 3 hours |
| **P3** | MIN-04 | Replace bare `except: pass` with logged exceptions | 1 hour |
| **P3** | MAJ-15 | Update USDC contract address | 15 min |
| **P3** | CRIT-13 | Rename `test_bet.py`, add confirmation prompt | 15 min |
| **P3** | MIN-21 | Consolidate name normalization into one canonical function | 2 hours |

---

## Appendix: Findings by Severity

| Severity | Count |
|----------|-------|
| Critical | 15 |
| Major | 21 |
| Minor | 25 |
| Nitpick | 14 |
| **Total** | **75** |

---

*Generated by Claude Opus 4.6 (1M context) — full-codebase automated audit*
*Files audited: 100+ source files, 35 test files, all config/deployment/script files*
*Method: Five parallel audit agents, each reading every file in their domain, with cross-referenced and deduplicated findings*
