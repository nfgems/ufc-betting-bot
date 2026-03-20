# UFC Betting Bot — Comprehensive Codebase Audit Report

**Date:** 2026-03-19
**Auditor:** Claude Opus 4.6 (automated, every file read)
**Scope:** Full codebase — src/, scripts/, tests/, config, deployment, model specs

---

## 1. Executive Summary

This codebase is a real-money UFC (and tennis) betting bot that scrapes data, trains XGBoost models, computes edges against Polymarket prices, and places live orders. The code is ambitious, well-structured in many areas, and shows genuine care for model evaluation rigor (promotion gates, control arms, walk-forward backtests). However, **several critical issues could cause direct money loss, data corruption, or security exposure in production**:

1. **The model's probability calibration is broken** — `CalibratedClassifierCV` is fitted on the same data XGBoost already trained on, producing overfit calibration that directly corrupts Kelly bet sizing.
2. **The web dashboard exposes all financial data without authentication** — wallet balance, positions, PnL, model probabilities, and even the server's IP/proxy config are readable by anyone who can reach the port.
3. **No crash recovery or concurrent-execution guard** — if the bot crashes mid-order or two instances run simultaneously, duplicate bets or orphaned orders result.
4. **Pickle model files in git** — arbitrary code execution risk via tampered `.pkl` files.
5. **Fuzzy name matching at dangerously low thresholds (0.70–0.75)** in data backfill scripts can silently assign wrong odds to wrong fights, corrupting training data.

The codebase is not unsafe to run in a limited capacity, but it needs hardening before scaling up real-money exposure.

---

## 2. Critical Issues

### C1. Probability Calibration Fitted on Training Data (Money Loss Risk)
- **File:** [train.py:306-312](src/model/train.py#L306-L312), [train_experimental.py:174-180](src/model/train_experimental.py#L174-L180)
- **What:** When `calibrate=True`, `CalibratedClassifierCV` is created and fitted on the **same** `X_train, y_train` that XGBoost was already trained on. The XGBoost model has already memorized this data, so the cross-validation splits inside `CalibratedClassifierCV` see optimistically biased predictions.
- **Why it matters:** Overfit calibration means the model's probability estimates are unreliable. Since Kelly criterion bet sizing is directly proportional to `(model_prob - market_prob)`, overconfident probabilities → oversized bets → money loss.
- **Fix:** Either (a) pass an **unfitted** XGBoost estimator to `CalibratedClassifierCV` so it trains from scratch with proper CV, or (b) hold out a separate calibration set that XGBoost never saw. The current `xgb.fit(X_train)` → `CalibratedClassifierCV(xgb).fit(X_train)` pattern defeats the purpose.

### C2. Calibration Data Leakage in Temporal Holdout Mode
- **File:** [model_variants.py:200-207](src/strategy/model_variants.py#L200-L207)
- **What:** In the `temporal_holdout` calibration path, `CalibratedClassifierCV` is called with `cv=[(cal_indices, cal_indices)]` — train and test indices are **identical**. Isotonic calibration is non-parametric and can memorize this.
- **Why it matters:** Same effect as C1 — calibration appears good in backtest but fails in production.
- **Fix:** Use `cv="prefit"` with `CalibratedClassifierCV(FrozenEstimator(xgb), cv="prefit")` and fit on the calibration set directly.

### C3. Web Dashboard Exposes All Financial Data Without Authentication
- **File:** [app.py:337-551](src/web/app.py#L337-L551)
- **What:** `/api/summary`, `/api/bets`, `/api/positions`, `/api/balance`, `/api/geoblock-status` and all other read endpoints have **zero authentication**. Only mutation endpoints (settle, close) require a token.
- **Why it matters:** Anyone who discovers the host+port can see: wallet balance, open positions, realized PnL, bet history, fighter names, model probabilities, edges, and even the server's IP address and proxy configuration (`/api/geoblock-status`).
- **Fix:** Add a `_require_read_auth()` check on all `/api/*` endpoints. At minimum gate `/api/balance` and `/api/geoblock-status`.

### C4. Default Bind to 0.0.0.0 (Public) + No Read Auth
- **File:** [serve.py:175](src/web/serve.py#L175)
- **What:** `host = os.environ.get("WEB_HOST", "0.0.0.0")` defaults to public bind. Combined with C3, deploying without setting `WEB_HOST` exposes everything.
- **Fix:** Default to `127.0.0.1`. Require explicit opt-in to public bind.

### C5. Pickle Files Tracked in Git — Arbitrary Code Execution
- **Files:** `models/*.pkl`, `models/backups/**/*.pkl`, `models/candidates/**/*.pkl`
- **What:** Python pickle files execute arbitrary code on load. These are tracked in git. Anyone who can submit a PR or tamper with the repo can embed a payload that runs when `joblib.load()` is called.
- **Why it matters:** A malicious pickle could exfiltrate `POLYMARKET_PRIVATE_KEY`, drain funds, or install a backdoor.
- **Fix:** Add `*.pkl` to `.gitignore`, remove from git history with `git filter-repo`. Store models in a separate artifact store. Consider switching to XGBoost's native `save_model()`/`load_model()` format.

### C6. No Crash Recovery or Concurrent Execution Guard
- **File:** [bot.py:1374-1783](src/bot.py#L1374-L1783)
- **What:** `cmd_duo_live` (the real-money entrypoint) has no top-level try/finally, no signal handler (SIGTERM/SIGINT), no cleanup logic, and no file lock preventing two instances from running simultaneously.
- **Why it matters:** If the bot crashes mid-order, orders are orphaned. If Railway redeploys overlap, two instances both read the same ledger, both see "no existing bet," and both place bets — doubling exposure.
- **Fix:** Add a process lock (file lock or PID file) at startup. Register SIGTERM handler that cancels pending orders. Add startup reconciliation that checks for resting orders from a previous run.

### C7. Conviction Bets Skip Positive-EV Check (Negative-Edge Bets)
- **File:** [triple_trader_backtest.py:482-493](src/strategy/triple_trader_backtest.py#L482-L493)
- **What:** The conviction trader only checks `model_prob >= threshold`. It does NOT verify `blended_prob > market_prob` (positive expected value). A `conviction_ev_fix` variant exists in `model_variants.py` specifically to address this, confirming it's a known issue.
- **Why it matters:** A conviction bet could be placed where the model says 75% but the market says 80%. This is a negative-EV bet that loses money in expectation.
- **Fix:** Add `blended_prob > market_prob` as a required condition for conviction bets.

### C8. Market Orders Without Fill-Price Verification
- **File:** [executor.py](src/polymarket/executor.py) (market order fallback path)
- **What:** When a limit order fails and the system falls through to a market order (FOK), the fill price is not checked against the expected price/edge threshold. In thin Polymarket markets, slippage can be substantial.
- **Why it matters:** A $50 bet at expected 2.5x odds that fills at 2.1x erases the edge entirely.
- **Fix:** After market order fills, verify actual fill price is within an acceptable slippage tolerance. Add a `max_slippage` parameter.

### C9. Backfill Scripts Use Dangerously Low Fuzzy Match Thresholds
- **Files:** [backfill_odds_api.py:87](scripts/backfill_odds_api.py#L87) (threshold 0.70), [merge_kaggle_data.py:90](scripts/merge_kaggle_data.py#L90) (threshold 0.75)
- **What:** These scripts fuzzy-match fighter names at thresholds where "Conor McGregor" could match "Cian McGregor." They then **overwrite** the master CSV in-place.
- **Why it matters:** Wrong odds assigned to wrong fights → corrupted training data → wrong model → wrong bets → money loss.
- **Fix:** Raise thresholds to ≥0.85. Add a dry-run/preview mode. Write to a separate output file, not the master.

### C10. `manual_test_bet.py` Has No Bet-Size Cap or Dry-Run Mode
- **File:** [manual_test_bet.py](scripts/manual_test_bet.py)
- **What:** No maximum order size guard. If `price_yes` is very small (close to 0), `shares = 1.0 / price` produces an extremely large order. The only safeguard is a "YES" confirmation prompt. No environment check (prod vs. test).
- **Fix:** Add `--dry-run`, cap maximum order size, validate price range (0.01–0.99), check for staging environment.

---

## 3. Major Issues

### M1. Unpinned Dependencies — Supply Chain Risk
- **File:** [requirements.txt](requirements.txt)
- **What:** All 17 dependencies use `>=` with no upper bound. A compromised release of `py-clob-client` or `requests` would be automatically pulled into the next Docker build.
- **Fix:** Pin exact versions (`==`) or use `pip-compile` with a lockfile.

### M2. Secrets Default to Empty String Instead of None
- **File:** [config.py:48-60](src/config.py#L48-L60)
- **What:** `POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")`. An empty key could cause a crypto library to behave unexpectedly rather than failing fast.
- **Fix:** Default to `None`. Validate at the point of use.

### M3. No Bankroll Sync with Actual Wallet Balance
- **File:** [config.py:89](src/config.py#L89)
- **What:** `INITIAL_BANKROLL = 500.00` is static. Kelly sizing derives from this, not from actual wallet balance. If the wallet has been partially drained, bets are oversized.
- **Fix:** Query actual balance before sizing: `min(INITIAL_BANKROLL, actual_balance)`.

### M4. Unbounded Log File Growth
- **File:** [bot.py:65](src/bot.py#L65)
- **What:** `FileHandler` appends indefinitely. On a long-running Railway deployment, `bot.log` grows without limit, eventually filling the volume and blocking ledger writes.
- **Fix:** Switch to `RotatingFileHandler` with `maxBytes` and `backupCount`.

### M5. Race Condition on Opening Lines JSON
- **File:** [line_tracker.py:114-173](src/data/line_tracker.py#L114-L173)
- **What:** `_load_opening_lines()` → modify → `_save_opening_lines()` with no file locking. Concurrent monitoring passes can overwrite each other's data.
- **Fix:** Use atomic write (write to temp file, rename) or file locking.

### M6. Rankings Scraper Conflates Men's and Women's Divisions
- **File:** [rankings_scraper.py:45-59](src/data/rankings_scraper.py#L45-L59)
- **What:** `_WC_ALIASES` maps "women's strawweight" → "strawweight", merging women's rankings into men's divisions.
- **Fix:** Keep separate canonical keys (e.g., "womens_strawweight").

### M7. Unbounded Memory Growth in Module-Level Caches
- **Files:** [fighter_lookup.py:41-44](src/data/fighter_lookup.py#L41-L44), [method_odds.py:55](src/data/method_odds.py#L55), [line_tracker.py](src/data/line_tracker.py), [fallback_scrapers.py](src/data/fallback_scrapers.py)
- **What:** Multiple unbounded `dict` caches at module level. In long-running sessions, these grow without limit.
- **Fix:** Use `functools.lru_cache` with a max size, or add TTL-based eviction.

### M8. Stale Feature Cache After Retraining
- **File:** [fighter_lookup.py:262-287](src/data/fighter_lookup.py#L262-L287)
- **What:** `_load_processed_feature_history` caches the features.csv DataFrame indefinitely. If the file is updated after retraining, stale data is served until `clear_cache()` is explicitly called.
- **Fix:** Check file mtime against cached version.

### M9. Blend Probability for Fighter B Uses Complement Instead of Independent Calculation
- **File:** [value.py](src/strategy/value.py)
- **What:** `blend_b = 1.0 - blend_a` instead of computing an independent dynamic blend weight for fighter B. A `blend_b_fix` variant exists in `model_variants.py` confirming this is a known issue.
- **Fix:** Compute `dyn_weight_b` independently and use it for `blend_b`.

### M10. No Absolute Dollar Cap on Bet Size
- **File:** [bankroll.py](src/strategy/bankroll.py)
- **What:** `BankrollManager` uses `MAX_BET_FRACTION` but no absolute cap. As bankroll grows, bet sizes grow without limit, risking market impact on thin Polymarket markets.
- **Fix:** Add `MAX_BET_ABSOLUTE` (e.g., $200) that caps regardless of bankroll percentage.

### M11. Historical Odds Feature Leakage Risk (Closing vs. Opening)
- **File:** [build_features.py:527-545](src/features/build_features.py#L527-L545)
- **What:** Implied probability features derived from odds columns. If these are closing odds (which incorporate outcome information), training on them inflates apparent accuracy. The `_add_odds_noise` mitigation (Gaussian noise std=0.04) is approximate, not principled.
- **Fix:** Clearly separate opening from closing odds in the pipeline. Train only on opening/pre-fight odds.

### M12. Weigh-In Logic: Weight Allowance Applied to Title Fights
- **File:** [live_monitor.py:409](src/data/live_monitor.py#L409)
- **What:** `effective_limit = limit + ALLOWANCE` is always applied, even for title fights. The comment says title fights get no 1lb allowance, but the code doesn't check `is_title_bout`.
- **Fix:** Check `fight.get("is_title_bout", False)` before adding allowance.

### M13. Fighter Name Matching in Weigh-In Uses Unsafe Substring Match
- **File:** [live_monitor.py:408](src/data/live_monitor.py#L408)
- **What:** `if fighter in fa or fa in fighter` — "Islam" matches any fighter with "Islam" in their name.
- **Fix:** Use `same_person_name()` from `name_utils.py`.

### M14. No Retry Logic on HTTP Requests (Except BetsAPI)
- **Files:** [scraper.py](src/data/scraper.py), [fighter_lookup.py](src/data/fighter_lookup.py), [method_odds.py](src/data/method_odds.py), [rankings_scraper.py](src/data/rankings_scraper.py)
- **What:** All `requests.get()` calls use `raise_for_status()` with no retry. A single transient network error causes the entire prediction pipeline to fail.
- **Fix:** Add `HTTPAdapter` with `urllib3.util.Retry`.

### M15. Ledger Recovery Writes Synthetic Bets with model_prob=0.0
- **File:** [app.py:1625-1627](src/web/app.py#L1625-L1627)
- **What:** CLOB recovery creates ledger entries with `model_prob=0.0`, `edge=0.0`. These pollute downstream analytics.
- **Fix:** Tag with `source="clob_recovery"` and filter in analytics.

### M16. V5 Spec Silently Drops 6 Ranking Features From V2
- **Files:** [full_live_contract_v5_fullfit_spec.json](models/full_live_contract_v5_fullfit_spec.json) vs [full_live_contract_v2_spec.json](models/full_live_contract_v2_spec.json)
- **What:** V2 includes `a_wc_rank_feat`, `b_wc_rank_feat`, `diff_wc_rank`, `a_pfp_rank_feat`, `b_pfp_rank_feat`, `diff_pfp_rank`. V5 (production) drops all of them with no documented rationale.
- **Fix:** If intentional, document. If accidental, re-add.

### M17. `merge_kaggle_data.py` Destructively Clears Post-2025 Data
- **File:** [merge_kaggle_data.py:274-280](scripts/merge_kaggle_data.py#L274-L280)
- **What:** Unconditionally sets odds and rank columns to NaN for all rows after 2025-01-01 before re-merging. Running the script multiple times wipes previously correct data.
- **Fix:** Track provenance of filled columns or use a staging file.

### M18. `scrape_bfo_moneyline.py` Last-Name-Only Matching
- **File:** [scrape_bfo_moneyline.py:292-303](scripts/scrape_bfo_moneyline.py#L292-L303)
- **What:** `_names_match` returns True if only last names match. Multiple fighters share last names (Silva, Johnson, Costa).
- **Fix:** Require at least first initial to also match.

### M19. WTA Winner Inference Uses Undocumented Modulo Arithmetic
- **File:** [tennis_data.py:422-424](src/data/tennis_data.py#L422-L424)
- **What:** `int(winner_flag) % 2 == 0` to decide winner side. This is an undocumented assumption about the WTA API.
- **Fix:** Use an explicit mapping with known values, log warnings for unknown values.

### M20. Tennis Target Label Uses Raw vs. Normalized Name Comparison
- **File:** [tennis_data.py:1328](src/data/tennis_data.py#L1328)
- **What:** `"target": int(player_a == row.get("winner_name"))` — `player_a` has been normalized but `winner_name` is raw. Accents or whitespace differences cause silent mislabeling.
- **Fix:** Compare using `normalize_player_name()` on both sides.

### M21. No Test for Ledger Recovery Path
- **File:** [app.py:1561-1637](src/web/app.py#L1561-L1637)
- **What:** `_recover_ledger_from_clob` creates real financial ledger entries. Line 1628: `1.0/price` will raise `ZeroDivisionError` if `price=0`. Zero tests exist for this function.
- **Fix:** Write unit tests covering normal recovery, `price=0`, missing `asset_id`, partial fills, and idempotency.

### M22. CI Only Runs 4 of ~40 Test Files
- **File:** [ci.yml:25-30](.github/workflows/ci.yml#L25-L30)
- **What:** CI explicitly lists only 4 test files. Regressions in strategy, trading, and polymarket modules are never caught.
- **Fix:** Run `pytest tests/` or mark integration tests with `@pytest.mark.integration`.

---

## 4. Minor Issues

### m1. `config.py` Creates Directories on Import
- **File:** [config.py:21-30](src/config.py#L21-L30)
- Importing `src.config` anywhere (tests, linters) creates 7+ directories as a side effect. Move to an explicit `ensure_dirs()` function.

### m2. `bot.py` Log File Handler Crashes on Missing Directory
- **File:** [bot.py:65](src/bot.py#L65)
- `logging.FileHandler(LOGS_DIR / "bot.log")` runs at import time. If `LOGS_DIR` doesn't exist, the entire app crashes including health checks.

### m3. `sys.path.insert` Hack in bot.py
- **File:** [bot.py:46](src/bot.py#L46)
- Fragile path manipulation. Use proper package installation (`pip install -e .`).

### m4. Exception Swallowing in Injury Detection
- **File:** [bot.py:667-668, 1557-1558](src/bot.py#L667-L668)
- `except Exception: pass` in injury detection means a fighter who was pulled could go undetected, causing bets on cancelled fights.

### m5. `_now_iso()` Uses Local Time, Not UTC
- **Files:** [method_odds.py:58-59](src/data/method_odds.py#L58-L59), [rankings_scraper.py:62-63](src/data/rankings_scraper.py#L62-L63), [line_tracker.py:237-238](src/data/line_tracker.py#L237-L238)
- Local time vs UTC comparisons cause stale-check errors.

### m6. Google Scraping for Weigh-In Data Will Almost Always Fail
- **File:** [live_monitor.py:313-331](src/data/live_monitor.py#L313-L331)
- `scrape_weighin_results` sends direct requests to Google Search, which blocks automated access. Function silently returns empty.

### m7. Silent Failure in `/api/summary` and `_compute_balance`
- **File:** [app.py:357-358, 491-498](src/web/app.py#L357-L358)
- `except Exception: pass` when fetching positions/balance. Dashboard silently shows $0 when the API call fails.

### m8. No Rate Limiting on Web Endpoints
- **File:** [app.py](src/web/app.py)
- An attacker can hammer `/api/balance` to exhaust CLOB API rate limits, causing the trading bot to fail.

### m9. `_parse_pct` Ambiguity Between 0-1 and 0-100 Ranges
- **File:** [kaggle_loader.py:83-94](src/data/kaggle_loader.py#L83-L94)
- Values ≤1.0 are multiplied by 100. A legitimate 0.45% accuracy becomes 45%.

### m10. EWM Rolling Stats Use min_periods=1
- **File:** [build_features.py:161-164](src/features/build_features.py#L161-L164)
- After a fighter's first fight, the rolling average is based on a single data point. Very noisy.

### m11. `fillna(0)` for Experimental Features Injects Fake Data
- **File:** [experimental_features.py:37, 63-64](src/features/experimental_features.py#L37)
- Unknown striking stats → 0 (very passive), unknown control time → 60s (above average). Should be NaN.

### m12. Feature Importance Mismatch with Indicator Columns
- **File:** [train_experimental.py:185](src/model/train_experimental.py#L185)
- `zip(feature_cols, xgb.feature_importances_)` silently drops indicator column importances when `impute_with_indicators=True`.

### m13. Walk-Forward Comparison Uses random_5fold for Baseline
- **File:** [compare.py:199](src/model/compare.py#L199)
- Random K-fold calibration on time-series data introduces temporal leakage in the baseline, making comparisons unfair.

### m14. BETSAPI Rate Limit Default is 0 (No Delay)
- **File:** [config.py:55](src/config.py#L55)
- `BETSAPI_REQUEST_MIN_INTERVAL_SECONDS` defaults to 0. Could lead to IP bans.

### m15. Hardcoded Referee Tendencies Include Retired Referees
- **File:** [live_monitor.py:435-457](src/data/live_monitor.py#L435-L457)
- Mario Yamasaki included (retired 2018). No update mechanism.

### m16. Settle Endpoint Silently Treats Invalid Results as Losses
- **File:** [app.py:461](src/web/app.py#L461)
- A typo in the result string (e.g., "winn") silently marks the bet as lost. No rejection of unrecognized values.

### m17. Weak Tennis Player Name Matching
- **Files:** [tennis_data.py:393-410](src/data/tennis_data.py#L393-L410), [tennis_bookmaker_audit.py:100-107](src/data/tennis_bookmaker_audit.py#L100-L107)
- Same surname + same first initial = match. "C. Gauff" matches "C. Garcia."

### m18. `entrypoint.sh` Missing `set -euo pipefail`
- **File:** [entrypoint.sh](entrypoint.sh)
- Silent migration failures mean the bot starts with missing ledger data.

### m19. No API Response Schema Validation
- **Files:** [odds_client.py](src/data/odds_client.py), [method_odds.py](src/data/method_odds.py), [betsapi_mma.py](src/data/betsapi_mma.py)
- API responses consumed with `.get()` calls, never validated. A schema change produces silently wrong data.

### m20. Odds Validation Missing — Division Errors Possible
- **File:** [odds_client.py:182-192](src/data/odds_client.py#L182-L192)
- No check that decimal odds > 1.0 before computing `1.0 / odds`. Odds of exactly 1.0 or negative values produce garbage.

### m21. Railway Restart Policy with No Alerting
- **File:** [railway.toml:6](railway.toml#L6)
- After 5 restart failures, Railway stops the service silently. No notification that the bot has stopped.

### m22. `use_label_encoder=False` Deprecated
- **Files:** [train.py:299](src/model/train.py#L299), [train_experimental.py:164](src/model/train_experimental.py#L164), [feature_selection.py:68](src/model/feature_selection.py#L68), [hyperparam_search.py:117](src/model/hyperparam_search.py#L117)
- Deprecated since XGBoost 1.6.0, generates warnings.

### m23. Vacuous Tests (Always Pass)
- **Files:** [test_phase2_schema_contract.py:1282-1320, 1323-1375, 1446-1486](tests/test_phase2_schema_contract.py#L1282-L1320)
- Three tests (`test_cmd_predict_skips_fights_without_live_event_context`, `test_cmd_predict_skips_fights_with_blank_weight_class_in_matched_context`, `test_cmd_duo_live_skips_fights_without_live_event_context`) have **no assertions**. They pass regardless of whether the feature works.

### m24. Audit Scripts Default to Stale V2 Spec
- **Files:** [audit_model_feature_nulls.py:401](scripts/audit_model_feature_nulls.py#L401), [audit_feature_provenance.py:63](scripts/audit_feature_provenance.py#L63)
- Default spec is `full_live_contract_v2`, but production is V5.

---

## 5. Nitpicks

- **n1.** f-strings in logger calls throughout `bot.py` — use lazy `%s` formatting for performance.
- **n2.** Duplicated `_explicit_model_path` function in [bot.py:113](src/bot.py#L113) and [live_control.py:63](src/live_control.py#L63).
- **n3.** `blend_weight_test.py` and `compare_models.py` in project root are scripts, not tests — should be in `scripts/`.
- **n4.** `compare_models.py` hardcodes paths to `models/old/xgboost_model.pkl` — will crash on any other machine.
- **n5.** `import re` inside functions in `kaggle_loader.py` rather than at module level.
- **n6.** `SweepConfig` defined in both `triple_trader_backtest.py` and `duo_trader_sweep.py` with different fields.
- **n7.** Stance encoded as ordinal numeric (0-4) — implies false ordering for logistic regression.
- **n8.** `datetime.now()` without timezone used throughout — mixing local/UTC timestamps.
- **n9.** Hardcoded `random_state=42` everywhere — fine for reproducibility but should be configurable for sensitivity analysis.
- **n10.** Empty `__init__.py` files throughout — consider adding `__all__` for public API definition.
- **n11.** `.gitignore` has `data/models/` but actual pkl files are under `models/` (not excluded).
- **n12.** No `pyproject.toml` — project is non-installable, forcing `sys.path` hacks.
- **n13.** Tennis features live under `src/features/` alongside UFC features — naming/organization concern.
- **n14.** Hardcoded year range `range(2022, 2027)` in [tennis_bookmaker_audit.py:794](src/data/tennis_bookmaker_audit.py#L794) — breaks in 2027.

---

## 6. Positive Observations

1. **Rigorous model evaluation pipeline.** The promotion gate, selection gate, control arm, and walk-forward comparison infrastructure is impressive. The codebase takes model governance seriously — there are multi-stage gates before a model reaches production.

2. **Feature provenance auditing.** The `feature_provenance.py` module and associated audit scripts show genuine care for tracking what features a model was trained with and verifying consistency at prediction time.

3. **Comprehensive training specs.** The `training_spec.py` system with JSON-based model contracts is well-designed. It enforces feature lists, hyperparameters, and dataset variants through a single source of truth.

4. **Preflight audit system.** The `preflight_audit.py` module runs sanity checks before live trading — checking for data staleness, feature nulls, and model version consistency.

5. **Duplicate bet prevention.** The executor has ledger-based deduplication to avoid placing the same bet twice. The test suite has specific regression tests for this (`test_executor_duplicates.py`).

6. **Bet ledger design.** The JSON-based ledger with per-bet tracking of model probability, edge, market ID, and outcome is well-structured for post-hoc analysis.

7. **Multi-source data strategy.** The codebase pulls from The Odds API, BetsAPI, BFO scrapers, Sherdog, Tapology, and Kaggle datasets. The fallback scraper chain is a good resilience pattern.

8. **Odds noise injection.** The `_add_odds_noise` function in training is a creative approach to the closing-odds leakage problem, even if the current implementation is approximate.

9. **Solid test coverage for schema contracts.** `test_phase2_schema_contract.py` is extensive and tests many command-line entry points with realistic mocking.

10. **Tennis expansion architecture.** The tennis modules are well-separated and follow the same patterns as the UFC modules, making the multi-sport expansion clean.

---

## 7. Recommended Priority Actions

### Tier 1 — Fix Before Next Real-Money Session
| # | Issue | Why | Effort |
|---|-------|-----|--------|
| 1 | **Fix CalibratedClassifierCV** (C1, C2) | Directly corrupts bet sizing — every bet may be oversized | 2 hours |
| 2 | **Add read auth to web dashboard** (C3, C4) | Financial data exposed to anyone on the network | 1 hour |
| 3 | **Add process lock + crash recovery** (C6) | Duplicate bets on redeploy, orphaned orders on crash | 4 hours |
| 4 | **Add conviction EV check** (C7) | Currently places negative-edge bets | 30 min |
| 5 | **Add market order slippage check** (C8) | Edge can be silently erased by slippage | 1 hour |

### Tier 2 — Fix This Week
| # | Issue | Why | Effort |
|---|-------|-----|--------|
| 6 | Pin dependencies exactly (M1) | Supply chain attack vector | 30 min |
| 7 | Remove pkl files from git (C5) | Arbitrary code execution risk | 1 hour |
| 8 | Fix blend_b calculation (M9) | Asymmetric edge errors | 30 min |
| 9 | Add absolute bet cap (M10) | Unbounded bet sizes as bankroll grows | 30 min |
| 10 | Sync bankroll with wallet (M3) | Bets sized against wrong balance | 1 hour |
| 11 | Fix log rotation (M4) | Volume fills up, blocks ledger writes | 15 min |
| 12 | Add `set -euo pipefail` to entrypoint (m18) | Silent migration failures | 5 min |

### Tier 3 — Fix Before Next Model Retrain
| # | Issue | Why | Effort |
|---|-------|-----|--------|
| 13 | Separate opening/closing odds (M11) | Training accuracy inflated vs. live performance | 4 hours |
| 14 | Raise fuzzy match thresholds (C9) | Wrong odds in training data | 30 min |
| 15 | Fix rankings conflation (M6) | Wrong ranking features for women's fights | 1 hour |
| 16 | Document V5 ranking feature drop (M16) | Unclear if intentional or regression | 30 min |
| 17 | Fix vacuous tests (m23) | False safety from tests that assert nothing | 1 hour |
| 18 | Expand CI to all tests (M22) | Most regressions not caught | 30 min |

### Tier 4 — Improve Over Time
| # | Issue | Why |
|---|-------|-----|
| 19 | Add retry logic to HTTP clients (M14) | Transient network errors crash predictions |
| 20 | Add cache TTL/eviction (M7, M8) | Memory growth + stale data in long sessions |
| 21 | Fix timestamp timezone consistency (m5) | Stale-check errors across timezones |
| 22 | Add rate limiting to web endpoints (m8) | API quota exhaustion vector |
| 23 | Test ledger recovery path (M21) | ZeroDivisionError and phantom bets possible |
| 24 | Replace Google scraping for weigh-ins (m6) | Currently non-functional in production |

---

*This report was generated by reading every source file in the repository. Findings are based on static analysis and code tracing — no tests were executed as part of this audit.*
