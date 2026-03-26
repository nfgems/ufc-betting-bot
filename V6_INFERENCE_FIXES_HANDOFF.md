# V6 Inference Fixes Handoff — 2026-03-25

## What Was Done

Three bugs were fixed that were degrading UFC betting ROI. All changes are tested and passing (68/68 schema contract tests, 42/42 profile enrichment tests).

### Fix 1: Sig Strikes Table Index (fighter_lookup.py)

**Problem:** `_scrape_fight_detail()` used `tables[1]` for the Significant Strikes breakdown, but UFCStats fight detail pages have 4 tables with class `b-fight-details__table`:
- `tables[0]`: Totals summary
- `tables[1]`: Totals per-round (has `js-fight-table` class)
- `tables[2]`: Sig Strikes summary
- `tables[3]`: Sig Strikes per-round (has `js-fight-table` class)

`tables[1]` was the per-round Totals table, not the Sig Strikes summary. This caused all sig strikes features (`head_landed`, `body_landed`, `leg_landed`, `distance_landed`, `clinch_landed`, `ground_landed`, and their opponent/attempted variants) to be NaN at inference time.

**Fix:** Filter out per-round tables before indexing:
```python
summary_tables = [t for t in tables if "js-fight-table" not in (t.get("class") or [])]
```
Then use `summary_tables[0]` for Totals and `summary_tables[1]` for Sig Strikes. More robust than hardcoding index 2 since it survives page structure changes.

**Files changed:** `src/data/fighter_lookup.py` (lines 1213-1280)

**Test added:** `test_scrape_fight_detail_sig_strikes_uses_summary_table` in `tests/test_v4_profile_enrichment.py` — mocks a 4-table HTML structure and verifies sig strikes are parsed from the correct summary table.

### Fix 2: Odds Look-Ahead Bias (ufc_refresh.py)

**Problem:** `_historical_moneyline_overlay()` sorted `offset_days` ascending and kept `first`, which picked closing odds (smallest offset = closest to fight). The model trained on closing odds it wouldn't have at bet time, inflating the odds baseline to 68.4% accuracy.

**Fix:** Changed `offset_days` sort to descending so `keep="first"` picks opening odds (largest offset, ~7 days out):
```python
elif column == "offset_days":
    ascending.append(False)  # descending: prefer opening (largest offset) over closing
```

**Files changed:** `src/data/ufc_refresh.py` (line 249)

**Impact:** The 4% odds noise injection in training already partially compensated, but training on opening odds directly is cleaner. Model accuracy vs the odds baseline may appear to shrink after retraining — this reflects real edge rather than inflated edge from closing-odds lookahead.

### Fix 3: Default Training Spec (bot.py)

**Problem:** `_default_training_spec()` imported and returned `full_live_contract_v6_spec` (untuned), meaning any bare `python -m src.bot train` would retrain with wrong hyperparameters. The tuned spec was promoted in the live model artifacts but the CLI default was stale.

**Fix:** Changed import and return to `full_live_contract_v6_tuned_spec`.

**Files changed:** `src/bot.py` (lines 139-141)

**Test updated:** `test_cmd_train_uses_full_live_contract_spec` in `tests/test_phase2_schema_contract.py` — assertion now expects `full_live_contract_v6_tuned`.

## What Needs To Be Done

### 1. Retrain the Model
The current model artifact was trained on closing odds. Retrain so it learns on opening odds:
```bash
python -m src.bot train
```
This will now use `full_live_contract_v6_tuned_spec` by default.

### 2. Re-scrape Fighter Data
Sig strikes features have been NaN for all previously scraped fights. Re-scrape to backfill:
```bash
python -m src.bot scrape
```
After scraping, spot-check a known fighter to verify `head_landed`, `body_landed`, etc. are populated (no longer all NaN).

### 3. Run Backtest with Opening Odds Baseline
Compare the retrained model against the opening odds baseline (not the inflated closing baseline):
```bash
python -m src.bot backtest
```
Both `closing_odds_baseline` and `opening_odds_baseline` are already reported in evaluation output (`src/strategy/run_evaluation.py` lines 741-753). After the odds fix, the model's training odds and the opening baseline are aligned — the gap between them represents real edge.

### 4. Evaluate Impact
After retraining + backtest:
- Compare opening vs closing baseline accuracy to quantify the true gap
- Verify `a_odds` values now correspond to opening snapshots (offset_days >= 5)
- Check that sig strikes features are flowing through to model predictions (no longer NaN in live inference)

### 5. Rebuild Features (if needed)
If the backfilled scrape data changes significantly, rebuild the features CSV:
```bash
python -m src.bot train --data data/processed/features.csv
```

## Files Modified (UFC-specific only)

| File | Change |
|------|--------|
| `src/data/fighter_lookup.py` | Filter summary tables, use `summary_tables[1]` for sig strikes |
| `src/data/ufc_refresh.py` | Sort `offset_days` descending in `_historical_moneyline_overlay()` |
| `src/bot.py` | Point `_default_training_spec()` to `full_live_contract_v6_tuned_spec` |
| `tests/test_v4_profile_enrichment.py` | Added `test_scrape_fight_detail_sig_strikes_uses_summary_table` |
| `tests/test_phase2_schema_contract.py` | Updated spec name assertion to `full_live_contract_v6_tuned` |

## Test Status
- All 68 schema contract tests: PASS
- All 42 profile enrichment tests: PASS
- Compile check: CLEAN
- Default spec verification: prints `full_live_contract_v6_tuned`
