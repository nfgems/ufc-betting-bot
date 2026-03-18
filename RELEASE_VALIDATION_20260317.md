# UFC Release Validation 2026-03-17

Promoted spec candidate: `full_live_contract_v2`

Status: `RETRAINED-NOT-PROMOTED` - retrain and post-train validation are complete, but no commit, promotion, or hydration restart has been performed yet.

## Current State

- Canonical repo: `C:\Users\Evan\betting-bot\ufc-betting-bot`
- Branch: `recovery/repo-cleanup-20260317`
- Full test suite: `369 passed, 1 skipped`
- Processed UFC artifacts were rebuilt from the promoted spec path on `2026-03-17T19:04:37`
- Trained model artifacts were regenerated on `2026-03-17T19:04:42`
- Embedded + sidecar spec metadata now records `git_hash=8c8dbbab8c2bacbe99249a2fef17f260a44225ad`
- Hydration remains intentionally paused
- Historical `full_live_contract_v1` is retained for explicit audit/compatibility use only

## What Changed In This Remediation Pass

- Raw `TitleBout` values now survive Kaggle normalization instead of being dropped to `NaN`.
- Added `best_of_both_full_history`, a full-history field-level merge variant that preserves richer legacy metadata/odds while backfilling official UFCStats fight-result and per-fight stat fields into the real pre-2022 train split.
- The promoted training spec now resolves to `full_live_contract_v2`:
  - dataset variant: `best_of_both_full_history`
  - feature count: `144`
  - no-odds feature count: `135`
  - line movement removed from the training contract pending real pre-2022 historical coverage
- The audit script now follows the promoted spec's dataset variant by default instead of silently auditing the stale legacy processed snapshot.
- The training path now stamps the active repo `git_hash` into the saved training spec and embedded model metadata.
- A real retrain was completed through `python -m src.bot train`, regenerating:
  - `data/processed/fights_cleaned.csv`
  - `data/processed/features.csv`
  - `data/processed/test_set.csv`
  - `models/xgboost_model.pkl`
  - `models/xgboost_no_odds_model.pkl`
  - `models/logistic_model.pkl`
  - `models/full_live_contract_v2_spec.json`

## Audit Results

### Historical failure reference

`full_live_contract_v1` still fails exactly as expected:

- dataset variant: legacy/default path
- `trainable_rows=3604`
- `train_split_rows=2612`
- `test_split_rows=992`
- `28` dead train-split contract columns
- dead columns include:
  - the full roll-stat family tied to missing per-fight historical stats
  - `is_title_bout`
  - the full line-movement family

### Current promoted candidate

`full_live_contract_v2` now passes the train-split dead-column gate:

- dataset variant: `best_of_both_full_history`
- `trainable_rows=4315`
- `train_split_rows=2878`
- `test_split_rows=1437`
- `dead_contract_columns_trainable=[]`
- `dead_contract_columns_train_split=[]`
- suspicious default fills: none reported

Important interpretation:

- The pre-2022 roll-stat family and `is_title_bout` are no longer structurally dead on the actual train split.
- Line movement remains available in live code and historical backfill utilities, but it is intentionally out of the promoted training contract until honest pre-2022 historical coverage exists.
- Remaining null-heavy areas are honest sparsity, not fabricated defaults:
  - pound-for-pound rankings
  - weight-class rankings
  - `is_empty_arena`
  - method odds / moneyline coverage on recent rows

## Post-Retrain Artifact Checks

- `models/full_live_contract_v2_spec.json`
  - `name=full_live_contract_v2`
  - `dataset_variant=best_of_both_full_history`
  - `git_hash=8c8dbbab8c2bacbe99249a2fef17f260a44225ad`
- `models/xgboost_model.pkl`
  - embedded `training_spec.name=full_live_contract_v2`
  - embedded `dataset_variant=best_of_both_full_history`
  - embedded `git_hash=8c8dbbab8c2bacbe99249a2fef17f260a44225ad`
  - feature count `144`
- `models/xgboost_no_odds_model.pkl`
  - embedded `training_spec.name=full_live_contract_v2`
  - embedded `dataset_variant=best_of_both_full_history`
  - embedded `git_hash=8c8dbbab8c2bacbe99249a2fef17f260a44225ad`
  - feature count `135`
- Offline inference smoke passed against the regenerated `xgboost` artifact on a local `test_set.csv` row.

## Validation Commands Run

- `$env:PYTHONPATH='.'; pytest -q`
- `python scripts/audit_model_feature_nulls.py --spec full_live_contract_v2 --json`
- `python scripts/audit_model_feature_nulls.py --spec full_live_contract_v1 --json --allow-dead-features`
- `$env:PYTHONPATH='.'; python -m src.bot train`

## Remaining Work Before Promotion

- Run live/API-backed smoke checks if you want a final pre-promotion sanity pass on real upcoming-card inference.
- Review model performance and promotion criteria before replacing any existing promoted runtime artifact.
- Only after that should hydration restart be considered, and only from the canonical repo.

## Promotion Decision

- `full_live_contract_v2` is now the current trained candidate because its train-split audit is clear and the retrain completed successfully.
- `full_live_contract_v1` remains non-promotable and should only be used for historical failure comparison.
- Promotion and hydration restart remain intentionally deferred until explicit go-ahead.
