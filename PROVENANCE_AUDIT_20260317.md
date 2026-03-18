# UFC Provenance Audit 2026-03-17

Strict candidate: `full_live_contract_v3`

Status: `TRAINED-OFFLINE-ONLY`

This note freezes the provenance comparison between the current retrained candidate
`full_live_contract_v2` and the provenance-strict successor candidate
`full_live_contract_v3`.

## Frozen Audit Artifacts

- `models/full_live_contract_v2_feature_provenance_audit.json`
- `models/candidates/full_live_contract_v3/feature_provenance_audit.json`
- `models/candidates/full_live_contract_v3/feature_null_audit.json`
- `models/candidates/full_live_contract_v3/full_live_contract_v3_spec.json`
- `models/candidates/full_live_contract_v3/xgboost_model.pkl`
- `models/candidates/full_live_contract_v3/xgboost_no_odds_model.pkl`
- `models/candidates/full_live_contract_v3/logistic_model.pkl`
- `data/processed/candidates/full_live_contract_v3/fights_cleaned.csv`
- `data/processed/candidates/full_live_contract_v3/features.csv`
- `data/processed/candidates/full_live_contract_v3/test_set.csv`

## Comparison Summary

### Current trained candidate: `full_live_contract_v2`

- dataset variant: `best_of_both_full_history`
- feature count: `144`
- train split rows: `2878`
- null audit: structurally clean, no dead train-split contract columns
- provenance result: still materially legacy-dependent

Train-split legacy dependence by family:

- `historical_performance`: `2868 / 2878` rows
- `experimental`: `2867 / 2878` rows
- `career_record`: `2822 / 2878` rows
- `event_context`: `2822 / 2878` rows
- `style_matchup`: `2822 / 2878` rows
- `market_odds`: `2718 / 2718` non-null rows
- `rankings`: `1205 / 1205` non-null rows
- `physical_profile`: `2718 / 2878` rows, with `160` default-only rows

Interpretation:

- `full_live_contract_v2` is honest and trainable.
- `full_live_contract_v2` is not provenance-strict.
- The unresolved dependence is no longer hidden; it is now quantified.

### Provenance-strict candidate: `full_live_contract_v3`

- dataset variant: `pulled_all`
- feature count: `113`
- train split rows: `3374`
- null audit: structurally clean, no dead train-split contract columns
- provenance result: zero legacy-dependent train rows in every included family

Train-split family summary:

- `historical_performance`: `0` legacy-dependent rows, `3373` repo-only rows, `1` default-only row
- `career_record`: `0` legacy-dependent rows, `3373` repo-only rows, `1` default-only row
- `experimental`: `0` legacy-dependent rows, `3373` repo-only rows, `1` default-only row
- `style_matchup`: `0` legacy-dependent rows, `3373` repo-only rows, `1` default-only row
- `event_context`: `0` legacy-dependent rows, `3374` repo-only rows, `0` default-only rows
- `rematch_h2h`: `0` legacy-dependent rows, `137` repo-only rows, `3237` default-only rows

Interpretation:

- The strict contract now trains only on repo-owned pulled/scraped history plus live-snapshot-compatible context.
- Remaining `default_only` rows are expected first-history / no-prior-rematch cases, not legacy carryover.

## Candidate Artifact Metadata

- `name=full_live_contract_v3`
- `dataset_variant=pulled_all`
- `git_hash=8c8dbbab8c2bacbe99249a2fef17f260a44225ad`
- `trained_at=2026-03-17T20:22:06.763954`
- `no_odds_feature_count=113` and matches the full feature set exactly because the strict contract excludes all market-derived columns
- offline inference smoke passed on a row from `data/processed/candidates/full_live_contract_v3/test_set.csv`
- example smoke output:
  - `prob_a=0.569933`
  - `prob_b=0.430067`
  - `predicted_winner=a`
  - `confidence=0.569933`

## Validation Run In This State

- `$env:PYTHONPATH='.'; python scripts/audit_model_feature_nulls.py --spec full_live_contract_v3 --json`
- `$env:PYTHONPATH='.'; python scripts/audit_feature_provenance.py --spec full_live_contract_v3 --json`
- `$env:PYTHONPATH='.'; python -m src.bot train --spec full_live_contract_v3 --output-subdir candidates/full_live_contract_v3`
- `$env:PYTHONPATH='.'; pytest -q`

Current full test result:

- `377 passed, 1 skipped`

## Remaining Blocker Before Promotion

`full_live_contract_v3` is provenance-clean offline, but live inference is not yet aligned to the same source contract.

Current live-path mismatch:

- `src/data/fighter_lookup.py` still fills several `roll_*` fields from current UFCStats profile averages instead of reconstructing them from the same repo-owned processed history semantics used by training.
- `src/data/fighter_lookup.py` still hardcodes `is_empty_arena=0`, while `full_live_contract_v3` removed that field from the contract entirely.

Because of that mismatch, the correct state is:

- strict offline training candidate: ready
- live/inference parity for strict candidate: not yet proven
- promotion / hydration restart: not ready

## Recommended Next Step

Implement a `full_live_contract_v3`-aligned live feature builder that:

- sources fighter history from repo-owned processed history / pulled data semantics
- reproduces the strict contract's `roll_*`, career-history, rematch, Elo momentum, and SoS behavior without profile-average shortcuts
- runs an API-backed smoke prediction using the strict candidate artifact only after that parity work is complete

No commit, push, promotion, or hydration restart was performed in this checkpoint.
