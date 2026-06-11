# Promotion Runbook — `full_live_contract_v6_durability`

**Status at 2026-06-10:** candidate selected on branch
`model-improvements-2026-06`. Beat the v6 control on identical walk-forward
folds under BOTH bases (legacy: +19.4% vs +14.0%; T-1+realistic: +16.6% vs
+11.7%; paired event bootstrap p=0.00 both). Seed-dispersion check:
`logs/track_c_batch/seed_dispersion.csv`.

**Evidence:** [EXPERIMENT_RESULTS_2026-06-10.md](EXPERIMENT_RESULTS_2026-06-10.md)

## Why the data rebuild is part of this promotion

The 9 new contract columns (`a/b/diff_loss_ko_rate`, `_loss_sub_rate`,
`_recent_ko_loss`) are produced by the updated `build_features.py` and do
NOT exist in the currently pinned `data/processed/features.csv`. The live
processed-snapshot path serves features from that file, so promotion
requires regenerating processed artifacts through the normal refresh
pipeline with the new code. (The live scrape path and roll-forward already
compute the new columns — see `fighter_lookup.py`; parity-tested.)

## Steps (in order)

1. **Merge the branch** (or cherry-pick) so the refresh/build code includes
   the new feature logic. Run full pytest first:
   `python -m pytest -q tests/` (expect ~915 passing).

2. **Regenerate processed data** through the standard refresh
   (`python -m src.bot <refresh command per PRODUCTION_RUNBOOK>`). Confirm
   the new columns exist:
   `python -c "import pandas as pd; df=pd.read_csv('data/processed/features.csv', nrows=5); print([c for c in df.columns if 'loss_ko' in c or 'recent_ko' in c])"`

3. **Parity replay on the new contract** (no network, ~5 min):
   `python scripts/parity_replay.py --mode exact --start 2025-01-01 --limit 250 --spec full_live_contract_v6_durability`
   Requirement: zero structural mismatches (time-aged fields excluded).
   Then run the CI tests: `python -m pytest tests/test_live_train_parity.py -q`.

4. **Formal evaluation under the honest standard:**
   `python -m src.strategy.run_evaluation --variants baseline --execution-mode realistic ...`
   (stage 3/4 now consume `--execution-mode`; the stage-4 control freeze
   must be regenerated under the same mode — the old frozen arm has no
   trading artifacts and blocks stage 4 regardless.)

5. **Train the production refit:** spec
   `full_live_contract_v6_durability_fullfit` (registered; tuned contract +
   durability columns, `train_cutoff_date=2027-01-01`), via the normal
   train/promotion CLI so the bundle manifest, spec JSON, sha256 pins, and
   rollback backup are produced by `production_bundle` as usual.

6. **Promotion gate:** the statistical bar is now default-on
   (`--legacy-gate` reverts). Compare against the freshly frozen v6 control
   under T-1+realistic.

7. **Post-promotion:** watch the first live cards for the no-odds-agreement
   and conviction behavior; the new features are NaN-honest (debutants and
   zero-loss fighters produce NaN, matching training).

## Standing decisions adopted with this promotion

- **Evaluation standard:** all future candidate adjudication uses
  T-1 pricing + `--execution-mode realistic` + the statistical gate.
  (Defaults in code remain legacy for reproducibility of old artifacts;
  the standard is a process rule, enforced by the gate's stage-4 context.)
- **Deferred follow-ups:** E9 fitted blend (bet-level test under realistic
  execution), E14 cadence sweep (`scripts/e14_cadence_sweep.py`), E5
  conviction gates (re-adjudicate with more sample), Railway fill-price
  pull to calibrate `assumed_half_spread`.
