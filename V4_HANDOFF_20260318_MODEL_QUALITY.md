# V4 Handoff 2026-03-18: Model-Quality Pass

## Objective

The data-completeness phase for the provenance-strict V4 line is done enough to stop scraping and move to model quality.

The next phase is:

1. Freeze the completeness hunt.
2. Treat the remaining V4 NaNs as accepted honest undefineds.
3. Keep the feature contract fixed at the audited `129`-feature V4 contract.
4. Run a model-quality pass on that stable `129` contract.
5. Compare tuned `129` candidates against:
   - `old_v4_122`
   - `v4_129_legacyreach`
6. Promote only if a tuned `129` candidate is clearly better overall.
7. Do not drift into `129 -> 144` feature recovery unless explicitly requested later.

This handoff is written to avoid the exact confusion that already happened once: there are multiple "V4" artifacts in the repo, and they are not the same thing.

## Freeze / Do Not Do

- Do not keep scraping for the last 4 unresolved fighter blanks.
- Do not try to force the remaining NaNs to zero or impute them away.
- Do not compare a new tuned `129` run against `models/full_live_contract_v2` and call that `old_v4_122`. That is wrong.
- Do not start `129 -> 144` recovery now.
- Do not revert unrelated dirty worktree changes.

## Baseline Map

There are 3 important baselines. Keep them separate.

### 1. Current promoted production artifact

- Spec/artifact: `models/full_live_contract_v2_spec.json`
- Path: `models/xgboost_model.pkl`
- Contract size: `144` features
- Dataset variant: `best_of_both_full_history`
- This is the currently promoted model path in the repo.
- This is **not** the `old_v4_122` comparison baseline.
- This spec carries much heavier missingness in rankings and odds families.

### 2. Comparison baseline used in refreshed V4 evals

- Label used in eval JSONs: `old_v4_122`
- Actual artifact directory: `models/candidates/full_live_contract_v4`
- Spec file: `models/candidates/full_live_contract_v4/full_live_contract_v4_spec.json`
- Contract size: `122` features
- Dataset variant: `pulled_all`
- This is the older V4 candidate baseline used in the refreshed V4 comparison pack.
- This baseline is relatively clean on nulls and is the correct "old V4" reference for the current `129`-feature V4 tuning work.

### 3. Current audited target contract

- Spec source of truth: `src/model/training_spec.py` -> `full_live_contract_v4_spec()`
- Current best trained `129` candidate: `models/candidates/full_live_contract_v4_20260318_tuned02_altxgb`
- Spec file: `models/candidates/full_live_contract_v4_20260318_tuned02_altxgb/full_live_contract_v4_spec.json`
- Contract size: `129` features
- Dataset variant: `pulled_all`
- This remains the audited `129` contract, and `tuned02_altxgb` became the prior `129` incumbent used for later `138` overlay comparisons.

## 122 -> 129 Delta

The current `129`-feature V4 contract adds exactly these 7 features on top of the older `122`-feature V4 baseline:

- `a_stance_enc`
- `b_stance_enc`
- `same_stance`
- `a_weight`
- `b_weight`
- `diff_weight`
- `is_empty_arena`

Nothing was removed from the old `122` contract.

## What Was Completed

### Core integrity / audit fixes

- Fixed reference-date leakage in fighter lookup so age and layoff do not depend on `datetime.now()` for historical/live feature construction.
- Fixed duplicate/shadowed no-odds / market-derived helper behavior in feature building.
- Fixed compatibility handling in `fighter_lookup` so older supported kwargs are not accidentally dropped.

### V4 profile recovery work

- Restored stance-family features cleanly.
- Added conservative legacy backfill for stable static lengths where a fighter had exactly one consistent legacy value.
- Added offline supplement paths using external sources:
  - Sherdog
  - Tapology
  - Wikipedia
  - MartialBot
  - FightDX
- Tightened parsing and placeholder rejection so garbage values are not accepted as real profile data.
- Rejected ambiguous/bad external leads instead of forcing them in.

### Audit tooling / regression coverage

- Added fighter-row profile gap audit script:
  - `scripts/audit_v4_profile_gaps.py`
- Added null-contract audit usage against the actual spec build path:
  - `scripts/audit_model_feature_nulls.py`
- Added regression coverage that rebuilds the real V4 train split and asserts:
  - no unexpected null families appear
  - rolling TD and strike-defense nulls are denominator-driven
  - the remaining profile/context nulls are the known honest exceptions
- New audit regression coverage lives in:
  - `tests/test_ufc_audit_regressions.py`

### Validation state

- Current full test status:
  - `415 passed, 1 skipped`
- Command:
  - `python -m pytest -q`
- Warnings are only the existing pandas fragmentation warnings in `ufc_refresh.py` / `build_features.py`.

## Current Audited 129 State

### Contract / dataset

- Spec: `full_live_contract_v4`
- Dataset variant: `pulled_all`
- Train cutoff: `2022-01-01`
- Train split rows: `3374`
- Test split rows: `1420`

### Null state

- Current audited train-split contract NaNs:
  - `1115 / 435246`
  - `0.26%`

- Top remaining train-split null buckets:
  - `diff_roll_td_acc`: `313`
  - `diff_roll_td_def`: `222`
  - `b_roll_td_acc`: `196`
  - `b_roll_td_def`: `134`
  - `a_roll_td_acc`: `132`
  - `a_roll_td_def`: `94`
  - `diff_roll_str_def`: `4`
  - `num_rounds_feat`: `3`
  - `diff_age`: `4`
  - `diff_reach`: `2`

### Why these NaNs are now accepted

- `roll_td_acc` / `roll_td_def` nulls are denominator-driven:
  - no prior TD attempts
  - or no opponent TD attempts
- `roll_str_def` nulls are denominator-driven:
  - no opponent significant-strike attempts
- `num_rounds_feat` nulls are real raw-source gaps on 3 old no-limit UFC bouts.
- The remaining profile gaps are true source blanks, not parser/merge failures.

These are now considered accepted honest undefineds.

## Remaining Frozen Profile Gaps

Do not keep hunting these unless explicitly re-opened.

- `gap_rows`: `7`
- `reach`: `2`
- `age`: `5`

Unresolved fighters:

- `Felix Lee Mitchell`: `age`, `reach`
- `Johnny Rhodes`: `age`, `reach`
- `Steve Nelmark`: `age`
- `Marcus Bossett`: `age`

These are source blanks in the canonical profile path plus failed narrow external archival search. No defensible exact values were found.

## Candidate Scoreboard

### Correct V4 comparison set

These are the metrics from the refreshed V4 comparison artifacts already on disk.

| model | feature_count | accuracy | log_loss | brier |
| --- | ---: | ---: | ---: | ---: |
| `old_v4_122` | 122 | 0.616901 | 0.653829 | 0.230927 |
| `v4_129_legacyreach` | 129 | 0.609859 | 0.651988 | 0.230081 |
| `v4_129_tuned02_altxgb` | 129 | 0.636620 | 0.645057 | 0.226851 |
| `v4_129_sherdogsupp` | 129 | 0.615493 | 0.653533 | 0.230801 |
| `v4_129_extprofiles` | 129 | 0.609859 | 0.653928 | 0.230962 |
| `v4_129_profilemax` | 129 | 0.604225 | 0.655597 | 0.231817 |

### Interpretation

- `v4_129_tuned02_altxgb` is the current best audited `129` candidate:
  - best accuracy
  - best log loss
  - best brier
- It clears the intended tuning gate by beating `v4_129_legacyreach` on calibration and eliminating the old accuracy gap versus `old_v4_122`.
- Therefore the forward comparison baseline is now:
  - incumbent comparison baseline: `old_v4_122`
  - best current `129`: `v4_129_tuned02_altxgb`

## Important Clarification About Missing Data

Earlier confusion happened because `old_v4_122` and promoted `v2` were mixed together. They are different.

### `old_v4_122`

- Path: `models/candidates/full_live_contract_v4`
- Train-split contract NaNs:
  - `1115 / 411628`
  - `0.27%`
- This is relatively clean.

### promoted `full_live_contract_v2`

- Path: `models/xgboost_model.pkl`
- Spec: `models/full_live_contract_v2_spec.json`
- Train-split contract NaNs:
  - `19748 / 414432`
  - `4.77%`
- Most of that missingness comes from rankings / moneyline / method-odds families.

So the correct takeaway is:

- `old_v4_122` was not "winning because it was dirty"
- promoted `v2` is the artifact with substantially heavier missing-data exposure

## Files Most Relevant To The Next Pass

### Spec / training

- `src/model/training_spec.py`
- `src/model/train.py`
- `src/bot.py`

### Feature/data path

- `src/features/build_features.py`
- `src/data/fighter_lookup.py`
- `src/data/ufc_refresh.py`
- `src/data/fallback_scrapers.py`
- `src/data/kaggle_loader.py`
- `src/model/feature_provenance.py`

### Audits / tests

- `scripts/audit_model_feature_nulls.py`
- `scripts/audit_v4_profile_gaps.py`
- `tests/test_ufc_audit_regressions.py`
- `tests/test_v4_profile_enrichment.py`
- `tests/test_phase2_schema_contract.py`

## Dirty Worktree Warning

The repo is currently dirty. Do not assume only one file changed.

Notable modified/untracked areas include:

- `src/bot.py`
- `src/config.py`
- `src/data/fallback_scrapers.py`
- `src/data/fighter_lookup.py`
- `src/data/kaggle_loader.py`
- `src/data/live_monitor.py`
- `src/data/scraper.py`
- `src/data/ufc_refresh.py`
- `src/features/build_features.py`
- `src/model/feature_provenance.py`
- `src/model/training_spec.py`
- `tests/test_external_data_snapshots.py`
- `tests/test_phase2_schema_contract.py`
- `tests/test_ufc_audit_regressions.py`
- `tests/test_v4_profile_enrichment.py`
- multiple candidate artifact folders under `models/candidates/`
- multiple processed candidate folders under `data/processed/candidates/`

Do not revert unrelated changes. Read first.

## Recommended Next Phase: Tune The Stable 129 Contract

Superseded by the later `138` overlay follow-up below. Keep this section as historical context only.

## 138 Follow-up: Current State Of Record

The user later explicitly reopened forward feature expansion. The winning direction was not the full `144`; it was the narrower `138` overlay:

- keep the audited `129` base
- add moneyline and method-odds families
- do **not** add rankings

Why:

- the added ranking features are still too sparse in current local historical sources to help
- the market/method-odds families add useful signal without dragging in the worst ranking missingness

### Fixed evaluation frame

All of the `138` comparisons below use the same post-`2022-01-01` test set:

- path: `data/processed/candidates/full_live_contract_v4_20260318_138_tuned02_nocal/test_set.csv`
- sha256: `a0359f6f6d423b791c298dab4cba825470f5af735b6f61e49cd5bd3658887608`
- test rows: `1420`

### Key discovery

The main problem with the early `138` run was calibration, not the feature set.

- `isotonic` underperformed
- `sigmoid` also underperformed
- `none` was materially better
- lowering `odds_noise_std` from `0.04` to `0.02` improved post-2022 accuracy further while still staying clearly ahead of the old `129` incumbent on log loss and brier

### 138 scoreboard on the fixed post-2022 test set

| model | feature_count | accuracy | log_loss | brier |
| --- | ---: | ---: | ---: | ---: |
| `v4_129_tuned02_altxgb` | 129 | 0.636620 | 0.645057 | 0.226851 |
| `v4_138_tuned02_overlay` | 138 | 0.628873 | 0.640975 | 0.225443 |
| `v4_138_tuned02_sigmoid` | 138 | 0.637324 | 0.647394 | 0.228037 |
| `v4_138_tuned02_nocal` | 138 | 0.643662 | 0.629848 | 0.219927 |
| `v4_138_tuned02_nocal_noise000` | 138 | 0.644366 | 0.631174 | 0.220538 |
| `v4_138_tuned02_nocal_noise002` | 138 | 0.646479 | 0.632022 | 0.220880 |
| `v4_138_tuned03_nocal_noise002_t180_lr0095` | 138 | 0.642254 | 0.629911 | 0.219934 |
| `v4_138_tuned03_nocal_noise002_d6` | 138 | 0.642254 | 0.634474 | 0.222021 |
| `v4_138_tuned03_nocal_noise002_d6m8` | 138 | 0.637324 | 0.628968 | 0.219484 |

### Current best `138` candidates

Accuracy-first current best candidate:

- label: `v4_138_tuned02_nocal_noise002`
- artifact dir: `models/full_live_contract_v4_20260318_138_tuned02_nocal_noise002`
- processed dir: `data/processed/full_live_contract_v4_20260318_138_tuned02_nocal_noise002`
- spec: `models/full_live_contract_v4_20260318_138_tuned02_nocal_noise002/full_live_contract_v4_138_spec.json`
- comparison: `models/full_live_contract_v4_20260318_138_tuned02_nocal_noise002/eval_against_refreshed_v4_test.json`

Best-calibrated sibling:

- label: `v4_138_tuned03_nocal_noise002_d6m8`
- artifact dir: `models/full_live_contract_v4_20260318_138_tuned03_nocal_noise002_d6m8`
- processed dir: `data/processed/full_live_contract_v4_20260318_138_tuned03_nocal_noise002_d6m8`
- this sibling now has the best log loss and brier among the tested `138` variants, but it gives back too much accuracy to replace `noise002` as the accuracy-first incumbent

### Audit state for the current best `138`

The structural audit state did not regress:

- dataset variant: `pulled_all_plus_legacy_market`
- train/test split: `3374 / 1420`
- dead train-split contract columns: none
- frozen profile-gap rows: `7`
- unresolved fighters remain:
  - `Felix Lee Mitchell`
  - `Johnny Rhodes`
  - `Steve Nelmark`
  - `Marcus Bossett`
- market-odds family is the only legacy-dependent family:
  - `2745` train rows with any non-null market-odds feature
  - `2745` train rows with any legacy dependency in that family

Important artifact note:

- the later `138` sweep wrote to `models/` and `data/processed/` root directories, not under `/candidates/`, because the isolated runner was called without a `candidates/` prefix in `--output-subdir`
- do not "fix" that by rerunning work; just be explicit about the paths you are using

### Recommended next work on the `138` line

Keep the current scope tight:

- keep the feature contract fixed at the current `138`
- keep the dataset variant fixed at `pulled_all_plus_legacy_market`
- do not resume scraping
- do not reopen the `144` path unless historical rankings coverage materially improves

The highest-value next experiments are:

- stay with `calibration_method=none` by default
- tune XGBoost around the current tuned02 parameter set
- sweep `odds_noise_std` near `0.0` to `0.02`
- keep `time_decay_half_life` near `730`
- use the current split between the two leaders on purpose:
  - `v4_138_tuned02_nocal_noise002` as the accuracy-first incumbent
  - `v4_138_tuned03_nocal_noise002_d6m8` as the calibration-first incumbent
- optionally run `129 + moneyline only` vs `129 + method odds only` ablations if feature-family attribution matters

The current promotion-style gate inside the V4 line is:

- beat `v4_138_tuned02_nocal_noise002` on post-2022 accuracy, or
- beat `v4_138_tuned03_nocal_noise002_d6m8` on log loss and brier without giving back too much accuracy
- do not introduce new audit failures
- pass full pytest if any code changes are made

## Suggested Commands

### Reconfirm current audit state

```powershell
python scripts/audit_model_feature_nulls.py --spec full_live_contract_v4_138 --null-threshold-pct 0 --json
python scripts/audit_v4_profile_gaps.py --spec full_live_contract_v4_138 --split train --sample-limit 20 --json
python -m pytest -q
```

### Train a new tuned `138` candidate in an isolated subdir

```powershell
$env:PYTHONPATH='.'
python scripts/train_v4_tuned_candidate.py `
  --base-spec full_live_contract_v4_138 `
  --output-subdir full_live_contract_v4_20260318_138_tuned03 `
  --candidate-label v4_138_tuned03 `
  --calibration-method none `
  --time-decay-half-life 730 `
  --odds-noise-std 0.02 `
  --test-set data/processed/candidates/full_live_contract_v4_20260318_138_tuned02_nocal/test_set.csv `
  --compare-model old_v4_122=models/candidates/full_live_contract_v4/xgboost_model.pkl `
  --compare-model v4_129_tuned02_altxgb=models/candidates/full_live_contract_v4_20260318_tuned02_altxgb/xgboost_model.pkl `
  --compare-model v4_138_tuned03_nocal_noise002_d6m8=models/full_live_contract_v4_20260318_138_tuned03_nocal_noise002_d6m8/xgboost_model.pkl `
  --compare-model v4_138_tuned02_nocal_noise002=models/full_live_contract_v4_20260318_138_tuned02_nocal_noise002/xgboost_model.pkl
```

This will write both:

- processed artifacts under:
  - `data/processed/full_live_contract_v4_20260318_138_tuned03/`
- model artifacts under:
  - `models/full_live_contract_v4_20260318_138_tuned03/`

### After a tuned run

At minimum, save:

- spec JSON
- null audit JSON
- provenance audit JSON
- profile gap audit JSON/CSV
- refreshed V4 comparison JSON

## State Of Record For The New Instance

If a new instance picks this up, it should assume:

- completeness hunting is over
- the remaining V4 NaNs are accepted honest undefineds
- the next job is model tuning, not more data collection
- the active tuning line is the `138` overlay contract, not the full `144`
- the current accuracy-first best candidate is `v4_138_tuned02_nocal_noise002`
- the current best-calibrated sibling is `v4_138_tuned03_nocal_noise002_d6m8`
- the meaningful baselines are:
  - `old_v4_122`
  - `v4_129_tuned02_altxgb`
  - `v4_138_tuned03_nocal_noise002_d6m8`
- the fixed refreshed-V4 evaluation frame is:
  - `data/processed/candidates/full_live_contract_v4_20260318_138_tuned02_nocal/test_set.csv`
  - sha256 `a0359f6f6d423b791c298dab4cba825470f5af735b6f61e49cd5bd3658887608`
- the latest full repo verification state is:
  - `418 passed, 1 skipped`
- the currently promoted model is still `full_live_contract_v2` unless and until a tuned V4 candidate clearly wins and is promoted on purpose

## 2026-03-18 Addendum: Coverage Gate Recovery And Training-Window Pivot

This addendum supersedes the earlier `138` direction.

The user later made two hard rules explicit:

- newly added feature families must have near-100% real coverage
- no missing data may be invented or estimated

Under that rule, the `138` and `144` overlay lines are not acceptable current candidates. The repo now has explicit external-family coverage gates, and those overlay specs correctly fail on the full-history train split.

### Recovery work completed after the original `138` pass

- exact fight-level gap inventories were generated for moneyline, method odds, and rankings
- the historical moneyline archive was expanded from `3776` to `3855` rows after a BestFightOdds scrape of the exact unresolved train-era cards
- that BFO pass scraped `256` cards and recovered `79` real moneyline fight rows
- full-history train moneyline coverage improved from `81.39%` to `83.73%`
- full-history train method-odds coverage stayed `67.25%`
- full-history train rankings coverage stayed effectively unusable at `0.98%`

### What the coverage numbers say now

On the `full_live_contract_v4_144` train split after the BFO merge:

| regime | rows | moneyline | method | rankings |
| --- | ---: | ---: | ---: | ---: |
| full history | 3374 | 83.73% | 67.25% | 0.98% |
| post-2014 only | 2260 | 97.92% | 90.58% | 1.42% |

Interpretation:

- cutting away pre-2014 dramatically improves the honest coverage picture
- rankings remain non-viable even in the modern window
- moneyline becomes nearly acceptable in the post-2014 regime
- method odds improve a lot in the post-2014 regime but still do not meet the near-complete bar

### Windowed `129` comparison on the fixed post-2022 test set

Because the sparse overlay families are still blocked, the clean comparison was rerun on the tuned `129` contract instead.

All of these use the same fixed post-2022 test set:

- path: `data/processed/candidates/full_live_contract_v4_20260318_tuned02_altxgb/test_set.csv`
- sha256: `fa83b157450e8787c1eaf17f0d9e2f450ab4c8f8c09fd37f8318445c20604068`

| model | train window | feature_count | accuracy | log_loss | brier |
| --- | --- | ---: | ---: | ---: | ---: |
| `v4_129_tuned02_altxgb` | full history | 129 | 0.636620 | 0.645057 | 0.226851 |
| `v4_129_tuned02_post2014` | `2014-01-01+` only | 129 | 0.638028 | 0.637847 | 0.223795 |
| `v4_129_tuned02_pre2014_nocal` | pre-`2014-01-01` only | 129 | 0.614789 | 0.658663 | 0.233166 |

### Important note on the pre-2014 diagnostic

The true apples-to-apples pre-2014-only run failed under the normal time-series isotonic calibration path.

Why:

- the pre-2014 training slice has `1114` rows
- target balance is `844` positive / `270` negative
- the earliest time-series folds contain only one class in the training portion

That is already evidence against relying on that era. To still get a directional read, the stored `pre2014` diagnostic was trained with `calibration_method=none`.

### Current state of record

The current best acceptable V4 candidate is now:

- label: `v4_129_tuned02_post2014`
- artifact dir: `models/full_live_contract_v4_20260318_129_tuned02_post2014`
- processed dir: `data/processed/full_live_contract_v4_20260318_129_tuned02_post2014`
- comparison: `models/full_live_contract_v4_20260318_129_tuned02_post2014/eval_against_refreshed_v4_test.json`

Why it is now the current best:

- it keeps the clean audited `129` contract
- it respects the rejection of pre-2014 training data
- it beats the prior full-history tuned `129` incumbent on all three tracked post-2022 metrics

### Recommended next direction

- treat `v4_129_tuned02_post2014` as the new V4 incumbent
- stop spending effort on pre-2014 recovery
- if forward expansion is reopened later, use the post-2014 regime as the only serious expansion window
- do not reopen rankings unless a truly dense dated rankings archive is found
