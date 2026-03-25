# Tennis ROI Handoff

Date: March 25, 2026

## Current status

- Real-money tennis execution should stay off.
- The strongest research candidate from selection is `optuna_objective_winner_13`.
- That candidate is better than `lean_hybrid_current`, but it is still not profitable.
- Conservative strategy ROI improved from `-0.075337` to `-0.051428`.
- That means the candidate loses less money, not that it makes money.

## Selected research baseline

Use this spec as the main baseline for the next tuning round:

- Model family: `logistic_elasticnet`
- Feature contract: `stage4_full_engineered`
- Train start: `2016-01-01`
- Min matches: `3`
- Half life: `365`
- Calibration: `sigmoid`
- CV: `timeseries_5fold`
- Retrain months: `6`
- `C=0.1266058971025309`
- `l1_ratio=0.7955692304713199`

Primary artifact references:

- `data/processed/tennis/tuning/selection_20260324_real_pass/final_selection_decision.json`
- `data/processed/tennis/tuning/selection_20260324_real_pass/final_comparison.csv`
- `data/processed/tennis/tuning/selection_20260324_real_pass/03_optuna_stage4_logistic/study_trials.csv`
- `data/processed/tennis/tuning/selection_20260324_real_pass/03_optuna_stage4_logistic/bundles/optuna_objective_winner_13/`

## What the last run proved

It proved three things:

1. The old live tennis baseline is not the best candidate anymore.
2. Stage-4 logistic models are the best family/feature zone found so far.
3. Predictive improvement alone is not enough. The model still does not beat the market well enough to create positive realized strategy ROI.

## What to do next

### 1. Freeze the model and tune the decision layer first

Do not jump back into broad model-family sweeps first.

Hold the selected research spec fixed and run a larger strategy-only search around execution rules:

- `min_edge`: test `0.04, 0.05, 0.06, 0.08, 0.10`
- `confidence_threshold`: test `0.62, 0.65, 0.68, 0.70`
- `min_bookmakers`: test `3, 4`
- `reference_edge_floor`: test `0.00, 0.01, 0.02, 0.03`
- `kelly_fraction`: test `0.02, 0.05, 0.10`
- Low-history handling: test stricter skips for `<=10` prior matches, not just `<=5`

Goal:

- Find out whether positive ROI exists with stricter execution rules before doing more model churn.
- If no positive region exists with the best current model, the core economics are likely not there yet.

### 2. Add better strategy diagnostics before trusting any ROI win

For every strategy sweep result, break ROI out by:

- ATP vs WTA
- Surface
- Price bucket
- Model edge bucket
- History bucket
- Rank-gap bucket
- Bookmaker-count bucket

Also record:

- Number of bets
- Total staked
- Average execution edge
- Median execution edge

Do not accept a positive ROI result driven by a tiny number of bets.

### 3. Compare against the market more directly

Right now the finalists still have negative market-relative Brier skill.

That means the model is still generally worse than `avg_fair_a` as a pure probability estimator.

Before chasing more model variants, verify:

- Whether execution price is materially better than the bookmaker reference used in backtests
- Whether the model only has value in specific mismatch buckets
- Whether any supposed ROI edge disappears once you require stronger market disagreement

If the model cannot show a stable positive pocket against market reference, more family sweeps will probably waste time.

### 4. Only then run the next model tuning pass

If strategy-only tuning finds a credible positive execution pocket, run a localized model search around the selected elastic-net zone:

- `train_start_date`: `2016-01-01`, `2018-01-01`
- `min_matches`: `3`, `5`
- `time_decay_half_life`: `365`, `730`, `1460`
- `retrain_months`: `3`, `6`
- `C`: concentrate near `0.05` to `0.5`
- `l1_ratio`: concentrate near `0.6` to `0.95`

Keep the objective anchored to OOS predictive quality, but reject anything that:

- Regresses lockbox Brier
- Regresses lockbox calibration
- Regresses market-relative skill
- Looks weak on conservative strategy ROI

### 5. Deprioritize XGBoost for now

The stage-4 XGBoost follow-up improved raw lockbox log loss, but it failed the market-skill gate.

Until the stage-4 logistic family is exhausted, do not spend the next cycle on broad XGBoost work.

## Minimal acceptance bar before any future promotion

Do not re-enable live tennis trading until a candidate can show all of this:

- Positive conservative ROI on lockbox
- Positive or near-flat market-relative skill, not deeply negative
- Improvement that is not concentrated in one tiny slice
- Reasonable bet count, not a tiny-sample fluke
- No obvious collapse after stricter edge thresholds or lower Kelly sizing

## Recommended sequence

1. Keep trading off.
2. Freeze `optuna_objective_winner_13`.
3. Run a strategy-only sweep around stricter execution rules.
4. Inspect slice-level ROI concentration.
5. If a real positive region appears, run localized stage-4 elastic-net tuning around that region.
6. Only consider promotion after both predictive and strategy evidence are positive.
