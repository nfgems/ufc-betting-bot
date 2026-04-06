# V6 Full-Fit ROI Handoff

## Situation

On March 26, 2026, the live production bundle was switched from the old promoted V6 tuned artifact to the new V6 full-fit artifact:

- old promoted bundle: `models/backups/20260326_v6_tuned_pre_fullfit_promotion/`
- old promoted manifest snapshot: `models/backups/20260326_v6_tuned_pre_fullfit_promotion/current_production_model.json`
- old promoted spec: `models/backups/20260326_v6_tuned_pre_fullfit_promotion/full_live_contract_v6_tuned_spec.json`
- new promoted manifest: `models/current_production_model.json`
- new promoted spec: `models/full_live_contract_v6_fullfit_spec.json`

The promotion was based on stronger honest predictive evidence, not on a true honest head-to-head ROI comparison.

## The Dilemma

The old promoted artifact had an honest realistic walk-forward ROI estimate:

- old V6 tuned realistic production-gated ROI: `0.2104`
  - source: `logs/pipeline_hardening_postchange_20260326/tuned_v6_realistic_walkforward_summary.csv`

The new promoted full-fit artifact does not have an equally honest ROI estimate.

Why:

- the old promoted artifact was still an evaluation-era model with `train_cutoff_date = 2022-01-01`
- the new promoted artifact is a true full-fit retrain with `train_cutoff_date = 2027-01-01`
- once the new model is trained on all available rows through the promotion window, any backtest on that same history is in-sample for the new artifact

So:

- we do have honest evidence that the new selected params beat baseline on the untouched `2025+` outer holdout on log loss
- we do not have an honest out-of-sample ROI comparison of the exact old live artifact versus the exact new live artifact

That means the promotion decision was defensible as a predictive-model-selection decision, but not proven as a better-ROI decision.

## What Counts As A True Honest ROI Comparison

A true honest ROI comparison from this point forward must satisfy all of the following:

1. Both model bundles are frozen.
2. Both models are evaluated only on fights that happen after the comparison start date.
3. Both models see the same feature snapshot and the same market snapshot at decision time.
4. Both models use the same stake sizing, filters, and execution assumptions.
5. No thresholds, params, or policy rules are changed mid-evaluation.
6. ROI is measured on future outcomes only.

If any historical rows that were already in the training data are used as the evaluation window for the new full-fit artifact, the ROI comparison is not honest.

## Recommended Path

Keep the new full-fit bundle live for now, but start a forward shadow comparison immediately.

### Primary recommendation: prospective shadow evaluation

Run the old bundle in shadow alongside the current live bundle for future UFC events only.

Required setup:

- current live bundle stays active: `full_live_contract_v6_fullfit`
- shadow comparison bundle: the backed-up old `full_live_contract_v6_tuned`
- comparison start date is the first event after promotion on March 26, 2026

For every eligible fight after the comparison start date, persist:

- event and fight identifiers
- timestamp of model decision
- market snapshot used for the decision
- both models' predicted probabilities
- both models' gated bet decisions
- quoted prices
- requested stake
- assumed or observed fill price
- settlement result
- realized P&L

Metrics to report:

- ROI
- total profit
- total wagered
- number of bets
- win rate
- CLV
- max drawdown
- fill rate if realistic execution is simulated
- log loss and Brier score as secondary predictive diagnostics

Important:

- ROI should be the headline metric
- CLV should be tracked as an early signal, because ROI is noisy over small samples
- do not declare a winner off one card or a tiny sample

### Why this is the only truly honest path now

The exact new promoted artifact is already full-fit on all pre-promotion data. Because of that, an exact old-vs-new backtest over prior history cannot answer the ROI question honestly.

Only future fights can do that honestly.

## Secondary Path: retrospective proxy, clearly labeled

If a faster directional read is wanted, run a retrospective walk-forward ROI comparison between:

- the old promoted V6 tuned setup
- the new trial-19 V6 setup under the same realistic execution rules

But this must be labeled as:

- a retrospective proxy for the model family / selected params
- not an honest ROI estimate of the exact promoted full-fit artifact

This can still be useful for context, but it is not sufficient to prove the promoted full-fit bundle is the better ROI model.

## Existing Evidence We Do Have

### Honest predictive evidence

From `data/processed/optuna_v6_best.json`:

- baseline outer-holdout log loss: `0.5958794283142894`
- best trial outer-holdout log loss: `0.5717407945354541`

Interpretation:

- the chosen trial beat baseline on an untouched `2025+` holdout
- this is the main reason the full-fit retrain was promoted

### Old live model ROI evidence

From `logs/pipeline_hardening_postchange_20260326/tuned_v6_realistic_walkforward_summary.csv`:

- old V6 tuned realistic ROI: `0.21041133301857104`

Interpretation:

- this is the best honest ROI estimate currently available for the old live bundle

### Evidence we do not have

- an honest forward ROI estimate yet for `full_live_contract_v6_fullfit`
- an honest exact old-vs-new ROI comparison for the two promoted artifacts

## Practical Next Steps

1. Freeze the comparison bundles and do not change them during the evaluation window.
2. Add a shadow-eval path that loads the old backed-up V6 tuned bundle while the new full-fit bundle remains live.
3. Persist one row per fight with both models' decisions and outcomes.
4. Produce rolling comparison reports after each settled card.
5. Do not use any historical old-vs-new ROI chart as the final answer for this question.
6. Make the eventual keep-or-rollback decision from forward ROI, with CLV as supporting evidence.

## Explicit Warning

Do not rely on `logs/old_vs_new_backtest.csv` for this question.

That file is from an older March 19, 2026 comparison workflow and is not an honest comparison of:

- old promoted `full_live_contract_v6_tuned`
vs
- new promoted `full_live_contract_v6_fullfit`

## Bottom Line

The current live bundle can stay promoted.

But if the goal is to answer "did the new promoted model actually improve ROI versus the old one?" honestly, the answer cannot come from another historical backtest of the exact promoted full-fit artifact.

It must come from a forward shadow comparison on future fights.
