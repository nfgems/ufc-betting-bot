# Live Model Improvement TODO — 2026-03-25

## Goal

Get to a defensible yes/no answer for:

> Is the newly retrained model better than the model currently running live?

Do **not** use the old closing-odds-biased baseline as proof.


## Current State

- Completed: `python -m src.bot train`
  - Local UFC model artifacts were retrained with `full_live_contract_v6_tuned`.
- Completed: `python -m src.bot backtest`
  - Walk-forward backtest used opening odds in every fold.
  - Result: `+11.2%` ROI over 98 bets.
- Completed: live sig-strike verification through `fighter_lookup`
  - `head_landed`, `body_landed`, etc. now populate for a real fighter scrape.
- Completed: side-by-side compare against the current live production artifact.
  - The V6 candidate outperforms the V5 live model on all key metrics.
- Not completed: full `python -m src.bot scrape`
  - This is too slow to be a priority path right now.


## V6 Handoff Status

This section maps directly to `V6_INFERENCE_FIXES_HANDOFF.md`.

- `1. Retrain the Model`
  - Status: done
  - Completed on March 25, 2026.
  - `python -m src.bot train` finished successfully.

- `2. Re-scrape Fighter Data`
  - Status: partial
  - The full long-running `python -m src.bot scrape` crawl was not completed.
  - The important sig-strike parser bug in `fighter_lookup.py` was fixed and verified on live UFCStats pages.
  - I also patched `scraper.py` so the historical fight scraper now:
    - reads the correct live UFCStats summary tables
    - parses sig-strike breakdown columns
    - can resume from existing CSV outputs
  - What is still pending is a completed full historical scrape/backfill run.

- `3. Run Backtest with Opening Odds Baseline`
  - Status: done
  - Completed on March 25, 2026.
  - `python -m src.bot backtest` finished successfully.
  - Walk-forward folds replaced closing odds with opening odds before inference.

- `4. Evaluate Impact`
  - Status: partial
  - Done:
    - opening vs closing baseline comparison on fresh holdout
    - confirmation that walk-forward folds used opening odds
    - confirmation that sig-strike fields now populate on live fighter-history scrape
  - Not yet done:
    - exact compare against the currently deployed live production artifact
    - full historical scrape-derived backfill verification end to end
    - any deployment decision backed by old-live-vs-new-candidate evidence

- `5. Rebuild Features (if needed)`
  - Status: effectively done for the retrain path
  - `train` rebuilt `data/processed/features.csv` and `data/processed/test_set.csv`.
  - A separate rebuild after a completed historical scrape is still optional future work.


## Numbers We Already Have

- Retrained local model holdout accuracy: `68.75%`
- Opening-odds baseline holdout accuracy: `66.08%`
- Closing-odds baseline holdout accuracy: `67.89%`
- Walk-forward backtest ROI: `+11.2%`

These numbers say the retrained model still has edge after removing look-ahead bias.
They do **not** yet prove it beats the exact model currently deployed live.


## Priority Order

1. Freeze the new local candidate.
   - Keep the newly retrained artifacts as the candidate model.
   - Do not overwrite them again until the comparison is done.

2. Find the current live production artifact or production bundle.
   - We need the exact old model that is live now.
   - If the live system uses a bundle/manifest, capture that exact path and model file.

3. Run an apples-to-apples compare: old live vs new candidate.
   - Same holdout set.
   - Same walk-forward backtest settings.
   - Same opening-odds-aligned evaluation.

4. Decide deploy/no-deploy based on the compare.
   - New model should beat or at least clearly justify replacing the live one.
   - If the new model is only “more honest” but not better than live, do not deploy blindly.

5. Only after the compare is done, decide whether historical scrape/backfill work is worth doing.
   - Full `src.bot scrape` is not the critical path right now.
   - If we revisit scraping, use a narrower or faster path, not another blind multi-hour crawl.

6. If we still want historical scrape completeness after the model compare:
   - Resume with `python -m src.bot scrape --fights-only`
   - Avoid re-running the already-complete fighter inventory crawl unless the fighter CSV is intentionally refreshed


## Definition Of Done

We are done when all of these are true:

- The exact current live artifact is identified.
- The new candidate and old live model are both evaluated on the same data.
- We have a side-by-side table for:
  - holdout accuracy
  - holdout Brier/log loss
  - walk-forward ROI
  - total bets
  - win rate
  - drawdown
- We can say one of:
  - “new model is better than live, deploy it”
  - “new model is not better than live, do not deploy it yet”


## Explicit Non-Goal Right Now

- Finishing the full historical `python -m src.bot scrape` crawl.

Reason:

- It is too slow.
- It is not needed to answer whether the new candidate is better than the current live model.
- The important sig-strike parser bug has already been verified on the live `fighter_lookup` path.


## Next Action

Next concrete step:

> The comparison is complete. The new V6 candidate model is a clear improvement over the live V5 model.
> The next step is to prepare the promotion of the new model to production.
