# Experiment Results — Tier 1-3 Execution vs `full_live_contract_v6_fullfit`

**Branch:** `model-improvements-2026-06` | **Date:** 2026-06-10
**Companion doc:** [MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md](MODEL_IMPROVEMENT_ANALYSIS_2026-06-10.md) (the pre-registration of every experiment)

Every experiment below was run against the production v6 contract with the
repo's own lab tooling, under the corrected evaluation regime built in
Track A. Negative results are reported as prominently as positive ones —
several "obvious" improvements failed honest testing.

---

## Headline verdicts

| # | Experiment | Verdict | One-line result |
|---|------------|---------|-----------------|
| E1 | Live/train parity fixes | **LANDED + verified** | Aggregate metrics neutral; 19% of fights flipped a 2% edge threshold under the stale fallback |
| E3 | Realistic execution in sweeps | **LANDED** | Frictionless basis overstates combined ROI by ~3.5pp and re-ranks configs |
| E5 | Conviction gates | **Directional, NOT promotable** | `c_min_edge=0.02` wins point estimates both bases; fails paired bootstrap (p=0.27) |
| E6 | Lab/production trainer parity | **LANDED** | Lab baseline was the v2 contract w/o mirror augmentation; every prior lab verdict was under the wrong regime |
| E7 | Statistical promotion gate | **LANDED + demonstrated** | Killed a would-have-promoted noise win in its first real use |
| E8 | Sweep selection bias | **Hypothesis REJECTED** | Best-of-2,916 vs forward-chained OOF gap is only +0.34pp |
| E2 | Full-data booster refit | **FAILED** | ROI +0.5% vs +13.6% baseline — sigmoid-transfer trap is real |
| E11 | Calibration ablation | **Neutral** (cal_none harmful) | Weighted/isotonic ≈ baseline; no calibration → negative ROI |
| E12 | Antithetic odds noise | **FAILED at std 0.06** | Worse log-loss (0.609 vs 0.602) and ROI (10.3% vs 13.6%) |
| E9 | Fitted logit stack | **PROMISING** | Beats heuristic blend on all OOF metrics; heuristic blend is worse than the raw market |
| E4 | Edge decay T-7/3/1 | _running_ | (fills-only + features+fills arms) |
| E10 | Defensive-vulnerability features | _running_ | (Track C) |
| E13 | Durability features | _running_ | (Track C) |
| E16 | Rankings backfill | _running_ | Archive installed (533 weekly snapshots 2013→2026-06) |
| E14 | Refit cadence sweep | **Ready to run** | `scripts/e14_cadence_sweep.py` (compute-heavy) |
| E15 | Line movement | **Blocked, blocker confirmed** | Line cols receive ZERO anti-leakage noise — must extend the noise set first |

---

## E1 — Live/train feature parity (the correctness fix)

**What landed** (all in `src/data/fighter_lookup.py`, guarded by
`tests/test_live_train_parity.py` which fails RED on the pre-fix code):

1. `opp_strength` now computes the training-exact recency-weighted
   strength-of-schedule (was: the opponent's `roll_won`, a semantically
   different quantity diverging on 97.5% of replayed fights).
2. Pre-UFC/amateur summaries resolve through canonical fighter names.
3. The live-scrape rolling stats re-roll over the training-seeded frame
   (pre-UFC supplement rows shift every EWM's weights under
   `ignore_na=False` — including interleaved all-NaN duplicate rows).
4. The processed-snapshot fallback rolls forward one fight using the
   training machinery itself; historical replays read the next-row oracle
   bit-exactly.
5. Exact-spelling group discipline (training groups by exact string —
   `same_person_name` was merging "Lance Gibson" with "Lance Gibson Jr.").

**Verification:** exact-mode replay = **202/202 features parity-perfect**
(250-fight sample); prefight-mode residuals are only the deliberate 1-day
time-shift fields and debutants (served by the scrape path live).

**Measured impact** (659 fights, 2025+, production bundle):

| variant | log-loss | brier | acc |
|---|---|---|---|
| training-parity features (reference) | 0.58561 | 0.19948 | 0.6889 |
| old SOS proxy | 0.58521 | 0.19932 | 0.6874 |
| fixed fallback (prefight) | 0.58558 | 0.19948 | 0.6889 |
| old stale fallback (prefight) | 0.58530 | 0.19937 | 0.6995 |

Aggregate metrics are **statistically indistinguishable** — the model is
heavily market-anchored, so even badly stale rolling features barely move
average quality. The defects' real cost is per-fight: the stale fallback
moved predictions by a mean of 2.3 probability points (65/659 fights > 5
points, max 11.8) and **124/659 fights (19%) flipped across a 2% raw-edge
threshold** — wrong bet/no-bet decisions whenever the fallback engaged
(silently, under the post-bdef682 advisory degraded mode).

## E3 + E7 + E8 — The honest evaluation layer

- **E3**: `_evaluate_config` now supports `execution_mode="realistic"`
  (half-spread quotes, synthetic order-book fills, liquidity/slippage caps,
  event-batched settlement; legacy byte-identical, regression-tested).
  Wired through `run_evaluation --execution-mode` into stage 3/4 with
  resume-state invalidation. Measured effect on the production anchor:
  combined ROI 9.2% → 5.7%; conviction trader hit hardest (8.8% → 4.8%,
  CLV → ~0.0001) because spread is a large fraction of favorite edges.
- **E7**: the trading bar now requires a paired event-clustered bootstrap
  win (P(candidate ≤ baseline) ≤ 0.25) and compares snapshot-distinct CLV
  (single-snapshot fights are structurally CLV=0 and are excluded —
  ~16% of bets in the test window had distinct snapshots missing).
  `--legacy-gate` restores the old behavior.
- **E8**: with all 2,916 grid configs evaluated per fold and selection
  forward-chained (choose on folds 1..k-1, score on fold k), the
  out-of-fold ROI of the selection PROCEDURE is +15.65% vs the in-sample
  best's +16.00%. **Selection bias ≈ 0.34pp — the sweep's numbers are
  essentially honest.** No reserved-window process change needed.

## E5 — Conviction gates (directional win, killed by the honest gate)

Under the corrected lab (E6) on identical folds, S-trader at production:

| basis | config | c_bets | c_roi | combined roi | max dd |
|---|---|---|---|---|---|
| legacy | production (no gates) | 441 | 8.8% | 9.2% | 63% |
| legacy | `c_min_edge=0.02` | 355 | **11.7%** | **11.8%** | 55% |
| realistic | production (no gates) | 441 | 4.8% | 5.7% | 66% |
| realistic | `c_min_edge=0.02` | 355 | **7.5%** | 8.3% | 60% |

The odds cap **never binds** (every conviction bet is a ≤1.8-odds
favorite). But the new gate's verdict on the gated config vs the anchor:
paired bootstrap p=0.27 (legacy) / 0.38 (realistic), 95% CI spans zero,
plus a trader-C overexposure flag (C is >80% of volume in this
configuration). **Directionally attractive, not statistically promotable
on 177 events.** Re-adjudicate after more live sample accumulates.

## E2 — Full-data booster refit: FAILED

The walk-forward A/B (production trainer, identical folds):
model log-loss 0.6037 vs 0.6016 baseline, strategy ROI **+0.5% vs +13.6%**,
bets 300 vs 227. The holdout-fitted sigmoid mis-maps the sharper full-data
booster's output distribution, manufacturing phantom edges. The exact trap
the pre-registration flagged ("do NOT refit the sigmoid against the full
booster's in-sample predictions") turns out to bite even in the
transfer direction. **Do not adopt as implemented.** If revisited: refit
requires a calibrator fitted on out-of-sample predictions OF THE FULL
BOOSTER (e.g. nested walk-forward calibration), not a transferred sigmoid.

## E11/E12 — Calibration and noise ablations: neutral-to-negative

| spec | model LL | model ECE | strat bets | strat ROI | CLV |
|---|---|---|---|---|---|
| v6_tuned (baseline) | 0.6016 | 0.051 | 227 | +13.6% | .0305 |
| cal_weighted | 0.6013 | 0.050 | 224 | +13.0% | .0288 |
| cal_isotonic | 0.6021 | 0.046 | 241 | +12.7% | .0293 |
| cal_none | 0.6027 | 0.034 | 252 | **−0.8%** | .0246 |
| noise_antithetic | 0.6092 | 0.054 | 259 | +10.3% | .0332 |

- Fixing the silent weight-drop in the calibrator (`temporal_holdout_weighted`)
  is metrically a wash — keep for hygiene consideration only.
- Removing calibration entirely improves raw ECE but **destroys bet
  selection** — calibration is load-bearing for the economics.
- Antithetic identity-preserving noise (a+b=1, diff=a−b, mirror-consistent)
  is mechanically correct (unit-tested) but **hurts** at std 0.06. The
  independent noise's identity-breaking apparently acts as useful extra
  regularization. A std re-tune under the antithetic scheme remains untested.
- Blend instrumentation (E11 phase 0) now ships in every walk-forward
  report: baseline blended-probability LL 0.5950 / ECE 0.041 / band ECE 0.023.

## E9 — Fitted logit stack: PROMISING

Walk-forward (fit on prior folds only, mirror-duplicated for symmetry),
pooled out-of-fold over 1,250 fights:

| probability | log-loss | brier | ECE |
|---|---|---|---|
| fitted stack | **0.58884** | **0.20198** | **0.03480** |
| production heuristic blend | 0.59281 | 0.20284 | 0.04660 |
| raw market | 0.59232 | 0.20307 | 0.03723 |

Two notable facts: (1) the hand-tuned heuristic blend is *worse than the
raw market* on log-loss — it has been subtracting probability quality;
(2) the stack's gain comes mostly from sharpening the market itself
(market coefficient 1.1-1.5 > 1; model coefficient near zero, drifting
positive in later folds). Intercept ≈ 0 confirms the symmetry construction.
Caveat: stack "edges" are mostly market-recalibration (favorite-longshot
correction), not model alpha — bet-level adjudication under realistic
execution required before any strategy change.

## E10 / E13 / E16 — Feature families (Track C)

All three implemented end-to-end (training build + live scrape path +
processed roll-forward + specs `v6_grapdef`, `v6_durability`,
`v6_plus_rankings`):

- **E10**: 12 opponent-allowed rolling stats (takedowns/control/sub-attempts
  conceded). No new data source — UFCStats opponent-side stats are already
  scraped per fight; only the differentials are new columns.
- **E13**: `loss_ko_rate`, `loss_sub_rate`, `recent_ko_loss` from per-fight
  methods (100% historical coverage; live scraper captures method per fight).
- **E16**: installed the martj42 historical rankings archive
  ([github.com/martj42/ufc_rankings_history](https://github.com/martj42/ufc_rankings_history),
  533 weekly snapshots 2013-02 → 2026-06, schema exactly matches the
  existing `_historical_rankings_overlay`). The Track C rebuild enriches
  rank coverage point-in-time with no look-ahead (backward merge_asof).

Results: see `logs/track_c_batch/track_c_summary.csv` (campaign running at
time of writing — RESULTS APPENDED BELOW when complete).

## E4 — Edge decay vs time-to-event

Plumbed end-to-end (`entry_offset_days` + `entry_offset_for_features`
through `_merge_historical_odds` → `_resolve_market_odds` →
`run_walkforward_strategy_comparison`/`run_backtest`); fills can now be
priced at the T-7/T-3/T-1 snapshot with CLV still measured vs closing.
Curve runs in flight — RESULTS APPENDED BELOW when complete.

## E14 / E15 — deferred with concrete next steps

- **E14**: `scripts/e14_cadence_sweep.py` is ready (cadence {1,2,4,6} months
  + never-swept `train_start_date` arms). Compute-heavy; run when idle.
- **E15**: blocker empirically confirmed — `ODDS_FEATURE_NAMES` does not
  include any line-movement column, so enabling `add_line_movement` today
  would feed noise-free near-closing-precision movement values (leakage).
  Extending the noise set (or an explicit documented waiver) is a hard
  prerequisite. Train/live semantics reconciliation additionally needs the
  Railway line-history snapshots.

## Operational notes

- **Railway ledger pull** (calibrating `assumed_half_spread` from real
  fills; conviction/LLM-operator ledger analysis) needs production access
  that this session was not authorized to use — run manually or grant
  access: `railway run --service ufc-bot -- <ledger export>`.
- **Control-arm re-freeze**: with realistic mode + the statistical gate
  landed, re-freeze a v6 control arm WITH trading artifacts (the current
  frozen arm has none, so stage-4 trading verdicts are blocked). Use
  `run_evaluation --execution-mode realistic` for the freeze run.
- **Process hazard hit during this work**: a `git add -A` commit raced a
  background task that had checked out an old file version, silently
  reverting the E1 fixes in one commit (repaired in `ada9917`). Commits in
  this repo should use explicit file lists whenever background jobs run.
