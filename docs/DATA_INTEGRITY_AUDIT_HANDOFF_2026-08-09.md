# Data-Integrity Audit — Verification Handoff (2026-08-09)

Audience: AI session tasked with independently double-checking and confirming/refuting the findings below. Not a remediation doc — verify first, fix nothing until the owner approves.

## Audit provenance

- Audited tree: commit `6d3e9b4` (main). During the audit `0fd84e4` ("Persist and restore prediction history") landed; it touches only `src/bot.py` (+35 lines, may shift bot.py line refs slightly), `src/prediction_history.py` (new), `src/web/app.py`, templates, tests. No finding's subject code was modified. Findings marked "verified at 0fd84e4" were re-checked after that commit.
- Method: 35-agent workflow — 2 mappers (pipeline trace + pandas coverage quantification), 8 dimension auditors (fabricated defaults, train/live parity, leakage, coverage/identity gaps, spec/contract integrity, eval honesty, live money path, prior-fix regression), adversarial verification (HIGH findings: 2 independent refuters with correctness + materiality lenses; MEDIUM: 1 refuter), completeness critic. 2 findings were refuted and killed (§6); severity corrections from verifiers are already applied below.
- Production contract audited: `models/full_live_contract_v6_durability_fullfit_spec.json` (211 feature_cols, `impute_strategy=native_nan`, `odds_noise_std=0.06 independent`, family gate `{moneyline_odds: 98.0}`), manifest `models/current_production_model.json` (bundle 2026-08-04, git d0c811d).

## Verification protocol for this session

1. Work read-only. Do not modify repo files. Run pandas/json checks with `python -X utf8 -c` from repo root, `PYTHONPATH=.`.
2. `pytest` is slow — run only single named test files with timeout ≥ 600000 ms. `tests/test_live_train_parity.py` takes ~4 min.
3. For each finding: reproduce the VERIFY steps, then attempt to refute (already-fixed at HEAD / intentional-by-design / dead code / mischaracterized). Record CONFIRMED or REFUTED with evidence.
4. Known non-findings (do not report): UFCStats plain HTTP is intentional; `models/*_spec.json` raw counts > builder counts is by-design Elo provenance; degraded-freshness betting path is a documented owner decision (bdef682); legacy frictionless execution default was examined and refuted as mischaracterized (§6).

---

## 1. P0 findings (confirm these first)

### F6 [HIGH] Same-name fighter identity merges corrupt training features and live lookups
`src/features/build_features.py:354` (also 427, 528, 1237, 1351); `src/data/fighter_lookup.py:1336`, `:1714-1740`
- Claim: Fighter identity is raw-name-keyed end-to-end (`fights_cleaned.csv` has no ID columns; `groupby("fighter")`; profile join first-row-wins on normalized name). Distinct fighters with identical names share one career. In the bundle-pinned `features.csv`: Jean Silva (FW, born 1996) carries age 45.9–48.3 across his 8 rows 2023–2026 (namesake born 1977 wins the DOB join) with `num_fights` continuing from a 2005 Gomi fight; Bruno Silva (FLW) + Bruno Silva (MW) form one merged 0→22 `num_fights` chain with MW KO losses contaminating FLW `loss_ko_rate` (0.273 on 2025-10-18); Mike Davis has a 2008 row at age 15.9; Victor Valenzuela 2026-04-25 collides. Live: `_lookup_processed_fighter` merges identical spellings; `search_fighter_url` returns first exact-token UFCStats index match, no DOB/record disambiguation. 11 canonical-key collision groups among 4,493 scraped fighters. `REVIEWED_FIGHTER_IDENTITIES` (name_utils.py) covers only 3 curated fighters, none of these. `a_age/diff_age/age_over_35/age_squared` are spec features in a durability-focused model; these fighters are active — real-money predictions consumed the wrong person's data. Worse than a NaN-rule violation: real observations of the WRONG person.
- Verify: pandas over `data/processed/features.csv` filtering `fighter_a/fighter_b == 'Jean Silva'` (ages 45.9–48.3 on 2023+ rows) and `'Bruno Silva'` (single 0..22 chain across FLW+MW weight classes); grep `data/raw/ufc_fighters_scraped.csv` for two UFCStats IDs per name with different DOBs; read the groupby sites.
- Status: UPHELD by 2/2 adversarial verifiers, both reproduced the numbers independently.

### F3 [HIGH] Promoted bundle predates the Aug-6 point-in-time repairs — live recompute diverges from training rows on 20/25 recent fights
`src/features/build_features.py:1333`; `models/current_production_model.json`
- Claim: The promoted bundle (built 2026-08-04, git d0c811d) was trained pre-repair; HEAD live code includes the 2026-08-06 repairs (commits `32e022f`, `4118b81`: `_filter_pre_ufc_rows_before_tracked_debut`, `_is_tracked_ufc_organization`, `_drop_quarantined_supplement_identities`, reviewed-history loading), which change pre-UFC seed rows and therefore EWM positions for the whole `roll_*` family. The repo's own parity replay bound to the production contract fails 20/25 sampled fights (drifts ~1–5%, up to 6.9 pts on `diff_roll_td_acc`; `diff_roll_td_avg` flips sign on Kim vs Mar Fan). The corrected promotion (`full_live_contract_v6_durability_corrected_20260805_fullfit`, `docs/CORRECTED_DURABILITY_PROMOTION_PACKET_2026-08-06.md`) is prepared but NOT deployed; a verifier confirmed the Railway runtime serves the mismatched Aug-4 bundle.
- Verify: `UFC_PARITY_TRAINING_SPEC=full_live_contract_v6_durability_fullfit python -m pytest -q tests/test_live_train_parity.py` → `test_exact_replay` PASSES, `test_prefight_replay` FAILS 20/25 (~4 min). Confirm local processed == promoted bundle: sha256 of CRLF-normalized `data/processed/features.csv` == manifest `processed_features_sha256` (`3f488cf5…`). `git diff 972911d..HEAD --stat -- src/features/build_features.py` (547 changed lines).
- Status: UPHELD 2/2; independently reproduced with a pre-repair control run proving causal attribution; Railway runtime state confirmed by one verifier.

### F5 [HIGH] Scheduled eval silently reverts to closing odds for ALL fights since 2026-03-07 (model features AND fill prices)
`src/strategy/backtest.py:249-251, 286-291, 484-511, 188-200`; `config/scheduled_refit_policy_v1.json:25-27`
- Claim: The weekly health gate + Track-C evidence run with `entry_offset_days=1.0, entry_offset_for_features=true`, but "opening" = max-offset snapshot per fight and "entry" requires `offset_days >= 1`. Every odds snapshot after 2026-03-07 has `offset_days=0` (Odds API archive dead — see F8; BFO recovery writes offset 0 only). So for post-2026-03-07 fights: the T-1 entry merge silently no-ops, the lone closing snapshot is treated as "opening", closing odds are written into `a/b/diff_implied_prob` model features (post-prediction-moment information), and the same closing snapshot becomes the fill price. Reproduced: 0/184 post-2026-03-07 fights have any offset≥1 snapshot; entry coverage 2025: 50.5%, 2026: 21%. No minimum entry-snapshot coverage check exists in `scripts/check_scheduled_refit_quality.py` or the policy, and the flattered region grows weekly.
- Verifier correction (apply when confirming): the core mechanism (closing odds as features + fills for the entire recent window) is CONFIRMED; the original sub-claim about the CLV-floor gate being progressively flattered was partially refuted — treat the CLV-dilution arithmetic as secondary, not load-bearing.
- Verify: `python -X utf8 -c "import pandas as pd,glob; [print(p, sorted(pd.read_csv(p)['offset_days'].unique())) for p in glob.glob('data/raw/historical_odds/historical_odds_bfo_recovered_*.csv')]"` → all `[0]`; `historical_odds.csv` max event_date = 2026-03-07; read the four backtest.py ranges above.
- Status: UPHELD 2/2 (one severity note as above).

### F14 [HIGH→MEDIUM] NaN no-odds probability bypasses the model-agreement gate and the conviction dual-model gate (fail-open)
`src/strategy/value.py:471-483, 1031`; `src/bot.py:4542-4552, 4755` (pre-0fd84e4 numbering)
- Claim: Both no-odds confirmation gates check `is None` then a `<` comparison — NaN passes both (`NaN is not None`; `NaN < x` is False). NaN is manufactured in production: when the no-odds prediction throws for one fight on a multi-fight card, `bot.py` leaves None and logs "model agreement filter will block this fight", but `pd.DataFrame(prediction_rows)` coerces the mixed column to float64 → NaN. `find_value_bets` then emits a real-money bet with REQUIRE_MODEL_AGREEMENT=True, and `find_conviction_bets` emits a conviction bet (up to 8% of C allocation) with `no_odds_prob=NaN, conviction_score=NaN`. Executed repro at HEAD returned 1 value bet + 1 conviction bet on the NaN row; the all-None single-row control correctly returned 0/0 (proving the design is fail-closed and defeated only by the None→NaN coercion). Note `prob_a/prob_b` in the same loops go through `_coerce_probability` which maps NaN→None — no_odds is read raw.
- Verify: run the 2-row DataFrame repro in the audit transcript (rows: one with `no_odds_prob_a/b=None` + valid probs/markets, one fully valid; call `find_value_bets(p, min_edge=0.02)` and `find_conviction_bets(p)`; expect 1 and 1). Then read the two gate sites.
- Status: UPHELD 2/2; one verifier corrected high→medium (trigger requires partial no-odds failure on a mixed card — real but narrower surface). Treat as P0 anyway: silent, real-money, trivially fixed.

---

## 2. P1 findings

### F4 [MEDIUM] Parity gates never replay HEAD code against the currently-promoted bundle (F3 ships invisibly)
`tests/test_live_train_parity.py:37-39`; `.github/workflows/ci.yml:28-35`
CI rebuilds fresh parity artifacts at HEAD and replays against them (code and artifacts always same version); weekly refit gates only pre-promotion candidates; local default spec is stale `full_live_contract_v6_fullfit`; CI pins `…corrected_20260805_fullfit`; nothing reads `models/current_production_model.json:model_spec_name`. Verify: read both files; run the parity test once against shipped `data/processed` (fails per F3) and once against a HEAD rebuild via `UFC_PARITY_PROCESSED_DATA_DIR` (passes, matching CI). Status: UPHELD.

### F1 [MEDIUM] Live UFCStats profile scrape turns missing per-fight stat cells into 0s feeding roll_* features (dormant but armed)
`src/data/fighter_lookup.py:2193-2233, 1524-1531` vs `src/data/kaggle_loader.py:98-105`
`scrape_fighter_fights()` (primary live history source; `prefer_live_refresh=True` at bot.py:3185/4496): kd/opp_kd → 0 when cell lacks `<p>` tags; `sig_str_landed`/`td_landed` init 0 and `_safe_float(text, 0)` so `'--'` → 0; sub_att → 0 on empty. Flows into slpm/sapm/td_avg/sub_avg recomputes → `_compute_rolling_for_fighter` → 21 spec features. Training parses the same data via kaggle_loader `_safe_float` → None/NaN. Fail-open on UFCStats layout drift: malformed ROW raises, malformed CELL silently zeros every fight; fabricated 0s don't register in `_live_prediction_quality_assessment`. Currently dormant: `'--'` cells absent from modern histories (0.00% NaN on 2014+ landed-count cols; 33 affected rows all ≤2012). Contrast with every sibling parser (NaN on missing) and the detail-scraper docstring "never estimated or fabricated". Status: UPHELD.

### F2 [MEDIUM] Live context fallback fabricates is_title_bout=0, is_empty_arena=0, num_rounds=3 — all three ARE v6 spec features
`src/bot.py:2644-2716` (post-0fd84e4 numbering); `src/data/fighter_lookup.py:3513-3518`
Three sub-mechanisms, statuses differ — verify each:
1. Off-card-history branch (bot.py:2687-2692): production-DEAD — all three production call sites (bot.py:3142/4211/4348) pass `allow_off_card_history_fallback=False`. Original auditor overstated this branch; verifier corrected.
2. Near-term-lookup branch (bot.py:2709-2714): **production-LIVE** — NOT gated by that flag (orchestrator re-verified at 0fd84e4). A bettable fight matched to no scraped card row but with an inferable weight class gets hardcoded `{is_title_bout: False, is_empty_arena: False, num_rounds: 3}`; `fighter_lookup.py:3513-3518` converts False→0.0 (the None→NaN path exists and is defeated by hardcoded False). Training preserves NaN for exactly these unknowns (`materialize_honest_context_features`, build_features.py:671-687; `infer_empty_arena` NaN when title+location missing).
3. Matched-row unparseable num_rounds (bot.py:2650-2652, live_monitor.py:881/963): guesses 5-if-main-event-or-title-else-3 by convention; training reads observed time_format, NaN when unknown. Also `best.get("is_title_bout", False)` defaults to False rather than NaN.
Spec membership settled by orchestrator at 0fd84e4: `is_title_bout` ✓, `num_rounds_feat` ✓, `is_empty_arena` ✓ (the completeness critic's claim that num_rounds is not in spec checked the wrong column name — ignore it). Status: UPHELD as medium via mechanisms 2+3.

### F15 [MEDIUM] BFO method-odds fail-closed fix (d993ec1) doesn't cover marker-less pages — cross-fight odds attribution
`src/data/method_odds.py:1440-1445, 194-203`
When no table carries `mu-` markers, `_parse_bfo_method_odds` falls back to page-wide `soup.select('tr')` guarded only by name matching; `_identify_fighter_side`'s last-name fallback attributes a "Silva wins by TKO/KO" row from a different fight to our fighter. Reproduced with synthetic marker-less HTML (Anderson Silva prop attributed to Bruno Silva, `a_ko_odds_prob=0.1`). Live production path (Odds API method source permanently disabled at :1529-1541; BFO is sole source); hole opens exactly on the markup-regression event d993ec1 defends against. All 6 method-odds features in spec. Status: UPHELD, repro confirmed.

### F10 [MEDIUM] Stage-3 winner's curse: max-ROI over ~2,916 correlated configs on one walk-forward window feeds the promotion gate
`src/strategy/run_evaluation.py:1541-1811, 2013-2016`; `src/strategy/duo_trader_sweep.py:719-726, 835-846`; `src/strategy/promotion_gate.py:394`
Broad grid (3·3·2·3·3·3·2·3=2,916) + narrow follow-up, all on ONE fold-prediction set, sorted by ROI; max row becomes `best_result` consumed by stage 4. Holdout re-check explicitly advisory ("do not override selection"). Bootstrap alpha=0.25. Absolute ROI the owner reads is a max order statistic over ~3k tries on ~250 bets. Status: UPHELD (verifier corrected high→medium: gate comparison is candidate-vs-control where both carry the bias; the ROI NUMBER is inflated, the gate decision less so).

### F11 [MEDIUM] model_lab predicts BEFORE the opening-odds swap; backtest's swap is a no-op on the 41% of rows carrying Kaggle closing odds
`src/strategy/model_lab.py:426-431, 567-578` vs `src/strategy/backtest.py:476-511`
Stage 1–4 orchestrator path calls `_predict_batch_with_model` before `_merge_historical_odds` — the look-ahead-prevention swap never runs there (despite model_lab.py:470 claiming to "mirror" backtest). And the swap only replaces rows having an opening snapshot; rows sourced from Kaggle CLOSING odds (1,091/2,654 fights 2022+ = 41.1%; 65.3% of 2026) are untouched. Where both present, implied_prob == opening 100% byte-identical (overlay prefers opening), so the swap is only load-bearing exactly where it can't fire. Mitigation verified: 0/227 bets in the latest believed run landed on closing-only rows. Status: UPHELD.

### F13 [MEDIUM] Promotion gate compares candidate vs incumbent across different fight windows; paired bootstrap zero-fills missing months
`src/strategy/gate_stats.py:86-100`; `data/frozen/control_arm_20260320/`
Union-of-event-dates with `.get(e, 0.0)` credits the candidate for months the frozen control never traded (control test range ends 2026-03-07). Currently moot in the orchestrator (arm has only control_metrics.json → stage 4 BLOCKED fail-closed) but the standalone CLI (`python -m src.strategy.promotion_gate --candidate-log/--baseline-log`) has no window-alignment check at all. Status: UPHELD.

### F7 [MEDIUM] Method-odds family (6/211 features) has received ZERO training signal since 2026-03-07; the family coverage gate that would catch it was dropped in v6
`src/data/ufc_refresh.py:491`; `data/raw/method_odds/historical_method_odds_all.csv`
Sole training source max event_date 2026-03-07; writers are only manual `scripts/backfill_v5_*` (no scheduled workflow appends). Coverage on train-eligible rows: 0.00% for all 167 rows after 2026-03-08 (features.csv monthly: 100% NaN every month since 2026-04). v4/v5 specs gated method_odds coverage; v6 gates moneyline only — every weekly refit silently widens the train/live regime gap (live still fetches real method odds). NaN rule honored; the issue is the silent decay + gate removal. Status: UPHELD.

---

## 3. P2 findings

### F8 [LOW, data-gap] Odds API moneyline archive dead since 2026-03-07; recent training odds are solely weekly BFO closing-line recovery
Same death date as F7. All-fights odds coverage 2026: 82% (2026-03: 65.4%, 2026-04: 60.0%) vs 87–94% 2017–2025; train-eligible 2026 moneyline coverage 96.0% vs ≥99.7% all prior years (98% gate still passes). Provenance regime shift: pre-2026-03 = opening-preferred Odds API snapshots; post = BFO closing (offset 0). Chinese cards (Road To UFC) are permanent BFO gaps. Status: UPHELD (medium→low corrected).

### F9 [LOW] Promoted production spec cannot pass its own strict staged validators (odds_noise_seed 42-vs-None)
`src/model/production_bundle.py:1260-1266, 1342-1350`; `scripts/build_staged_production_bundle.py:58, 1658`; `src/model/train.py:1049-1050`; `src/model/training_spec.py:924-938`
Registry leaves seed None; train backfills 42 into the embedded spec; two strict validators strip only git_hash/trained_at → diff `{'odds_noise_seed': (42, None)}` → raise. Two sibling validators normalize deferred seeds (contradictory policies). Fail-closed: the live artifact can't be re-validated/re-staged/rolled back through the strict path. Corrected spec (`…corrected_20260805`) pins seed=42 explicitly; the promoted one doesn't. Status: UPHELD (medium→low).

### F12 [LOW] `_resolve_market_odds` silently prices unmatched rows at no-vig Kaggle CLOSING; `allow_closing_odds` guard fires only frame-wide; per-row provenance erased
`src/strategy/backtest.py:163-224`
One fight with opening data disables the guard for the whole frame; `odds_source` stamps one concatenated frame-wide label (all 227 recent bets carry the identical string); per-bet marker is only `clv.isna()`. Latent (0 bets used the layer in the believed run). No caller ever passes `allow_closing_odds`. Status: UPHELD.

### C1 [LOW, critic] Kaggle legacy overlay merges a_/b_ columns on an ORDER-INDEPENDENT fight key with no orientation swap
`src/data/ufc_refresh.py:1036-1058, 1421-1435` (contrast the orientation-aware siblings at :279-288, :549-564)
6/6,197 matched fights have reversed orientation; on ALL 6 the raw method-odds columns carry the opponent's values. Moneyline damage rescued by the swapped historical overlay; residue confined to method-odds raw columns at HEAD. Structural wrong-side-data bug worth fixing despite small blast radius. Status: critic-reported, spot-checked empirically by the critic, NOT adversarially verified — verify from scratch.

### C2 [LOW, critic] BetsAPI challenger-feature builder assigns home→slot-a over unordered fight keys with zero reorientation
`src/data/betsapi_mma.py:151-158, 1104-1115, 2302-2506`
Every side-specific `betsapi_*` eval feature is wrong-side whenever BetsAPI listing order differs from the frame's. NOT in the production contract (verified: zero betsapi_ cols in spec) — experiment/challenger stack only (run_evaluation, model_lab, compare_matrix). Data impact unquantified (0 local event_snapshot rows). Status: critic-reported, NOT adversarially verified — verify from scratch.

---

## 4. Minor/info findings (passthrough, not adversarially verified)

- Sherdog fallback parse fabricates 0-0-0 record when winloses widget absent — `fallback_scrapers.py:2901` [LOW]
- `_h2h_summary` returns is_rematch=0/h2h_record_diff=0 when both lookups returned no data — `fighter_lookup.py:3002` [LOW]
- SHAP explanation vectors median-imputed → displayed explanations describe a different vector than the model scored — `bot.py:~4590` [INFO]
- Training rows with absent UFCStats bout string get title_bout=0 not NaN — `ufc_refresh.py:1480` [INFO]
- wc_move: debutants 0 in training vs NaN live — `fighter_lookup.py:2941` [INFO]
- 'independent' odds noise trains off the no-vig manifold every live row satisfies — `train.py:804` [INFO]
- Pre-2022 "historical opening odds" file is Kaggle closing relabeled as T-1 — `historical_backfill.py:428` [LOW]
- Nickname/suffix canonicalization creates 4 cross-source false-positive identity merges — `name_utils.py:122` [LOW] (adjacent to F6 — check together)
- 7.7% of 2014+ fights (515/6,703) have no odds anywhere; concentrated in debut/DWCS/Road-To-UFC; mostly unrecoverable [INFO, gap]
- Manifest git_sha stale by 4.5 months across three refits — `current_production_model.json:17` [LOW]
- Draw/NC fights dropped from every eval path; Polymarket draw resolution unmodeled — `backtest.py:1599` [LOW]
- Stage-1 "closing_odds_baseline" is actually opening for ~half its rows; default frozen control can't support stages 2/4 — `run_evaluation.py:741` [INFO]
- Conviction trader experience gate fails open on unknown fight counts (value path fails closed) — `value.py:1041` [LOW]
- `parse_fight_market` fabricates missing outcome price as 1−other_price — `markets.py:293` [LOW]
- Offline/experimental trainers still fit slot-A-biased rows without mirror augmentation — `train_experimental.py:117` [LOW] (production trainer is clean)
- Manifest hash pins are LF raw-byte hashes; Windows CRLF mismatch is not drift — [INFO]
- CLAUDE.md "don't commit .pkl" contradicts actual deployment mechanism — [INFO]
- Degraded freshness path can bet on stale snapshot during dual-source outage — by design, loudly logged — [INFO]

## 5. Key coverage facts (quantified at HEAD, features.csv 2014+ window, 6,715 rows)

- All 211 spec cols present. 55/211 cols >20% NaN — worst: diff_loss_ko/sub_rate 45.2%, diff_roll_control_per_td 44.7%. diff_roll_ family mean 23.1% NaN, flat-to-improving by year (structural: debut/short-history fighters; consistent with NaN rule — NOT an outage).
- Odds/market family by year: ~9–16% NaN 2017–2025, **2026: 57.9%** (implied_prob 17.7%, method-odds 78% → 100% since Apr). Moneyline capture degraded windows: Aug–Oct 2025, Mar–May 2026.
- Label balance: slot-A win 57.3% overall (63.2% in 2014) in the STORED csv — raw artifact remains winner-biased; mitigation is on-load: mirror augmentation `_mirror_augment_training_rows` (train.py:442) + symmetrized prediction (`predict.py:_predict_prob_a_symmetrized`). Confirmed applied in the v6 production training path; slot A is NOT alphabetical in the stored data (49.7%).
- Row integrity: 0 duplicate fights in window; 329 surplus dup rows all 1994–2010 (outside window); fights_cleaned↔features key parity exact (9,338); only zero-fight month is 2020-04 (COVID).

## 6. Refuted findings — do NOT re-report

1. "Frictionless legacy execution is the default across the eval stack" — mechanically true that legacy is the default entry-point mode, but the load-bearing impact claim was false at HEAD; the believed headline numbers come from realistic-mode runs.
2. "Headline +13–15% ROI is statistically fragile / contradicted by its own negative CLV" — raw numbers reproduce (n=227, ROI +14.91%, event-clustered bootstrap CI [+1.6%, +28.5%]), but the damning interpretations mischaracterized the code (negative CLV is partly mechanical from the realistic execution model). Note F10/F11 still apply to how that number was selected.

## 7. Areas verified clean (spot-check only if time permits)

Storage retention (atomic, protects newest valid snapshots); line_tracker/line_history_archive (S3-before-delete, required in prod); the now-retired LLM operator gate (historically fail-closed BLOCK in real mode, PASS/BLOCK only, never resized); prediction_history (display-only, drops post-start rows); tracker auto-settle (settles only from Polymarket resolution — no phantom-win vector); dashboard PnL endpoints (degradation flagged, win_rate from settled only); rankings path (not in v6 spec; NaN on failure); executor `_place_bet_locked` price/edge/Kelly chain (no NaN fail-open found — the fail-open is upstream in value.py, F14); event_context.infer_empty_arena (NaN when unknowable, shared train/live); training-side NaN discipline in build_features (absence → NaN throughout); calibration group-aware splits (mirror twins kept together).

## 8. Suggested confirmation order

F14 (5-min repro) → F6 (pandas, 10 min) → F5 (pandas + code read) → F3 (parity test, ~4 min runtime) → F2/F1/F15 (code reads + small repros) → F4/F7/F10/F11/F13 → C1/C2 (unverified, need full verification) → P2/minors as time allows.

Full agent transcripts: `C:\Users\Evan\.claude\projects\c--Users-Evan-betting-bot-ufc-betting-bot\befd3426-3868-4fcd-bd57-e0d182f7b59d\subagents\workflows\wf_17a180ba-a30\journal.jsonl` (one result line per agent, includes evidence and exact repro commands).
