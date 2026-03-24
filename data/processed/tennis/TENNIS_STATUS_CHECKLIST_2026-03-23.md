# Tennis Status Checklist

Date: 2026-03-23

## Current Read

The tennis system is no longer an early prototype.

It currently has:
- historical ATP/WTA ingestion
- modeling-safe match exports
- odds matching and diagnostics
- player-profile enrichment
- rankings-history enrichment
- feature generation
- trained tennis models
- anchored OOS evaluation
- strict lockbox evaluation
- live odds discovery
- Polymarket market matching
- automation controls
- Gemini veto support
- dry-run live artifacts

It does **not** yet have a clean, fully-audited, clearly-documented production boundary.

Best label for the project right now:
- `late experimental / pre-production hardening`

## Verified In This Tree

- Tennis processed tables currently span `2022-01-03` through `2026-03-18`.
- `data/processed/tennis/matches.csv` has `24,676` rows.
- `data/processed/tennis/matches_modeling_safe.csv` has `24,656` rows.
- `data/processed/tennis/features.csv` has `24,656` rows.
- Tennis model files exist in `models/tennis/`:
  - `lean_hybrid.pkl`
  - `surface_elo.pkl`
- Anchored OOS evaluation exists for `lean_hybrid`:
  - `17,216` prediction rows
  - log loss `0.6256`
  - accuracy `0.6423`
  - ECE `0.0089`
- Strict 2026 lockbox evaluation exists for `lean_hybrid`:
  - lockbox start `2026-01-01`
  - `1,577` lockbox rows
  - log loss `0.6136`
  - accuracy `0.6493`
  - ECE `0.0164`
- Latest local tennis test run in this worktree:
  - `pytest tests -k tennis -q`
  - result: `138 passed`
- Recent live artifact snapshots exist:
  - `live_reference_decisions.csv`
  - `live_reference_auto_eligible.csv`
  - `live_reference_auto_skipped.csv`
  - `live_execution_decisions.csv`
  - `live_execution_tradeable.csv`
  - `live_execution_auto_skipped.csv`

## High-Confidence Checklist

- [x] Historical ATP/WTA data pipeline exists.
- [x] Modeling-safe base match table exists.
- [x] Odds-joined modeling-safe table exists.
- [x] Player profile enrichment exists and has high fill coverage.
- [x] Rankings-history enrichment exists and has high fill coverage.
- [x] Tennis feature pipeline exists.
- [x] Tennis training command exists.
- [x] Tennis prediction command exists.
- [x] Tennis dry-run live command exists.
- [x] Polymarket tennis market discovery exists.
- [x] Polymarket tennis market matching exists.
- [x] Automation states exist: `auto_eligible`, `auto_skip`, `auto_block`, `not_eligible`.
- [x] Gemini veto layer exists.
- [x] Tennis tests are green in the current tree.

## Still Open

- [x] README and env docs match actual tennis runtime behavior.
- [x] A single explicit tennis live-trading policy boundary is enforced.
- [ ] Tennis has a promoted production manifest equivalent to the UFC line.
- [x] A matched-market end-to-end validation session is captured and signed off in `TENNIS_MATCHED_MARKET_AUDIT_2026-03-23.md`.
- [ ] A truly independent second-source feed is wired in instead of relying mainly on hooks / execution-price cross-checks.
- [ ] The repo has a clean commit boundary for tennis-only work.

## Important Mismatch To Fix First

The main policy confusion that existed earlier is now narrowed:

- docs now describe the experimental tennis trader correctly
- shared-wallet tennis participation is controlled by `TENNIS_TRADER_ENABLED`
- any non-dry-run tennis execution now also requires:
  - `TENNIS_TRADING_ARMED=1`
  - `TENNIS_TRADING_CONFIRMATION=EXPERIMENTAL_TENNIS_TRADING_ENABLED`

That means the remaining work is less about policy ambiguity and more about proof, audit evidence, and production promotion discipline.

## Recommended Next 3 Commits

### Commit 1

`docs: sync tennis runtime policy with current code`

Purpose:
- make the written contract match the runtime contract

Scope:
- `README.md`
- `CLAUDE.md`
- `.env.example`
- `src/config.py` comments
- `data/processed/tennis/TENNIS_STATUS_CHECKLIST_2026-03-23.md`

Changes:
- remove stale `TENNIS_LIVE_TRADING_ARMED` guidance for tennis
- document `TENNIS_TRADER_ENABLED`
- document `TENNIS_PORTFOLIO_SHARE`
- change wording from "dry-run only" to something accurate:
  - recommended wording: `experimental tennis execution path exists but is disabled by default and not yet production-approved`

Done when:
- a new reader can tell, from docs alone, whether tennis can place live orders and what flag controls it

### Commit 2

`feat: enforce one explicit tennis live-trading safety gate`

Purpose:
- eliminate ambiguity between "implemented" and "allowed"

Recommended approach:
- keep `TENNIS_TRADER_ENABLED=0` as the default
- add a second explicit confirmation gate for real tennis order placement
- require both:
  - `TENNIS_TRADER_ENABLED=1`
  - a tennis-specific confirmation string for non-dry-run execution

Suggested scope:
- `src/config.py`
- `src/bot.py`
- `tests/test_bot_tennis_live.py`
- `tests/test_bot_live_counts.py`

What this commit should decide:
- either tennis is truly dry-run only for now, in which case hard-block live orders
- or tennis is allowed experimentally, in which case require an explicit arming confirmation similar to UFC

Done when:
- there is only one correct answer to the question "can tennis place live orders right now?"

### Commit 3

`test: prove tennis live path with matched-market audit evidence`

Purpose:
- move from "code exists" to "execution path is proven"

Scope:
- tennis live audit / regression tests
- a short audit handoff note with exact observed results
- only intentionally versioned evidence, not random transient CSV noise

Minimum proof target:
- one real session with active tennis Polymarket matchup markets
- at least one fully matched bookmaker/Polymarket candidate
- second-source logic exercised
- LLM veto path observed, either vetoing or returning `NO_VETO`
- saved summary of counts and one or two concrete examples

Likely files:
- `src/strategy/tennis_decision.py`
- `src/strategy/tennis_llm_operator.py`
- `src/polymarket/tennis_markets.py`
- `tests/test_tennis_decision.py`
- `tests/test_tennis_llm_operator.py`
- `tests/test_tennis_markets.py`
- new handoff note under `data/processed/tennis/`

Done when:
- you can point to one audit artifact and say "this exact live path was exercised successfully"

## Recommended Working Order

1. Finish Commit 1 first.
2. Do not touch live deployment behavior until Commit 1 is merged.
3. After Commit 1, choose the live-policy answer in Commit 2.
4. Only after Commit 2, run the matched-market proof pass for Commit 3.

## Short Version

If you want the fastest honest summary:

- the tennis model/data system is real
- the tennis live shell is real
- the tennis production policy is not clean yet
- the next job is not more model work
- the next job is tightening the runtime boundary and documenting it correctly
