# Tennis Live-Readiness Handoff

Date: 2026-03-22

## Goal

Get the tennis live-trading shell to a state that is safe to commit, audit repeatedly, and only then promote to live use.

This handoff is about the shell around the core tennis model:
- automated skip/block logic
- second-source confirmation hooks
- Gemini veto integration
- live command outputs and artifacts

The core tennis probability model was intentionally left untouched in this phase.

## Current State

The tennis shell no longer has a manual-review path.

Current control flow:
- `auto_eligible`
- `auto_skip`
- `auto_block`
- `not_eligible`

Current key files:
- `src/strategy/tennis_decision.py`
- `src/strategy/tennis_llm_operator.py`
- `src/bot.py`
- `src/config.py`
- `tests/test_tennis_decision.py`
- `tests/test_tennis_llm_operator.py`
- `tests/test_bot_tennis_live.py`
- `data/processed/tennis/auto_blocklist.csv`

## What Was Implemented

### 1. Manual review removed

The old tennis manual-review states were replaced with deterministic automation states.

Behavior now:
- suspicious model-vs-market mismatches auto-skip
- low-history matches auto-skip
- thin 3-book markets need automatic second-source confirmation or they auto-skip
- explicit blocklist hits auto-block

### 2. Second-source confirmation hooks added

The shell now supports second-source confirmation at the decision-layer level.

Current practical behavior:
- if explicit second-source probabilities are present, they can be used
- if a live execution market is present, the execution price can act as the current second-source check
- if a thin market has no second source, it auto-skips

Important nuance:
- this is a real hook and a useful cross-market check
- it is not yet a truly independent second bookmaker/feed integration

### 3. Gemini veto-only layer added

The new Gemini operator is veto-only.

Allowed outputs:
- `AUTO_BLOCK`
- `AUTO_SKIP`
- `NO_VETO`

Design intent:
- Gemini never "approves" a bad structured candidate
- Gemini only blocks, skips, or stays out of the way
- if enabled without a `GEMINI_API_KEY`, it fails closed to `AUTO_SKIP`

Relevant env vars:
- `TENNIS_LLM_VETO_ENABLED`
- `TENNIS_LLM_VETO_FAIL_CLOSED`
- `TENNIS_LLM_VETO_MODEL`
- `GEMINI_API_KEY`

### 4. Live artifacts updated

Current live reference artifacts:
- `data/processed/tennis/live_reference_decisions.csv`
- `data/processed/tennis/live_reference_auto_skipped.csv`
- `data/processed/tennis/live_reference_auto_eligible.csv`

Important fix made during audit:
- `live_reference_auto_skipped.csv` now contains only actual `auto_skip` / `auto_block` rows
- it no longer mixes in ordinary `not_eligible` rows

## Audit Findings Fixed During This Pass

### Finding 1

Problem:
- after an LLM veto, `automation_status` changed but `structured_status` could stay `auto_eligible`

Why it mattered:
- live outputs and downstream audits could show stale internal state

Fix:
- vetoed rows now update both `structured_status` and `automation_status`

### Finding 2

Problem:
- the file named `live_reference_auto_skipped.csv` was initially capturing all non-auto-eligible rows, including `not_eligible`

Why it mattered:
- the artifact name was operationally misleading

Fix:
- snapshot filters now only write actual `auto_skip` / `auto_block` rows into the auto-skipped artifact

## Verification Completed

### Test coverage

Commands run successfully:
- `pytest tests -k tennis -q`
- `pytest tests/test_tennis_decision.py tests/test_tennis_llm_operator.py tests/test_bot_tennis_live.py -q`
- `python -m py_compile src/strategy/tennis_decision.py src/strategy/tennis_llm_operator.py src/bot.py tests/test_tennis_decision.py tests/test_tennis_llm_operator.py`

Latest results:
- full tennis suite: `119 passed`
- targeted bot/decision/veto suite: `24 passed`

### Live command verification

Commands run successfully:
- `python -m src.bot tennis-predict --model lean_hybrid`
- `python -m src.bot tennis-live --dry-run --model lean_hybrid`

Latest live reference result on 2026-03-22:
- `22` matches
- `14` `auto_eligible`
- `5` `not_eligible`
- `3` `auto_skip`

Observed examples:
- `Qinwen Zheng vs Madison Keys` auto-skipped as suspicious mismatch
- `Martin Landaluce vs Karen Khachanov` auto-skipped as suspicious mismatch
- `Jiri Lehecka vs Taylor Fritz` auto-skipped because it was a thin 3-book market with no second source

Verified artifact content:
- `live_reference_auto_skipped.csv` now contains only the actual auto-skipped rows

### Fail-closed veto behavior

Directly verified:
- if `TENNIS_LLM_VETO_ENABLED=1` and `GEMINI_API_KEY` is missing, the tennis veto layer returns `AUTO_SKIP`

## What Is Still Not Proven

These are real remaining gaps. Do not gloss over them.

### 1. No real matched execution-market validation yet

On the latest dry-run check:
- there were `0` active tennis Polymarket matchup winner markets

Meaning:
- the live shell handled the no-market case correctly
- but the full matched execution path was not exercised on a real candidate

### 2. Second source is still only a hook plus execution-market cross-check

The current system can:
- require confirmation
- consume explicit second-source columns
- use execution price as a cross-market confirmation source

But it still does not have:
- a dedicated second independent bookmaker/feed wired in automatically

### 3. Gemini real-world live use is not yet fully exercised

The Gemini veto layer is implemented and tested, but a real live trade candidate was not passed through a live Gemini veto call during this pass.

Meaning:
- code path exists
- fail-closed logic exists
- mocked/unit verification exists
- but real live candidate + real Gemini + real matched execution market is still unproven

## Push/Commit Blockers

The tennis shell is closer, but there are still blockers before a clean live push.

### Blocker 1: Dirty worktree

The repository currently has many unrelated tracked and untracked changes outside this narrow tennis shell scope.

Implication:
- do not do a blind `git add .`
- do not assume the current worktree is a clean commit candidate

### Blocker 2: Need a commit boundary

Recommended commit scope for this shell work:
- `src/bot.py`
- `src/config.py`
- `src/strategy/tennis_decision.py`
- `src/strategy/tennis_llm_operator.py`
- `tests/test_tennis_decision.py`
- `tests/test_tennis_llm_operator.py`
- `tests/test_bot_tennis_live.py`
- `data/processed/tennis/auto_blocklist.csv`

Recommended not to commit as code changes:
- generated live snapshot CSVs
- transient processed outputs unless intentionally versioned for audit

### Blocker 3: No end-to-end matched-market dry run yet

Before calling this live-ready, get at least one real dry-run session where:
- Polymarket tennis matchup markets are active
- a match is fully matched
- second-source logic runs in the real execution path
- Gemini veto can be observed on a real candidate if enabled

## Recommended Next Steps

### Phase 1: Prepare a clean commit shell

1. Isolate the intended commit scope.
2. Exclude generated live CSV snapshots unless you explicitly want them tracked.
3. Re-run:
   - `pytest tests -k tennis -q`
   - `python -m src.bot tennis-predict --model lean_hybrid`
   - `python -m src.bot tennis-live --dry-run --model lean_hybrid`

Goal:
- make sure the exact staged diff still passes after cleanup

### Phase 2: Audit round 1

Audit focus:
- code-path consistency
- fail-closed behavior
- snapshot truthfulness
- env misconfiguration behavior
- no stale manual-review assumptions anywhere in the tennis path

Specific checks:
- verify `TENNIS_LLM_VETO_ENABLED=1` with no key causes skip, not pass-through
- verify `auto_blocklist.csv` entries really hard-block
- verify thin 3-book matches skip without confirmation
- verify reference-only outputs and execution outputs stay internally consistent

### Phase 3: Fix findings from audit round 1

Do not batch blindly.

For each finding:
- fix the code
- add or update regression coverage
- re-run tennis tests

### Phase 4: Audit round 2 on a live slate

Wait for a slate where tennis matchup markets are actually active on Polymarket.

Required observations:
- real matched execution candidates exist
- second-source confirmation runs in the execution path
- trade-ready vs auto-skip vs not-eligible counts look sane
- if Gemini is enabled, inspect veto logs for one real candidate

### Phase 5: Decide whether execution-market confirmation is enough

Make an explicit decision:
- either accept execution-market confirmation as the current second-source policy
- or add a truly independent second bookmaker/feed before live promotion

This should be a written go/no-go decision, not an implicit assumption.

### Phase 6: Final go-live audit

Only after the above:
- re-run tests
- re-run live dry-run on an active market slate
- inspect saved artifacts
- confirm env vars
- confirm blocklist path
- confirm Gemini mode
- confirm no manual review dependency remains

## Recommended Go-Live Checklist

Use this as the final gate:

- core tennis model artifact loads cleanly
- tennis tests pass
- live reference prediction runs cleanly
- live dry-run matches real Polymarket matchup markets
- at least one real matched execution candidate observed
- second-source logic observed on real execution candidate
- Gemini veto behavior observed or intentionally disabled
- `auto_blocklist.csv` path confirmed
- no snapshot/logging inconsistency found
- commit diff isolated to intended files
- commit message and rollback plan prepared

## Bottom Line

The tennis shell is materially safer than before and the core logic is now internally consistent.

But the correct statement is:
- commit-ready is close after worktree cleanup and audit isolation
- live-ready is not yet fully proven until there is at least one real matched execution-market dry run
- if you want stronger market-confirmation standards, a truly independent second source is still the next structural upgrade
