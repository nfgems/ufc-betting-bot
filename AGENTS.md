# Current saved source - October 16, 2026

The main checkout now represents the currently deployed 6068bb7d release source.
Read docs/CURRENT_RELEASE.md and release-source-manifest.json for exact source,
the immutable private model/data base, verification and archived research.
Source-control changes do not authorize an automatic Railway deployment.
Preserve raw hash-bound source/config bytes. The active image starts
/bin/bash /app/refresh-entrypoint.sh; do not restore the older legacy startup.
The older instructions below remain preserved; the canonical living handoff
records later completed milestones and the current continuation.

---

# AGENTS.md — UFC Betting Bot

## What This Is

ML-powered UFC fight prediction and Polymarket execution bot. Covers data collection, feature engineering, model training, backtesting, live prediction, and a Flask dashboard.

## Quick Reference

```bash
# Run tests
pytest -q tests/

# CI smoke checks (run before pushing)
python -m compileall src tests

# Main CLI
python -m src.bot <command>    # scrape, train, predict, live, backtest, etc.
```

## Railway

- Railway CLI access is available in this workspace and should be checked before claiming otherwise.
- Typical linked hosted target: project `ufc-betting-bot`, environment `production`, service `ufc-bot`.
- For production/runtime issues, verify `railway whoami` and `railway status` first, then inspect Railway runtime before relying on local state.

## How Things Fit Together

- The CLI orchestrator in `src/bot.py` is the main entry point — all commands route through here.
- All settings, thresholds, and env var loading live in `src/config.py`. Strategy params belong here, not scattered around.
- Model training is driven by named training contracts (feature lists + hyperparams). Every promoted model must have a reproducible spec.
- The production web entrypoint starts Flask + a background monitor + the betting loop.
- UFC real-money trading requires multiple separate env vars to arm — look at the live control module.

## Conventions

- `PYTHONPATH=.` — all imports are `from src.*`.
- Logging via `logging.getLogger(__name__)`, no print in library code.
- Feature hierarchy: **real observed value > NaN > NEVER median/default**. Only features producible at live inference time belong in training specs.
- UFCStats uses HTTP intentionally (they dropped TLS) — don't "fix" this to HTTPS.
- Odds features get training noise to prevent closing-odds leakage — don't bypass this.
- The CLOB client has thread-safety handling — check before making concurrent Polymarket calls.

## Don't

- **Don't commit `.env`, ledger files, or model `.pkl` artifacts** — these are runtime/sensitive data.
- **Don't add features to training specs without confirming they're available at live inference time** — this is the most important invariant in the project and it's easy to violate.

## Testing

- `pytest` with no special config. Many tests use monkeypatching.
- There are schema contract tests that enforce feature contract integrity — check them when touching model specs or feature builders.
- CI: compile check → artifact integrity → pytest → Docker build.

## Roster continuation: Railway context and production protection

- Read the newest correction in the canonical living handoff at
  `C:/Users/Evan/betting-bot/ufc-betting-bot/docs/PRACTICAL_MODEL_DATA_RESET_LIVING_HANDOFF.md`
  before following older roster next-work lists. The September 6, 2026 u47b1
  correction records existing Railway storage and supersedes the assumption
  that generic machine-reboot recovery must be the immediate next milestone.
- Railway is the existing intended host. Read-only checks observed `ufc-bot`
  online with `ufc-bot-volume` at `/app/logs`; `UFC_DATA_DIR` and `UFC_LOGS_DIR`
  also pointed there. Recheck these dated observations through the Railway
  skill and whoami/status. Connect recovery design to actual deployed startup
  and persistent paths before adding new storage/clock/witness machinery.
- Production protection is not a reason to ignore authorized read-only Railway
  inspection. Inspect only bounded relevant status, nonsecret path settings and
  evidence; avoid commands or application entrypoints that trigger active work.
- For this roster continuation, local isolated implementation/testing and
  read-only Railway inspection are authorized. Production deploys, restarts,
  configuration/volume/permission changes, hosted trials or test writes,
  scheduling/betting changes, source requests and publication require separate
  authorization. Do not use the live volume as a synthetic fixture. Preserve
  the older enabled refresh loop and existing production state.
- An attached volume proves an existing storage facility, not complete new-roster
  recovery or role isolation. Preserve uncertainty, exact independent evidence,
  original timing/accounting rules and consumed IDs. Distinguish app restart,
  local WSL boot change and hosted replacement; do not infer a production data
  loss incident from local test behavior. Keep the next milestone bounded and
  explain any remaining prerequisite in plain English.

## Current roster milestone correction — u56b1, September 7, 2026

- The milestone is **“integrate and verify the new refresh workflow on Railway.”**
  Railway already runs as a hosted service. Do not reopen a generic question of
  whether Railway runs independently of the owner's computer, or use “server
  independence remains unproved” as the outstanding milestone.
- The owner's computer and Codex can remain on throughout every test. Verify
  the new workflow through relevant Railway configuration, processes, logs and
  saved records. Turning the computer off adds no necessary proof. Never impose
  a computer-off, browser-off or Codex-off test, observation condition or gate.
- Read the newest entry in the canonical living handoff named above, then
  `docs/CODEX_BRIEF_RAILWAY_WORKFLOW_INTEGRATION_POST_U56A1.md`. This correction
  supersedes computer-power-state and broad hosted/server-independence framing
  in older briefs, handoffs and future-work lists, including u56a1's wording.
  Preserve those historical records. The portable worktree's older living
  handoff is historical; the main workspace's canonical handoff is current.
- u56a1 completed the local test-window launcher and final-receipt interruption
  rehearsal. The remaining work concerns this new workflow's Railway startup,
  permissions, persistent storage and correct operation. Its hosted integration
  has not yet been verified; later-cycle provisioning and recurrence stay
  separate. Do not repeat completed local prerequisites from older lists.
- This documentation correction grants no hosted deployment, test, restart or
  production change. Continue already-authorized local implementation/testing,
  independent review and bounded read-only Railway inspection under the existing
  restrictions. Preserve the older enabled production refresh loop.
