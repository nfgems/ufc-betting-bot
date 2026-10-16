# Fresh-session handoff — connect the live updater to the betting bot

Date: October 11, 2026, America/New_York. This is a continuation instruction, not a claim that live betting is ready or enabled.

## Latest owner approval and checkpoint rule

The owner's exact latest requests were:

> can you write a handoff for a fresh session to proceed with the work/plan/fixes. it has my permission and approval to do whatver it needs to do

> it shhould also stop at logical checkpoints and not try to complete all remaining things in one run

The preceding owner instruction explicitly authorized downloading needed data, publishing validated updates and activating the completed refresh workflow. They withdrew assistant-created elaborate review and once-only-helper procedures and asked for normal development tools, inspection, tests and debugging. They authorized simpler implementation while preserving accepted history, the working model and existing accounting. Keep stakes and risk settings unchanged.

In the context of the immediately preceding discussion about finishing the actual live-betting workflow, the new approval covers necessary implementation, fixes, data/market collection, tests, legitimate authenticated access, builds, image publication, Railway deployments/restarts and required startup/configuration changes. It also authorizes eventual live-betting activation after the execution path passes technical validation, using the existing stakes, risk limits and entry rules. Do not ask for the same authorization again. This is authority to do the work; it is not evidence that the work has passed.

**Work one logical checkpoint per session, then stop and report. The next session should complete checkpoint 1 only.** Do not automatically proceed through all checkpoints or launch an open-ended goal. If a checkpoint becomes too large, stop at a smaller useful, tested boundary and record the exact remaining blocker. Authorization persists across these stops; the owner can resume the next checkpoint without granting it again. Do not turn the stop into an additional permission gate.

This handoff supersedes incompatible old pause-only, no-deploy/no-source/no-activation and assistant-created process restrictions for this continuation. Preserve old records, failures, artifacts and provenance as historical evidence. Genuine system requirements and the project's existing trading controls still apply. Do not bypass an actual control just to get a green result.

## What is live, and the correction to earlier wording

The automatic **data refresh and prediction service** is deployed and running on Railway. The accepted model is working with refreshed data. The **market-monitoring/order-execution betting workflow is not running** in this deployment.

Earlier wording that "the bot is live" was too broad. The live program is `python -m src.data.refresh_service serve`. Its module explicitly says it never trains, places an order or starts a betting loop. The current startup rejects any `LIVE_TRADING_MODE` other than `off`. Simply changing that variable will fail startup; removing the check alone will not integrate the trader.

Fresh read-only checks in this documentation session confirmed Railway online on the image/deployment below and `/readyz` returning `ready: true`, generation `probe-20261010`, 9,503 rows, 13 predictions and `trading_enabled: false`. `/api/refresh-status` reported `waiting`, no pending job and no last error. These are observations on October 11, not perpetual guarantees; recheck in the next session.

The first scheduled cycle after activation is Tuesday, October 13, 2026, at **11 a.m. America/New_York / 15:00 UTC**. It had not occurred when this handoff was written. We tested planner behavior and an actual full manual refresh; do not claim the upcoming scheduled trigger has already fired.

## Read first, without restarting the historical investigation

1. This complete handoff and the newest prefix of MAIN's `docs/PRACTICAL_MODEL_DATA_RESET_LIVING_HANDOFF.md`.
2. `C:/Temp/ufc-refresh-live-20261010/verification/COMPLETED_REFRESH_RESULT.json`.
3. Relevant original acceptance/production-handover sections of `docs/MASTER_COMPLETION_AND_PRODUCTION_RELEASE_PLAN.md` and the actual current code paths named below.
4. Additional old reports only when needed for a concrete implementation question. Old master-plan September statuses and checkpoint48 are historical where superseded by actual October deployment evidence and the latest owner instructions.

Use the Railway runtime-access skill at `C:/Users/Evan/.codex/skills/railway-runtime-access/SKILL.md`; run `railway whoami` and `railway status` before claims about access or current production state. Railway is the existing host. No computer-off, browser-off or Codex-off proof is required.

## Current release, code and persistent state

- Isolated implementation/package: `C:/Temp/ufc-refresh-live-20261010`.
- MAIN: `C:/Users/Evan/betting-bot/ufc-betting-bot`. It has substantial unrelated dirty/untracked work. Do not reset, clean, overwrite it or assume its files equal the deployed package.
- Live image: `ghcr.io/nfgems/ufc-betting-bot@sha256:9bd1740761599376880b199a2576e3e45757ee446be6c6a92da13248991eae83`.
- Deployment at last check: `2507f2e9-f17a-4bb7-9e83-adb33ccb0acb`.
- Last good pre-refresh base image, retained: `ghcr.io/nfgems/ufc-betting-bot@sha256:7fd5806186bec041629a0e683f20fa593fc00f0426d071bd809a05f7e83fc12b`.
- Railway project `0f1fd486-bc96-4fb7-8c23-3f56ff8dbb2b`, production environment `7b128011-3c33-4ada-8602-a8ab1863ad3e`, service `702f574c-389b-48ea-911f-a1054b3e4405` (`ufc-bot`).
- Volume `ufc-bot-volume` mounted at `/app/logs`; last check about 4.5 GB / 9.8 GB.
- Public URL: `https://ufc-bot-production.up.railway.app/predictions`. Read-only status endpoints: `/readyz`, `/api/refresh-status`, `/api/predictions`.
- Actual Railway startup override: `/bin/bash /app/entrypoint.sh`. The current Dockerfile replaces that path and also copies the refresh script to `/app/refresh-entrypoint.sh` for default Docker startup.
- Original entrypoint is retained in the image as `/app/legacy-entrypoint.sh`. The isolated package's root `entrypoint.sh` is the original source; the active replacement source is `refresh-entrypoint.sh`. Do not mix these up.
- Current Python refresh process runs as UID 100. There is one active refresh worker; preserve sole-writer behavior.
- Original accepted payload: `/app/release_payload` in the image; identical local `release_payload` in the isolated package.
- Live refresh state: `/app/logs/incremental_refresh`; local verified counterpart: `C:/Temp/ufc-refresh-live-20261010/workflow`.

Inside that refresh root:

- `published.json` is the final serving transaction: exact accepted data snapshot plus immutable prediction report. Trading consumers must use a consistent final snapshot, not a partially completed candidate.
- `published_reports/probe-20261010.json` is the immutable served report.
- `service.json` retains original intent/recovery and completion; its pending job is null.
- `data/accepted.json` is the producer's internal accepted data pointer. It can lead the final served pair during pending prediction work; do not confuse it with the final publication.
- `data/generations/probe-20261010/processed` contains the two accepted successor CSVs.
- `data/runs/probe-20261010/journal.json` is COMPLETED.
- `data/cache/probe-20261010` contains saved response envelopes and request intents.
- `prediction_runs/probe-20261010/predictions.json` retains the original calculation result. Baseline preflight under `prediction_checks` was never the served report.
- `.seed-installed.json` prevents startup from reinstalling an old seed over valid saved progress. Preserve it and subsequent valid state.

Use an isolated successor workspace based on the verified package/deployed source for implementation. Preserve the current package and evidence. Runtime Python is 3.11.16; the earlier Windows verification used 3.14.3. Both had XGBoost 3.2.0 and scikit-learn 1.8.0. Verify compatibility in the actual Linux image, not only on Windows.

## Accepted data/model facts to preserve and reuse

The actual completed refresh added **50 new fights** from four validated UFCStats completed events through October 10. Both final artifacts contain **9,503 rows and 9,502 distinct historical identities**. Each original accepted prefix has **9,453 rows and 9,452 distinct identities**, including an inherited duplicate. Its bytes, cells, order, occurrences and provenance remain unchanged. Additions are unique and disjoint. The two historical CSV orders differ; check each prefix independently and compare identity occurrence multisets where appropriate.

All 100 new-fight participant profiles and 26 nearest-card profiles were collected/validated. The October 17 card has 13 bouts, with primary and no-odds predictions. The primary contract has 211 exact feature keys; no-odds has 202. Missing values stay null/native NaN, never invented values or medians. Models were not retrained.

Selected bundle: `v9-monthly07-20260927-carried-s01`; feature contract `full_live_contract_v6_durability_2022`.

| Artifact | SHA-256 |
| --- | --- |
| Original fights CSV | `a41f1e40d978429db6c767b28ddd90fff772c10c4a3d483d252d80156b5e469e` |
| Original features CSV | `93664d94ebdb7dd867cfc407a7c3f159db79bef54c33a593d2951f7d86c8cace` |
| Current fights CSV | `b17d8207503463852614aec71711a659eee231f9816ee932ec739f31236bbabf` |
| Current features CSV | `4196f5619c4cb50520bb7519f8b353f8f38df8d0307cf05ad49faeb3f5f2cdfb` |
| Accepted primary model | `e296e3d04d005c79c7ef77f7879e693a3ab71359a501409acafd053653219435` |
| Accepted no-odds model | `5c7092692bb5a14b60080ac58a9cb0c3aed7c31b83a5259363d7b01bd5d408c2` |
| Accepted logistic model | `1185a50a169032cf0e4e02bcd9441d1d086de2e9b0e622dab7126314829f4471` |
| Original served prediction report | `5fe8143c132bb63cc24f5648743047319c620a0d03bc55333f4bdc9a7345f914` |

The original service job was `probe-20261010`, decision instant `2026-10-11T03:36:27.489511+00:00`; final data/prediction publication completed at `2026-10-11T04:08:01.462170+00:00`. Do not rewrite those records or reuse the old job to make a fresh request. Current market/inference work needs its actual current observations and decision instant. The saved upcoming-card report is not evidence of a current executable price or a filled order.

Historical blank IDs and namesake ambiguity were handled with authenticated calculation-only associations while preserving accepted cells. The Anthony Romero historical ID `6f81399074fcdbba` must remain separate from current namesake ID `4419acb81e6f0ea4`. Reuse the retained association/provenance instead of guessing, merging namesakes or downloading/rebuilding all old history.

## Existing proof; its scope and limits

The 73 tests passed: producer 16, predictor 30, service 23, seed installer 4. Actual additions were independently reproduced from retained raw sources. Windows/Linux/live model probability reproduction passed 26 comparisons, maximum absolute difference `1.1102230246251565e-16`.

Local staged recovery completed with zero source GETs. Forced local Linux interruption/restart and an actual Railway restart preserved saved work. Across the hosted restart, 380 new refresh-state files retained identical bytes and mtimes. Across deployment, 900 original bundle/accounting files totaling 2,563,566,495 bytes remained identical. All 52 existing application settings and all 5 risk settings were unchanged; trading remained off. Those unchanged-setting observations are historical evidence, not a prohibition on the now-authorized necessary activation/auth/startup changes. Preserve risk settings and document necessary operational changes without exposing secrets.

Verification records are under `C:/Temp/ufc-refresh-live-20261010/verification`, especially `actual_refresh_additions_verified.json`, `integrated-local-completion.json`, `package-verified.json`, `railway-actual-verified.json`, `railway-restart-verified.json` and `COMPLETED_REFRESH_RESULT.json`. Baseline/after accounting and settings records are retained at the package root.

These prove the refresh/prediction workflow, not current market matching, trading-loop operation, order placement, fills, settlement or profitability. No fabricated order or profitable-edge claim should be made. The original master plan does not require a new profitability research project as an activation gate. No retraining is needed merely because data advanced. If a demonstrated defect truly requires model replacement, preserve the original, use a reproducible spec and validate the replacement before applying it. Automatic monthly promotion must not be enabled as a side effect.

Local validation containers were stopped after the proof to prevent duplicate scheduled work. Their fixtures/evidence remain. MAIN application source and the old Source51/67/71/77/83/102/103/104 cuts, genuine reviews, handoff49 and adoption proposals were not changed by the refresh implementation. Do not rerun old consumed experimental helpers or turn them into new prerequisites; ordinary new development/test runs are allowed.

## Checkpoint 1 — actual local integration and fixes; next session's scope

Trace the existing ordinary bot startup and prediction/market/execution consumers, then implement the smallest coherent connection to the new final accepted snapshot. Reuse the existing bot and refresh work. Finish an actual callable local path, not another schema, uncalled seam or review-only result. Production stays on the current working updater with trading off during this checkpoint.

Relevant package files include:

- `src/data/incremental_refresh.py`, `src/data/refresh_predictions.py`, `src/data/refresh_service.py`, `src/features/build_features.py`.
- `src/bot.py`, `src/config.py`, `src/live_control.py`.
- `src/web/serve.py`, `src/web/app.py`, `src/web/paused.py`.
- `src/model/predict.py`, `src/model/production_bundle.py`, `src/data/runtime_selected_inputs.py`.
- `src/data/live_monitor.py`, `src/polymarket/monitor.py`, `src/polymarket/markets.py`, `src/polymarket/market_lookup.py`, `src/polymarket/client.py`, `src/polymarket/executor.py`, `src/polymarket/tracker.py`.
- `src/strategy/execution_audit.py`, actual ledger/sizing/exposure consumers discovered through these entrypoints.
- `refresh-entrypoint.sh`, original `entrypoint.sh`, Dockerfile, `scripts/install_incremental_seed.py`.

Prove the real consumer reads the final served data/model generation together, current inputs are honestly observed, current market outcomes map to the right fighter/token, and model preprocessing/order match the accepted contract. Resolve current-price use correctly: weekly saved predictions are not permission to invent odds features or treat an old quote as executable. Preserve the existing no-odds selection/fallback rules and genuine missing values.

Add meaningful integration and interruption tests for the changed behavior. Test a completed refresh, pending candidate, failed prediction and saved published pair; the trader must retain the last good consistent pair. Test original intent/order identifiers and recovery, including ambiguous submission outcomes: reconcile against actual order/account state before considering another submission. Do not blindly resend an uncertain order or fabricate successful fills. Verify request reuse where applicable, sole writer, locks and that restart does not create another monitor/order worker. Preserve CLOB thread-safety handling.

Keep orders disabled in local tests, using the real production API boundaries with bounded test clients/fixtures. Do not use the live volume as a synthetic fixture. Run relevant tests and compile checks; reuse the 73 refresh tests where the connection changes their consumers/startup. Broaden testing when an actual change/failure warrants it. Preserve all accepted history/models and real accounting. Do not rebuild old historical features.

**Stop after a tested local integration/fix result.** Record changed paths, actual commands/results, preserved invariants, concrete unresolved blockers and the exact checkpoint 2 starting point. Do not deploy or activate betting in this first session just because later work is also authorized.

## Checkpoint 2 — combined Railway operation with betting off

In a subsequent session, finish and verify combined startup/container composition locally, then publish/deploy it under the existing approval with betting off. Keep one refresh owner and the intended monitor/execution owner. Do not start the old legacy updater alongside the new one or silently return the trader to old data. Preserve the existing volume and valid successor state; rollback code while retaining valid data/accounting, never restore an old volume or erase newer progress.

Verify actual live current-card/market discovery, accepted data/model selection, observed prices, account connectivity/exposure reads, prediction delivery, monitoring, intended order selection/sizing with real submissions disabled and interruption/restart reconciliation. An honest no-action result outside the existing 72-hour earliest-entry rule is acceptable; do not weaken that rule, wait for a fabricated eligibility test or create a wager solely to prove activity. Document unexercised actionable paths separately.

The existing `src/live_control.py` requires public hosted trading to have a nonblank `WEB_DASHBOARD_TOKEN`. It was blank in the last verified updater configuration; the current prediction-only API therefore followed the inherited public configuration. Address actual authenticated trading/control startup using the existing secret mechanism rather than deleting this check. Secure token creation/configuration is within the approval; never print a token/private key or save credentials in the handoff, source or Git. Check actual available credentials without dumping secrets. Ask for input only if a required credential/access decision is genuinely unavailable.

Retain concrete before/after accounting/model/data evidence and check live readiness/logs/processes. Stop when this hosted betting-off checkpoint passes or a genuine blocker is documented. Do not arm real orders in this checkpoint.

## Checkpoint 3 — validated live-betting activation

In a subsequent session, reconcile current account/orders/positions and saved bot state, finish any remaining demonstrated execution fixes, and activate the tested execution path under the owner's approval. Use the actual `src/live_control.py` checks, including `LIVE_TRADING_MODE`, `LIVE_MODEL`, `LIVE_TRADING_ARMED` and `LIVE_TRADING_CONFIRMATION`; do not bypass them or guess which combination starts the right model/workers. Startup/configuration changes needed for this checkpoint are authorized.

Keep the existing stakes, risk/exposure limits, entry timing, emergency stops, account/order ownership and ledger conventions. Do not activate unrelated BTC trading, change strategies, increase funding or enable automatic model promotion as a side effect. Existing multi-strategy/shared-account positions and manual holdings require correct attribution; do not treat an account balance/position display as an empty-account proof.

Verify the intended model and accepted snapshot, actual trading mode/controls, authenticated dashboard/control access, healthy monitor/executor, account reconciliation and durable intent/ledger behavior. Real eligible orders may be handled by the established bot after activation under its unchanged rules; do not force a bet, fill or sale for demonstration. If no order is eligible, report "activated and waiting; actual order/fill path not yet observed" rather than claiming filled-order proof.

Stop at the activation checkpoint with precise evidence and a clear pause/rollback route that retains current valid state. If a safety/correctness check fails, keep or restore betting off, fix within scope where feasible and state the concrete failure; never invent acceptance.

## Checkpoint 4 — later observed operation and remaining broader tracks

Observe the actual scheduled refresh, real order/fill/settlement lifecycle when naturally available and accounting/recovery together. Fix demonstrated issues without duplicate requests/orders or rewriting settled history. This is its own checkpoint; do not keep an earlier session running until Tuesday or until a wager becomes eligible.

Keep separate any remaining original master-plan tracks: retained pre-fight market observations/results/corrections, Sunday discovery/roster mapping, monthly candidate training/evaluation/saving and explicit model selection. Monthly provider/policy remain absent and promotion month blank in the current refresh service. Do not claim those features exist, enable them incidentally or make every research/automation follow-up a prerequisite for the straightforward betting integration. Choose a bounded later checkpoint for genuinely needed work.

## How to communicate and close each checkpoint

Give brief plain-English updates: what now works, the concrete remaining blocker and what is being done about it. Avoid new infrastructure, mass archive reading or documents as substitutes for executable behavior. No approval workflow is needed merely because an earlier assistant invented one.

At each stop, save a short continuation result beside the working successor: exact code/package locations, tests and actual runtime checks, active/off state, preserved data/model/accounting, unresolved limits and next checkpoint. Update the canonical handoff with a concise newest entry while preserving its entire historical bytes. Do not stage/commit `.env`, ledger files or model `.pkl` artifacts. Do not claim future scheduled cycles, real orders, profitability or complete master-plan acceptance without corresponding evidence.

This documentation session did not change application code, deploy, restart or enable trading. It performed bounded read-only status/health checks and wrote this handoff plus a canonical navigation/status prefix. The prior canonical checkpoint48 bytes are retained intact.
