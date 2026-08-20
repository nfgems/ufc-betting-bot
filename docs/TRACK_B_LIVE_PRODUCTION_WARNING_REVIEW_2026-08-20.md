# Track B live-production warning review — 2026-08-20

Status: local repairs are complete and tested but are not live. An exact two-source-file Railway deployment was attempted on 2026-08-20, became unreachable during the existing cold-start data refresh, and was rolled back successfully to the prior production image. The narrow follow-up found no demonstrated application or configuration defect; an immediate or unattended redeployment remains a no-go. The owner has chosen to finish one offline Track A candidate model and its validation before returning to Track B. Track B integration and deployment remain separately gated behind the controlled maintenance window.

## Scope and production reference

- Reviewed the warnings supplied by the owner against Railway production logs and the exact deployed source.
- Railway target: project `ufc-betting-bot`, environment `production`, service `ufc-bot`.
- Latest successful deployment reviewed: `614846a1-aeb7-4c21-8d6a-eb2e9505615d`.
- Deployed Git commit: `3eacba726d5f0fd6c497492a18c74a85d8001044`.
- Local repair checkout: `C:/Users/Evan/betting-bot/ufc-betting-bot-main-production-warning-repair-20260813`.
- Railway and GitHub Actions were inspected read-only. No runtime, order, workflow, deployment, model, or production-pointer state was changed.

## Two confirmed repairs

### 1. Reject expired wallet-position snapshots

The Polymarket positions endpoint returned HTTP 429 on 2026-08-17. The monitor correctly treated its snapshot as incomplete, but the executor reused a cached wallet snapshot that was 1,131 seconds old. Production recovered by 13:32, and the affected cycle placed no orders and showed no wallet/ledger mutation, so no damage was observed.

The executor had a 60-second cache limit, but its cooldown and final-error branches bypassed that limit. The local repair makes 60 seconds a hard freshness boundary everywhere. A fresh cache is still usable; an expired cache now returns no snapshot, causing the existing caller to stop reconciliation and fail the live cycle safely.

Changed files:

- `src/polymarket/executor.py`
- `tests/test_polymarket_ledger_regressions.py`

### 2. Let the Tapology alternate runner handle a reader block

Railway's direct Tapology fetch is intentionally disabled; the GitHub Actions supplement workflow owns this refresh. The latest scheduled workflow, run `31927852134` on 2026-08-16, failed after the primary runner's reader returned a `runtime_block`. That condition was mislabeled as `discovery_failed`, so the workflow skipped its existing alternate runner.

The local repair classifies the existing `runtime_block` diagnostic as `hosted_egress_blocked`. The workflow already treats that result as retryable and will try the alternate runner. No workflow structure, scraper routing, or Railway behavior was changed. Read-only inspection of run `31927852134` confirmed that the primary job failed on the Harrison Garcia `runtime_block`, the alternate-pool job was skipped because the primary outcome was recorded as ordinary `failed`, and the finalizer exited without a result or data commit. The run used `main` at `3eacba726d5f0fd6c497492a18c74a85d8001044`.

Changed files:

- `scripts/refresh_tapology_profile_supplement.py`
- `tests/test_tapology_profile_refresh.py`

## Reviewed warnings that do not need code changes

| Warning | What the evidence showed | Decision |
| --- | --- | --- |
| Donte Johnson cancellation disagreement | Polymarket's cancel response and canonical order view briefly disagreed. The bot kept the ledger and reserved cash unchanged until canonical confirmation. The supplied order was reconciled as canceled at 15:19:13, about four minutes after the warning. | Leave the fail-closed logic unchanged. |
| No trusted model view | These were cancel-only maintenance passes for lower-priority resting orders, not evidence that the production model was missing. Partial fills and uncertain order states remain protected. The warnings stopped after 2026-08-14 03:49. | No safety fix needed. |
| Runtime data freshness | The guard correctly paused live betting while a completed card was absent, requested a refresh, and later accepted the refreshed bundle. | Safeguard worked; no change. |
| Method odds 11/12 | The source omitted one expected fight. The collector reported partial coverage and left unavailable values missing instead of inventing them. | Keep the warning. Investigate the named missing fight only if this persists within 48 hours of an event. |
| Stan Dorsainvil short-notice replacement | The replacement was detected, the active pairing changed, a prediction was produced for the new matchup, and Stan was included in the targeted profile supplement. | Informational warning; no change. |
| Robertson–Dern steam and Puga–Trembley line movement | These are intended market-intelligence alerts, not failures. | Keep the alerts unchanged. |
| Tapology 403 in Railway | Direct Railway access is deliberately disabled and delegated to GitHub Actions. | Do not re-enable Railway scraping. Fix only the hosted fallback classification described above. |

## Validation

- `90 passed`: wallet-ledger regressions, Tapology profile refresh, and Tapology workflow contract.
- `226 passed`: limit-order refresh, position pagination/strict failure, runtime freshness, and web runtime regressions.
- Compile check passed for all four modified Python files.
- An independent read-only review approved both changes with no blocking findings.
- `tests/test_clob_v2_migration.py` could not be collected in the mandated CI virtual environment because `py_clob_client_v2` is not installed there. No CLOB-v2 file was changed; the relevant cancellation behavior passed in `tests/test_limit_refresh.py` and was also verified from production logs.

## Recommended next decision

These repairs do not depend on another model fit or model promotion. The owner decided on 2026-08-20 to finish one offline Track A candidate fit and validation first, while leaving the production pointer and production state unchanged.

1. Keep Track B parked while the offline Track A candidate is built and validated from the already frozen, hash-locked Track A feature snapshot. The failed Tapology workflow does not block that fit; it does block any claim that a newly refreshed Tapology supplement is healthy/current and therefore blocks a new data rebuild or final model promotion until resolved.
2. Treat failed scheduled workflow run `31927852134` as part of Track B. The local Tapology classifier repair directly addresses its skipped alternate runner. No manual rerun is authorized now.
3. A Railway deployment alone cannot fix the GitHub Actions job. The repaired Tapology script must also reach GitHub's default branch under separate commit/push or merge authorization. Because a change to `main` can trigger Railway deployment, coordinate that integration with the controlled Track B window; do not push it casually. The next natural schedule is `2026-08-23T04:30:00Z`, and it may repeat the known failure if `main` still lacks the repair.
4. When the Track A candidate is ready, keep the wallet and Tapology repairs together for Track B integration, execute only the controlled deployment plan below, and then verify the naturally scheduled Tapology run or a separately authorized probe. Do not combine the Track B deployment with model promotion.
5. If positions 429s continue after this safety repair is eventually live, open a separate follow-up to reduce duplicate requests or share backoff state. That larger refactor remains deliberately deferred.

## Deployment attempt and rollback

The owner authorized deploying only the two repairs without committing unrelated work. A temporary bundle was built from deployed commit `3eacba726d5f0fd6c497492a18c74a85d8001044` and overlaid with exactly these two modified source files:

- `src/polymarket/executor.py`
- `scripts/refresh_tapology_profile_supplement.py`

The bundle delta was verified against a clean archive of that commit. It excluded the modified tests, this handoff, all other local changes, `.env`, bytecode, and caches. No commit or push was made.

- Attempted deployment: `b8915159-a0fa-43ce-a6ad-a98a393c40a3`
- Result: image build succeeded, but `/readyz` became unavailable while the existing startup freshness guard launched a scheduled UFC data refresh.
- Safety response: no retry was attempted. The deployment was rolled back.
- Rollback deployment: `ba12c1d7-3f0e-4c3a-9c51-7e678558f786`
- Rollback result: `SUCCESS`; `/readyz` returned HTTP 200 and the original production source/image was restored.
- Production source after rollback: commit `3eacba726d5f0fd6c497492a18c74a85d8001044`.

The two repairs therefore remain local and are **not deployed**. Production correctly paused betting while its scheduled data refresh ran; startup wallet reconciliation found the five live positions already tracked, and resting-order maintenance reported zero cancellations, placements, or reconciliation changes. The rollback refresh completed successfully at `2026-08-20T16:05:33Z`, rebuilt 10,823 processed fight rows through 2026-08-18, and returned the normal betting loop to prediction building. The final `/readyz` check returned HTTP 200 with `ready=true` and zero health errors.

## Cold-start readiness diagnosis and controlled deployment plan

The narrow follow-up did not demonstrate a code or configuration defect that justifies another repair.

- The candidate bound Flask at `2026-08-20T15:24:00Z`; CLOB was running by `15:24:02Z`, the monitor by `15:24:10Z`, and the betting loop by `15:24:15Z`. The freshness guard requested the missing completed-card refresh at `15:24:32Z`, safely paused trading, made zero resting-order changes, and started the existing scheduled refresh.
- Candidate application logs contain no completed `GET /readyz`, no application-generated HTTP 503, no crash, no out-of-memory event, and no refresh exception before rollback. Railway HTTP metrics instead record one 15.007-second 5xx and two approximately 9.967-second 4xx responses while the application handler logged no request. That is transport/non-response evidence, not evidence that the readiness calculation returned `ready=false`.
- Candidate CPU stayed around one core against a 24-core limit and memory remained far below the 24 GB limit. Resource-limit exhaustion is not demonstrated.
- The rollback ran the same freshness guard and same scheduled refresh. It returned `/readyz` HTTP 200 repeatedly while the refresh was running, completed normally after about 29 minutes 53 seconds, and returned HTTP 200 afterward. The prior successful production deployment on 2026-08-13 also started the refresh immediately and served HTTP 200 throughout it.
- The executable readiness calculation in `src/web/app.py` treats the betting loop, monitor, and CLOB as critical; the UFC refresh loop is deliberately non-critical. The freshness pause keeps the betting component running when guarded resting-order maintenance succeeds. The two Track B source repairs do not touch startup, readiness, freshness, refresh, or web-serving code.
- Five focused local regressions passed in 0.70 seconds: freshness-pause maintenance, delayed-refresh readiness exemption, first-betting-cycle refresh gating, guarded gate release, and refresh-loop startup gating.

Conclusion: the failed probes overlapped the cold refresh, but the recorded evidence does not establish the refresh as their application-level cause. The observed failure is best classified as a transient candidate transport/ingress non-response during a volume-backed stop/start deployment. Changing readiness logic, increasing its timeout, delaying the freshness repair, or moving the refresh into a new framework would be speculative.

For a later separately authorized deployment:

1. Reserve a 60-minute quiet maintenance window. The observed build took about 10 minutes and the normal cold refresh about 30 minutes; the remaining time is for readiness observation or rollback.
2. Rebuild the deployment input from exact commit `3eacba726d5f0fd6c497492a18c74a85d8001044` with only the two repaired source files. Reverify their recorded hashes and exclude tests, this handoff, secrets, caches, models, and all unrelated local files.
3. Before deployment, require the current rollback deployment to be `SUCCESS`, `/readyz` to return HTTP 200 with zero errors, and the processed snapshot to cover 2026-08-18 or a later completed event. Do not manually rerun the refresh or disable the freshness guard.
4. During startup, record the old-container stop, candidate start, Flask bind, and the CLOB, monitor, and betting-loop states. Once the three critical components are running, poll external `/readyz` every 30 seconds with a 15-second client timeout and retain both the response and its corresponding application request log.
5. Roll back if the candidate has no ingress/application request and no HTTP 200 within two minutes after the critical components are running; if `/readyz` returns an application 503, preserve its JSON and roll back unless the named critical component recovers within two minutes. Also roll back for a crash, out-of-memory event, refresh exception, unexpected order-state mutation, or three consecutive readiness timeouts.
6. If readiness is established, keep observing it throughout the existing cold refresh. Let the freshness guard pause betting and let the normal refresh finish; do not trigger a workflow or a second refresh.
7. Require successful refresh completion, a processed maximum event date at least as current as the pre-deployment snapshot, zero refresh health errors, three consecutive post-refresh `/readyz` HTTP 200 responses, and one normal post-refresh prediction cycle before accepting the deployment.
8. Ensure the Tapology script repair reaches GitHub `main` under the same separately authorized integration plan; the Railway source overlay by itself does not change scheduled Actions. Account for any automatic Railway deployment triggered by merging to `main`. Then observe the next naturally scheduled Tapology workflow to verify the alternate-runner repair, or use a separately authorized probe if waiting for the schedule is impractical. Do not manually rerun it without that authorization. If wallet-position 429s recur, confirm the new 60-second fail-closed behavior from ordinary logs; do not manufacture an outage or touch orders for a test.

Plain-English recommendation: **no-go for an immediate or unattended redeployment; conditional go only for the controlled maintenance-window procedure above after separate owner authorization.** No additional source repair is warranted by the current evidence.

No deployment, restart, Railway mutation, workflow rerun, order action, model fit, model-pointer edit, promotion, schedule activation, commit, or push was performed during this diagnosis or the read-only GitHub run review.
