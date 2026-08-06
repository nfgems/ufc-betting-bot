# Corrected Durability Production Promotion Packet — 2026-08-06

## Status and authority boundary

This packet supersedes `docs/DURABILITY_PROMOTION_RUNBOOK.md`. The June
`p=0.00` promotion claim predates the corrected historical-odds and
point-in-time repairs and must not be used for this release.

The preparation described here was performed locally and with read-only
production inspection. As of this packet, **nothing has been committed,
pushed, deployed, restarted, activated on the Railway volume, or newly
scheduled externally**. No external workflow state was enabled, disabled, or
dispatched.

Important: a pre-existing GitHub workflow named `Weekly Model Retrain` is
already enabled on `main` with `0 6 * * 2`, write permission, and a final
commit/push-to-main deployment step. Scheduled runs occurred on 2026-08-04 and
2026-07-28. This packet did not alter or disable it because cron changes and
external actions were not authorized. Production promotion is blocked until a
separate decision either pauses that workflow or completes and reviews its
safe replacement.

## Recommendation in one sentence

The candidate is a corrected-data durability model with a small, consistent
predictive lift and plausible but seed-sensitive simulated trading upside. It
is defensible as a corrected-data production upgrade, not proof of superior
future profit.

The comparison control below is `full_live_contract_v6_tuned` rebuilt on the
same corrected data. It is not the older durability binary currently live.

## What the proposed model actually did

Lower log loss, Brier score, and ECE are better.

| Seed | Accuracy: control -> durability | Log loss: control -> durability | Brier: control -> durability | ECE: control -> durability | Predictive gate |
|---:|---:|---:|---:|---:|---|
| 7 | 68.65% -> 68.82% | 0.602173 -> 0.601803 | 0.206934 -> 0.206806 | 0.03664 -> 0.03636 | Pass |
| 42 | 68.20% -> 68.82% | 0.602423 -> 0.600678 | 0.207056 -> 0.206212 | 0.03608 -> 0.03568 | Pass |
| 2026 | 68.31% -> 68.65% | 0.603317 -> 0.602946 | 0.207487 -> 0.207258 | 0.03616 -> 0.03999 | Fail: ECE +10.6% |

| Seed | Bets: control -> durability | Win rate: control -> durability | ROI: control -> durability | Profit: control -> durability | Difference |
|---:|---:|---:|---:|---:|---:|
| 7 | 247 -> 240 | 59.11% -> 61.67% | 7.46% -> 8.94% | $219.34 -> $286.60 | +$67.25 |
| 42 | 233 -> 229 | 65.24% -> 64.63% | 13.56% -> 16.10% | $403.81 -> $517.64 | +$113.83 |
| 2026 | 224 -> 227 | 64.29% -> 62.11% | 13.41% -> 11.84% | $443.28 -> $378.68 | -$64.60 |

These rows replay the same fights under different random seeds. They are
robustness checks and must not be added as independent profits. All six arms
were profitable historically, but durability beat the control on simulated
profit only at seeds 7 and 42.

The paired event bootstrap intervals both include zero:

- Seed 42: +$113.83, 95% interval -$48.75 to +$274.40, one-sided `p=.08`.
- Seed 7: +$67.25, 95% interval -$88.21 to +$224.44, one-sided `p=.19`.

Those passed the repository's permissive one-sided `alpha=.25` gate, not a
conventional `.05` standard. Seed 2026 lost $64.60 relative to control and was
byte-identical on rerun. Small probability changes moved bets around hard
strategy thresholds; the preregistered sign-stability rule was therefore not
a clean automatic pass. Durability was selected later with this uncertainty
known.

The evidence used 1,761 fights, ten equal walk-forward folds, dates from
2022-01-15 through 2026-08-01, six-month historical retraining intervals, and
one-day-prior odds/features. The final artifact itself is a seed-42 full fit on
all 4,094 eligible corrected rows through 2026-08-01; it has no private
held-out rows. The historical evidence assesses the recipe, not future profit
from the final binary.

The stored evaluation outputs and data identities support the numbers above,
but the historical evaluation source is not fully byte-for-byte reproducible.
At least one tracked source fingerprint differs from the current tree, and a
Track-C result did not record its own complete source fingerprint before later
source edits. This does not change the recorded outputs, but it is an evidence
provenance limitation that must be accepted explicitly or closed by a newly
frozen rerun before promotion.

## Proposed release identity

| Item | Exact identity |
|---|---|
| Evaluation recipe | `full_live_contract_v6_durability` |
| Full-fit spec | `full_live_contract_v6_durability_corrected_20260805_fullfit` |
| Bundle | `ufc-production-20260801-full_live_contract_v6_durability_corrected_20260805_fullfit` |
| Staging manifest SHA-256 | `2f9a8529989d90e247ae2b184618d683e4230468a8be3d999c26abda5e43ee60` |
| Primary model SHA-256 | `e62f35830e0c70ef25bd56d3628169e65ad567707b433e1ec4ecdb69f30fe26b` |
| No-odds model SHA-256 | `0acfd773dabff87b8699bb2e7b8fd64c236e10161f2144f7e1b28438d85bc3c4` |
| Logistic model SHA-256 | `05e47c1871ec46f75ef135542a79c13a01ba2244b5c5d9b6b4efd56e30cd7374` |
| Immutable training fights SHA-256 | `77a9071d8991a2458644fcf8a3b41b681d9da5f266b47deff4dd8614ef8e6f75` |
| Immutable training features SHA-256 | `5c6b4cb328e4e7f66e13305a381614ca4023d545df560bb4fc1d7a84e6423183` |
| Corrected evaluation features SHA-256 | `7949168f55996d9510023e928b319beffecb57eaf90beb94ef9983235b18872b` |
| XGBoost / odds-noise seeds | `42 / 42` |
| Feature count | `211` |
| Training cutoff / buffer at build | `2027-01-01 / 148 days` |

The portable staging source is
`.codex_stage/corrected_durability_prod_20260805_v1_bundle`. It is ignored by
Git on purpose; model `.pkl` files are delivered to an isolated volume upload
location and are never committed.

The production allowlist intentionally excludes research-only evaluation
code. A supplemental 11-file post-run source snapshot exists at
`.codex_stage/corrected_durability_prod_20260805_v1_evaluation_source.tar.gz`
with SHA-256
`aa5737fd99d44a0a19399b096c7dcba1e52923b2e0ed4f68205ab0d0ddeb6cb6`.
It preserves the current reviewed source bytes, not a proven exact copy of all
source bytes used for model selection. It is evidence only, not executable
runtime input, and should accompany the staged bundle in a separate inert
evidence upload if promotion is later authorized.

## Captured predecessor artifacts for model/data rollback

Read-only production inspection captured the currently served model, immutable
training data, and lookup artifacts under
`.codex_stage/previous_production_runtime_20260806_v1`. The deployed Git SHA is
recorded separately as code-deployment evidence; it is not restored by the
artifact installer:

| Item | Exact identity |
|---|---|
| Live bundle | `ufc-production-20260801-full_live_contract_v6_durability_fullfit` |
| Deployed Git SHA | `0efb2cefd1fc6b32b077ba00c79dca9ee91c5c42` |
| Capture source manifest SHA-256 | `1435968903b6bd9943f539d8541e0b4404892a79aa73ce408d634572920f48e1` |
| Captured runtime manifest SHA-256 | `e9fc112c6f01ae88a2b9459c84084b2038e5f1361c71ce74bc5ce1c65327305b` |
| Capture inventory SHA-256 | `45a67035a7bf17fdf853471a46e46e4b9ad273e9374a980be9213386c7225c90` |
| Primary model SHA-256 | `5c196935937d2c0847e16183a2e063d223bff48dd659197c7a5cc9b8e7de3530` |
| No-odds model SHA-256 | `a3a4cfb0df525cd8cedaec5f6ca10558cca5a1798129e695a2bd28b3914ae7eb` |
| Logistic model SHA-256 | `d36583c59a1b965e099547fbdabc05bb5e64c7c12db49965a265278f71fe5ba2` |
| Runtime fights SHA-256 | `554a45930060ec7504f5f0313f6a2dddb512ccaa7440c7eff9c229de55c7741e` |
| Runtime features SHA-256 | `42381a2ca16458cea0f3891da899df351d14a56554ffb1fd0f15c08d5a9cc07a` |

The installer fails closed if the captured artifact predecessor, any stored
model, its captured lookup, or its readiness evidence no longer matches this
identity. It does not restore an earlier Railway deployment or shared raw-data
volume bytes. Before the first store-backed restart, the running legacy process
still uses the shared `/app/logs/processed` lookup, not the inert store copy.
Therefore the operator must repeat live `/readyz` plus direct remote model/file
hashes immediately before initialization/promotion. If those live bytes
changed, recapture and rebind the packet; do not rely on the older store copy
or bypass the check.

## Verification completed locally

- Rich manifest: 33 exact allowlisted files, all declared sizes and hashes
  reconciled.
- Python 3.11.15 with the exact requirement versions: all three models loaded
  and emitted 32/32 finite probabilities.
- Exact replay: 25 fights x 211 features, zero forbidden mismatches.
- Established-fighter prefight replay: 25 fights x 211 features, zero
  structural/non-time mismatches; 250 expected time-aging cells allowlisted.
- The immutable training snapshot contains 4,094 observed rows (8,188 after
  mirrored augmentation), with snapshot date 2026-08-01.
- Focused runtime/bundle/refit/staging/readiness checks pass.
- The complete bounded pytest result and atomic promotion/rollback rehearsal
  are recorded in the final verification section below.

The Python 3.11 check was on Windows and is not a Linux-container attestation.
The local Docker daemon was unavailable and was not started. An exact Linux
image build/startup smoke remains a mandatory later go-live gate after the
reviewed artifact-delivery commit exists.

## Atomic hosted layout

The fixed production store root is
`/app/logs/production_bundle/store` on Railway volume
`a9c56c45-be3f-4ff6-b0e8-208204bb2712`.

- `releases/<release-id>/` is a hash-inventoried, read-only model plus immutable
  training-snapshot unit.
- The active generation has its own mutable live lookup directory.
- A rollback generation uses a separately copied and hash-pinned lookup
  snapshot, so a partial or later-mutated lookup cannot be activated.
- `active_bundle.json` is the single atomic selection point. Models, training
  data, lookup identity, and rollback identity are never activated by
  independent file replacements.
- Startup falls back to the existing image bundle only when no store directory
  is detected. A detected but partial, tampered, or unresolvable store stops
  startup. Dangling-link behavior still requires the mandatory Linux startup
  proof before go-live.
- Within the bundle store, runtime code may write only the active lookup. It
  receives read/traverse access to the selected release but no write ownership
  of immutable models or training evidence.

## Later operator sequence — not executed

Each numbered phase requires fresh approval. Finishing one phase does not
authorize the next.

1. Under separate approval, pause the existing write-enabled `Weekly Model
   Retrain` workflow or finish and approve its safe replacement. Do not promote
   while it can independently push and redeploy `main`.
2. Under separate approval, quiesce the current deployment by setting
   `LIVE_TRADING_MODE=off` and `UFC_REFRESH_ENABLED=0`, which causes a restart.
   Keep both disabled through code-deploy overlap and candidate verification.
   Require two matching `/readyz` responses showing effective mode `off`,
   trading disabled, and betting/refresh loops disabled. Confirm exactly one
   RUNNING instance and `numReplicas=1`; one configured replica alone does not
   rule out old/new deploy overlap.
3. Bind a direct read-only CLOB `get_open_orders(max_attempts=1,
   read_timeout_seconds=8, total_budget_seconds=12)` query to that exact
   deployment-instance id. Require a successful explicit list, not an
   exception treated as empty, and apply the approved no-resting-order policy.
   Record `/api/positions` exposure separately. The preparation-time probe
   against instance `f016d80c-4a87-4869-9a71-ce6e6bd2a0a9` returned zero open
   orders, but this must be repeated after quiescence.
4. Re-read production `/readyz` and direct remote model/file hashes. If they
   differ from the predecessor above, stop, recapture, and rebuild the
   predecessor binding. Keep the capture-to-pointer window controlled so a
   background refresh cannot silently move the shared legacy lookup.
5. Commit only the exact reviewed allowlist. Do not use `git add -A`.
6. Push the reviewed commit only after separate push approval.
7. Before changing the deployment, record and prove a Railway-side rollback
   path to the exact predecessor deployment that does not depend on the app
   staying healthy. Also decide whether shared raw-data changes are explicitly
   forward-compatible or require a separate snapshot. The artifact installer
   alone is not a full-service rollback.
8. Build/deploy the delivery code while retaining the old image model. With no
   store present, startup must continue serving the old model bundle. This is
   not volume-read-only: existing startup behavior can overlay the reviewed
   same-name BFO recovery files and perform normal raw/profile maintenance.
   Prove the exact Linux image, obtain two matching old-bundle `/readyz`
   responses, and then recapture the live raw/lookup predecessor before any
   store promotion.
9. Upload the candidate and predecessor directories to unique inert
   `/app/logs/production_bundle/incoming/...` container paths on the named
   volume. Do not overwrite an existing path.
10. Run `initialize-legacy` to seed the exact old artifact unit, then `resolve`
   and compare every returned identity. The running process remains unchanged.
11. Run `promote` with explicit expected source, active bundle, release, and
   installed-manifest identities. This performs one atomic pointer replacement
   but does not change the already-running process.
12. Before activation, prove a break-glass path that can run the installer
    against the mounted volume if the application crash-loops; for example, a
    separately authorized recovery deployment/job using the same reviewed
    image and volume. Do not activate if rollback requires SSH into a healthy
    application container.
13. Only after separate artifact-activation approval, restart/redeploy while
    trading and refresh remain off. Require two consecutive `/readyz` responses
    that agree on deployed Git SHA, all three model hashes, bundle id, immutable
    training hashes, live lookup hashes, disabled loops, and one RUNNING
    instance.
14. Re-arm real trading and the approved refresh setting only under a final
    separate approval. Its restart-time position reconciliation must pass.
    Again require one RUNNING instance plus two matching `/readyz` responses
    before treating the release as live.
15. If model/data activation checks fail, use the proven normal or break-glass
    path to run the exact-identity artifact `rollback` pointer swap, then
    restart only with separate rollback/restart authority. If code/startup
    checks fail, use the separately proven Railway deployment rollback as well.
    Treat shared raw-data restoration as a separate action. Require two
    matching predecessor-artifact `/readyz` responses and the intended deployed
    Git SHA; never imply that a pointer swap changed the deployed code.

Never delete the predecessor release, either lookup snapshot, or either upload
until the release has completed an explicitly chosen observation period.

## Exact future Git allowlist

The final allowlist is intentionally explicit and is recorded after the local
rehearsal. Candidate/rollback `.pkl` files and processed snapshots are volume
artifacts, not Git inputs. Cron workflow changes, ledgers, unrelated dirty-tree
work, and candidate output directories are excluded.

The reviewed allowlist contains exactly these 41 paths:

```text
.dockerignore
entrypoint.sh
railway.toml
data/raw/historical_odds/historical_odds_bfo_recovered_20260319.csv
data/raw/historical_odds/historical_odds_bfo_recovered_20260529_fullfit_gap.csv
data/raw/historical_odds/historical_odds_bfo_recovered_20260711_guard_gap.csv
data/raw/historical_odds/historical_odds_bfo_recovered_auto_20260722_run29887204421_1.csv
data/raw/historical_odds/historical_odds_bfo_recovered_auto_20260728_run30341844205_1.csv
data/raw/historical_odds/historical_odds_bfo_recovered_auto_20260804_run30891790168_1.csv
data/raw/historical_odds/bfo_revalidation_20260805_head_source.provenance.jsonl
docs/DURABILITY_PROMOTION_RUNBOOK.md
docs/CORRECTED_DURABILITY_PROMOTION_PACKET_2026-08-06.md
scripts/bootstrap_runtime_production_bundle.py
scripts/parity_replay.py
scripts/recover_bfo_moneyline_gaps.py
scripts/revalidate_bfo_recovery_file.py
scripts/build_model_input_inventory.py
scripts/build_staged_production_bundle.py
scripts/install_staged_production_bundle.py
src/bot.py
src/config.py
src/data/fighter_lookup.py
src/features/build_features.py
src/model/production_bundle.py
src/model/training_spec.py
src/web/app.py
tests/test_build_staged_production_bundle.py
tests/test_config.py
tests/test_corrected_durability_training_spec.py
tests/test_entrypoint_atomic_copy.py
tests/test_install_staged_production_bundle.py
tests/test_model_input_inventory.py
tests/test_parity_replay_cli.py
tests/test_phase2_schema_contract.py
tests/test_production_bundle.py
tests/test_railway_config.py
tests/test_refit_automation.py
tests/test_revalidate_bfo_recovery_file.py
tests/test_staged_production_bundle.py
tests/test_ufc_audit_regressions.py
tests/test_web_runtime_api.py
```

The provenance ledger intentionally preserves historical absolute
`C:\Users\Evan\...` `output_batch` strings. They are non-secret provenance;
changing them would change the frozen input and staging identities and requires
a rebuilt packet.

Explicitly excluded are the weekly workflow and its refit/deployment-wait and
scheduled-quality files/tests, research/tuning and model-lab changes, the four
historical `CODEX_*` handoff documents, every candidate processed/model
directory, `.codex_stage/**`, `logs/**`, all `.pkl` files, runtime ledgers, and
environment files.

## Final verification record

This record is local preparation evidence, not activation approval.

- `python -m compileall -q src tests scripts`, `bash -n entrypoint.sh`, and
  `git diff --check` passed.
- `scripts/check_tracked_artifact_integrity.py` passed.
- `scripts/check_production_refit_contract.py --manifest
  .codex_stage/corrected_durability_prod_20260805_v1_bundle/staging_manifest.json`
  passed.
- The focused dirty-worktree suite passed `320` tests with one Windows
  symlink-privilege skip.
- The complete suite ran in four deterministic slices: `2,016` passed, `2`
  skipped, and `1` failed. The only failure was
  `tests/test_live_train_parity.py::test_prefight_replay_matches_training_rows`
  against the intentionally unchanged canonical `data/processed` snapshot.
  The same parity file, pointed through `UFC_PROCESSED_DIR` at the frozen
  candidate snapshot, passed both tests (`2 passed`). Canonical artifacts were
  not replaced to make the suite green.

The exact allowlist was overlaid onto a clean clone of base Git commit
`0efb2cefd1fc6b32b077ba00c79dca9ee91c5c42`. It produced exactly `41` changed
or new paths and no other dirty paths. In that isolated tree, compile, shell
syntax, and diff checks passed; the focused suite passed `319` tests with two
expected skips (Windows symlink privilege and the deliberately absent local
predecessor capture). An initial run from a deeply nested pytest temp path hit
three Windows path-length `FileNotFoundError`s; the identical tree rerun with a
short unique `--basetemp` passed, so those were not assertion or dependency
failures. After this packet is frozen, the authoritative per-path SHA-256,
byte-size, Git-blob, and synthetic tree identities are written to
`.codex_stage/corrected_durability_promotion_allowlist_20260806_v1.json`.
Any later byte or base-commit change invalidates that record and requires a
rebuild and rerun.

The Python 3.11.15 compatibility report is
`logs/corrected_durability_prod_20260805_v1/python311_compatibility_verification.json`
(`1,919` bytes, SHA-256
`5dda957cebf0a8592a869ab2ed7a527fa2119c88dae465d2b03df3f376065203`).
It records strict 33-file manifest validation plus finite predictions from all
three models (`32/32` each). Preserve this file in the inert evidence upload;
it is local-only until a separately authorized upload occurs. The model-review
report is `4,789` bytes with SHA-256
`b511149904a7ae64792222ee992efbec8aab279cab8f929061547a56dd283ee6`.

The disposable atomic rehearsal used this exact command chain, with every
required `--expected-*` identity supplied and no default source/target paths:

```text
initialize-legacy -> resolve -> promote -> resolve
-> rollback (candidate to predecessor) -> resolve
-> rollback (predecessor to candidate) -> resolve
```

It resolved these target-path-dependent installed identities:

| Item | Rehearsed identity |
|---|---|
| Predecessor release | `r-1435968903b6bd9943f5` |
| Predecessor installed manifest | `e5df5588f53bcf43810dbe24504e871589a0cf768dd23de9cecf45da1cfbd010` |
| Candidate release | `r-2f9a8529989d90e247ae` |
| Candidate installed manifest | `9e6c45a0fcd00130ee8715a06441ffcef858e7a804e31da8eea279f6e3f8e748` |
| Final active/rollback state | candidate / predecessor |
| Final `active_bundle.json` SHA-256 | `7b273bf5bc655c012a4b3eb44196551471c6ce0a7df619f664d149a521996ad0` |

The rehearsal proved initialize, promote, exact artifact/lookup rollback, and
roll-forward in a disposable local store. Installed-manifest hashes depend on
their final paths and therefore must be resolved and rebound on Railway; the
local values above must not be copied into a remote promote command.

Remaining go-live blockers are the active write-enabled weekly workflow, the
mandatory exact Linux image/startup/break-glass proof, fresh quiesced
single-instance/open-order/live-hash capture, and an explicit decision to
accept the historical evaluation-source provenance limitation or rerun the
evaluation from a newly frozen source inventory.

## Cron remains out of scope

Do not enable, dispatch, disable, or finish the Tuesday 06:00 UTC weekly refit
from this packet. The old write-enabled schedule is already active and is a
promotion blocker, but changing its external state requires separate approval.
Its fixed-model policy, publication transaction, baseline migration, bounded
tests, and final container proof still require a separate review and explicit
authorization.
