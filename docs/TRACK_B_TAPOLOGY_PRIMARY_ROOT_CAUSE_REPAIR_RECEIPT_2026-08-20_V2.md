# Track B Tapology primary root-cause repair receipt — 2026-08-20 V2

Recorded at `2026-08-20T20:41:10.658Z`.

This receipt supersedes the repair conclusion in
`TRACK_B_TAPOLOGY_PRIMARY_ROOT_CAUSE_DIAGNOSTIC_RECEIPT_2026-08-20_V1.md`.
It preserves V1's failed-run evidence and adds one authorized, fail-closed
GitHub-hosted probe plus the smallest justified local repairs. It does not
claim that Harrison Garcia has a Tapology profile or that Tapology's origin is
reachable.

## Scope and immutable anchors

- Track B worktree only:
  `C:/Users/Evan/betting-bot/ufc-betting-bot-main-production-warning-repair-20260813`
- Track B base and `HEAD` remained
  `3eacba726d5f0fd6c497492a18c74a85d8001044`.
- The original dirty checkout was not touched.
- Every Python command used
  `C:/Users/Evan/betting-bot/_ci_venv/Scripts/python.exe`.
- No model, schedule, order, secret, or production configuration was changed.

Railway was inspected read-only as required. The authenticated target remained
project `ufc-betting-bot`, environment `production`, service `ufc-bot`, on the
successful rollback deployment for commit `3eacba726d5f0fd6c497492a18c74a85d8001044`.

## Authorized hosted probe

One diagnostic-only workflow dispatch was run against the exact production
commit:

- Run: `32414164187`
- URL: `https://github.com/nfgems/ufc-betting-bot/actions/runs/32414164187`
- Ref/SHA: `main` / `3eacba726d5f0fd6c497492a18c74a85d8001044`
- Inputs: `probe_only=true`, `sync_active_roster=false`, `limit=1`,
  `runner=ubuntu-latest`, `probe_name=Harrison Garcia`, and known parser canary
  `probe_url=https://www.tapology.com/fightcenter/fighters/230675-feng-peng-zhao-winged-tiger`.
- Runtime: GitHub-hosted Ubuntu 24, direct route, no configured proxy, and no
  configured Brave search key.
- Artifact: `tapology-primary-32414164187-1`
- Artifact digest:
  `ac739707d7c31b9b7804c33979519c27dd9e6c805b85a5e359b2076d8caa040b`

The probe was intentionally fail-closed. `--probe-only` returned before roster
sync or supplement refresh, packaging was skipped, the final commit step was
skipped, and repository `main` remained exactly `3eacba726d5f0fd6c497492a18c74a85d8001044`.

Downloaded diagnostic hashes:

- `attempt.json`:
  `78621519084afdc34e0adb804baa18416b6a246a3fa32e118968bf378bcba3ca`
- `attempt.log`:
  `37519bc5d4327781fc575748bd37b16ca6795dda28c7b9c6fa781949ff1904c2`
- `runtime.txt`:
  `21d6ac22330d5ce2b420fe7a32711b4217ce02aa9fea9b4413919e4a0e142777`

## Probe result and route separation

The new hosted probe established the following without mutating data:

1. Tapology origin search was still Cloudflare `403`.
2. Hosted headed Chrome was also Cloudflare `403`.
3. The reader initially returned invalid non-search content for Harrison, then
   retried within the existing bounded policy and returned a valid Tapology
   `Search Results (0)` page. The diagnostic recorded `healthy=true`,
   `candidate_count=0`, `result=no_results`, and `reader_circuit_open=false`.
4. The known Feng origin profile was blocked, but the reader fetched and parsed
   the profile as `Pengchao Feng` with physical fields. Its mismatch against the
   requested Harrison identity correctly kept the probe red.

This rules out current parser drift and a current reader-wide outage. It does
not make origin access healthy, and it does not create a Harrison profile.

## Proven root causes

The failed scheduled run `31927852134` combined two independent problems:

1. A target-specific reader `403` was treated as reader-wide. Feng and other
   reader targets had just succeeded, but Harrison's default and `x-no-cache`
   requests both returned `403`. The code then opened a global 900-second
   circuit, preventing later fighters from using the reader. The deeper cause
   of that transient external `403` is not recoverable from the old diagnostics,
   but the global circuit decision is a repository-code defect.
2. Harrison Garcia was not a valid recovery candidate. The cached UFC row is a
   hollow archive/CMS placeholder: `Active`, `0-0-0`, no UFCStats identity, no
   profile record, no birthplace, and no height/reach/weight. Tapology exact and
   reasonable alias searches return zero results. Current production card
   discovery returned 75 bouts and 150 fighter names, with no Harrison match.

The 978-row cached roster contains exactly 12 eligible rows with the same
strong hollow signature. None matched the current cards. This is archive
contamination, not evidence that twelve current MMA fighters need Tapology
recovery.

## Minimal local repairs

No proxy machinery, new provider, canary system, per-URL cache, counter, or
workflow refactor was added.

### Reader circuit scope

In `src/data/fallback_scrapers.py`:

- Reader `403` remains bounded to the existing default and `x-no-cache`
  variants, then fails only that target.
- A raw `403` or reader-relayed Cloudflare `403` no longer opens the global
  reader circuit.
- The failure result retains the full sanitized exception text, including
  `status 403`, for workflow classification and evidence.
- Global circuit behavior remains for true reader-wide signals: `401`, `451`,
  and exhausted transport failures.

### Candidate eligibility

In `scripts/refresh_tapology_profile_supplement.py`:

- The temporary hosted Tapology candidate path now quarantines an `Active`,
  `0-0-0`, default-`mma` row that has no classification reason, UFCStats
  identity, profile record, birthplace, height, reach, or weight.
- The visible reason is
  `unverified_hollow_zero_record_not_on_current_card`.
- An unambiguous current-card match bypasses this quarantine, preserving a real
  booked newcomer.
- The canonical roster is not mutated and no name denylist was added.

The pre-existing one-line `runtime_block` classifier repair remains useful for
alternate-runner routing, but it is not described as primary-path recovery.

## Changed file hashes

- `scripts/refresh_tapology_profile_supplement.py`:
  `e33b239ef5dcbb40b94f56ef2e2f75622f3e299fd3c7af2d6efad625d7f04486`
- `src/data/fallback_scrapers.py`:
  `8844e124b250ab7677005201c400cc3ed2973f6744848eb9aae290295536fb13`
- `tests/test_tapology_profile_refresh.py`:
  `2e35f8d8a2fed337d2053e840ddfe7d56735c5abffbc252a056aa576d3b6909d`
- `tests/test_v4_profile_enrichment.py`:
  `e32475389c9f79836228f71e277f29bfed5bf04e510e5db2a8effe3f48c3cb6d`

The unrelated, pre-existing Track B wallet-cache repair remained intact and
was not altered during this investigation.

## Validation

All commands below used the mandated Python interpreter.

- Five focused reader/candidate regressions: `5 passed`.
- Entire hosted refresh test file: `57 passed`.
- Reader-focused scraper selection: `17 passed, 215 deselected`.
- Entire scraper enrichment file: `231 passed`; one test failed during import
  because the mandated local CI venv does not contain `selenium`. The failure
  was `ModuleNotFoundError` before repository code executed and is the same
  known environment limitation recorded in V1.
- Existing wallet-cache regression file: `25 passed`.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed; Git emitted only existing Windows line-ending
  conversion warnings.

The actual cached roster was evaluated with the new predicate: 12 hollow rows
were identified, Harrison was one of them, current discovery returned 75 bouts
and 150 names, and Harrison was not an unambiguous current-card match.

## Remaining risks and decision

- Tapology origin and hosted Chrome remain Cloudflare-blocked.
- The old Harrison reader `403` was transient/external and its service-layer
  cause remains unproven because the old workflow did not preserve response
  headers or a body hash.
- Harrison currently has no Tapology search result or profile, so the original
  acceptance requirement for Harrison identity plus physical fields cannot be
  satisfied honestly.
- These repairs are local and uncommitted; they have not been exercised on a
  hosted runner.

No deployment, commit, push, production configuration change, roster data
write, supplement update, schedule change, model action, or order action was
performed.

**Outcome: `TAPOLOGY_PRIMARY_NOT_READY`.**
