# Track B Tapology access and missing-profile decision — 2026-08-20 V3

This receipt supplements
`TRACK_B_TAPOLOGY_PRIMARY_ROOT_CAUSE_REPAIR_RECEIPT_2026-08-20_V2.md` after
the owner clarified that a genuinely absent Tapology profile may remain blank
and be reported as missing.

## Plain-language decision

Direct Tapology origin and hosted Chrome remain Cloudflare-blocked. The
configured reader route is nevertheless a usable Tapology access path: hosted
run `32414164187` returned a valid Tapology zero-result page for Harrison Garcia
and fetched and parsed the known Feng profile with identity and physical fields.

Harrison's result is therefore not an access failure. Tapology currently has no
discoverable Harrison profile. His profile fields remain blank/NaN and are not
fabricated. A healthy zero-result search is treated as zero recovery and zero
source errors. An unhealthy search, transport error, identity mismatch, or
profile parser failure remains visibly red.

## Minimal hardening

- The standalone script now uses the same known-good Feng Pengchao name and
  exact profile URL already used by the scheduled workflow. A green probe still
  requires healthy discovery, that exact profile, matching identity, and at
  least one real physical field.
- Hosted diagnostics now include the count and public names of hollow zero-record
  off-card rows quarantined from Tapology recovery. Harrison is therefore
  explicitly visible as intentionally missing/unverified rather than silently
  treated as a source outage.
- The canonical roster and supplement are not given a blank synthetic row or a
  permanent negative cache. A future unambiguous current-card match still
  restores eligibility.
- No workflow refactor, proxy system, second canary, alternate data provider, or
  name-specific denylist was added.

## Validation

- Focused default-canary, quarantine, healthy-zero, and unhealthy-discovery
  checks: `5 passed`.
- Entire hosted refresh test file: `58 passed`.
- Tapology scraper selection, excluding the previously documented local
  Selenium-import-only test: `54 passed, 179 deselected`.
- The first parallel invocation encountered only a shared pytest temporary-root
  race on Windows; the same selection passed cleanly when rerun serially.
- `python -m compileall -q src scripts tests`: passed with the mandated Python.
- `git diff --check`: passed with only existing Windows line-ending warnings.

Current file hashes:

- `scripts/refresh_tapology_profile_supplement.py`:
  `ed8333d3ac31d8bfee6ab5572a69f3c6c356737f4b367728b2004e06231a18ee`
- `src/data/fallback_scrapers.py`:
  `8844e124b250ab7677005201c400cc3ed2973f6744848eb9aae290295536fb13`
- `tests/test_tapology_profile_refresh.py`:
  `d1c79d1a8c48ff3dbdf93071db50cd2e9c8d0458b31a2131f3325a08c14d9267`
- `tests/test_v4_profile_enrichment.py`:
  `f52a07563a017c76e9f97dff8df6b633e87d0bd45e54407c4c24bb8fd98f5d31`

## Status

Under the owner's revised acceptance decision, the configured hosted Tapology
path is usable: Feng proves access and parsing, while Harrison is an honest
missing-profile outcome. The local 403 circuit and candidate-eligibility repairs
are ready for a separate integration/deployment decision.

No commit, push, deployment, production/Railway mutation, data write, model
action, schedule activation, or order action was performed.

**Outcome: `TAPOLOGY_PRIMARY_FIX_READY_FOR_SEPARATE_DEPLOYMENT_AUTHORIZATION`.**
