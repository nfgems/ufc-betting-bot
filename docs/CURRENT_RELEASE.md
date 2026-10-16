# Current saved production source

This checkout saves the code running on Railway at the October 16, 2026 source
reconciliation. Its reference image and exact file hashes are recorded in
`release-source-manifest.json`. The application code, configuration, dependencies
and startup scripts come from immutable image `6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94`.
This source-control update does not deploy, restart, change settings or start work.

The Dockerfile inherits that exact private image, including the accepted model,
data payload and seed, and copies only versioned source. It does not copy local
models, ledgers, credentials or live data. One source-request provenance record
needed by identity-evidence validation is retained only in that pinned image;
its path/hash is declared as inherited in the manifest. The checkout verifier
checks all 285 repository files; the image verifier checks all 286 runtime files.
This requires access to the existing private GHCR image; it is not a from-scratch
model/data rebuild or a new model release.

Run `python tools/verify_release_source.py` to check source bytes. Build with
`docker build --network none -t ufc-saved-source:test .`. CI compiles and runs the
saved release's complete current test suite in the image, with networking and
orders disabled and without application startup. Historical tests and source
remain available on `archive/main-before-production-sync-20261016`; the pending
model research is saved on `archive/research-before-production-sync-20261016`.
They are separate versions and are not mixed into this release's test tree.

All 196 pending files were also backed up under the sibling directory
`../ufc-git-reconciliation-20261016/research-backup`. The research source snapshot
is not an independently authorized or self-contained rerun bundle. Its ignored
datasets, original observations, authority/consumption records and closed
experiments remain at their original local paths. Never discard or reset them.

GitHub's existing scheduled workflow definitions are retained unchanged. This
reconciliation does not run or alter them, enable monthly promotion, or replace
the existing Railway image/volume. Source publication should preserve that
separation; the active Railway service remains pinned to the reference image.
