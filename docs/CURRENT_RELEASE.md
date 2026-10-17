# Saved production baseline and current release evidence

`release-source-manifest.json` preserves the exact October 16 deployed source
hashes, immutable image `6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94`
and deployment `c48776b2-475a-45d3-bc85-1faca04d4c6c`. That manifest is unchanged.
The account-read diagnostics overlay is declared separately in
`source-change-manifest.json`. It adds safe failure-stage/HTTP-status diagnostics
and mode-neutral warnings; retry, sizing and trading policies remain unchanged.

A source manifest proves bytes, not rollout. The [CI production job](https://github.com/nfgems/ufc-betting-bot/actions/workflows/ci.yml)
and its `railway-release-<commit>` artifact identify the commit, immutable image,
previous image, Railway deployment and health/readiness result. The built image
also contains `/app/release-build.json` and an OCI revision label. Compare these
with the actual live deployment when checking what is running. A skipped/failed
production job means the push has not been verified as deployed.

The production Dockerfile inherits the original private image, accepted
model/data payload, seed and source-request provenance record. It copies only
versioned source and writes build identity. It never copies local models,
ledgers, credentials or live data. `python tools/verify_release_source.py`
checks all 285 repository source files against the baseline plus declared
overlay. Image verification checks all 286 runtime files, including inherited
provenance. The source status `declared_overlay` makes no deployment claim.

Build locally with `docker build --network none -t ufc-saved-source:test .`
using authorized private GHCR access. Public CI uses `Dockerfile.ci`, public
dependencies and synthetic tests without private-image or account access.
Tests run with networking and orders disabled, without application startup.
The separate protected production job tests again against exact private-image
dependencies before publication and deployment. See [Deployment workflow](DEPLOYMENT_WORKFLOW.md).

Historical source/tests, proposals and operational handoffs remain preserved
in the owner's local archive branches and backups. They are not an approved
release queue. The separate fee-accounting proposal remains unapplied; historical
fee repair and model/data research are outside this automation release.
Private handoffs are excluded from the current tree; older published Git history
still contains earlier notes. Preserve ignored datasets, original observations,
authority/consumption records and closed experiments at their original paths.

Historical scheduled workflow definitions and disabled states remain unchanged.
The restored release job does not collect new source data, train/promote models,
alter trading controls, or restore old volume contents.
