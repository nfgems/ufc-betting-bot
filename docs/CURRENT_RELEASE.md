# Saved production baseline and current release evidence

`release-source-manifest.json` preserves the exact October 16 deployed source
hashes, immutable image `6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94`
and deployment `c48776b2-475a-45d3-bc85-1faca04d4c6c`. That manifest is unchanged.
The account-read diagnostics and observed entry-cost overlays are declared
separately in `source-change-manifest.json`. Diagnostics retain safe failure-stage
and HTTP-status fields. Accounting uses independently matched cash debits, including
observed entry charges, for original open real fills and future settlement.
Original gross fill records and already-settled history remain intact. Missing or
ambiguous cost delays financial finalization. See [Fee accounting](FEE_ACCOUNTING.md).

The declared consumer overlay preserves current market observations when an
unresolved original order blocks execution. It reports the blocker explicitly,
keeps trading disabled, and separates service readiness from trading readiness.
The dashboard distinguishes unavailable coverage from observed empty markets
and dates market checks using actual observations. Rollout still requires the
release receipt and runtime verification described below.

Evaluation alerts have separate recovery lifecycles for observation failures
and unresolved original orders. A completed fresh observation can recover an
earlier market-fetch warning while the order blocker stays active. Recovery
survives a worker restart, preserves alert history, and never changes accounting
records or trading controls. Gamma fetch results and their alert transitions
are serialized so an older success cannot clear a newer failure.

The dashboard also reports independently observed card and publication freshness
holds alongside order recovery. These messages come only from a coherent, fresh
consumer report and use fixed public wording. They do not reconcile an order,
refresh a publication, or change execution and readiness checks.

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
checks the 285 original repository source files plus explicitly declared additions
against the baseline and overlay. Image verification also checks inherited
provenance. The source status `declared_overlay` makes no deployment claim.

Build locally with `docker build --network none -t ufc-saved-source:test .`
using authorized private GHCR access. Public CI uses `Dockerfile.ci`, public
dependencies and synthetic tests without private-image or account access.
Tests run with networking and orders disabled, without application startup.
The separate protected production job tests again against exact private-image
dependencies before publication and deployment. See [Deployment workflow](DEPLOYMENT_WORKFLOW.md).

Historical source/tests, proposals and operational handoffs remain preserved
in the owner's local archive branches and backups. They are not an approved
release queue. The accepted fee-accounting correction is declared in this source;
use release/runtime evidence above to establish deployment. Historical fee repair
and model/data research remain outside this release.
Private handoffs are excluded from the current tree; older published Git history
still contains earlier notes. Preserve ignored datasets, original observations,
authority/consumption records and closed experiments at their original paths.

Historical scheduled workflow definitions and disabled states remain unchanged.
The restored release job does not collect new source data, train/promote models,
alter trading controls, or restore old volume contents.
