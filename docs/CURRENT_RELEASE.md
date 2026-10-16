# Current saved production source

This checkout saves the application source running on Railway at the October 16,
2026 reconciliation. `release-source-manifest.json` records its immutable image
and exact file hashes. The application code, configuration, dependencies and
startup scripts match image `6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94`.
Publishing this source does not deploy, restart or reconfigure the online bot.

The production Dockerfile inherits that existing private image, including its
accepted model/data payload and seed, and copies only versioned source. It does
not copy local models, ledgers, credentials or live data. One source-request
provenance record is retained only in that pinned image and declared as inherited
in the manifest. The checkout verifier checks all 285 repository source files;
the production-image verifier checks all 286 runtime files.

Run `python tools/verify_release_source.py` to check source bytes. Rebuilding the
production image with `docker build --network none -t ufc-saved-source:test .`
requires authorized access to the existing private GHCR image. It is not a
from-scratch model/data rebuild or a new model release.

Public CI uses `Dockerfile.ci` to install public dependencies and copy only public
source, configuration and synthetic test fixtures. It receives no private-image
access or account secrets. The tests run with networking and orders disabled,
without application startup. Reproduce with `docker build -f Dockerfile.ci -t
ufc-public-tests .` and `docker run --rm --network none ufc-public-tests`.
Source integrity, Python compilation, artifact integrity and dashboard JavaScript
checks also run on pushes and pull requests.

Historical source/tests, pending research and full operational handoffs remain
preserved in the owner's local archive branches and verified backups. Operational
handoffs are excluded from the current source tree. They were present in earlier
published Git history; this cleanup does not rewrite that history. Archive
branches remain local. Their
ignored datasets, original observations, authority/consumption records and
closed experiments remain at their original local paths. Never discard, reset
or infer new rerun authority from them. The private canonical handoff remains
the current record when working in the owner's workspace.

Existing scheduled workflow definitions and disabled states are unchanged. This
publication does not run them, enable model promotion or replace the existing
Railway image/volume. Historical workflows require their corresponding archived
source and separate authorization before use.
