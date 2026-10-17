# GitHub main to Railway

The owner authorized restoring automatic deployment on October 17, 2026.
Only reviewed release changes belong on `main`. A push runs the `CI` workflow:

1. Public source/artifact checks, compilation, all tests and dashboard checks run
   without private registry or account credentials. PRs run these checks only.
2. The production job requires those checks and the `production` environment,
   restricted to `main`. A shared concurrency group serializes releases without
   interrupting a deployment already in progress.
3. The job checks that its exact commit is still GitHub main and captures the
   current Railway image/deployment. It builds from the pinned private baseline,
   verifies installed source, and runs the full suite offline against the exact
   production dependencies with orders off and synthetic temporary state.
4. Only a tested image is published under `git-<full commit>`. Its registry digest
   becomes the Railway source. The helper rechecks main and production before a
   single source update, then waits for that image's deployment and health/readiness.
   It performs no forced refresh, order, account write, volume restore or extra restart.

The private GHCR package stays private and unlinked to this public repository.
Do not grant the public repository package Actions access: forks can gain access
through that grant. Public tests must not receive private image access.
`GHCR_TOKEN` and `RAILWAY_TOKEN` are production-environment secrets, never repository
secrets or container build arguments. Registry login is confined to the production
job; Railway credentials are supplied only to its preparation/deployment steps.
The GHCR credential must be valid for package read/write. A package-only classic
PAT is preferable when rotating an existing broader authenticated registry token.

Open the [CI runs](https://github.com/nfgems/ufc-betting-bot/actions/workflows/ci.yml)
for the commit and inspect **Deploy tested main to Railway**, its job summary and
`railway-release-<commit>` artifact. The receipt contains exact image/deployment
identity and the prior immutable image for rollback. Failed/skipped deployment
is not a successful release, even if public checks passed. Compare the receipt
to Railway's currently active deployment; later releases can supersede it.
The image records the commit in `/app/release-build.json` and its OCI revision label.

Runtime/source verification uses read-only inspection, including installed source
hashes and existing persistent paths. Do not provoke an account failure or force
trading/refresh work to demonstrate a diagnostics change. Preserve current controls,
original IDs/fills, pending intents, accepted publications and model/data state.

For an authorized rollback, inspect current production and select the previous
immutable image recorded in the release receipt. Update only the service image
and verify activation. Never restore an old volume or overwrite newer accounting
records. Keep `/app/logs`, startup and configured controls unchanged. A later main
push will release the newly tested commit, so reconcile source before resuming.

The initial audit found source.repo unset and source.image pinned to the October 16
release. The September 19, 2026 exact-image V9 cutover intentionally removed the
Git deployment trigger; later public CI contained no replacement deploy job.
Image-based deployment is retained, now driven by the tested main workflow.
