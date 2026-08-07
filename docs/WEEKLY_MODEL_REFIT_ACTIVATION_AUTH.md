# Weekly model refit activation authentication

The protected scheduled activation keeps the existing Railway volume publisher
and in-container installer. That path preserves the exact parent
compare-and-swap check, atomic pointer update, rollback, restart, and two exact
readiness samples. Railway volume-file operations alone cannot run those locked
installer checks, so the workflow does not replace them with client-side file
moves.

## Protected GitHub environment contract

The repository must contain an environment named exactly `production`, limited
to the `main` branch, with exactly these two secrets:

- `RAILWAY_TOKEN`: a Railway project token scoped to this project's production
  environment. It is not an account or workspace token.
- `RAILWAY_SSH_PRIVATE_KEY`: a dedicated, unencrypted Ed25519 private key used
  only by the weekly-refit GitHub runner. Its public half is registered once as
  a Railway workspace SSH key.

The SSH key is workspace-wide because that is Railway's SSH-key scope. The
project token remains environment-scoped and cannot add, remove, or list SSH
keys. Do not store an account/workspace Railway API token in GitHub for this
workflow.

The publisher does not call `railway whoami`: that command asks for account
identity and is not a valid project-token authentication test. It instead
requires the exact project, environment, service, deployment Git SHA, single
running instance, volume, active bundle state, and public readiness identity.

## Fresh-runner SSH trust

Railway CLI `5.30.4` delegates ordinary SSH commands to the system OpenSSH
client. Railway does not publish stable authoritative host-key fingerprints or
SSHFP records. On each fresh runner the workflow therefore:

1. resolves `ssh.railway.com` once and uses that one relay IP for the job;
2. writes the dedicated private key with mode `0600`;
3. enables `BatchMode`, `IdentitiesOnly`, and
   `StrictHostKeyChecking accept-new` for only `ssh.railway.com`;
4. retains the accepted key for the whole job and fails if it changes; and
5. removes the runner key/config files in an `always()` step.

This is trust on first use, matching the Railway CLI's own noninteractive SSH
forwarding precedent. It is not an out-of-band cryptographic pin. The workflow
must fail closed if Railway presents inconsistent relay keys; it must never use
`StrictHostKeyChecking=no` or `/dev/null` as `UserKnownHostsFile`.

## Activation sequence

1. Keep `Weekly Model Retrain` disabled.
2. Merge the authentication change and require green exact-head and merge-SHA
   CI plus a healthy Railway deployment of that merge SHA.
3. Create the exact `production` environment, limit it to `main`, and provision
   only the two secret names above through secure interfaces.
4. Manually run `Weekly Model Refit Railway Access Check` from `main`. It may
   read status, volume metadata, installer state, and `/readyz`; it must not
   upload, promote, roll back, restart, delete, or rename anything.
5. Enable the schedule only after that fresh-runner check succeeds.
6. Do not manually dispatch the schedule-only workflow. Enabling it must not
   create a deployment or change any production identity.

Primary Railway references:

- <https://docs.railway.com/cli/login>
- <https://docs.railway.com/cli/ssh>
- <https://github.com/railwayapp/cli/tree/v5.30.4/src/commands/ssh>
