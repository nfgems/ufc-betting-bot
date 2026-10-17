FROM ghcr.io/nfgems/ufc-betting-bot@sha256:6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94

# Inherit accepted private payload/models/seed, exact dependencies and startup.
# Only the explicitly versioned source is copied from the checkout.
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY --chmod=755 entrypoint.sh refresh-entrypoint.sh legacy-entrypoint.sh /app/
COPY requirements.txt refresh_seed_manifest.json release-source-manifest.json /app/
COPY source-change-manifest.json /app/
COPY tools/verify_release_source.py /tmp/verify_release_source.py
RUN python /tmp/verify_release_source.py --root /app --image

# Trace a tested release to Git without granting the public repo private-package access.
ARG SOURCE_COMMIT=local-unpublished
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT
RUN SOURCE_COMMIT="$SOURCE_COMMIT" python -c 'import json,os,re; from pathlib import Path; commit=os.environ["SOURCE_COMMIT"]; assert commit=="local-unpublished" or re.fullmatch("[0-9a-f]{40}",commit); Path("/app/release-build.json").write_text(json.dumps({"git_commit":commit})+"\n")'
