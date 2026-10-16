FROM ghcr.io/nfgems/ufc-betting-bot@sha256:6068bb7d8941e32dd4bf6a4f1c00fc9bc22148aa9cb23a6c45d146d37fedab94

# Inherit accepted private payload/models/seed, exact dependencies and startup.
# Only the explicitly versioned source is copied from the checkout.
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY --chmod=755 entrypoint.sh refresh-entrypoint.sh legacy-entrypoint.sh /app/
COPY requirements.txt refresh_seed_manifest.json release-source-manifest.json /app/
COPY tools/verify_release_source.py /tmp/verify_release_source.py
RUN python /tmp/verify_release_source.py --root /app --image
