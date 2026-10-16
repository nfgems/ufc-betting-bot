#!/bin/bash
# Final web dispatch, after the existing startup migrations and ownership guard.
set -euo pipefail

if [ -z "${UFC_ROSTER_PACKAGE_SHA256:-}${UFC_ROSTER_STARTUP_REQUEST:-}${UFC_ROSTER_STARTUP_REQUEST_SHA256:-}" ]; then
    exec su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python -m src.web.serve"
fi

# An incomplete explicit request is a configuration error, never an implicit
# prepare/provision/default invocation. No original is constructed here.
: "${UFC_ROSTER_PACKAGE_SHA256:?Explicit runtime package SHA256 required}"
: "${UFC_ROSTER_STARTUP_REQUEST:?Explicit original startup request required}"
: "${UFC_ROSTER_STARTUP_REQUEST_SHA256:?Explicit original request SHA256 required}"

# The enclosing shell/image is already trusted. Refuse loader overrides before
# the first packaged Python executable; its launcher child separately receives
# the exact clean u59 initial environment. Web settings retain their values.
for name in "${!LD_@}" "${!DYLD_@}"; do
    if [ -n "$name" ]; then
        echo '[startup] loader override refused for root web composition' >&2
        exit 1
    fi
done
cd /
exec /opt/ufc-roster/runtime/python/bin/python3.11 -I -S -B \
    /opt/ufc-roster/runtime/source/scripts/roster_web_service.py \
    --package-sha256 "$UFC_ROSTER_PACKAGE_SHA256" \
    --request "$UFC_ROSTER_STARTUP_REQUEST" \
    --request-sha256 "$UFC_ROSTER_STARTUP_REQUEST_SHA256" < /dev/null
