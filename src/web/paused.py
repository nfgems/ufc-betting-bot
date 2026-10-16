"""Read-only release readiness; no dashboard, monitor, scheduler or trading."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import re
import sys

logger = logging.getLogger(__name__)

PAUSE_REASON = "Release is paused for verification. Activation requires a separate deployment decision."
DISABLED_COMPONENTS = (
    "trading", "btc5m_trading", "monitor", "clob", "refresh", "monthly_schedule",
    "source_collection", "model_training", "automatic_promotion", "position_reconciliation",
)


def _pinned_manifest(environ) -> tuple[Path, str]:
    raw = str(environ.get("UFC_RELEASE_MANIFEST", "")).strip()
    expected = str(environ.get("UFC_RELEASE_MANIFEST_SHA256", "")).strip().lower()
    if not raw or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Paused startup requires an explicit release manifest and SHA256")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Release manifest must be an absolute regular non-symlink file")
    resolved = path.resolve()
    if resolved != path.absolute():
        raise ValueError("Release manifest ancestry must not contain symlinks")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Release manifest SHA256 mismatch")
    return resolved, expected


def validate_release(environ=None) -> dict:
    """Fully validate explicit saved inputs before the diagnostics server binds."""
    env = os.environ if environ is None else environ
    if env.get("UFC_STARTUP_MODE") != "paused":
        raise ValueError("The paused server requires UFC_STARTUP_MODE=paused")
    manifest, manifest_sha256 = _pinned_manifest(env)
    from src.model.production_bundle import load_production_bundle, validate_production_bundle

    bundle = load_production_bundle(manifest)
    payload_root = manifest.parent
    for label, path in (
        ("primary model", bundle.model_path),
        ("no-odds model", bundle.no_odds_model_path),
        ("logistic model", bundle.logistic_model_path),
        ("processed snapshot", bundle.processed_dir),
    ):
        if path is None or not Path(path).resolve().is_relative_to(payload_root):
            raise ValueError(f"Paused {label} must be inside the pinned release payload")
    summary = validate_production_bundle(
        bundle,
        expected_models_dir=bundle.model_path.parent,
        expected_processed_dir=bundle.processed_dir,
    )
    selected_path = summary.get("selected_runtime_manifest_path")
    selected_hash = summary.get("selected_runtime_manifest_sha256")
    if not selected_path or not selected_hash or not summary.get("selected_release_id"):
        raise ValueError("Paused candidate requires a validated selected-input release binding")
    # production_bundle validates the release inventory and selector pin.
    # The prediction adapter below validates all nested selected-input bindings.
    selected_path = Path(selected_path)
    if not selected_path.resolve().is_relative_to(payload_root):
        raise ValueError("Selected runtime manifest escaped the release payload")
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != selected_hash:
        raise ValueError("Selected runtime manifest changed after validation")
    import joblib
    from src.data.runtime_selected_inputs import RuntimeSelectedInputs

    loaded = {}
    for role, path in (("primary", bundle.model_path), ("no_odds", bundle.no_odds_model_path),
                       ("logistic", bundle.logistic_model_path)):
        result = joblib.load(path)
        if not isinstance(result, dict) or "model" not in result:
            raise ValueError(f"The {role} artifact is not a loaded model bundle")
        loaded[role] = {**result, "artifact_path": str(Path(path).resolve())}
    selected = RuntimeSelectedInputs.from_summary(
        summary, model_result=loaded["primary"], no_odds_result=loaded["no_odds"]
    )
    if selected is None:
        raise ValueError("Paused candidate has no selected runtime binding")
    return {
        "status": "paused",
        "ready": True,
        "readiness_scope": "saved release inputs verified; no operational loops started",
        "reason": PAUSE_REASON,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": manifest_sha256,
        "bundle": summary,
        "selected_input_binding": selected.binding,
        "components": {name: {"state": "disabled", "reason": "release_paused"}
                       for name in DISABLED_COMPONENTS},
        "automatic_resume": False,
    }


def make_handler(status: dict):
    """Expose only saved status; no general dashboard routes are registered."""
    encoded = json.dumps(status, sort_keys=True, default=str).encode("utf-8")

    class PausedHandler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def _send(self, code, data, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                body = ("UFC bot release is paused.\n" + PAUSE_REASON + "\n"
                        "Saved release inputs are verified. Trading, monitoring and updates are disabled.\n").encode()
                self._send(200, body, "text/plain")
            elif self.path in {"/healthz", "/readyz", "/runtime-status"}:
                self._send(200 if status.get("ready") else 503, encoded)
            else:
                self._send(404, b'{"error":"route unavailable while release is paused"}')

        def _deny_mutation(self):
            self._send(405, b'{"error":"read-only paused release; activation is unavailable"}')

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = do_CONNECT = do_TRACE = _deny_mutation

        def log_message(self, fmt, *args):
            logger.info("Paused HTTP %s", fmt % args)

    return PausedHandler


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        status = validate_release()
        host = os.environ.get("WEB_HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "5050"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT is outside the valid range")
    except Exception:
        logger.exception("Paused release validation failed; server was not started")
        raise SystemExit(1)
    logger.info("Verified paused release: %s", json.dumps(status, sort_keys=True, default=str))
    with HTTPServer((host, port), make_handler(status)) as server:
        logger.info("Serving read-only paused release status on %s:%s", host, port)
        server.serve_forever()


if __name__ == "__main__":
    main()
