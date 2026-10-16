"""Original response preservation for explicitly bound ordinary data sources.

Capture establishes transport facts only. Independent caller inputs establish
fighter/page ownership and temporal eligibility in the source finalizers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from urllib.parse import urlparse

from src.data.method_observation_history import _digest, _json_bytes, _safe_path, _write_immutable


class SourceCaptureRefusal(ValueError):
    """A received response could not be preserved/processed; never retry source."""

    def __init__(self, message: str, *, captures=()):
        super().__init__(message)
        self.captures = tuple(captures)


def _utc_now():
    return datetime.now(timezone.utc)


def _transport_url(value):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise SourceCaptureRefusal("missing original transport URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise SourceCaptureRefusal("unsupported original transport URL")
    return value


class SourceResponseCapture:
    def __init__(self, source, *, history_root, capture_subdir, capture_log=None):
        if source not in {"bestfightodds", "ufc_rankings"}:
            raise SourceCaptureRefusal("unsupported observation source")
        if not isinstance(capture_subdir, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", capture_subdir):
            raise SourceCaptureRefusal("invalid capture directory")
        if capture_log is not None and not isinstance(capture_log, list):
            raise SourceCaptureRefusal("capture log must be caller-owned list")
        self.source = source
        self.root = _safe_path(Path(history_root)) / capture_subdir
        self.captures = []
        self.capture_log = capture_log

    def __call__(self, response, *, requested_url, format="html"):
        received_at = _utc_now().isoformat().replace("+00:00", "Z")
        try:
            body = getattr(response, "content", None)
            if not isinstance(body, bytes):
                raise SourceCaptureRefusal("original response bytes unavailable")
            body_path = self.root / "bodies" / f"{_digest(body)}.body"
            _write_immutable(body_path, body)
            status = getattr(response, "status_code", None)
            if type(status) is not int or not 100 <= status <= 599:
                raise SourceCaptureRefusal("original response status unavailable")
            requested = _transport_url(requested_url)
            issued = _transport_url(getattr(getattr(response, "request", None), "url", None))
            effective = _transport_url(getattr(response, "url", None))
            history = getattr(response, "history", None)
            if not isinstance(history, (list, tuple)):
                raise SourceCaptureRefusal("original redirect history unavailable")
            redirects = [_transport_url(getattr(hop, "url", None)) for hop in history]
            if redirects and redirects[-1] != effective:
                redirects.append(effective)
            if format != "html":
                raise SourceCaptureRefusal("unsupported original response format")
            envelope = {"schema": 1, "status_code": status, "requested_url": requested, "issued_url": issued,
                        "effective_url": effective, "redirect_chain": redirects,
                        "received_at": received_at, "format": format,
                        "response_bytes": len(body), "response_sha256": _digest(body)}
            envelope_body = _json_bytes(envelope)
            envelope_path = self.root / "envelopes" / f"{_digest(envelope_body)}.json"
            _write_immutable(envelope_path, envelope_body)
            descriptor = {"schema": 1, "source": self.source,
                          "raw_response": {"path": str(body_path), "bytes": len(body), "sha256": _digest(body)},
                          "response_envelope": {"path": str(envelope_path), "bytes": len(envelope_body), "sha256": _digest(envelope_body)}}
            descriptor_body = _json_bytes(descriptor)
            _write_immutable(self.root / "descriptors" / f"{_digest(descriptor_body)}.json", descriptor_body)
            self.captures.append(descriptor)
            if self.capture_log is not None:
                self.capture_log.append(descriptor)
            return descriptor
        except Exception as exc:
            raise SourceCaptureRefusal(f"original response capture refused: {exc}", captures=self.captures) from exc
