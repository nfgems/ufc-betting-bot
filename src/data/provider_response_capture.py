"""Preserve exact provider transport responses for the existing retained reader.

This boundary records evidence, including unsuccessful HTTP responses. It makes
no membership, identity, field-permission or acceptance decision. Callers must
feed its descriptor to the retained processors with independent reviewed inputs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from src.data.method_observation_history import (
    ObservationRefusal,
    _json_bytes,
    _safe_path,
    _write_immutable,
)
from src.data.provider_observation_history import ProviderObservationError, _provider_number

CAPTURE_SUBDIR = "provider_response_captures_v1"


class ProviderCaptureRefusal(ProviderObservationError):
    """Capture/processing is incomplete; it must not trigger another request."""

    def __init__(self, message: str, *, captures: tuple[dict, ...] = ()) -> None:
        super().__init__(message)
        self.captures = captures


class ProviderAcquisitionError(RuntimeError):
    """A source request failed; any completed captures remain original evidence."""

    def __init__(self, message: str, *, captures: tuple[dict, ...] = ()) -> None:
        super().__init__(message)
        self.captures = captures


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _url(value: object) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ProviderCaptureRefusal("missing original transport URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ProviderCaptureRefusal("unsupported original transport URL")
    return value


class ProviderResponseCapture:
    """A call-local sink; immutable files are written before soup/text parsing."""

    def __init__(self, source: str, *, history_root: Path, capture_log: list | None = None) -> None:
        if source not in {"sherdog", "tapology"}:
            raise ProviderCaptureRefusal("unsupported provider capture source")
        if capture_log is not None and not isinstance(capture_log, list):
            raise ProviderCaptureRefusal("capture log must be an explicit caller-owned list")
        self.source = source
        self.root = _safe_path(Path(history_root)) / CAPTURE_SUBDIR
        self.captures: list[dict] = []
        self.capture_log = capture_log
        self.last_status: int | None = None

    def __call__(self, response: object, *, requested_url: str, format: str) -> dict:
        # This time is observed immediately on return from the GET boundary;
        # neither a page date nor an archive/reader URL supplies receipt time.
        received_at = _utc_now().isoformat().replace("+00:00", "Z")
        try:
            body = getattr(response, "content", None)
            if not isinstance(body, bytes):
                raise ProviderCaptureRefusal("original response body bytes are unavailable")
            # Preserve available original bytes even if later context validation
            # or envelope publication fails. Such a partial is never a descriptor.
            body_path = self.root / "bodies" / f"{_digest(body)}.body"
            _write_immutable(body_path, body)
            status = getattr(response, "status_code", None)
            if type(status) is not int or not 100 <= status <= 599:
                raise ProviderCaptureRefusal("original HTTP status is unavailable")
            requested = _url(requested_url)
            effective = _url(getattr(response, "url", None))
            history = getattr(response, "history", None)
            if not isinstance(history, (list, tuple)):
                raise ProviderCaptureRefusal("original transport redirect history is unavailable")
            redirects = [_url(getattr(hop, "url", None)) for hop in history]
            # Existing schema-1 redirect semantics list observed redirect
            # responses followed by the actual final transport URL.
            if redirects and redirects[-1] != effective:
                redirects.append(effective)
            if format not in {"html", "tapology_reader"} or (format == "tapology_reader" and self.source != "tapology"):
                raise ProviderCaptureRefusal("unsupported original response format")
            envelope = {"schema": 1, "status_code": status,
                        "requested_url": requested, "effective_url": effective,
                        "redirect_chain": redirects, "received_at": received_at,
                        "response_sha256": _digest(body), "response_bytes": len(body),
                        "format": format}
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
            self.last_status = status
            return descriptor
        except ProviderCaptureRefusal:
            raise
        except (OSError, TypeError, ValueError, AttributeError, ObservationRefusal) as exc:
            raise ProviderCaptureRefusal(f"provider response capture refused: {exc}") from exc


def acquire_provider_response(
    source: str,
    url: str,
    *,
    history_root: Path,
    transport=None,
    response_format: str = "html",
    capture_log: list | None = None,
) -> dict:
    """Use the existing acquisition boundary and return its original descriptor.

``transport`` is an explicit GET callable for synthetic transport injection.
``capture_log`` receives every complete response, including earlier failed HTTP
attempts before eventual success, so caller checkpoints can retain those inputs.
Normal collection still uses its existing source throttles and status handling.
Browser DOM, Wayback soup and legacy text caches cannot provide this evidence.
No fallback to those routes is permitted while a capture sink is attached.
"""
    from src.data import fallback_scrapers as scrapers

    sink = ProviderResponseCapture(source, history_root=history_root, capture_log=capture_log)
    _url(url)
    if not _provider_number(source, url):
        raise ProviderCaptureRefusal("unsupported provider origin/fighter route before acquisition")
    if response_format not in {"html", "tapology_reader"} or (response_format == "tapology_reader" and source != "tapology"):
        raise ProviderCaptureRefusal("unsupported original response format")
    if transport is not None and not callable(transport):
        raise ProviderCaptureRefusal("provider transport must be an explicit GET callable")
    try:
        if response_format == "tapology_reader":
            scrapers._get_tapology_markdown_with_reader(url, response_capture=sink, request_get=transport)
        elif source == "sherdog":
            scrapers._get_sherdog_soup(url, response_capture=sink, request_get=transport)
        else:
            scrapers._get_tapology_soup(url, response_capture=sink, request_get=transport)
    except ProviderCaptureRefusal as exc:
        exc.captures = tuple(sink.captures)
        raise
    except (scrapers.requests.exceptions.RequestException, scrapers.SherdogRequestError, scrapers.TapologyRequestError) as exc:
        raise ProviderAcquisitionError(str(exc), captures=tuple(sink.captures)) from exc
    except Exception as exc:
        # A parser failure after a completed HTTP 200 is processing failure,
        # not authorization to charge another source request.
        if sink.captures:
            raise ProviderCaptureRefusal(f"provider response processing refused: {exc}", captures=tuple(sink.captures)) from exc
        raise ProviderAcquisitionError(str(exc), captures=tuple(sink.captures)) from exc
    if not sink.captures:
        raise ProviderCaptureRefusal("provider helper returned no complete original response capture")
    return sink.captures[-1]
