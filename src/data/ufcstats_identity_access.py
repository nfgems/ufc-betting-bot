"""Reviewed, bounded UFCStats acquisition for offline identity matching.

This module never decides fighter identity. Each acquired GET becomes an
embedded, hash-checked input for the existing captured-evidence matcher. There
is no browser, persistent cookie, retry, redirect or application import here.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit


HOST = "ufcstats.com"
POST_URL = "http://ufcstats.com/__c"
SUPPORTED_TEMPLATE_SHA256 = "73bc5d16d71bc3a30a4670dc73f8ec14e64135d6b48b1c7c1f90f792bd1cc850"
NONCE_RE = re.compile(rb'var nonce="([0-9a-f]{16})"')
NETWORK_GATE = {"enabled": False, "address": None}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class AccessError(RuntimeError):
    """A bounded access or evidence-control requirement failed."""


def _require(condition, reason):
    if not condition:
        raise AccessError(reason)


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _safe_path(path, *, exists=True):
    path = Path(path).absolute()
    _require(path.resolve() == path, "Path resolves through a link")
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            _require(not stat.S_ISLNK(info.st_mode) and not
                     getattr(info, "st_file_attributes", 0) & 1024, "Reparse or symbolic link")
    if exists:
        _require(path.exists(), "Required path is absent")
    return path


def file_pin(path):
    path = _safe_path(path)
    _require(path.is_file(), "Pin is not a regular file")
    body = path.read_bytes()
    return {"path": path.as_posix(), "bytes": len(body), "sha256": _sha(body)}


def _readonly(path):
    info = Path(path).stat()
    return bool(info.st_file_attributes & 1) if hasattr(info, "st_file_attributes") else not bool(info.st_mode & 0o222)


def _check_pin(item, *, readonly=False):
    _require(isinstance(item, dict) and _HASH.fullmatch(str(item.get("sha256", ""))), "Invalid pin")
    actual = file_pin(item["path"])
    _require(actual == {key: item[key] for key in ("path", "bytes", "sha256")}, "Pinned file drift")
    _require(not (readonly or item.get("read_only")) or _readonly(item["path"]), "Pinned evidence is writable")
    return Path(item["path"])


def _immutable(path, body):
    path = _safe_path(path, exists=False)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(stat.S_IREAD)
    return file_pin(path)


def _append_ledger(path, record):
    path = _safe_path(path, exists=False)
    with path.open("ab") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _document_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.netloc != HOST or parsed.port is not None
                or parsed.username or parsed.password or parsed.fragment):
            return False
    except (TypeError, ValueError):
        return False
    return bool((re.fullmatch(r"/(fighter|event)-details/[a-f0-9]{16}", parsed.path) and not parsed.query)
                or (parsed.path == "/statistics/fighters" and re.fullmatch(r"char=[a-z]&page=all", parsed.query)))


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _template(body):
    _require(len(NONCE_RE.findall(body)) == 1, "Unsupported challenge nonce")
    normalized = NONCE_RE.sub(b'var nonce="NORMALIZED"', body)
    _require(_sha(normalized) == SUPPORTED_TEMPLATE_SHA256, "Unsupported challenge template")
    return normalized


def verify_authority(plan_path, review_path):
    """Verify a sealed plan, exact code, evidence and carried ledgers using stdlib."""
    plan_path, review_path = _safe_path(plan_path), _safe_path(review_path)
    _require(_readonly(plan_path) and _readonly(review_path), "Plan and review must be sealed")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    _require(plan.get("schema_version") == 1 and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(plan.get("plan_id", ""))), "Invalid plan identity")
    _require(review.get("status") == "passed" and review.get("blockers") == [], "Independent review did not pass")
    _require(review.get("plan") == file_pin(plan_path), "Review does not bind exact plan")
    code = plan.get("code_pins")
    _require(isinstance(code, list) and code and review.get("code_pins") == code, "Review does not bind code")
    _require(any(item.get("path") == Path(__file__).absolute().as_posix() for item in code), "Access module is not pinned")
    for item in code:
        _check_pin(item)
    evidence = plan.get("evidence_pins", [])
    _require(isinstance(evidence, list), "Invalid evidence pins")
    for item in evidence:
        _check_pin(item, readonly=True)
    ledgers = plan.get("prior_ledgers")
    _require(isinstance(ledgers, list), "Missing carried ledgers")
    _require(len({item["path"] for item in ledgers}) == len(ledgers), "Repeated prior ledger")
    prior, rosterwatch = 0, 0
    for item in ledgers:
        path = _check_pin(item, readonly=True)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            _require(isinstance(row, dict), "Invalid prior ledger row")
            if row.get("event") == "request_started":
                prior += 1
                host = (urlsplit(row.get("url", "")).hostname or "").removeprefix("www.")
                _require(host in {HOST, "ufc.com", "roster.watch"}, "Unknown prior source host")
                rosterwatch += host == "roster.watch"
    _require(_integer(plan.get("prior_sends"), 0, 200) and plan["prior_sends"] == prior, "Carried send count mismatch")
    _require(_integer(plan.get("prior_rosterwatch_sends"), 0, 80) and plan["prior_rosterwatch_sends"] == rosterwatch, "Carried rosterwatch count mismatch")
    _require(plan.get("session_send_ceiling") == 200 and plan.get("rosterwatch_send_ceiling") == 80, "Source ceilings changed")
    requests = plan.get("requests")
    _require(isinstance(requests, list) and requests, "Empty document plan")
    _require(all(isinstance(row, dict) and row.get("method") == "GET" and _document_url(row.get("url")) for row in requests), "Request outside UFCStats document scope")
    _require(len({row["url"] for row in requests}) == len(requests), "Duplicate planned document")
    challenge = plan.get("challenge", {})
    _require(challenge.get("enabled") in (True, False) and type(challenge.get("enabled")) is bool, "Invalid challenge switch")
    _require(challenge.get("post_url") == POST_URL and challenge.get("max_rounds") == 1
             and challenge.get("max_difficulty") == 2 and challenge.get("max_iterations") == 100000, "Challenge bounds changed")
    template_path = _check_pin(challenge["template_pin"], readonly=True)
    _template(template_path.read_bytes())
    required = len(requests) + (2 if challenge["enabled"] else 0)
    _require(_integer(plan.get("send_ceiling"), required, 200) and prior + plan["send_ceiling"] <= 200, "Insufficient reserved request budget")
    interval = plan.get("min_interval_seconds")
    _require(type(interval) in (int, float) and 2.2 <= interval <= 60, "Invalid request spacing")
    _require(plan.get("timeout_seconds") == 20 and plan.get("max_response_bytes") == 8388608, "Transport bounds changed")
    ua = plan.get("user_agent")
    _require(isinstance(ua, str) and 1 <= len(ua) <= 256 and ua.isascii()
             and all(32 <= ord(c) < 127 for c in ua) and "mozilla/" not in ua.casefold(), "Invalid identifying user agent")
    root = _safe_path(plan["evidence_root"], exists=False)
    _require(root.parent.is_dir(), "Evidence parent must already exist")
    for item in code + evidence + ledgers + [challenge["template_pin"], file_pin(plan_path), file_pin(review_path)]:
        _require(root not in Path(item["path"]).absolute().parents, "Input evidence lies inside output root")
    _require(not any(os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")), "Proxy inheritance forbidden")
    return plan


def is_challenge(body):
    return any(token in body[:10000].lower() for token in (
        b"checking your browser", b"requires javascript", b"verify you are human", b"captcha",
        b"challenge-platform", b"cf-chl-", b"just a moment", b"access denied"))


def _solve(body, template):
    _require(_template(body) == _template(template), "Unsupported challenge template")
    nonce = NONCE_RE.findall(body)[0].decode()
    difficulty = re.search(rb"target=new Array\((\d+)\+1\)\.join\('0'\)", body)
    _require(difficulty is not None and 1 <= int(difficulty[1]) <= 2, "Unsupported challenge difficulty")
    for number in range(100000):
        if _sha((nonce + ":" + str(number)).encode()).startswith("0" * int(difficulty[1])):
            return urlencode({"nonce": nonce, "n": number}).encode()
    raise AccessError("Challenge iteration ceiling")


def _cookie(headers):
    values = [v for k, v in headers if k.casefold() == "set-cookie"]
    _require(len(values) == 1, "Expected one fresh challenge cookie")
    parts = [part.strip() for part in values[0].split(";")]
    _require(re.fullmatch(r"_fmc=[A-Za-z0-9._~-]{1,4096}", parts[0]), "Invalid challenge cookie")
    seen = set()
    for attribute in parts[1:]:
        key, _, value = attribute.partition("=")
        key = key.casefold()
        _require(key not in seen, "Duplicate cookie attribute")
        seen.add(key)
        if key == "domain":
            _require(value.casefold().lstrip(".") == HOST, "Foreign cookie domain")
        if key == "path":
            _require(value == "/", "Unexpected cookie path")
        _require(key != "secure", "Secure cookie cannot be sent over HTTP")
    return parts[0]


def header_safety(headers, size):
    lengths = [v for k, v in headers if k.casefold() == "content-length"]
    transfer = [v.casefold() for k, v in headers if k.casefold() == "transfer-encoding"]
    coding = [v.casefold() for k, v in headers if k.casefold() == "content-encoding"]
    _require(not coding or coding == ["identity"], "Unexpected content coding")
    _require(not (lengths and transfer), "Ambiguous response framing")
    _require(not transfer or transfer == ["chunked"], "Unsupported transfer coding")
    _require(len(lengths) <= 1, "Duplicate Content-Length")
    if lengths:
        _require(re.fullmatch(r"[0-9]{1,10}", lengths[0]) and int(lengths[0]) == size, "Incomplete Content-Length")


def _one_connection(address, timeout, source_address=None):
    _require(address == (HOST, 80) and source_address is None, "Unexpected connection target")
    addresses = socket.getaddrinfo(HOST, 80, socket.AF_INET, socket.SOCK_STREAM)
    _require(bool(addresses), "No IPv4 address")
    family, kind, protocol, _, target = addresses[0]
    NETWORK_GATE["address"] = target
    sock = socket.socket(family, kind, protocol)
    try:
        sock.settimeout(timeout)
        sock.connect(target)
        return sock
    except BaseException:
        sock.close()
        raise


def real_wire(method, url, headers, form, plan):
    """One connection/send; preserve any partial entity and never follow/retry."""
    _require(not NETWORK_GATE["enabled"], "Nested network dispatch")
    _require((method == "GET" and _document_url(url)) or (method == "POST" and url == POST_URL), "Unexpected wire URL")
    parsed = urlsplit(url)
    conn = http.client.HTTPConnection(HOST, 80, timeout=plan["timeout_seconds"])
    conn._create_connection = _one_connection
    body = bytearray()
    result = {"status": None, "headers": [], "complete": False, "error": None}
    NETWORK_GATE.update(enabled=True, address=None)
    try:
        conn.request(method, parsed.path + (("?" + parsed.query) if parsed.query else ""), body=form, headers=headers)
        response = conn.getresponse()
        result.update(status=response.status, headers=response.getheaders())
        while True:
            chunk = response.read(min(65536, plan["max_response_bytes"] + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            _require(len(body) <= plan["max_response_bytes"], "Response byte ceiling")
        header_safety(result["headers"], len(body))
        result["complete"] = True
    except http.client.IncompleteRead as exc:
        body.extend(exc.partial)
        result["error"] = "IncompleteRead"
    except Exception as exc:
        result["error"] = type(exc).__name__
    finally:
        conn.close()
        NETWORK_GATE.update(enabled=False, address=None)
    result["body"] = bytes(body)
    return result


def _document_supported(url, body, headers):
    media = [v.split(";", 1)[0].strip().casefold() for k, v in headers if k.casefold() == "content-type"]
    if media != ["text/html"] or is_challenge(body):
        return False
    lower = body.lower()
    if "/fighter-details/" in url:
        return b"b-content__title-highlight" in lower and b"b-list__box-list" in lower
    if "/event-details/" in url:
        return b"b-fight-details__table" in lower and b"b-list__box-list" in lower
    return b"b-statistics__table" in lower and b"fighter-details/" in lower


def acquire_reviewed(plan_path, review_path, requested_urls, *, wire=None, verifier=None,
                     clock=time.monotonic, sleeper=time.sleep):
    """Consume one sealed acquisition plan and return immutable replay evidence.

    ``requested_urls`` must come from the unchanged matcher. A production caller
    also supplies its protected-state verifier. Wire injection is for isolated
    tests only and is deliberately absent from the command-line interface.
    """
    plan = verify_authority(plan_path, review_path)
    _require(wire is not None or callable(verifier), "Production protected verifier required")
    _require(verifier is None or callable(verifier), "Invalid protected verifier")
    urls = [row["url"] for row in plan["requests"]]
    _require(set(urls) <= set(requested_urls), "Planned document is not an evidenced matcher request")
    verifier = verifier or (lambda: None)
    verifier()
    plan_pin, review_pin = file_pin(plan_path), file_pin(review_path)
    root = _safe_path(plan["evidence_root"], exists=False)
    _require(not root.exists(), "Plan evidence root already consumed")
    root.mkdir()  # Exclusive first mutation: even a subsequent failure consumes this plan.
    _immutable(root / "consumed.json", _json_bytes({"plan": plan_pin, "review": review_pin,
               "plan_id": plan["plan_id"], "consumed_at": _utc(), "rerun_permitted": False}))
    ledger = root / "request-ledger.jsonl"
    wire = real_wire if wire is None else wire
    template = Path(plan["challenge"]["template_pin"]["path"]).read_bytes()
    records, captures = [], []
    count, wire_sends, rounds, last_start, cookie = 0, 0, 0, None, None

    def send(method, url, form=None):
        nonlocal count, wire_sends, last_start
        _require((method == "GET" and url in urls and form is None) or
                 (method == "POST" and url == POST_URL and rounds == 1 and form is not None), "Unplanned send")
        _require(count < plan["send_ceiling"] and plan["prior_sends"] + count < 200, "Request ceiling reached")
        _require(file_pin(plan_path) == plan_pin and file_pin(review_path) == review_pin, "Plan or review changed")
        _require(verify_authority(plan_path, review_path) == plan, "Authority changed")
        verifier()
        if last_start is not None:
            delay = plan["min_interval_seconds"] - (clock() - last_start)
            if delay > 0:
                sleeper(delay)
        headers = {"User-Agent": plan["user_agent"], "Accept": "text/html", "Accept-Encoding": "identity", "Connection": "close"}
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if cookie:
            headers["Cookie"] = cookie
        ordinal = count + 1
        started = {"event": "request_started", "plan_id": plan["plan_id"], "ordinal": ordinal,
                   "cumulative_ordinal": plan["prior_sends"] + ordinal, "prior_sends": plan["prior_sends"],
                   "method": method, "url": url, "host": HOST, "at_utc": _utc()}
        # Conservatively charge the intent even if the append/fsync fails after
        # partially reaching storage. A failed append can never reach the wire.
        count = ordinal
        _append_ledger(ledger, started)
        last_start = clock()
        try:
            wire_sends += 1
            result = wire(method, url, headers, form, plan)
        except Exception as exc:
            result = {"status": None, "headers": [], "body": b"", "complete": False, "error": type(exc).__name__}
        body = bytes(result["body"])
        raw = _immutable(root / f"{ordinal:03d}-body.bin", body)
        _require(file_pin(raw["path"]) == raw, "Persisted body changed")
        error = result.get("error")
        complete = result.get("complete") is True
        try:
            header_safety(result["headers"], len(body))
            _require(len(body) <= plan["max_response_bytes"], "Response byte ceiling")
        except AccessError as exc:
            complete, error = False, str(exc)
        try:
            body.decode("utf-8", errors="strict")
            decoded = True
        except UnicodeDecodeError:
            decoded, error = False, "Invalid UTF-8"
        eligible = bool(method == "GET" and complete and decoded and not error and result["status"] == 200
                        and _document_supported(url, body, result["headers"]))
        captured_at = _utc()
        record = {**started, "event": "request_completed", "status": result["status"], "error": error,
                  "complete": complete, "raw": raw, "completed_at_utc": captured_at,
                  "request_headers": {k: "_fmc=<redacted>" if k == "Cookie" else v for k, v in headers.items()},
                  "response_headers": [[k, {"redacted": True, "sha256": _sha(v.encode())} if k.casefold() == "set-cookie" else v] for k, v in result["headers"]],
                  "form_sha256": _sha(form) if form else None,
                  "challenge_detected": is_challenge(body), "semantic_evidence_eligible": eligible}
        _immutable(root / f"{ordinal:03d}-response.json", _json_bytes(record))
        records.append(record)
        if method == "GET":
            captures.append({"url": url, "status_code": result["status"], "body_base64": base64.b64encode(body).decode("ascii"),
                             "decoded_sha256": raw["sha256"], "decoded_size": len(body), "captured_at": captured_at,
                             "capture_complete": complete, "decode_complete": decoded,
                             "ufcstats_challenge_detected": is_challenge(body), "semantic_evidence_eligible": eligible,
                             "error": error})
        _append_ledger(ledger, {key: record[key] for key in ("event", "plan_id", "ordinal", "method", "url", "status", "error", "complete", "semantic_evidence_eligible")})
        _require(complete and not error, "Incomplete or failed response")
        return result

    status, error = "failed", None
    try:
        for url in urls:
            result = send("GET", url)
            _require(result["status"] == 200, "Unexpected document status; redirects are not followed")
            _require([v.split(";", 1)[0].strip().casefold() for k, v in result["headers"]
                      if k.casefold() == "content-type"] == ["text/html"], "Unexpected document media")
            if is_challenge(result["body"]):
                _require(plan["challenge"]["enabled"] and rounds == 0, "Challenge disabled or recurring")
                form = _solve(result["body"], template)
                rounds += 1
                posted = send("POST", POST_URL, form)
                _require(type(posted["status"]) is int and 200 <= posted["status"] < 300, "Challenge POST rejected")
                cookie = _cookie(posted["headers"])
                result = send("GET", url)
                _require(result["status"] == 200 and not is_challenge(result["body"]), "Challenge reload unsuccessful")
            _require(_document_supported(url, result["body"], result["headers"]), "Expected UFCStats document markers or media missing")
        verify_authority(plan_path, review_path)
        verifier()
        status = "passed"
    except Exception as exc:
        error = str(exc) if isinstance(exc, AccessError) else type(exc).__name__
    finally:
        cookie = None
        if ledger.exists():
            ledger.chmod(stat.S_IREAD)
        terminal = {"schema_version": 1, "plan_id": plan["plan_id"], "status": status, "error": error,
                    "plan": plan_pin, "review": review_pin, "new_sends": count, "wire_sends": wire_sends,
                    "count_basis": "conservatively charged pre-send intents", "prior_sends": plan["prior_sends"],
                    "cumulative_sends": plan["prior_sends"] + count, "challenge_rounds": rounds,
                    "records": records, "captures": captures, "ledger": file_pin(ledger) if ledger.exists() else None,
                    "redirects_followed": 0, "retries": 0, "publication_requested": False, "rerun_permitted": False}
        terminal_pin = _immutable(root / "terminal.json", _json_bytes(terminal))
    return {**terminal, "terminal": terminal_pin}
