"""Bounded HTTP transport preparation; no installed live factory or activation.

The callable exchange accepts an explicit connection factory for isolated tests.
The real single-address connection primitive is deliberately disconnected from
the broker and CLI until a reviewed parent socket boundary is installed. Passing
an injected factory is not evidence of containment or permission to use sources.
"""
from __future__ import annotations

import http.client
import ctypes
import errno
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import ssl
import stat
import sys
import threading
import time
from urllib.parse import urlsplit

from src.data.ufcstats_identity_access import POST_URL, _document_url, header_safety

MEMBERSHIP_URL = "https://www.roster.watch/"
MAX_BODY = 8 * 1024 * 1024
# HTTP framing shares the existing per-response byte ceiling. Raw overhead can
# therefore stop a response before its decoded-body ceiling; never discard it.
MAX_RAW_RESPONSE = MAX_BODY + 1
_PINNED_LIBRARIES = {}


class WireError(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise WireError(reason)


def _target(method, url):
    require(isinstance(url, str) and url.isascii() and not any(ord(c) <= 32 or ord(c) == 127 for c in url),
            "Invalid request target")
    parsed = urlsplit(url)
    require(not parsed.username and not parsed.password and not parsed.fragment,
            "Userinfo or fragment forbidden")
    require((method == "GET" and (url == MEMBERSHIP_URL or _document_url(url)))
            or (method == "POST" and url == POST_URL), "Unsupported canonical target")
    require(parsed.netloc == ("www.roster.watch" if url == MEMBERSHIP_URL else "ufcstats.com"),
            "Noncanonical hostname or port")
    return parsed.hostname, 443 if parsed.scheme == "https" else 80, parsed.scheme == "https", parsed.path + ("?" + parsed.query if parsed.query else "")


def _request_headers(method, url, headers, form):
    require(isinstance(headers, dict), "Explicit headers required")
    base = {"User-Agent", "Accept", "Accept-Encoding", "Connection"}
    require(set(headers) <= base | {"Content-Type", "Cookie"} and base <= set(headers),
            "Request header outside broker contract")
    require(all(isinstance(k, str) and isinstance(v, str) and v.isascii()
                and 0 < len(v) <= 8192 and all(32 <= ord(c) < 127 for c in v)
                for k, v in headers.items()), "Unsafe request header")
    require(1 <= len(headers["User-Agent"]) <= 256 and "mozilla/" not in headers["User-Agent"].lower(),
            "Identifying user agent required")
    require(headers["Accept"] == "text/html" and headers["Accept-Encoding"] == "identity"
            and headers["Connection"] == "close", "Header contract drift")
    if method == "POST":
        require(url == POST_URL and isinstance(form, bytes)
                and re.fullmatch(rb"nonce=[a-f0-9]{16}&n=[0-9]{1,5}", form)
                and headers.get("Content-Type") == "application/x-www-form-urlencoded"
                and "Cookie" not in headers, "Unexpected verification POST")
    else:
        require(form is None and "Content-Type" not in headers, "GET body forbidden")
    if "Cookie" in headers:
        require(url != MEMBERSHIP_URL and method == "GET"
                and re.fullmatch(r"_fmc=[A-Za-z0-9_-]{1,256}", headers["Cookie"]), "Unexpected cookie")


def _safe_response_headers(headers):
    return isinstance(headers, (list, tuple)) and len(headers) <= 100 and all(
        isinstance(row, (tuple, list)) and len(row) == 2 and all(
            isinstance(value, str) and len(value) <= 8192 and "\r" not in value and "\n" not in value
            for value in row) for row in headers)


class _Deadline:
    def __init__(self, seconds, monotonic=time.monotonic):
        self.monotonic = monotonic
        self.started = monotonic()
        self.ends = self.started + seconds

    def remaining(self):
        remaining = self.ends - self.monotonic()
        if remaining <= 0:
            raise TimeoutError("Exchange deadline")
        return remaining


def _bounded_dns(host, port, deadline, resolver=None):
    """Terminate and reap a stalled libc resolver; no surviving timeout thread.

    This disconnected primitive requires a single-threaded Linux parent. The
    future network helper still needs its own reviewed DNS/egress policy. Tests
    inject saved resolver answers and never contact a name server.
    """
    require(sys.platform == "linux" and len(list(Path("/proc/self/task").iterdir())) == 1,
            "DNS helper requires a single-threaded Linux process")
    deadline.remaining()
    read_fd, write_fd = os.pipe()
    child = None
    try:
        child = os.fork()
        if child == 0:
            try:
                os.close(read_fd)
                # Resolver children inherit no store/input/other IPC handles.
                for entry in Path("/proc/self/fd").iterdir():
                    fd = int(entry.name)
                    if fd != write_fd:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                answers = (resolver or socket.getaddrinfo)(host, port, socket.AF_INET, socket.SOCK_STREAM)
                payload = json.dumps({"answers": answers}, separators=(",", ":")).encode()
                if len(payload) > 16384:
                    payload = b'{"error":"DNS answer byte ceiling"}'
                offset = 0
                while offset < len(payload):
                    offset += os.write(write_fd, payload[offset:])
            except BaseException:
                try:
                    os.write(write_fd, b'{"error":"DNS resolution failed"}')
                except OSError:
                    pass
            finally:
                os._exit(0)
        os.close(write_fd)
        write_fd = None
        payload = bytearray()
        while True:
            readable, _, _ = select.select([read_fd], [], [], deadline.remaining())
            if not readable:
                raise TimeoutError("DNS exchange deadline")
            chunk = os.read(read_fd, 16385 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
            require(len(payload) <= 16384, "DNS answer byte ceiling")
        deadline.remaining()
        result = json.loads(payload)
        require(set(result) == {"answers"}, "DNS resolution failed")
        return result["answers"]
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if child:
            # Always reap, including malformed output and expired deadlines.
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child, 0)


def _public_addresses(host, port, resolver=None, *, deadline=None):
    """Validate the complete answer set, then pick one address without fallback."""
    answers = (resolver(host, port, socket.AF_INET, socket.SOCK_STREAM) if resolver is not None
               else _bounded_dns(host, port, deadline or _Deadline(20)))
    require(isinstance(answers, list) and 0 < len(answers) <= 32, "Bounded IPv4 DNS answer required")
    addresses = []
    for family, kind, protocol, canonical, address in answers:
        require(family == socket.AF_INET and kind == socket.SOCK_STREAM
                and protocol in (0, socket.IPPROTO_TCP) and len(address) == 2
                and address[1] == port, "Unexpected resolved address")
        ip = ipaddress.ip_address(address[0])
        require(ip.version == 4 and ip.is_global and not ip.is_multicast
                and not ip.is_reserved and not ip.is_unspecified, "Nonpublic source address")
        addresses.append((str(ip), port))
    return sorted(set(addresses))


def _sealed_bytes(pin, *, maximum):
    """Load the exact sealed bytes which will actually be used, once."""
    require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}, "Explicit file pin required")
    path = Path(pin["path"])
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(), "Canonical pinned file required")
    require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= maximum, "Pinned file size limit")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o222,
                "Sealed regular unshared file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(maximum + 1)
        require(len(payload) == pin["bytes"] and hashlib.sha256(payload).hexdigest() == pin["sha256"],
                "Pinned file drift")
        return payload
    finally:
        os.close(fd)


def pinned_tls_context(ca_bundle_pin):
    """Only the exact supplied PEM roots; no environment/system trust fallback."""
    pem = _sealed_bytes(ca_bundle_pin, maximum=1024 * 1024)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(cadata=pem.decode("ascii", errors="strict"))
    require(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            and context.cert_store_stats()["x509_ca"] > 0, "Pinned CA verification required")
    return context


def _pinned_library(pin):
    """dlopen the verified snapshot, immune to pathname or inode replacement."""
    import fcntl
    require(sys.platform == "linux" and os.uname().machine == "x86_64", "Linux x86_64 library sealing required")
    payload = _sealed_bytes(pin, maximum=8 * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    if digest in _PINNED_LIBRARIES:
        return _PINNED_LIBRARIES[digest][0]
    require(len(_PINNED_LIBRARIES) < 2, "Only two pinned control libraries allowed")
    fd = os.memfd_create("roster-pinned-control", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        # Linux fcntl UAPI values. The pinned portable Python was compiled
        # without these names, although this kernel implements and verifies them.
        seals = 0x0008 | 0x0004 | 0x0002 | 0x0001  # WRITE | GROW | SHRINK | SEAL
        fcntl.fcntl(fd, 1033, seals)  # F_ADD_SEALS
        require(fcntl.fcntl(fd, 1034) == seals, "Pinned library memory seals failed")  # F_GET_SEALS
        library = ctypes.CDLL("/proc/self/fd/" + str(fd), use_errno=True)
        # Retain both descriptors for process lifetime. dlopen caches its input
        # pathname, so reusing an fd number could otherwise select an old DSO.
        _PINNED_LIBRARIES[digest] = (library, fd)
        return library
    except BaseException:
        os.close(fd)
        raise


def install_parent_network_deny(*, seccomp_library_pin, libc_library_pin):
    """Irreversibly deny networking in the current *disabled* controller.

    Pipes remain usable; sockets and io_uring must be absent before installing.
    This is not an outbound allowlist or a live helper installation. A native
    network helper would have to be separately contained before this boundary.
    """
    require(sys.platform == "linux" and os.uname().machine == "x86_64",
            "Linux x86_64 parent boundary required")
    require(len(list(Path("/proc/self/task").iterdir())) == 1,
            "Parent boundary requires a single-threaded process")
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:  # the enumerating directory handle is closed
            continue
        require(not target.startswith("socket:") and "io_uring" not in target,
                "Inherited socket or io_uring descriptor forbidden")
    libc = _pinned_library(libc_library_pin)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    seccomp = _pinned_library(seccomp_library_pin)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
    seccomp.seccomp_attr_set.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    denied = ("socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg "
              "recvfrom recvmsg recvmmsg shutdown getsockopt setsockopt "
              "fork vfork clone clone3 execve execveat ptrace process_vm_readv process_vm_writev pidfd_getfd "
              "unshare setns mount umount2 pivot_root chroot open_by_handle_at name_to_handle_at "
              "bpf io_uring_setup io_uring_enter io_uring_register userfaultfd").split()
    context = seccomp.seccomp_init(0x7fff0000)
    require(bool(context), "Parent seccomp allocation failed")
    try:
        for name in denied:
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            require(number >= 0, "Parent syscall unavailable in pinned library: " + name)
            require(seccomp.seccomp_rule_add(context, 0x50000 | errno.EPERM, number, 0) == 0,
                    "Parent seccomp rule failed: " + name)
        require(seccomp.seccomp_attr_set(context, 4, 1) == 0, "Parent seccomp TSYNC unavailable")
        require(libc.prctl(38, 1, 0, 0, 0) == 0, "Parent no_new_privs failed")
        require(seccomp.seccomp_load(context) == 0, "Parent seccomp enforcement failed")
    finally:
        seccomp.seccomp_release(context)
    return {"schema_version": 1, "network": "denied", "process_creation": "denied",
            "inherited_sockets": "rejected", "no_new_privs": True, "seccomp_tsync": True,
            "library_load": "verified_snapshot_in_sealed_memfd",
            "denied_syscalls": denied, "seccomp_library": dict(seccomp_library_pin),
            "libc_library": dict(libc_library_pin), "live_network_helper_installed": False}


class _RecordingReader(io.RawIOBase):
    """Capture every decrypted HTTP byte before a parser can buffer/discard it."""
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def readable(self):
        return True

    def readinto(self, buffer):
        owner = self.owner
        room = owner.raw_limit - len(owner.raw_response)
        require(room > 0, "Raw response byte ceiling")
        owner.sock.settimeout(owner.deadline.remaining())
        received = owner.sock.recv_into(memoryview(buffer)[:room])
        owner.raw_response.extend(memoryview(buffer)[:received])
        owner.deadline.remaining()
        return received


class _ResponseSocket:
    def __init__(self, owner):
        self.owner = owner

    def makefile(self, mode):
        require(mode == "rb", "Unexpected response reader")
        return io.BufferedReader(_RecordingReader(self.owner), buffer_size=8192)


class _PinnedConnection(http.client.HTTPConnection):
    """One resolved IPv4 socket with TLS SNI/verification for roster.watch.

    Not selected by any current launcher. DNS and connection happen only inside
    request(), after the exchange's precharge verifier. No proxies or retry.
    """
    def __init__(self, host, port, secure, timeout, *, ca_bundle_pin=None):
        super().__init__(host, port, timeout=timeout)
        self.secure = secure
        self.ca_bundle_pin = ca_bundle_pin
        self.peer = None
        self.dns_addresses = []
        self.deadline = _Deadline(timeout)
        self.raw_response = bytearray()
        self.raw_limit = MAX_RAW_RESPONSE
        self._recording_response = None

    def bind_deadline(self, deadline, body_limit):
        require(self.sock is None and self.peer is None, "Cannot rebind a started native exchange")
        self.deadline = deadline
        self.raw_limit = min(MAX_RAW_RESPONSE, body_limit + 1024 * 1024 + 1)

    def connect(self):
        require(self.sock is None and self.peer is None, "Connection is single use")
        require((self.host, self.port, self.secure) in {
            ("ufcstats.com", 80, False), ("www.roster.watch", 443, True)}, "Unexpected socket target")
        # Validate exact roots before any DNS/connect. SSL_CERT_FILE, SSL_CERT_DIR
        # and OpenSSL defaults cannot silently choose a different trust store.
        context = pinned_tls_context(self.ca_bundle_pin) if self.secure else None
        self.deadline.remaining()
        self.dns_addresses = _public_addresses(self.host, self.port, deadline=self.deadline)
        self.peer = self.dns_addresses[0]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        try:
            sock.settimeout(self.deadline.remaining())
            sock.connect(self.peer)
            if self.secure:
                sock.settimeout(self.deadline.remaining())
                sock = context.wrap_socket(sock, server_hostname=self.host)
            self.deadline.remaining()
            self.sock = sock
        except BaseException:
            sock.close()
            raise

    def send(self, data):
        if self.sock is None:
            self.connect()
        self.sock.settimeout(self.deadline.remaining())
        super().send(data)
        self.deadline.remaining()

    def getresponse(self):
        # HTTPConnection's ordinary buffered reader can lose bytes when status,
        # headers, chunks or body reads raise. Record below that entire parser.
        require(self.sock is not None and self._recording_response is None,
                "Request must precede one response")
        self._recording_response = http.client.HTTPResponse(_ResponseSocket(self))
        self._recording_response.begin()
        return self._recording_response

    def close(self):
        if self._recording_response is not None:
            self._recording_response.close()
        super().close()


class BoundedSourceExchange:
    """Parent-selected request, precharge check, one exchange, complete evidence.

    A missing factory is a hard disabled gate. A supplied test factory must not
    be relabeled as live. This class supplies no source authorization, accounting
    store lock, worker socket permission, recurring schedule or publication.
    """
    def __init__(self, *, precharge_verifier, connection_factory=None,
                 timeout_seconds=20, max_response_bytes=MAX_BODY, monotonic=time.monotonic):
        require(callable(precharge_verifier), "Precharge verifier required")
        require(connection_factory is None or callable(connection_factory), "Invalid factory")
        require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 30, "Finite timeout required")
        require(type(max_response_bytes) is int and 0 < max_response_bytes <= MAX_BODY, "Bounded response required")
        self.precharge_verifier = precharge_verifier
        self.factory = connection_factory
        self.timeout = timeout_seconds
        self.limit = max_response_bytes
        self.monotonic = monotonic
        self.lock = threading.Lock()

    def __call__(self, method, url, headers, form, plan):
        require(self.factory is not None, "Live parent socket boundary is not installed; exchange disabled")
        host, port, secure, path = _target(method, url)
        _request_headers(method, url, headers, form)
        require(isinstance(plan, dict) and plan.get("execution_mode") == "synthetic"
                and plan.get("publication_requested") is False, "Only injected synthetic validation is enabled")
        allowed = ([MEMBERSHIP_URL] if plan.get("membership") is True else []) + plan.get("documents", [])
        require((method == "GET" and url in allowed) or
                (method == "POST" and plan.get("challenge", {}).get("enabled") is True), "Request outside exact plan")
        require(self.lock.acquire(blocking=False), "Concurrent exchange forbidden")
        result = {"status": None, "headers": [], "complete": False, "error": None,
                  "wire_metadata": {"transport": "injected_http_validation", "synthetic": True,
                                    "redirects_followed": 0, "retries": 0}}
        body = bytearray()
        connection = None
        deadline = _Deadline(self.timeout, self.monotonic)
        try:
            # Verifier must inspect the durable pending intent for this exact
            # plan/method/URL; it is called before factory/DNS/connect/send.
            require(self.precharge_verifier(method, url, plan) is True, "Exact durable precharge not verified")
            deadline.remaining()
            connection = self.factory(host, port, secure, self.timeout)
            deadline.remaining()
            if isinstance(connection, _PinnedConnection):
                connection.bind_deadline(deadline, self.limit)
            elif getattr(connection, "sock", None) is not None:
                connection.sock.settimeout(deadline.remaining())
            connection.request(method, path, body=form, headers=dict(headers))
            deadline.remaining()
            if getattr(connection, "sock", None) is not None:
                connection.sock.settimeout(deadline.remaining())
            response = connection.getresponse()
            result.update(status=response.status, headers=response.getheaders())
            require(type(result["status"]) is int and 100 <= result["status"] <= 599, "Invalid response status")
            require(_safe_response_headers(result["headers"]), "Response metadata limit")
            while True:
                deadline.remaining()
                if getattr(connection, "sock", None) is not None:
                    connection.sock.settimeout(deadline.remaining())
                remaining = self.limit + 1 - len(body)
                require(remaining > 0, "Response byte ceiling")
                chunk = response.read(min(65536, remaining))
                require(isinstance(chunk, bytes) and len(chunk) <= min(65536,remaining), "Read exceeded bound")
                if not chunk:
                    break
                body.extend(chunk)
                require(len(body) <= self.limit, "Response byte ceiling")
            deadline.remaining()
            header_safety(result["headers"], len(body))
            result["complete"] = True
        except http.client.IncompleteRead as exc:
            room = self.limit + 1 - len(body)
            body.extend(exc.partial[:room])
            result["error"] = "IncompleteRead"
            result["wire_metadata"]["partial_exception_bytes"] = len(exc.partial)
            result["wire_metadata"]["partial_exception_truncated"] = len(exc.partial) > room
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:512]
        finally:
            if connection is not None:
                result["wire_metadata"].update(peer=getattr(connection,"peer",None),
                                                dns_addresses=getattr(connection,"dns_addresses",[]))
                if isinstance(connection, _PinnedConnection):
                    raw = bytes(connection.raw_response)
                    result["raw_response"] = raw
                    result["wire_metadata"].update(
                        raw_response_sha256=hashlib.sha256(raw).hexdigest(),
                        raw_response_bytes=len(raw), raw_response_scope="decrypted_http_response",
                        raw_response_limit=connection.raw_limit,
                        raw_response_limit_reached=len(raw) == connection.raw_limit,
                        ca_bundle_pin=connection.ca_bundle_pin if secure else None,
                        deadline_scope="dns_connect_tls_send_status_headers_body")
                    if not _safe_response_headers(result["headers"]):
                        # The malformed original is preserved exactly in the raw
                        # blob; keep the structured envelope within its old cap.
                        result["headers"] = []
                        result["wire_metadata"]["parsed_headers_rejected"] = True
                try:
                    connection.close()
                except Exception as exc:
                    result["complete"] = False
                    closing = "Close failed: " + type(exc).__name__
                    result["wire_metadata"]["close_error"] = closing
                    result["error"] = (result["error"] + "; " if result["error"] else "") + closing
            self.lock.release()
        result["body"] = bytes(body)
        result["wire_metadata"]["overflow_witness"] = len(body) > self.limit
        result["wire_metadata"]["elapsed_seconds"] = self.monotonic() - deadline.started
        return result


def activation_readiness():
    """Machine-readable remaining controls; no bool/env flag activates transport."""
    return {"schema_version": 1, "status": "disabled_preparation", "source_network_enabled": False,
            "publication_enabled": False, "schedule_enabled": False,
            "live_factory_installed": False, "live_broker_adapter_installed": False,
            "remaining_gates": ["reviewed_parent_socket_boundary", "schema2_semantic_need_recomputation",
                                "exact_native_plan_store_image_binding", "finite_cumulative_budget_and_precharge",
                                "hosted_containment_and_recovery", "independent_complete_volume_backup",
                                "exclusive_legacy_loop_ownership", "separate_owner_live_trial_authorization"]}
