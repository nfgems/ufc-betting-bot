"""Explicit saved-response helper over preopened pipes; live egress stays disabled.

The trusted native launcher owns process/UID setup and supplies both endpoints
before installing the existing parent network/process guard. Neither endpoint
creates processes, sockets, or network requests. This is local helper plumbing,
not an installed service, live transport, or a claim of hosted containment.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import stat
import struct
import threading
import time

from src.data.roster_hosted_store import (
    StoreError, _json, _new_file, _safe, assess_source_freshness,
)
from src.data.roster_hosted_wire import (
    MAX_BODY, MAX_RAW_RESPONSE, MEMBERSHIP_URL, _request_headers,
    _safe_response_headers, _sealed_bytes, _target,
)

MAX_REQUEST = 32768
# Base64 body and raw HTTP witness plus bounded metadata; below 128 MiB/file.
MAX_RESPONSE = 24 * 1024 * 1024
MAX_FIXTURE = 128 * 1024 * 1024
MAX_ENTRIES = 10000
MAX_OUTPUT = 1024 * 1024 * 1024
RESERVE = 512 * 1024 * 1024


class HelperError(StoreError):
    pass


def _require(ok, reason):
    if not ok:
        raise HelperError(reason)


def _stamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(isinstance(value, datetime) and value.tzinfo is not None,
             "Aware helper timestamp required")
    return value.astimezone(timezone.utc)


def _pipe(fd):
    _require(type(fd) is int and fd >= 0 and stat.S_ISFIFO(os.fstat(fd).st_mode),
             "Preopened anonymous pipe required")
    _require(os.readlink('/proc/self/fd/' + str(fd)).startswith('pipe:['),
             "Named or substituted IPC pipe forbidden")
    os.set_blocking(fd, False)
    return os.fstat(fd).st_dev, os.fstat(fd).st_ino


def _remaining(deadline):
    result = deadline - time.monotonic()
    if result <= 0:
        raise TimeoutError("Helper exchange deadline")
    return result


def _read_exact(fd, count, deadline, retained):
    body = bytearray()
    while len(body) < count:
        ready, _, _ = select.select([fd], [], [], _remaining(deadline))
        if not ready:
            raise TimeoutError("Helper exchange deadline")
        try:
            piece = os.read(fd, min(65536, count - len(body)))
        except BlockingIOError:
            continue
        _require(piece, "Helper pipe closed during frame")
        body.extend(piece)
        retained.extend(piece)
    return bytes(body)


def _receive(fd, maximum, deadline, retained, initial=b''):
    _require(isinstance(initial, bytes) and len(initial) <= 1, "Invalid initial helper frame byte")
    retained.extend(initial)
    size = struct.unpack('!I', initial + _read_exact(fd, 4 - len(initial), deadline, retained))[0]
    _require(0 < size <= maximum, "Helper frame byte ceiling")
    return json.loads(_read_exact(fd, size, deadline, retained))


def _send(fd, value, maximum, deadline):
    payload = _json(value)
    _require(0 < len(payload) <= maximum, "Helper frame byte ceiling")
    framed = struct.pack('!I', len(payload)) + payload
    offset = 0
    while offset < len(framed):
        _, ready, _ = select.select([], [fd], [], _remaining(deadline))
        if not ready:
            raise TimeoutError("Helper exchange deadline")
        try:
            offset += os.write(fd, framed[offset:offset + 65536])
        except BlockingIOError:
            continue


def _load(pin, maximum):
    return json.loads(_sealed_bytes(pin, maximum=maximum))


def _responses(pin):
    value = _load(pin, MAX_FIXTURE)
    _require(isinstance(value, dict) and set(value) == {'schema', 'mode', 'responses'}
             and value['schema'] == 1 and value['mode'] == 'saved_synthetic_only'
             and isinstance(value['responses'], list) and len(value['responses']) <= 12,
             "Exact bounded saved synthetic fixture required")
    result = {}
    required = {'method', 'url', 'status', 'headers', 'body_base64', 'complete', 'error'}
    for row in value['responses']:
        _require(isinstance(row, dict) and set(row) in (required, required | {'raw_response_base64'})
                 and row['method'] == 'GET' and row['url'] not in result,
                 "Only unique explicit saved GET responses supported")
        _target(row['method'], row['url'])
        _require(type(row['complete']) is bool and _safe_response_headers(row['headers'])
                 and (row['status'] is None or type(row['status']) is int and 100 <= row['status'] <= 599)
                 and (row['error'] is None or isinstance(row['error'], str) and len(row['error']) <= 8192),
                 "Invalid saved response envelope")
        body = base64.b64decode(row['body_base64'], validate=True)
        raw = base64.b64decode(row.get('raw_response_base64', ''), validate=True)
        _require(len(body) <= MAX_BODY and len(raw) <= MAX_RAW_RESPONSE,
                 "Saved response byte ceiling")
        result[row['url']] = deepcopy(row)
    return result


class SavedHelperClient:
    """A finite synthetic broker wire; uses only already-open IPC after guard.

    A failed exchange closes the client permanently. The retained IPC prefix is
    diagnostic metadata, never mislabeled as raw HTTP response evidence.
    """
    execution_mode = 'synthetic'

    def __init__(self, read_fd, write_fd, *, store_root, synthetic_responses_pin,
                 timeout_seconds=20):
        _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 20,
                 "Helper deadline must not exceed existing twenty seconds")
        self.read_fd, self.write_fd = read_fd, write_fd
        self.identities = (_pipe(read_fd), _pipe(write_fd))
        _require(self.identities[0] != self.identities[1], "Two distinct IPC pipes required")
        self.store_root = _safe(store_root).as_posix()
        self.synthetic_responses_pin = deepcopy(synthetic_responses_pin)
        self.timeout = timeout_seconds
        self.sequence, self.failed = 0, False
        self.lock = threading.Lock()
        self.last_ipc_prefix = b''

    def exchange(self, plan_pin, review_pin):
        plan = _load(plan_pin, 1024 * 1024)
        review = _load(review_pin, 1024 * 1024)
        _require(plan.get('execution_mode') == 'synthetic' and plan.get('publication_requested') is False
                 and plan.get('store_root') == self.store_root and review.get('plan') == plan_pin
                 and review.get('status') == 'passed' and review.get('blockers') == [],
                 "Exact unpublished synthetic plan/review required")

        def wire(method, url, headers, form, scope):
            _require(scope == plan and method == 'GET' and form is None,
                     "Helper only accepts exact saved GET plan scope")
            _target(method, url)
            _request_headers(method, url, headers, form)
            _require(self.lock.acquire(blocking=False), "Concurrent helper exchange forbidden")
            received = bytearray()
            try:
                _require(not self.failed, "Helper client closed; never resend")
                self.failed = True
                _require((_pipe(self.read_fd), _pipe(self.write_fd)) == self.identities,
                         "Helper IPC descriptor replaced")
                self.sequence += 1
                message = {'schema': 1, 'sequence': self.sequence, 'operation': 'saved_get',
                    'plan_id': plan['plan_id'], 'plan_sha256': plan_pin['sha256'],
                    'review_sha256': review_pin['sha256'],
                    'fixture_sha256': self.synthetic_responses_pin['sha256'],
                    'method': method, 'url': url, 'headers': headers}
                deadline = time.monotonic() + self.timeout
                _send(self.write_fd, message, MAX_REQUEST, deadline)
                answer = _receive(self.read_fd, MAX_RESPONSE, deadline, received)
                _require(isinstance(answer, dict) and answer.get('schema') == 1
                         and answer.get('sequence') == self.sequence
                         and answer.get('request_sha256') == hashlib.sha256(_json(message)).hexdigest(),
                         "Unbound helper response")
                _require(answer.get('status') == 'passed', 'Saved helper rejected request: ' + str(answer.get('error')))
                result = deepcopy(answer['result'])
                result['body'] = base64.b64decode(result.pop('body_base64'), validate=True)
                raw = result.pop('raw_response_base64', None)
                if raw is not None:
                    result['raw_response'] = base64.b64decode(raw, validate=True)
                _require(len(result['body']) <= MAX_BODY
                         and len(result.get('raw_response', b'')) <= MAX_RAW_RESPONSE
                         and _safe_response_headers(result.get('headers', [])),
                         "Helper violated response bounds")
                self.failed = not (result.get('complete') is True and not result.get('error'))
                return result
            except Exception as exc:
                return {'status': None, 'headers': [], 'body': b'', 'complete': False,
                        'error': type(exc).__name__ + ': ' + str(exc),
                        'wire_metadata': {'synthetic': True, 'helper': True, 'source_requests': 0,
                            'ipc_prefix_base64': base64.b64encode(received[:1048576]).decode(),
                            'ipc_received_bytes': len(received),
                            'ipc_received_sha256': hashlib.sha256(received).hexdigest(),
                            'ipc_prefix_truncated': len(received) > 1048576,
                            'retry_permitted': False}}
            finally:
                self.last_ipc_prefix = bytes(received)
                self.lock.release()
        return wire


class SavedResponseAuthority:
    """Trusted helper whose exact charge head is independently acknowledged.

    The native launcher gives this process a different UID, independently owned
    response/ledger namespace and read-only access to the mutable store. Root
    ownership is acceptable for this local prototype; hosted custody is unproved.
    ``confirmed_head`` must consult the independent authority, not caller state.
    """
    def __init__(self, store, synthetic_responses_pin, journal_root, *,
                 controller_uid, confirmed_head, clock=lambda: datetime.now(timezone.utc)):
        _require(type(controller_uid) is int and controller_uid >= 0
                 and controller_uid != os.geteuid() and callable(confirmed_head),
                 "Independent helper UID and explicit confirmed-head authority required")
        self.store, self.pin = store, deepcopy(synthetic_responses_pin)
        self.root = _safe(journal_root)
        self.controller_uid, self.confirmed_head, self.clock = controller_uid, confirmed_head, clock
        _require(self.root.is_dir() and self.root not in store.root.parents
                 and store.root not in self.root.parents and self.root != store.root,
                 "Independent preprovisioned helper evidence namespace required")
        self.identity = (self.root.stat().st_dev, self.root.stat().st_ino)
        self._custody()
        self._fixture_custody()
        _require(not list(self.root.glob('*.tmp')), "Unacknowledged helper evidence remains")
        self.responses = _responses(self.pin)
        self.sequence, self.failed = 0, False

    def _fixture_custody(self):
        path = _safe(self.pin['path'])
        _require(path.stat().st_uid != self.controller_uid,
                 "Saved fixture belongs to controller")
        for parent in path.parents:
            info = parent.lstat()
            _require(info.st_uid != self.controller_uid
                     and (not info.st_mode & 0o022 or info.st_mode & stat.S_ISVTX),
                     "Saved fixture lacks independent native custody")

    def _custody(self):
        _require((self.root.stat().st_dev, self.root.stat().st_ino) == self.identity,
                 "Helper journal root replaced")
        for path in (self.root, *self.root.parents):
            info = path.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid != self.controller_uid,
                     "Helper custody belongs to controller")
            # A sticky shared ancestor (e.g. /tmp) protects another UID's entry.
            _require(not info.st_mode & 0o022 or info.st_mode & stat.S_ISVTX,
                     "Writable helper custody ancestor")
        _require(self.root.stat().st_uid == os.geteuid() and not self.root.stat().st_mode & 0o077,
                 "Private native helper-owned journal required")

    def _capacity(self, needed):
        entries = list(self.root.iterdir())
        _require(len(entries) + 1 <= MAX_ENTRIES, "Helper evidence entry ceiling")
        used = 0
        for path in entries:
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid()
                     and not info.st_mode & 0o222, "Helper evidence changed or aliased")
            used += info.st_size
        _require(used + needed <= MAX_OUTPUT and shutil.disk_usage(self.root).free >= needed + RESERVE,
                 "Helper evidence output/free-space ceiling")

    def answer(self, message):
        _require(not self.failed, "Helper authority closed after failure")
        self.failed = True
        self._custody()
        required = {'schema', 'sequence', 'operation', 'plan_id', 'plan_sha256', 'review_sha256',
                    'fixture_sha256', 'method', 'url', 'headers'}
        _require(isinstance(message, dict) and set(message) == required
                 and message['schema'] == 1 and message['sequence'] == self.sequence + 1
                 and message['operation'] == 'saved_get'
                 and message['fixture_sha256'] == self.pin['sha256'], "Unbound helper request")
        _require(isinstance(message['plan_id'], str)
                 and re.fullmatch(r'[A-Za-z0-9_-]{1,70}', message['plan_id']), "Invalid helper plan identity")
        claim = self.store.claimed_authority(message['plan_id'])
        _require(claim['plan_pin']['sha256'] == message['plan_sha256']
                 and claim['review_pin']['sha256'] == message['review_sha256'],
                 "Helper request differs from exact claimed authority")
        plan = _load(claim['plan_pin'], 1024 * 1024)
        review = _load(claim['review_pin'], 1024 * 1024)
        now = _stamp(self.clock())
        _require(plan.get('kind') == 'roster_source_collection_v1'
                 and plan.get('plan_id') == message['plan_id']
                 and plan.get('execution_mode') == 'synthetic' and plan.get('publication_requested') is False
                 and plan.get('store_root') == self.store.root.as_posix()
                 and review.get('status') == 'passed' and review.get('blockers') == []
                 and review.get('plan') == claim['plan_pin'], "Invalid synthetic helper authority")
        _require(_stamp(plan['issued_at']) <= now < _stamp(plan['expires_at'])
                 and 0 < (_stamp(plan['expires_at']) - _stamp(plan['issued_at'])).total_seconds() <= 86400,
                 "Expired or future helper plan")
        method, url = message['method'], message['url']
        _target(method, url)
        _request_headers(method, url, message['headers'], None)
        _require(method == 'GET' and message['headers']['User-Agent'] == plan['user_agent']
                 and 'Cookie' not in message['headers'] and plan['challenge'] == {'enabled': False}
                 and (url == MEMBERSHIP_URL and plan['membership'] is True or url in plan['documents']),
                 "Request outside finite saved-only scope")
        state = self.store.snapshot()
        _require(type(plan.get('membership')) is bool and isinstance(plan.get('documents'), list)
                 and len(set(plan['documents'])) == len(plan['documents'])
                 and type(plan.get('max_requests')) is int
                 and plan['max_requests'] == int(plan['membership']) + len(plan['documents'])
                 and 0 < plan['max_requests'] <= state['policy']['plan_ceiling'],
                 "Helper finite plan ceiling changed")
        for document in plan['documents']:
            _target('GET', document)
            _require(document != MEMBERSHIP_URL, "Membership cannot masquerade as document")
        _require(assess_source_freshness(plan['observed_at'], now=now,
                 max_age_seconds=state['policy']['max_source_age_seconds'])['eligible'], "Stale helper observation")
        pending = state['pending_requests']
        _require(len(pending) == 1 and state['uncertain_requests'] == pending,
                 "Exactly one charged pending request required")
        request_id = pending[0]
        charge = state['requests'][request_id]
        plan_charges = [row for row in state['requests'].values() if row['plan_id'] == plan['plan_id']]
        _require(charge['plan_id'] == plan['plan_id'] and charge['method'] == method and charge['url'] == url
                 and charge['source'] == ('roster.watch' if url == MEMBERSHIP_URL else 'ufcstats.com')
                 and 0 < len(plan_charges) <= plan['max_requests']
                 and request_id == plan['plan_id'] + '-' + str(len(plan_charges))
                 and sum(row['url'] == url for row in plan_charges) == 1,
                 "No exact durable charge for helper request")
        acknowledgement = self.confirmed_head(deepcopy(state['head']))
        _require(isinstance(acknowledgement, dict) and acknowledgement.get('phase') == 'current'
                 and acknowledgement.get('store_head') == state['head'],
                 "Exact charge head not independently acknowledged")
        _require(self.store.snapshot()['head'] == state['head'], "Store changed during helper authorization")
        # Re-read exact sealed bytes at use, never trust a mutable cached object.
        self._fixture_custody()
        responses = _responses(self.pin)
        _require(url in responses, "No pinned saved response for request")
        row = responses[url]
        result = {key: row[key] for key in ('status', 'headers', 'body_base64', 'complete', 'error')}
        if 'raw_response_base64' in row:
            result['raw_response_base64'] = row['raw_response_base64']
        result['wire_metadata'] = {'synthetic': True, 'helper': True, 'fixture': self.pin,
            'request_id': request_id, 'store_head': state['head'], 'source_requests': 0,
            'socket_created': False, 'retry_permitted': False}
        response = {'schema': 1, 'sequence': message['sequence'], 'status': 'passed',
            'request_sha256': hashlib.sha256(_json(message)).hexdigest(), 'result': result}
        record = _json({'schema': 1, 'request': message, 'response': response,
                        'charge': charge, 'confirmed_head': acknowledgement})
        _require(len(record) <= MAX_RESPONSE, "Helper receipt byte ceiling")
        self._capacity(len(record))
        # Exclusive creation and both fsyncs precede output. If output is lost,
        # this consumed identity survives helper restart and remains blocked.
        _new_file(self.root / (request_id + '.json'), record, immutable=True)
        _require(_stamp(self.clock()) < _stamp(plan['expires_at']),
                 "Helper response finished after authority expiry")
        self.sequence = message['sequence']
        self.failed = not (row['complete'] is True and not row['error'])
        return response

    def serve_one(self, read_fd, write_fd, *, timeout_seconds=20, _initial=b''):
        _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 20,
                 "Helper deadline must not exceed existing twenty seconds")
        _pipe(read_fd)
        _pipe(write_fd)
        deadline = time.monotonic() + timeout_seconds
        received = bytearray()
        message = None
        try:
            message = _receive(read_fd, MAX_REQUEST, deadline, received, initial=_initial)
            answer = self.answer(message)
            _send(write_fd, answer, MAX_RESPONSE, deadline)
            return answer
        except BaseException:
            self.failed = True
            # The exclusive response receipt, when created, is never removed.
            # Preserve malformed/partial IPC too; it grants no retry authority.
            failure = _json({'schema': 1, 'sequence': self.sequence + 1,
                'ipc_prefix_base64': base64.b64encode(received).decode(), 'request': message})
            self._custody()
            self._capacity(len(failure))
            _new_file(self.root / ('failure-' + str(self.sequence + 1) + '-' +
                      str(time.time_ns()) + '.json'), failure, immutable=True)
            raise


def serve_saved_helper(authority, request_fd, response_fd, *, maximum_requests=200):
    """One preopened channel under the launcher's enclosing job supervisor.

    Several minutes of controller validation may pass between requests. Only an
    actual first request byte starts the existing twenty-second exchange bound.
    EOF ends the service; restarting it cannot erase its immutable served IDs.
    """
    _require(type(maximum_requests) is int and 0 < maximum_requests <= 200,
             "Helper request ceiling may not exceed inherited accounting limit")
    _pipe(request_fd)
    _pipe(response_fd)
    for _ in range(maximum_requests):
        select.select([request_fd], [], [])
        first = os.read(request_fd, 1)
        if not first:
            return {'source_requests': 0, 'synthetic_requests': authority.sequence,
                    'source_network_enabled': False}
        authority.serve_one(request_fd, response_fd, _initial=first)
        if authority.failed:
            return {'source_requests': 0, 'synthetic_requests': authority.sequence,
                    'source_network_enabled': False, 'uncertainty_blocks_reuse': True}
    raise HelperError('Helper cumulative request ceiling reached')
