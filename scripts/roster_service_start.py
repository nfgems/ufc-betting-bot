"""One explicit root service process for the pinned disabled roster launcher.

The enclosing trusted service starts process_spec with its EXACT initial env,
root identity, cwd, null stdin and closed descriptors. This entry then execs the
existing startup adapter once, preserving its PID and launcher-owned lifecycle.
It provisions no authority and never retries or schedules another invocation.
The already-running interpreter/bootstrap and OS remain trusted prerequisites.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import stat
import sys

PACKAGE = Path('/opt/ufc-roster/runtime')
PYTHON = PACKAGE / 'python/bin/python3.11'
ENTRY = PACKAGE / 'source/scripts/roster_service_start.py'
ADAPTER = PACKAGE / 'source/scripts/roster_railway_startup.py'
INITIAL_ENV = {
    'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
    'PYTHON_DOTENV_DISABLED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONNOUSERSITE': '1', 'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'UFC_REFRESH_ENABLED': '0', 'UFC_MONTHLY_CANDIDATE_ENABLED': '0',
    'UFC_UPDATE_CALENDAR_ENABLED': '0', 'LIVE_TRADING_MODE': 'off',
    'LIVE_TRADING_ARMED': '0',
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _inputs(package_sha256, request, request_sha256):
    for value in (package_sha256, request_sha256):
        _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
                 'Exact external SHA256 required')
    _require(isinstance(request, str), 'Explicit original request path required')
    path = Path(request)
    _require(path.is_absolute() and request == path.as_posix()
             and request == os.path.normpath(request) and not request.startswith('//')
             and Path('/app/logs/roster_refresh') in path.parents,
             'Canonical original request in reserved persistent namespace required')
    return ['--package-sha256', package_sha256, '--request', request,
            '--request-sha256', request_sha256]


def process_spec(package_sha256, request, request_sha256):
    """Render the exact service execution contract; do not start or issue work.

This is a finite component contract, not an OCI image or provider configuration.
The service manager must apply these values before the FIRST executable starts.
It must preserve the original request and result; restart='never' forbids retries.
"""
    return {
        'schema': 1, 'kind': 'roster_service_process_v1',
        'args': [str(PYTHON), '-I', '-S', '-B', str(ENTRY),
                 *_inputs(package_sha256, request, request_sha256)],
        'env': dict(INITIAL_ENV), 'cwd': '/', 'uid': 0, 'gid': 0,
        'stdin': 'null', 'close_fds': True, 'restart': 'never',
    }


def _entry_admission():
    _require(sys.platform == 'linux' and os.geteuid() == 0 and os.getegid() == 0
             and not set(os.getgroups()) - {0}, 'Native root service identity required')
    _require(sys.version_info[:3] == (3, 11, 16) and sys.flags.isolated
             and sys.flags.no_site and sys.dont_write_bytecode
             and not sys.flags.optimize, 'Packaged CPython3.11.16 -I -S -B required')
    _require(Path(sys.executable) == PYTHON and Path(__file__) == ENTRY
             and Path(sys.prefix) == PACKAGE / 'python'
             and Path(sys.base_prefix) == PACKAGE / 'python',
             'Exact packaged service bootstrap required')
    _require(dict(os.environ) == INITIAL_ENV and os.getcwd() == '/',
             'Exact clean initial service environment and root cwd required')
    for path in (Path('/'), Path('/app'), PACKAGE, *PACKAGE.parents):
        info = path.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                 and not info.st_mode & 0o022, 'Protected root service ancestry required')
    for path in (PYTHON, ENTRY, ADAPTER):
        for parent in path.parents:
            info = parent.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                     and not info.st_mode & 0o022, 'Protected service source ancestry required')
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == 0
                 and not info.st_mode & 0o222, 'Immutable service executable/source required')
    info, null = os.fstat(0), os.stat('/dev/null')
    _require(stat.S_ISCHR(info.st_mode) and info.st_rdev == null.st_rdev,
             'Service stdin must be /dev/null')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package-sha256', 'request', 'request-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        options = _inputs(args.package_sha256, args.request, args.request_sha256)
        _entry_admission()
        os.umask(0o077)
        # This service owns its helper; the unchanged launcher creates all exact
        # anonymous role pipes. No ambient descriptor becomes a borrowed helper.
        for name in os.listdir('/proc/self/fd'):
            descriptor = int(name)
            if descriptor > 2:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if exc.errno != errno.EBADF:
                        raise
        command = [str(PYTHON), '-I', '-S', '-B', str(ADAPTER),
                   '--package-manifest', str(PACKAGE / 'runtime-manifest.json'), *options]
        os.execve(str(PYTHON), command, dict(INITIAL_ENV))
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({'status': 'service_start_refused', 'error': str(exc),
                          'retry_authorized': False, 'activation_authorized': False}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
