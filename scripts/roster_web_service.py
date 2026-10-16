"""Keep the ordinary web service alive around one explicitly requested launcher.

The trusted enclosing image starts this fixed packaged interpreter with -I -S
-B as root after the existing entrypoint has prepared ordinary web paths. Its
environment belongs to the web service. The finite root launcher receives only
roster_service_start.INITIAL_ENV, before its first executable. This component
never provisions authority, retries a launcher, or schedules another cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import stat
import subprocess
import sys
import time

PACKAGE = Path('/opt/ufc-roster/runtime')
PYTHON = PACKAGE / 'python/bin/python3.11'
ENTRY = PACKAGE / 'source/scripts/roster_web_service.py'
ATTEMPTS = Path('/app/logs/roster_refresh/service-attempts')
WEB_UID, WEB_GID = 100, 101


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def event(name, **values):
    try:
        print(json.dumps(dict(event=name, **values), sort_keys=True), flush=True)
    except OSError:
        # A closed log consumer cannot bypass child cleanup or undo a fence.
        pass


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_once(path, value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode() + b'\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def bootstrap(package_sha256, *, web_child=False):
    require(sys.platform == 'linux' and sys.version_info[:3] == (3, 11, 16)
            and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
            and not sys.flags.optimize, 'Packaged CPython3.11.16 -I -S -B required')
    require(Path(sys.executable) == PYTHON and Path(__file__) == ENTRY
            and Path(sys.prefix) == PACKAGE / 'python'
            and Path(sys.base_prefix) == PACKAGE / 'python', 'Exact packaged bootstrap required')
    require(not any(key.startswith(('LD_', 'DYLD_')) for key in os.environ),
            'Loader environment must be absent before the trusted bootstrap')
    identity = (WEB_UID, WEB_GID) if web_child else (0, 0)
    require((os.geteuid(), os.getegid()) == identity
            and (not os.getgroups() if web_child else not set(os.getgroups()) - {0}),
            'Exact service identity with no additional supplementary groups required')
    for path in (Path('/'), Path('/app'), PACKAGE, *PACKAGE.parents):
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
                'Protected application/runtime ancestry required')
    # These first interpreter/bootstrap bytes are an enclosing-image trust input.
    # Verification below cannot authenticate code that has already executed.
    adapter_path = PACKAGE / 'source/scripts/roster_railway_startup.py'
    module_spec = importlib.util.spec_from_file_location('roster_web_checked_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    root, manifest = adapter.verify_package(PACKAGE / 'runtime-manifest.json', package_sha256)
    require(root == PACKAGE, 'Fixed complete runtime package required')
    application = PACKAGE / ('web-source' if web_child else 'source')
    adapter._trusted(application, directory=True)
    # The web component comes from its own exact clean deployed Git revision.
    # Do not mix its imports with the older frozen roster source component.
    sys.path[:0] = [str(application), str(PACKAGE / 'dependencies')]
    return adapter, manifest


def reserve_attempt(adapter, request_path, request_sha256, process):
    """A persistent no-retry fence, never a source claim or recovery authority.

Any existing name (including incomplete/corrupt evidence) suppresses invocation.
A crash after mkdir and before Popen intentionally leaves uncertain execution.
Deleting a fence or restoring an older volume is outside this component's scope.
"""
    from src.data.roster_railway_paths import persistent_path
    path = persistent_path(request_path)
    original = adapter._pin(path, request_sha256)
    request = json.loads(original)
    require(isinstance(request, dict) and request.get('kind') == 'roster_railway_startup_request_v1'
            and request.get('mode') == 'startup', 'Only an explicit original startup request is allowed')
    persistent_path(str(ATTEMPTS))
    try:
        ATTEMPTS.mkdir(mode=0o700)
        sync_directory(ATTEMPTS.parent)
    except FileExistsError:
        pass
    adapter._trusted(ATTEMPTS, directory=True)
    info = ATTEMPTS.lstat()
    require(stat.S_IMODE(info.st_mode) == 0o700 and info.st_gid == 0,
            'Protected service-attempt directory required; no permission repair')
    attempt = ATTEMPTS / request_sha256
    try:
        attempt.mkdir(mode=0o700)
    except FileExistsError:
        event('launcher_suppressed', reason='persistent_attempt_exists', request_sha256=request_sha256,
              retry_authorized=False)
        return None
    sync_directory(ATTEMPTS)
    # Save exact original bytes and the execution contract before process creation.
    save_once(attempt / 'request-original.json', original)
    save_once(attempt / 'process-original.json', process)
    save_once(attempt / 'intent.json', dict(schema=1, kind='roster_service_attempt_v1',
        request_path=str(path), request_sha256=request_sha256,
        package_sha256=process['args'][process['args'].index('--package-sha256') + 1],
        recorded_utc_ns=time.time_ns(), execution='not_yet_observed', retry_authorized=False,
        authority_created=False, recovery_authorized=False))
    return attempt


def finished(child):
    # Keep web's original unreaped leader reserved until its final group signal.
    return os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def stop_launcher(child):
    if child is None or child.poll() is not None:
        return True
    child.terminate()
    try:
        # The original launcher owns its role groups and bounded cleanup. Do not
        # replace that cleanup with a parent kill that strands those role groups.
        child.wait(timeout=35)
        return True
    except subprocess.TimeoutExpired:
        event('launcher_cleanup_unconfirmed', pid=child.pid, retry_authorized=False)
        return False


def record_terminal(attempt, launcher, streams, *, reason):
    errors = []
    for stream in streams:
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    terminal = dict(exit_code=launcher.returncode if launcher is not None else None,
        recorded_utc_ns=time.time_ns(), retry_authorized=False, reason=reason,
        launcher_reaped=launcher is not None and launcher.returncode is not None,
        source_success_inferred=False, output_preservation_errors=errors)
    try:
        save_once(attempt / 'terminal.json', terminal)
    except OSError as exc:
        event('launcher_terminal_save_failed', error=str(exc), retry_authorized=False)
    event('launcher_exited', **terminal)


def compose(package_sha256, request, request_sha256):
    adapter, _ = bootstrap(package_sha256)
    require(os.getcwd() == '/', 'Root composition working directory required')
    null = os.stat('/dev/null')
    info = os.fstat(0)
    require(stat.S_ISCHR(info.st_mode) and info.st_rdev == null.st_rdev, 'Null service stdin required')
    os.umask(0o077)
    from scripts.roster_service_start import process_spec
    process = process_spec(package_sha256, request, request_sha256)
    stopped = []

    def stopping(signum, _frame):
        stopped.append(signum)

    old_handlers = {signum: signal.signal(signum, stopping) for signum in (signal.SIGTERM, signal.SIGINT)}
    web = launcher = None
    attempt = None
    streams = []
    observed_terminal = False
    try:
        # Match the existing ordinary account numerically; no root web imports.
        # No ambient descriptor can become a roster role's anonymous pipe.
        web = subprocess.Popen([str(PYTHON), '-I', '-S', '-B', str(ENTRY),
            '--web-child', '--package-sha256', package_sha256],
            env=dict(os.environ), cwd='/app', stdin=subprocess.DEVNULL, close_fds=True,
            user=WEB_UID, group=WEB_GID, extra_groups=[], start_new_session=True)
        event('web_started', pid=web.pid, uid=WEB_UID, gid=WEB_GID)
        try:
            if not stopped and finished(web) is None:
                attempt = reserve_attempt(adapter, request, request_sha256, process)
            if attempt is not None and not stopped and finished(web) is None:
                for name in ('launcher.stdout', 'launcher.stderr'):
                    descriptor = os.open(attempt / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    streams.append(os.fdopen(descriptor, 'wb'))
                launcher = subprocess.Popen(process['args'], env=process['env'], cwd=process['cwd'],
                    user=process['uid'], group=process['gid'], extra_groups=[],
                    stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True,
                    stdout=streams[0], stderr=streams[1])
                save_once(attempt / 'started.json', dict(pid=launcher.pid, recorded_utc_ns=time.time_ns(),
                    retry_authorized=False, original_request_sha256=request_sha256))
                event('launcher_started', pid=launcher.pid, request_sha256=request_sha256)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            # Web availability is separate from roster startup admission/outcome.
            # A partially created fence remains permanent, including spawn failure.
            event('launcher_admission_failed', error=type(exc).__name__ + ': ' + str(exc), retry_authorized=False)
        while not stopped and finished(web) is None:
            if launcher is not None and launcher.poll() is not None and not observed_terminal:
                observed_terminal = True
                record_terminal(attempt, launcher, streams, reason='launcher_exited_web_continues')
            time.sleep(0.1)
        if stopped:
            return 128 + stopped[0]
        result = finished(web)
        return (result.si_status or 1) if result.si_code == os.CLD_EXITED else 128 + result.si_status
    finally:
        # Defer subsequent service signals until the existing launcher has had its
        # cleanup opportunity. ALWAYS may restart web; the durable fence survives.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        launcher_stopped = web_stopped = False
        try:
            launcher_stopped = stop_launcher(launcher)
        except (OSError, subprocess.TimeoutExpired) as exc:
            event('launcher_cleanup_unconfirmed', error=str(exc), retry_authorized=False)
        try:
            # Reuse the original PID-reserving bounded group cleanup for web's
            # group. This does not change launcher role creation or its cleanup.
            from src.data.roster_hosted_launcher import _cleanup
            cleanup = _cleanup({'web': web}) if web is not None else {'quiescent': True}
            web_stopped = cleanup['quiescent']
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            event('web_cleanup_unconfirmed', error=str(exc), retry_authorized=False)
        if attempt is not None and not observed_terminal:
            record_terminal(attempt, launcher, streams, reason='enclosing_web_service_stopped')
        for stream in streams:
            for operation in (stream.flush, lambda: os.fsync(stream.fileno()), stream.close):
                try:
                    operation()
                except (OSError, ValueError) as exc:
                    event('launcher_output_close_failed', error=str(exc), retry_authorized=False)
        event('service_stopped', web_reaped=web_stopped, launcher_reaped=launcher_stopped,
              retry_authorized=False)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        require(web_stopped and launcher_stopped,
                'Service descendant cleanup unconfirmed; enclosing container must stop remaining processes')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-sha256', required=True)
    parser.add_argument('--request')
    parser.add_argument('--request-sha256')
    parser.add_argument('--web-child', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.web_child:
            require(args.request is None and args.request_sha256 is None, 'Web has no launcher request interface')
            bootstrap(args.package_sha256, web_child=True)
            runpy.run_module('src.web.serve', run_name='__main__')
            return 0
        return compose(args.package_sha256, args.request, args.request_sha256)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        event('service_refused', error=type(exc).__name__ + ': ' + str(exc), retry_authorized=False)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
