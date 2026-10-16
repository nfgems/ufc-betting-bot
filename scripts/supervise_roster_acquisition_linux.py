"""Linux offline preparation runner; deliberately has no source-network mode.

Run only an externally reviewed, pinned Python worker in a fresh work directory.
This is a control primitive, not an installed acquisition job or a scheduler.
The unchanged acquisition CLI still needs a reviewed durable transport adapter
and a separately approved hosted integration trial. No project imports occur in
this module. Kernel controls are mandatory, never replaced by an audit-only run.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import secrets
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone


class ControlError(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise ControlError(reason)


def safe_path(value, *, exists=True):
    require(isinstance(value, (str, os.PathLike)), 'path required')
    path = Path(value)
    require(path.is_absolute() and path == path.resolve(), 'absolute link-free path required')
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            require(not part.is_symlink(), 'symlink denied')
    require(not exists or path.exists(), 'missing path')
    return path


def pin(value):
    path = safe_path(value)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'regular unaliased file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': info.st_size, 'sha256': digest}


def check_pin(item, *, sealed=False):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'invalid pin')
    require(pin(item['path']) == item, 'pin drift: ' + str(item['path']))
    require(not sealed or not (Path(item['path']).stat().st_mode & 0o222), 'unsealed input')


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def utc():
    return datetime.now(timezone.utc).isoformat()


def instant(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None, 'zoned timestamp required')
    return stamp.astimezone(timezone.utc)


def authority(plan_path, review_path, review_sha256):
    require(sys.platform == 'linux' and os.uname().machine == 'x86_64', 'Linux x86_64 required')
    require(re.fullmatch('[a-f0-9]{64}', review_sha256) is not None, 'external review digest required')
    require(pin(review_path)['sha256'] == review_sha256, 'external review pin drift')
    review = json.loads(Path(review_path).read_bytes())
    require(review.get('status') == 'passed' and review.get('blockers') == [], 'review did not pass')
    require(review.get('plan') == pin(plan_path), 'review does not bind plan')
    check_pin(pin(plan_path), sealed=True)
    check_pin(pin(review_path), sealed=True)
    plan = json.loads(Path(plan_path).read_bytes())
    require(plan.get('schema_version') == 1 and plan.get('mode') == 'offline', 'only offline preparation supported')
    require(re.fullmatch('[A-Za-z0-9_-]{1,100}', str(plan.get('plan_id', ''))) is not None, 'invalid plan id')
    require(plan.get('network_requests') == 0 and plan.get('publication_requested') is False,
            'network and publication authority forbidden')
    code, inputs = plan.get('code_pins'), plan.get('input_pins')
    require(isinstance(code, list) and code and isinstance(inputs, list), 'pins required')
    require(review.get('code_pins') == code, 'review does not bind code')
    require(len({item['path'] for item in code + inputs}) == len(code + inputs), 'duplicate pins')
    for item in code + inputs:
        check_pin(item, sealed=True)
    # Opt-in import roots never grant wildcard reads. Every file beneath them
    # must already be in the sealed code list; the audit hook checks each open.
    roots = plan.get('python_roots', [])
    require(isinstance(roots, list) and len(set(roots)) == len(roots), 'invalid Python roots')
    pinned = {item['path'] for item in code}
    for value in roots:
        root = safe_path(value)
        require(root.is_dir(), 'Python root must be a directory')
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'linked Python dependency')
            if path.is_file():
                require(str(path) in pinned, 'unsealed dependency file: ' + str(path))
    if 'dependency_profile' in plan:
        require(plan['dependency_profile'] == 'roster_candidate_py311_v1', 'unsupported dependency profile')
        timezone_pin = plan.get('utc_timezone')
        require(timezone_pin in code + inputs, 'regular UTC timezone must be pinned')
        require(Path(timezone_pin['path']).name == 'UTC'
                and Path(timezone_pin['path']).read_bytes().startswith(b'TZif'), 'invalid UTC timezone artifact')
    else:
        require('utc_timezone' not in plan, 'timezone needs explicit dependency profile')
    require(pin(Path(__file__).resolve()) in code, 'supervisor not pinned')
    require(pin(Path(sys.executable).resolve()) in code, 'interpreter not pinned')
    require(plan.get('worker') in code, 'worker not pinned')
    require(plan.get('seccomp_library') in code, 'seccomp library not pinned')
    require(plan.get('libc_library') in code, 'libc library not pinned')
    require(isinstance(plan.get('argv'), list) and all(isinstance(v, str) for v in plan['argv']), 'invalid argv')
    for name, low, high in [('wall_seconds', 1, 1800), ('memory_bytes', 67108864, 4294967296),
                            ('file_bytes', 1024, 134217728), ('cpu_seconds', 1, 1800),
                            ('output_bytes', 16384, 1073741824), ('output_entries', 8, 10000)]:
        require(type(plan.get(name)) is int and low <= plan[name] <= high, 'invalid limit: ' + name)
    now = datetime.now(timezone.utc)
    issued, expires = instant(plan['issued_at']), instant(plan['expires_at'])
    require(issued <= now < expires and 0 < (expires - issued).total_seconds() <= 86400, 'expired or future plan')
    return plan


def kernel_boundary(work, plan, expected_parent):
    """Deny writes outside work, all networking and process creation/escape.

    Landlock ABI >= 3 covers truncation. Seccomp also denies metadata mutation
    not covered by that ABI. Worker sealing therefore belongs to a future
    narrow broker, not a chmod allowance here. See the hosted release gate.
    """
    libc = ctypes.CDLL(plan['libc_library']['path'], use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0, 'parent-death signal failed')
    require(os.getppid() == expected_parent, 'supervisor disappeared before containment')
    require(libc.prctl(38, 1, 0, 0, 0) == 0, 'no_new_privs failed')
    abi = libc.syscall(444, 0, 0, 1)
    require(abi >= 3, 'Landlock ABI 3 or newer required')
    # Linux uapi landlock.h: all v3 filesystem rights except read file/directory.
    rights = ((1 << 15) - 1) & ~((1 << 2) | (1 << 3))
    ruleset_attr = ctypes.c_uint64(rights)
    ruleset = libc.syscall(444, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
    require(ruleset >= 0, 'Landlock ruleset creation failed')
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    work_fd = os.open(work, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # No executable or device nodes, FIFOs, sockets, symlinks or cross-dir links.
        allowed = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 14)
        rule = PathRule(allowed, work_fd)
        require(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0) == 0, 'Landlock path rule failed')
        require(libc.syscall(446, ruleset, 0) == 0, 'Landlock enforcement failed')
    finally:
        os.close(work_fd)
        os.close(ruleset)
    seccomp = ctypes.CDLL(plan['seccomp_library']['path'], use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = seccomp.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    require(bool(ctx), 'seccomp allocation failed')
    denied = ('socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg '
              'fork vfork clone clone3 execve execveat setsid setpgid ptrace process_vm_writev '
              'mount umount2 pivot_root chroot unshare setns open_by_handle_at name_to_handle_at '
              'bpf io_uring_setup io_uring_enter io_uring_register userfaultfd '
              'chmod fchmod fchmodat fchmodat2 chown fchown lchown fchownat '
              'utime utimes futimesat utimensat setxattr lsetxattr fsetxattr '
              'removexattr lremovexattr fremovexattr kill tkill tgkill pidfd_send_signal '
              'reboot kexec_load kexec_file_load init_module finit_module delete_module '
              'swapon swapoff acct quotactl flock prctl '
              'setuid setgid setresuid setresgid setreuid setregid setfsuid setfsgid').split()
    try:
        for name in denied:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number == -1:  # Unknown to this pinned library; no guessed syscall numbers.
                continue
            require(seccomp.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0) == 0,
                    'seccomp rule failed: ' + name)
        require(seccomp.seccomp_load(ctx) == 0, 'seccomp enforcement failed')
    finally:
        seccomp.seccomp_release(ctx)
    return {'landlock_abi': abi, 'seccomp': True, 'no_new_privs': True,
            'network': 'denied', 'process_creation': 'denied', 'parent_death_signal': 'SIGKILL'}


class AuditBoundary:
    def __init__(self, work, code, inputs):
        self.work = work
        self.readable = {item['path'] for item in code + inputs}
        self.denials = []
        self.stdlib = Path(os.__file__).resolve().parent

    def inside(self, value):
        require(not isinstance(value, int), 'unscoped descriptor path')
        path = safe_path(Path(os.fsdecode(value)).absolute(), exists=False)
        return path == self.work or self.work in path.parents

    def audit(self, event, args):
        try:
            if event == 'open':
                path, mode, flags = args
                require(not isinstance(path, int), 'unscoped descriptor open')
                path = safe_path(Path(os.fsdecode(path)).absolute(), exists=False)
                write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                if write:
                    require(self.inside(path), 'write outside work')
                else:
                    require(str(path) in self.readable or self.inside(path) or
                            (self.stdlib in path.parents and 'site-packages' not in path.parts),
                            'unpinned read: ' + str(path))
            elif event.startswith(('socket.', 'ctypes.', 'subprocess.')) or event in {
                'os.system', 'os.fork', 'os.forkpty', 'os.exec', 'os.posix_spawn',
                'os.kill', 'os.killpg', 'os.putenv', 'os.unsetenv', 'sys.addaudithook',
                'os.link', 'os.symlink', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate', 'fcntl.flock',
            }:
                raise ControlError('operation denied: ' + event)
            elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chdir'}:
                require(self.inside(args[0]), 'mutation outside work')
                if event != 'os.chdir':
                    require(args[-1] in (-1, None), 'dir_fd mutation denied')
            elif event == 'os.rename':
                require(self.inside(args[0]) and self.inside(args[1]) and args[2:] == (-1, -1),
                        'rename outside work or via descriptor')
        except (ControlError, ValueError, TypeError) as exc:
            self.denials.append({'event': event, 'reason': str(exc)})
            raise ControlError(str(exc)) from exc


def worker_main(args):
    plan = authority(args.plan, args.review, args.review_sha256)
    run = safe_path(args.run)
    require(run == safe_path(plan['control_root']) / plan['plan_id'], 'child run is not plan-bound')
    require(os.getppid() == args.parent_pid, 'child parent mismatch')
    require(args.launch_fd is not None and args.lock_fd is not None, 'inherited launch authority required')
    require(stat.S_ISFIFO(os.fstat(args.launch_fd).st_mode), 'launch authority must be a pipe')
    token = os.read(args.launch_fd, 33)
    os.close(args.launch_fd)
    require(len(token) == 32, 'invalid launch token')
    claim = json.loads((run / 'consumed.json').read_bytes())
    require(claim['launch_sha256'] == hashlib.sha256(token).hexdigest()
            and claim['plan'] == pin(args.plan) and claim['review'] == pin(args.review), 'unbound launch token')
    lock_info = os.fstat(args.lock_fd)
    named_lock = (Path(plan['control_root']) / 'execution.lock').stat()
    require((lock_info.st_dev, lock_info.st_ino) == (named_lock.st_dev, named_lock.st_ino), 'wrong inherited lock')
    # This exclusive durable record also rejects a second child with an inherited token.
    immutable(run / 'worker-started.json', {'pid': os.getpid(), 'parent_pid': args.parent_pid,
              'started_at': utc(), 'launch_sha256': claim['launch_sha256']})
    work = safe_path(run / 'work')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (plan['memory_bytes'], plan['memory_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE, (plan['file_bytes'], plan['file_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU, (plan['cpu_seconds'], plan['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    os.chdir(work)
    # No user/site startup, proxies, credentials, dotenv or inherited app flags.
    sys.dont_write_bytecode = True
    # Python still probes cache paths with -B. Redirect absent cache probes to
    # disposable work; do not authorize unpinned reads in the sealed code tree.
    sys.pycache_prefix = str(work / 'python-cache')
    sys.path[:] = [str(Path(plan['worker']['path']).parent)] + [
        path for path in sys.path if path and 'site-packages' not in path and 'dist-packages' not in path]
    sys.path[:0] = plan.get('python_roots', [])
    if plan.get('dependency_profile') == 'roster_candidate_py311_v1':
        # The worker has no networking capability. Prevent urllib3's optional
        # startup IPv6 probe without allowing any socket syscall or audit event.
        import socket
        socket.has_ipv6 = False
    kernel = kernel_boundary(work, plan, args.parent_pid)
    boundary = AuditBoundary(work, plan['code_pins'], plan['input_pins'])
    sys.addaudithook(boundary.audit)
    result = {'started_at': utc(), 'kernel': kernel, 'status': 'failed', 'source_requests': 0}
    code = Path(plan['worker']['path']).read_bytes()
    require(hashlib.sha256(code).hexdigest() == plan['worker']['sha256'], 'worker changed before execution')
    sys.argv = [plan['worker']['path'], *plan['argv']]
    try:
        if plan.get('dependency_profile') == 'roster_candidate_py311_v1':
            # Real third-party imports remain AFTER the kernel/audit boundary.
            # A pinned regular UTC copy avoids dateutil's system symlink path.
            import dateutil.tz
            dateutil.tz.tz.TZPATHS = [str(Path(plan['utc_timezone']['path']).parent)]
        exec(compile(code, plan['worker']['path'], 'exec'),
             {'__name__': '__main__', '__file__': plan['worker']['path']})
        result['status'] = 'passed'
    except SystemExit as exc:
        result['status'] = 'passed' if exc.code in (0, None) else 'failed'
        result['exit'] = exc.code if isinstance(exc.code, (int, str)) else repr(exc.code)
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    result['denials'] = boundary.denials
    if boundary.denials:
        result['status'] = 'failed'
    result['completed_at'] = utc()
    # Child writes only work; the trusted parent seals this file after exit.
    with (work / 'worker-result.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(work)
    return 0 if result['status'] == 'passed' else 1


def supervise(args, *, broker=None):
    plan = authority(args.plan, args.review, args.review_sha256)
    require((broker is None) == ('broker_plan' not in plan), 'explicit bound parent broker required')
    if broker is not None:
        require(getattr(broker, 'execution_mode', None) == 'synthetic', 'only synthetic parent broker supported')
    control = safe_path(args.control_root)
    require(str(control) == plan.get('control_root'), 'control root is not bound by plan')
    require(control.is_dir() and not control.stat().st_mode & 0o022, 'private control root required')
    capacity = os.statvfs(control)
    require(capacity.f_bavail * capacity.f_frsize >= plan['output_bytes'] + 536870912,
            'insufficient free disk reserve')
    lock_path = control / 'execution.lock'
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(lock_fd).st_nlink == 1, 'aliased lock')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControlError('another workflow holds the exclusive lock') from exc
        sync_dir(control)
        run = control / plan['plan_id']
        run.mkdir(mode=0o700)  # Never reuse a consumed plan, including startup failures.
        sync_dir(control)
        token = secrets.token_bytes(32)
        immutable(run / 'consumed.json', {'plan': pin(args.plan), 'review': pin(args.review),
                  'claimed_at': utc(), 'replay_permitted': False,
                  'launch_sha256': hashlib.sha256(token).hexdigest()})
        work = run / 'work'
        work.mkdir(mode=0o700)
        sync_dir(run)
        if broker is not None:
            broker.bind(work, plan)
        for item in plan['code_pins'] + plan['input_pins']:
            require(control not in Path(item['path']).parents, 'inputs inside mutable control root')
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TZ': 'UTC',
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
               'PYTHON_DOTENV_DISABLED': '1', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OPENBLAS_MAIN_FREE': '1', 'MKL_NUM_THREADS': '1',
               'TMPDIR': str(work), 'HOME': str(work),
               'UFC_DATA_DIR': str(work / 'data'), 'UFC_PROCESSED_DIR': str(work / 'processed'),
               'UFC_MODELS_DIR': str(work / 'models'), 'UFC_LOGS_DIR': str(work / 'logs'),
               'UFC_REFRESH_ENABLED': '0', 'UFC_UPDATE_CALENDAR_ENABLED': '0',
               'UFC_MONTHLY_CANDIDATE_ENABLED': '0', 'LIVE_TRADING_MODE': 'off'}
        for name in ('data', 'processed', 'models', 'logs'):
            (work / name).mkdir(mode=0o700)
        launch_read, launch_write = os.pipe()
        os.write(launch_write, token)
        os.close(launch_write)
        command = [str(Path(sys.executable).resolve()), '-I', '-S', '-B', str(Path(__file__).resolve()),
                   '--worker-child', '--plan', str(args.plan), '--review', str(args.review),
                   '--review-sha256', args.review_sha256, '--run', str(run),
                   '--parent-pid', str(os.getpid()), '--launch-fd', str(launch_read), '--lock-fd', str(lock_fd)]
        terminal = {'plan_id': plan['plan_id'], 'started_at': utc(), 'status': 'failed',
                    'replay_permitted': False, 'source_requests': 0}
        child = None
        old_handlers = {}
        def interrupted(signum, frame):
            raise ControlError('supervisor interrupted by signal ' + str(signum))
        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                old_handlers[sig] = signal.signal(sig, interrupted)
            with (run / 'stdout.bin').open('xb') as stdout, (run / 'stderr.bin').open('xb') as stderr:
                child = subprocess.Popen(command, env=env, cwd=work, stdin=subprocess.DEVNULL,
                                         stdout=stdout, stderr=stderr, close_fds=True,
                                         pass_fds=(lock_fd, launch_read), start_new_session=True)
                terminal['pid'] = child.pid
                try:
                    deadline = time.monotonic() + plan['wall_seconds']
                    while child.poll() is None:
                        check_output_bounds(run, plan)
                        if broker is not None:
                            broker.poll()
                        if time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(command, plan['wall_seconds'])
                        time.sleep(.05)
                    check_output_bounds(run, plan)
                except subprocess.TimeoutExpired:
                    terminal['timed_out'] = True
                    raise ControlError('worker deadline exceeded')
                terminal['exit_code'] = child.returncode
                result_path = work / 'worker-result.json'
                require(result_path.is_file(), 'worker result missing')
                result = json.loads(result_path.read_bytes())
                require(result.get('status') == 'passed' and result.get('denials') == []
                        and child.returncode == 0, 'worker failed or attempted a denied operation')
                # The immutable inputs must still be exact after native execution.
                require(authority(args.plan, args.review, args.review_sha256) == plan, 'poststate authority drift')
                terminal['status'] = 'passed'
        except BaseException as exc:
            terminal['error'] = type(exc).__name__ + ': ' + str(exc)
        finally:
            os.close(launch_read)
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                child.wait(timeout=10)
                terminal['exit_code'] = child.returncode
                terminal['child_reaped'] = True
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            if broker is not None:
                try:
                    terminal['broker'] = broker.close()
                except BaseException as exc:
                    terminal['status'] = 'failed'
                    terminal['broker_error'] = type(exc).__name__ + ': ' + str(exc)
            # Output cap is also enforced by the worker's RLIMIT_FSIZE.
            for path in sorted(run.rglob('*')):
                require(not path.is_symlink(), 'unexpected output symlink')
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    path.chmod(0o400)
            terminal['completed_at'] = utc()
            immutable(run / 'terminal.json', terminal)
        return 0 if terminal['status'] == 'passed' else 1
    finally:
        os.close(lock_fd)  # The stable lock inode is never unlinked or replaced.


def check_output_bounds(run, plan):
    """Watchdog bound in addition to kernel per-file and process limits.

    This is not a filesystem quota; overshoot between scans remains possible
    (including scan time). A hosted volume quota and low-space stop remain release
    requirements. Do not follow output links, even when collecting failures.
    """
    total = count = 0
    for directory, directories, files in os.walk(run, followlinks=False):
        for name in directories + files:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                # Normal atomic tempfile replacement can remove an entry seen
                # by scandir. Final scan still enforces the completed output.
                continue
            require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), 'unexpected output object')
            require(not stat.S_ISREG(info.st_mode) or info.st_nlink == 1, 'output hardlink')
            count += 1
            total += info.st_size if stat.S_ISREG(info.st_mode) else 0
            require(count <= plan['output_entries'] and total <= plan['output_bytes'], 'output budget exceeded')
    capacity = os.statvfs(run)
    require(capacity.f_bavail * capacity.f_frsize >= 536870912, 'free disk reserve exhausted')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--control-root')
    parser.add_argument('--worker-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--launch-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_child:
            require(args.run is not None and args.parent_pid is not None, 'missing child context')
            return worker_main(args)
        require(args.control_root is not None, 'control root required')
        return supervise(args)
    except (ControlError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
