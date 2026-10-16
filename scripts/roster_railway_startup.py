"""Explicit disabled refresh startup from an externally pinned native package.

Invoke the packaged CPython with -I -S -B. The enclosing trusted image/caller
must establish this bootstrap and interpreter before invocation. Verification
here precedes application imports; it does not authenticate its own boot chain
or the operating system. No default action, scheduler or provisioning of initial
authority is provided. --check-only reports packaging and input placement only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

PATHS = {'source': 'source', 'python': 'python/bin/python3.11',
         'dependencies': 'dependencies', 'libc': 'native/libc.so',
         'seccomp': 'native/libseccomp.so'}
MAX_ENTRIES = 10000
MAX_FILE = 134217728
MAX_TOTAL = 1073741824


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _trusted(path, *, directory=False):
    raw = os.fspath(path)
    path = Path(raw)
    _require(path.is_absolute() and not raw.startswith('//') and raw == path.as_posix()
             and raw == os.path.normpath(raw), 'Canonical package/input path required')
    for parent in path.parents:
        info = parent.lstat()
        # Only the fixed reserved input volume may have sticky writable ancestry.
        sticky_volume = parent == Path('/app/logs') and bool(info.st_mode & stat.S_ISVTX)
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0
                 and (not info.st_mode & 0o022 or sticky_volume), 'Untrusted package/input ancestry')
    info = path.lstat()
    _require(info.st_uid == 0 and not info.st_mode & (0o022 if directory else 0o222)
             and (stat.S_ISDIR(info.st_mode) if directory else
                  stat.S_ISREG(info.st_mode) and info.st_nlink == 1),
             'Immutable native root-owned package/input required')
    return path


def _bytes(path):
    _trusted(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(info.st_size <= MAX_FILE, 'Package/input file ceiling')
        raw = stream.read(MAX_FILE + 1)
    _require(len(raw) == info.st_size, 'Package/input changed during read')
    return raw


def _pin(path, digest, size=None):
    raw = _bytes(path)
    _require(hashlib.sha256(raw).hexdigest() == digest and
             (size is None or len(raw) == size), 'Exact external package/input pin mismatch')
    return raw


def verify_package(manifest_path, digest):
    """Hash the complete finite package before any of its application imports."""
    manifest_path = _trusted(manifest_path)
    root = _trusted(manifest_path.parent, directory=True)
    _require(manifest_path.name == 'runtime-manifest.json', 'Fixed package manifest required')
    _require(not any(p == root or p in root.parents for p in
                    map(Path, ('/app', '/tmp', '/var/tmp', '/mnt'))),
             'Runtime package must be outside managed and convenience paths')
    mounts = []
    escapes = {'040': ' ', '011': '\t', '012': '\n', '134': '\\'}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = Path(re.sub(r'\\(040|011|012|134)', lambda m: escapes[m[1]], left.split()[4]))
        _require(root not in mount.parents, 'Nested mount inside runtime package')
        if mount == root or mount in root.parents:
            mounts.append((len(mount.parts), right.split()[0]))
    deepest = max(row[0] for row in mounts)
    covering = [row for row in mounts if row[0] == deepest]
    _require(len(covering) == 1 and covering[0][1] in {'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'overlay', 'zfs'},
             'Native nonvolatile runtime package filesystem required')
    value = json.loads(_pin(manifest_path, digest))
    _require(set(value) == {'schema', 'kind', 'paths', 'files'} and type(value['schema']) is int and value['schema'] == 1
             and value['kind'] == 'roster_runtime_package_v1' and value['paths'] == PATHS,
             'Exact runtime package format required')
    expected = {}
    total = manifest_path.stat().st_size
    for row in value['files']:
        _require(set(row) == {'relative', 'bytes', 'sha256'}, 'Exact package file record required')
        relative = row['relative']
        p = Path(relative)
        _require(not p.is_absolute() and relative == p.as_posix() and '..' not in p.parts
                 and relative != '.' and relative != 'runtime-manifest.json'
                 and relative not in expected, 'Unsafe or duplicate package member')
        _require(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_FILE, 'Package file ceiling')
        expected[relative] = row
        total += row['bytes']
        _require(len(expected) < MAX_ENTRIES and total <= MAX_TOTAL, 'Package capacity ceiling')
    seen, pending, count = set(), [root], 1
    while pending:
        directory = pending.pop()
        _trusted(directory, directory=True)
        for path in directory.iterdir():
            count += 1
            _require(count <= MAX_ENTRIES, 'Package entry ceiling')
            if stat.S_ISDIR(path.lstat().st_mode):
                pending.append(path)
            elif path != manifest_path:
                relative = path.relative_to(root).as_posix()
                _require(relative in expected, 'Unlisted runtime package file')
                row = expected[relative]
                _pin(path, row['sha256'], row['bytes'])
                seen.add(relative)
    _require(seen == set(expected), 'Missing runtime package file')
    for name in ('source', 'dependencies'):
        _trusted(root / PATHS[name], directory=True)
    required = {PATHS['python'], PATHS['libc'], PATHS['seccomp'],
                'source/scripts/roster_railway_startup.py',
                'source/scripts/launch_roster_hosted_server.py',
                'source/scripts/run_roster_hosted_server.py',
                'source/scripts/run_roster_hosted_saved_helper.py'}
    _require(required <= seen, 'Incomplete launcher runtime package')
    _require((root / PATHS['python']).stat().st_mode & 0o111, 'Packaged interpreter is not executable')
    return root, value


def build_arguments(root, manifest, request, *, helper_read_fd=None, helper_write_fd=None):
    """Place named original inputs; unchanged roles decide semantic authority."""
    from src.data.roster_railway_paths import persistent_path
    fields = {'schema', 'kind', 'mode', 'evidence_directory', 'controls', 'destination',
              'controller_binding_path', 'configuration', 'helper_journal_root', 'recovery_evidence'}
    _require(set(request) == fields and type(request['schema']) is int and request['schema'] == 1
             and request['kind'] == 'roster_railway_startup_request_v1', 'Exact explicit startup request required')
    mode = request['mode']
    _require(mode in {'provision', 'successor', 'prepare', 'startup'}, 'Explicit launcher mode required')
    _require((helper_read_fd is None) == (helper_write_fd is None), 'Both original helper pipe ends required')
    borrowed = helper_read_fd is not None
    _require(not borrowed or mode == 'startup' and request['helper_journal_root'] is None,
             'Borrowed original helper channels are startup-only')
    _require(mode == 'prepare' or request['recovery_evidence'] is None,
             'Original calendar evidence is preparation-only')

    def placed(value):
        return str(persistent_path(value))

    def original(pin):
        _require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'},
                 'Complete original external input pin required')
        return json.loads(_pin(placed(pin['path']), pin['sha256'], pin['bytes']))

    evidence = placed(request['evidence_directory'])
    _trusted(evidence, directory=True)
    _require(not list(Path(evidence).iterdir()), 'Unused original launcher evidence directory required')
    args = [mode, '--evidence-directory', evidence, '--dependencies', str(root / PATHS['dependencies']),
            '--timeout-seconds', '300']
    pins = {row['relative']: row for row in manifest['files']}
    for name in ('libc', 'seccomp'):
        rel = PATHS[name]
        args += ['--' + name + '-library', str(root / rel),
                 '--' + name + '-library-sha256', pins[rel]['sha256']]
    if mode == 'provision':
        _require(request['configuration'] is None and request['helper_journal_root'] is None
                 and not borrowed, 'Provision requires only original custody inputs')
        controls = original(request['controls'])
        args += ['--destination', placed(request['destination']), '--controls', request['controls']['path'],
                 '--controls-sha256', request['controls']['sha256'],
                 '--controller-binding-path', placed(request['controller_binding_path'])]
    elif mode == 'successor':
        _require(request['controls'] is None and request['helper_journal_root'] is None
                 and not borrowed, 'Successor requires the exact previous configuration and no helper channels')
        config = original(request['configuration'])
        controls = original(config['controls'])
        for key in ('destination', 'capsules'):
            placed(config[key]['path'])
        placed(config['controller_binding_path'])
        args += ['--configuration', request['configuration']['path'],
                 '--configuration-sha256', request['configuration']['sha256'],
                 '--destination', placed(request['destination']),
                 '--controller-binding-path', placed(request['controller_binding_path'])]
    else:
        _require(all(request[k] is None for k in ('controls', 'destination', 'controller_binding_path')),
                 'Preparation/startup requires only the original configuration')
        config = original(request['configuration'])
        controls = original(config['controls'])
        for key in ('destination', 'capsules'):
            placed(config[key]['path'])
        placed(config['controller_binding_path'])
        args += ['--configuration', request['configuration']['path'],
                 '--configuration-sha256', request['configuration']['sha256']]
        if mode == 'prepare':
            _require(request['helper_journal_root'] is not None, 'Original owned preparation helper required')
        if request['helper_journal_root'] is not None:
            args += ['--helper-journal-root', placed(request['helper_journal_root'])]
    _require(controls['kind'] == 'native_root_roster_custody_v1'
             and type(controls['worker_uid']) is int and controls['worker_uid'] == 10001,
             'Original separate root/worker custody required')
    # Follow only live schema relationships, never arbitrary historic embedded pins.
    for key in ('root', 'records', 'lock', 'initial_checkpoint'):
        placed(controls[key]['path'])
    for key in ('accepted_roots', 'accepted_inventory', 'accepted_pins'):
        for item in controls[key]:
            placed(item['path'])
    for item in (controls.get('controller_controls') or {}).values():
        if item is not None:
            placed(item['path'])
    retention = original(controls['retention_binding'])
    for key in ('mutable_root', 'lock', 'accepted_pointer', 'initial_checkpoint'):
        placed(retention[key]['path'])
    placed(retention['store_root'])
    for item in retention['directories'].values():
        placed(item['path'])
    for suffix in ('', 'data', 'processed', 'models', 'logs'):
        placed(str(Path(retention['mutable_root']['path']) / 'launcher-runtime' / suffix))
    if request['recovery_evidence'] is not None:
        original(request['recovery_evidence'])
        args += ['--recovery-evidence', request['recovery_evidence']['path'],
                 '--recovery-evidence-sha256', request['recovery_evidence']['sha256']]
    if borrowed:
        args += ['--helper-read-fd', str(helper_read_fd), '--helper-write-fd', str(helper_write_fd)]
    return args


def dispatch(arguments):
    """Delegate once to the existing guarded launcher and retain its exit result."""
    from scripts.launch_roster_hosted_server import main as launch
    return launch(arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package-manifest', 'package-sha256', 'request', 'request-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--helper-read-fd', type=int)
    parser.add_argument('--helper-write-fd', type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        _require(sys.platform == 'linux' and os.geteuid() == 0 and sys.version_info[:3] == (3, 11, 16)
                 and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
                 'Native root packaged CPython3.11.16 with -I -S -B required')
        root, manifest = verify_package(args.package_manifest, args.package_sha256)
        _require(Path(sys.executable) == root / PATHS['python']
                 and Path(__file__) == root / 'source/scripts/roster_railway_startup.py',
                 'Must start the exact packaged interpreter and bootstrap')
        python_root = root / 'python'
        _require(Path(sys.prefix) == python_root and Path(sys.base_prefix) == python_root
                 and all(python_root in Path(value).parents for value in sys.path),
                 'Packaged standard-library search paths required')
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and module is not sys.modules[__name__]:
                _require(python_root in Path(path).parents,
                         'Unexpected preloaded module outside packaged standard library')
        os.environ.clear()
        os.environ.update(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHON_DOTENV_DISABLED='1',
            PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', UFC_REFRESH_ENABLED='0',
            UFC_MONTHLY_CANDIDATE_ENABLED='0', UFC_UPDATE_CALENDAR_ENABLED='0',
            LIVE_TRADING_MODE='off', LIVE_TRADING_ARMED='0')
        sys.path[:0] = [str(root / PATHS['source']), str(root / PATHS['dependencies'])]
        from src.data.roster_railway_paths import persistent_path
        persistent_path(args.request)
        request = json.loads(_pin(args.request, args.request_sha256))
        arguments = build_arguments(root, manifest, request,
            helper_read_fd=args.helper_read_fd, helper_write_fd=args.helper_write_fd)
        if args.check_only:
            print(json.dumps({'status': 'package_and_placement_checked_only', 'arguments': arguments,
                'semantic_authority_validated': False, 'workflow_started': False}))
            return 0
        return dispatch(arguments)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc), 'activation_authorized': False}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
