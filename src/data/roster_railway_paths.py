"""Read-only placement checks for the explicitly reserved Railway volume branch.

Mount and ownership observations are local admission evidence, not continuity,
accounting, provider volume identity, recovery or durability authority.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

RESERVED = Path('/app/logs/roster_refresh')
VOLUME = Path('/app/logs')


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _canonical(value):
    raw = os.fspath(value)
    path = Path(raw)
    _require(path.is_absolute() and raw == path.as_posix()
             and raw == os.path.normpath(raw), 'Canonical absolute refresh path required')
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode), 'Refresh path contains a symbolic link')
    return path


def _mounts(text):
    rows = []
    escapes = {'040': ' ', '011': '\t', '012': '\n', '134': '\\'}
    for line in text.splitlines():
        left, right = line.split(' - ', 1)
        fields, tail = left.split(), right.split()
        decode = lambda s: re.sub(r'\\(040|011|012|134)', lambda m: escapes[m[1]], s)
        rows.append((int(fields[0]), fields[2], decode(fields[3]),
                     Path(decode(fields[4])), tail[0]))
    return rows


def persistent_path(value):
    """Require exact /app/logs mount placement, including future child paths.

The reservation and its ancestry must already exist. Worker descendants retain
their original separate ownership; role-specific APIs validate their identities.
Nested mounts are refused even when they share a device with the volume.
"""
    path = _canonical(value)
    _require(path == RESERVED or RESERVED in path.parents,
             'Refresh role path must stay in /app/logs/roster_refresh')
    for parent in (Path('/'), Path('/app'), VOLUME, RESERVED):
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0,
                 'Refresh reservation ancestry must be native root owned')
        _require(not info.st_mode & 0o022 or
                 (parent == VOLUME and bool(info.st_mode & stat.S_ISVTX)),
                 'Refresh reservation ancestry permits replacement')
    rows = _mounts(Path('/proc/self/mountinfo').read_text())
    _require(not any(row[3] == RESERVED or RESERVED in row[3].parents for row in rows),
             'Nested mount inside the protected refresh reservation')
    volumes = [row for row in rows if row[3] == VOLUME]
    _require(len(volumes) == 1, 'Exact unambiguous /app/logs mount required')
    volume = volumes[0]
    _require(volume[4] not in {'9p', 'drvfs', 'tmpfs', 'ramfs'},
             'Native nonvolatile refresh volume required')
    covering = [row for row in rows if path == row[3] or row[3] in path.parents]
    deepest = max(len(row[3].parts) for row in covering)
    _require([row for row in covering if len(row[3].parts) == deepest] == [volume],
             'Refresh role path leaves the original log mount through a nested mount')
    existing = path
    while not existing.exists():
        existing = existing.parent
    device = existing.stat().st_dev
    _require(volume[1] == str(os.major(device)) + ':' + str(os.minor(device)),
             'Refresh native device disagrees with its covering mount')
    return path
