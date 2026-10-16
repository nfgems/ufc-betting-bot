"""Install a validated refresh seed once, resuming an interrupted file copy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def sync_directory(path):
    if os.name == 'nt':
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(path):
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        sync_directory(directory.parent)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def install(seed, manifest, target):
    seed, manifest, target = Path(seed).resolve(), Path(manifest).resolve(), Path(target).resolve()
    make_directory(target)
    sync_directory(target.parent)
    installed = target / '.seed-installed.json'
    pending = target / '.seed-install.json'
    if installed.is_file():
        return 'existing_saved_state'
    if not manifest.is_file():
        if any(target.iterdir()):
            return 'existing_saved_state'
        return 'no_seed'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or not isinstance(record.get('files'), list) or not record['files']:
        raise ValueError('Invalid refresh seed inventory')
    manifest_hash = digest(manifest)
    if pending.is_file():
        if json.loads(pending.read_text()).get('manifest_sha256') != manifest_hash:
            raise ValueError('Interrupted seed install belongs to a different seed')
    else:
        present = list(target.iterdir())
        if present:
            # A kill before the first marker rename leaves only its temporary.
            # Preserve that partial and finish installing from the same seed.
            first_marker_partials = all(
                item.is_file() and item.name.startswith(pending.name + '.')
                and item.name.endswith('.pending') for item in present)
            if not first_marker_partials:
                if ((target / 'service.json').is_file()
                        or (target / 'data' / 'accepted.json').is_file()
                        or (target / 'data' / 'active.json').is_file()):
                    return 'existing_saved_state'
                raise ValueError('Unrecognized saved state before seed installation')
        atomic_json(pending, {'manifest_sha256': manifest_hash, 'state': 'copying'})
    seen = set()
    for item in record['files']:
        relative = item['path']
        part = Path(relative)
        if (not isinstance(relative, str) or not relative or part.is_absolute()
                or '..' in part.parts or '\\' in relative or ':' in relative or relative in seen):
            raise ValueError('Unsafe or duplicate seed path')
        seen.add(relative)
        source, destination = seed / part, target / part
        if (source.is_symlink() or not source.is_file() or source.stat().st_size != item['bytes']
                or digest(source) != item['sha256']):
            raise ValueError('Refresh seed file mismatch: ' + relative)
        make_directory(destination.parent)
        if destination.exists():
            if (destination.is_symlink() or not destination.is_file()
                    or destination.stat().st_size != item['bytes'] or digest(destination) != item['sha256']):
                raise ValueError('Saved seed partial differs: ' + relative)
        else:
            temporary = destination.with_name(destination.name + '.seed-copy')
            with source.open('rb') as incoming, temporary.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if digest(temporary) != item['sha256']:
                raise ValueError('Refresh seed copy mismatch: ' + relative)
            os.replace(temporary, destination)
            sync_directory(destination.parent)
    atomic_json(installed, {'manifest_sha256': manifest_hash, 'files': len(seen), 'state': 'installed'})
    return 'installed_validated_seed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    print(install(args.seed, args.manifest, args.target))
