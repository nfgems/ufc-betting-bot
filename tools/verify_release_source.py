"""Verify source bytes; deployment status requires a separate release receipt."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def _source_path(root: Path, name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or '..' in path.parts
            or '\\' in name or ':' in name
            or any(part.rstrip(' .') != part for part in path.parts)
            or (root / name).resolve() != root / name
            or not (root / name).resolve().is_relative_to(root)):
        raise ValueError('Source-change manifest path escapes or aliases the source root')
    return path


def _hash_record(name: str, record: dict) -> None:
    if (not isinstance(record, dict) or set(record) != {'bytes', 'sha256'}
            or type(record['bytes']) is not int or record['bytes'] < 0
            or not isinstance(record['sha256'], str) or len(record['sha256']) != 64
            or any(char not in '0123456789abcdef' for char in record['sha256'])):
        raise ValueError(name + ': invalid source-change hash record')


def _source_changes(root: Path, manifest: dict, manifest_bytes: bytes) -> tuple[dict, dict]:
    path = root / 'source-change-manifest.json'
    if not path.is_file():
        return {}, {}
    changes = json.loads(path.read_bytes())
    if type(changes.get('schema_version')) is not int or changes['schema_version'] != 1:
        raise ValueError('Source-change manifest requires schema_version 1')
    if changes.get('base_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Source-change manifest does not match the saved release manifest')
    if changes.get('status') not in {'undeployed', 'declared'}:
        raise ValueError('Source changes must be marked declared or undeployed, never deployed')
    if not isinstance(changes.get('description'), str) or not changes['description'].strip():
        raise ValueError('Source-change manifest requires a description')
    files = changes.get('files')
    additions = changes.get('additions', {})
    if not isinstance(files, dict) or not isinstance(additions, dict) or not (files or additions):
        raise ValueError('Source-change manifest requires declared files')
    for name, record in files.items():
        _source_path(root, name)
        baseline = manifest['files'].get(name)
        if baseline is None or baseline['provided_by'] != 'repository':
            raise ValueError(name + ': source changes require an existing repository file')
        _hash_record(name, record)
        if record['bytes'] == baseline['bytes'] and record['sha256'] == baseline['sha256']:
            raise ValueError(name + ': declared source change matches the deployed baseline')
    known_names = {name.casefold() for name in manifest['files']}
    for name, record in additions.items():
        path = _source_path(root, name)
        if len(path.parts) < 2 or path.parts[0] not in {'src', 'scripts', 'config'}:
            raise ValueError(name + ': source additions must be within src, scripts or config')
        if name.casefold() in known_names:
            raise ValueError(name + ': source addition replaces or aliases an existing file')
        known_names.add(name.casefold())
        _hash_record(name, record)
    return files, additions


def verify(root: Path, *, image: bool = False) -> dict:
    root = root.resolve()
    manifest_bytes = (root / 'release-source-manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    replacements, additions = _source_changes(root, manifest, manifest_bytes)
    changes = {**replacements, **additions}
    expected = dict(manifest['files'])
    expected.update({name: dict(record, provided_by='repository')
                     for name, record in changes.items()})
    errors, inherited = [], []
    checked = 0
    for name, record in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes source root')
        if record['provided_by'] == 'pinned_base_image' and not image:
            inherited.append(name)
            continue
        if not path.is_file():
            errors.append(name + ': missing')
            continue
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            declared_source = 'declared source overlay' if name in changes else 'saved deployed release'
            errors.append(name + ': bytes differ from ' + declared_source)
        checked += 1
    for directory in ('src', 'scripts', 'config'):
        for path in (root / directory).rglob('*'):
            if path.is_symlink():
                errors.append(path.relative_to(root).as_posix() + ': source symlink is not permitted')
                continue
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.relative_to(root).as_posix() not in expected:
                errors.append(path.relative_to(root).as_posix() + ': unexpected source file')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'passed': True, 'checked_files': checked, 'inherited_only': inherited,
            'reference_image': manifest['reference_image'],
            'source_status': 'declared_overlay' if changes else 'saved_deployed_release',
            'source_changes': sorted(changes), 'source_additions': sorted(additions)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, image=args.image)))
