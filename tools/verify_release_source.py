"""Verify the deployed source baseline and declared, undeployed source changes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def _source_changes(root: Path, manifest: dict, manifest_bytes: bytes) -> dict:
    path = root / 'source-change-manifest.json'
    if not path.is_file():
        return {}
    changes = json.loads(path.read_bytes())
    if type(changes.get('schema_version')) is not int or changes['schema_version'] != 1:
        raise ValueError('Source-change manifest requires schema_version 1')
    if changes.get('base_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Source-change manifest does not match the saved release manifest')
    if changes.get('status') != 'undeployed':
        raise ValueError('Source changes must be explicitly marked undeployed')
    if not isinstance(changes.get('description'), str) or not changes['description'].strip():
        raise ValueError('Source-change manifest requires a description')
    files = changes.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Source-change manifest requires declared files')
    for name, record in files.items():
        source_path = Path(name)
        if (source_path.is_absolute() or '..' in source_path.parts or '\\' in name
                or not (root / source_path).resolve().is_relative_to(root)):
            raise ValueError('Source-change manifest path escapes or aliases the source root')
        baseline = manifest['files'].get(name)
        if baseline is None or baseline['provided_by'] != 'repository':
            raise ValueError(name + ': source changes require an existing repository file')
        if (not isinstance(record, dict) or set(record) != {'bytes', 'sha256'}
                or type(record['bytes']) is not int or record['bytes'] < 0
                or not isinstance(record['sha256'], str) or len(record['sha256']) != 64
                or any(char not in '0123456789abcdef' for char in record['sha256'])):
            raise ValueError(name + ': invalid source-change hash record')
        if record['bytes'] == baseline['bytes'] and record['sha256'] == baseline['sha256']:
            raise ValueError(name + ': declared source change matches the deployed baseline')
    return files


def verify(root: Path, *, image: bool = False) -> dict:
    root = root.resolve()
    manifest_bytes = (root / 'release-source-manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    changes = _source_changes(root, manifest, manifest_bytes)
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
            declared_source = 'declared undeployed source' if name in changes else 'saved deployed release'
            errors.append(name + ': bytes differ from ' + declared_source)
        checked += 1
    for directory in ('src', 'scripts', 'config'):
        for path in (root / directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.relative_to(root).as_posix() not in expected:
                errors.append(path.relative_to(root).as_posix() + ': unexpected source file')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'passed': True, 'checked_files': checked, 'inherited_only': inherited,
            'reference_image': manifest['reference_image'],
            'source_status': 'undeployed' if changes else 'saved_deployed_release',
            'source_changes': sorted(changes)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, image=args.image)))
