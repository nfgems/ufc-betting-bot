"""Verify saved deployed source without importing or starting the bot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def verify(root: Path, *, image: bool = False) -> dict:
    root = root.resolve()
    manifest = json.loads((root / 'release-source-manifest.json').read_bytes())
    expected = manifest['files']
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
            errors.append(name + ': bytes differ from saved deployed release')
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
            'reference_image': manifest['reference_image']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, image=args.image)))
