import hashlib
import json

import pytest

from tools.verify_release_source import verify


def _record(data, *, provided_by=None):
    result = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if provided_by is not None:
        result['provided_by'] = provided_by
    return result


@pytest.fixture
def release(tmp_path):
    original = b'original source\n'
    stable = b'{"stable": true}\n'
    (tmp_path / 'src').mkdir()
    (tmp_path / 'config').mkdir()
    (tmp_path / 'src' / 'example.py').write_bytes(original)
    (tmp_path / 'config' / 'stable.json').write_bytes(stable)
    manifest = {
        'reference_image': 'example.invalid/image@sha256:fixture',
        'files': {
            'src/example.py': _record(original, provided_by='repository'),
            'config/stable.json': _record(stable, provided_by='repository'),
            'private/provenance.json': _record(b'private', provided_by='pinned_base_image'),
        },
    }
    baseline = tmp_path / 'release-source-manifest.json'
    baseline.write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path


def _changes(root, **overrides):
    data = b'changed source\n'
    (root / 'src' / 'example.py').write_bytes(data)
    manifest = {
        'schema_version': 1,
        'base_manifest_sha256': hashlib.sha256(
            (root / 'release-source-manifest.json').read_bytes()).hexdigest(),
        'status': 'undeployed',
        'description': 'Synthetic source change',
        'files': {'src/example.py': _record(data)},
    }
    manifest.update(overrides)
    (root / 'source-change-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def test_saved_release_is_verified_without_change_manifest(release):
    result = verify(release)
    assert result['source_status'] == 'saved_deployed_release'
    assert result['source_changes'] == []
    assert result['inherited_only'] == ['private/provenance.json']


def test_declared_changes_preserve_baseline_and_private_image_checks(release):
    baseline = (release / 'release-source-manifest.json').read_bytes()
    _changes(release)
    result = verify(release)
    assert result['source_status'] == 'undeployed'
    assert result['source_changes'] == ['src/example.py']
    assert result['reference_image'] == 'example.invalid/image@sha256:fixture'
    assert (release / 'release-source-manifest.json').read_bytes() == baseline
    with pytest.raises(ValueError, match='private/provenance.json: missing'):
        verify(release, image=True)
    (release / 'private').mkdir()
    (release / 'private' / 'provenance.json').write_bytes(b'private')
    assert verify(release, image=True)['inherited_only'] == []


@pytest.mark.parametrize('with_overlay', [False, True])
def test_undeclared_source_mutations_are_rejected(release, with_overlay):
    if with_overlay:
        _changes(release)
    (release / 'config' / 'stable.json').write_bytes(b'changed without declaration')
    with pytest.raises(ValueError, match='config/stable.json: bytes differ'):
        verify(release)


def test_overlay_does_not_hide_unexpected_source(release):
    _changes(release)
    (release / 'src' / 'unexpected.py').write_bytes(b'unexpected')
    with pytest.raises(ValueError, match='src/unexpected.py: unexpected source file'):
        verify(release)


@pytest.mark.parametrize('name', ['../outside.py', 'src/../example.py', 'src\\example.py'])
def test_source_changes_reject_escaped_or_aliased_paths(release, name):
    _changes(release, files={name: _record(b'changed')})
    with pytest.raises(ValueError, match='path escapes or aliases'):
        verify(release)


@pytest.mark.parametrize('name', ['src/new.py', 'private/provenance.json'])
def test_source_changes_cannot_override_new_or_inherited_files(release, name):
    _changes(release, files={name: _record(b'changed')})
    with pytest.raises(ValueError, match='existing repository file'):
        verify(release)


def test_source_changes_are_bound_to_original_manifest(release):
    _changes(release)
    baseline = release / 'release-source-manifest.json'
    baseline.write_bytes(baseline.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='does not match the saved release manifest'):
        verify(release)


@pytest.mark.parametrize('override, message', [
    ({'schema_version': 2}, 'schema_version 1'),
    ({'status': 'deployed'}, 'explicitly marked undeployed'),
    ({'description': ''}, 'requires a description'),
    ({'files': {}}, 'requires declared files'),
    ({'files': {'src/example.py': {'bytes': -1, 'sha256': 'invalid'}}}, 'invalid source-change hash'),
    ({'files': {'src/example.py': _record(b'original source\n')}}, 'matches the deployed baseline'),
])
def test_source_change_metadata_is_validated(release, override, message):
    _changes(release, **override)
    with pytest.raises(ValueError, match=message):
        verify(release)
