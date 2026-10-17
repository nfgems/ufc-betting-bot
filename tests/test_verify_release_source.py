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
        'status': 'declared',
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
    assert result['source_status'] == 'declared_overlay'
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
    ({'status': 'deployed'}, 'marked declared or undeployed'),
    ({'description': ''}, 'requires a description'),
    ({'files': {}}, 'requires declared files'),
    ({'files': {'src/example.py': {'bytes': -1, 'sha256': 'invalid'}}}, 'invalid source-change hash'),
    ({'files': {'src/example.py': _record(b'original source\n')}}, 'matches the deployed baseline'),
])
def test_source_change_metadata_is_validated(release, override, message):
    _changes(release, **override)
    with pytest.raises(ValueError, match=message):
        verify(release)


def test_explicit_addition_is_verified_without_changing_baseline(release):
    baseline = (release / 'release-source-manifest.json').read_bytes()
    new_source = b'def cashflow():\n    return None\n'
    (release / 'src' / 'cashflow.py').write_bytes(new_source)
    _changes(release, additions={'src/cashflow.py': _record(new_source)})
    result = verify(release)
    assert result['source_changes'] == ['src/cashflow.py', 'src/example.py']
    assert result['source_additions'] == ['src/cashflow.py']
    assert result['checked_files'] == 3
    (release / 'private').mkdir()
    (release / 'private' / 'provenance.json').write_bytes(b'private')
    assert verify(release, image=True)['checked_files'] == 4
    assert (release / 'release-source-manifest.json').read_bytes() == baseline


def test_addition_only_overlay_is_supported(release):
    new_source = b'new source\n'
    (release / 'src' / 'cashflow.py').write_bytes(new_source)
    _changes(release, files={}, additions={'src/cashflow.py': _record(new_source)})
    (release / 'src' / 'example.py').write_bytes(b'original source\n')
    assert verify(release)['source_changes'] == ['src/cashflow.py']


@pytest.mark.parametrize('damage', ['missing', 'changed', 'undeclared_sibling'])
def test_addition_does_not_weaken_exact_source_inventory(release, damage):
    new_source = b'new source\n'
    path = release / 'src' / 'cashflow.py'
    path.write_bytes(new_source)
    _changes(release, additions={'src/cashflow.py': _record(new_source)})
    if damage == 'missing':
        path.unlink()
        message = 'cashflow.py: missing'
    elif damage == 'changed':
        path.write_bytes(b'changed without updating pin')
        message = 'cashflow.py: bytes differ'
    else:
        (release / 'src' / 'another.py').write_bytes(new_source)
        message = 'another.py: unexpected source file'
    with pytest.raises(ValueError, match=message):
        verify(release)


@pytest.mark.parametrize('name', ['src/example.py', 'src/EXAMPLE.py', 'config/stable.json'])
def test_additions_cannot_replace_or_alias_baseline_files(release, name):
    _changes(release, additions={name: _record(b'new')})
    with pytest.raises(ValueError, match='replaces or aliases an existing file'):
        verify(release)


def test_additions_cannot_alias_each_other_by_case(release):
    _changes(release, additions={'src/New.py': _record(b'new'), 'src/new.py': _record(b'new')})
    with pytest.raises(ValueError, match='replaces or aliases an existing file'):
        verify(release)


@pytest.mark.parametrize('name', [
    'private/provenance.json', 'private/new.json', 'models/new.pkl', 'tests/new.py',
    'requirements.txt', 'src',
])
def test_additions_are_limited_to_verified_runtime_directories(release, name):
    _changes(release, additions={name: _record(b'new')})
    with pytest.raises(ValueError, match='within src, scripts or config'):
        verify(release)


@pytest.mark.parametrize('name', [
    '../new.py', '/src/new.py', 'src/../new.py', 'src\\new.py',
    'src//new.py', 'src/./new.py', 'C:/src/new.py', 'src/new.py:stream',
    'src/new.py.', 'src/new.py ',
])
def test_additions_reject_escaping_and_noncanonical_paths(release, name):
    _changes(release, additions={name: _record(b'new')})
    with pytest.raises(ValueError, match='path escapes or aliases'):
        verify(release)


@pytest.mark.parametrize('outside', [False, True])
def test_additions_reject_symlink_aliases(release, outside):
    target = ((release.parent / (release.name + '-outside.py')) if outside
              else (release / 'src' / 'example.py'))
    if outside:
        target.write_bytes(b'outside source')
    link = release / 'src' / 'linked.py'
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip('This platform does not permit creating a test symlink')
    _changes(release, additions={'src/linked.py': _record(target.read_bytes())})
    with pytest.raises(ValueError, match='path escapes or aliases'):
        verify(release)


@pytest.mark.parametrize('record', [
    {'bytes': True, 'sha256': 'a' * 64},
    {'bytes': 1, 'sha256': 'invalid'},
    {'bytes': 1, 'sha256': 'a' * 64, 'provided_by': 'pinned_base_image'},
])
def test_addition_hash_records_are_exact_and_repository_owned(release, record):
    _changes(release, additions={'src/new.py': record})
    with pytest.raises(ValueError, match='invalid source-change hash record'):
        verify(release)


def test_addition_cannot_replace_runtime_inherited_file(release):
    baseline_path = release / 'release-source-manifest.json'
    baseline = json.loads(baseline_path.read_bytes())
    baseline['files']['config/inherited.json'] = _record(b'private', provided_by='pinned_base_image')
    baseline_path.write_text(json.dumps(baseline), encoding='utf-8')
    _changes(release, additions={'config/inherited.json': _record(b'changed')})
    with pytest.raises(ValueError, match='replaces or aliases an existing file'):
        verify(release)


def test_undeclared_directory_symlink_cannot_hide_runtime_files(release):
    target = release / 'hidden'
    target.mkdir()
    (target / 'hidden.py').write_bytes(b'not declared')
    try:
        (release / 'src' / 'hidden').symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('This platform does not permit creating a test symlink')
    with pytest.raises(ValueError, match='source symlink is not permitted'):
        verify(release)
