import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('refresh_seed', Path(__file__).parents[1] / 'scripts/install_incremental_seed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    seed, target = tmp_path / 'seed', tmp_path / 'target'
    seed.mkdir()
    records = []
    for name, body in [('data/accepted.json', b'{"generation":"new"}\n'), ('data/generation/fights.csv', b'original\naddition\n')]:
        path = seed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        records.append({'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'files': records}))
    return seed, manifest, target


def test_seed_restart_preserves_newer_runtime_progress(tmp_path):
    seed, manifest, target = fixture(tmp_path)
    assert module.install(seed, manifest, target) == 'installed_validated_seed'
    current = target / 'data/accepted.json'
    current.write_bytes(b'{"generation":"later"}\n')
    assert module.install(seed, manifest, target) == 'existing_saved_state'
    assert current.read_bytes() == b'{"generation":"later"}\n'


def test_interrupted_copy_resumes_without_replacing_valid_partial(tmp_path):
    seed, manifest, target = fixture(tmp_path)
    target.mkdir()
    module.atomic_json(target / '.seed-install.json', {'manifest_sha256': module.digest(manifest), 'state': 'copying'})
    partial = target / 'data/accepted.json'
    partial.parent.mkdir()
    partial.write_bytes((seed / 'data/accepted.json').read_bytes())
    assert module.install(seed, manifest, target) == 'installed_validated_seed'
    assert (target / 'data/generation/fights.csv').read_bytes() == b'original\naddition\n'


def test_existing_unseeded_progress_and_different_partials_are_preserved(tmp_path):
    seed, manifest, target = fixture(tmp_path)
    target.mkdir()
    current = target / 'service.json'
    current.write_bytes(b'working-progress')
    assert module.install(seed, manifest, target) == 'existing_saved_state'
    assert current.read_bytes() == b'working-progress'
    module.atomic_json(target / '.seed-install.json', {'manifest_sha256': module.digest(manifest), 'state': 'copying'})
    partial = target / 'data/accepted.json'
    partial.parent.mkdir()
    partial.write_bytes(b'different-progress')
    with pytest.raises(ValueError, match='Saved seed partial differs'):
        module.install(seed, manifest, target)
    assert partial.read_bytes() == b'different-progress'


def test_kill_before_initial_marker_rename_keeps_partial_and_installs_seed(tmp_path, monkeypatch):
    seed, manifest, target = fixture(tmp_path)
    original = module.os.replace
    def interrupted(source, destination):
        if Path(destination).name == '.seed-install.json':
            raise KeyboardInterrupt('before initial marker rename')
        return original(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, 'replace', interrupted)
        with pytest.raises(KeyboardInterrupt):
            module.install(seed, manifest, target)
    temporary = list(target.glob('.seed-install.json.*.pending'))
    assert len(temporary) == 1
    partial = temporary[0].read_bytes()
    assert module.install(seed, manifest, target) == 'installed_validated_seed'
    assert temporary[0].read_bytes() == partial
    assert (target / 'data/generation/fights.csv').read_bytes() == b'original\naddition\n'
