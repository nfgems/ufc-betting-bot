"""Resume exact accepted-baseline materialization without overwriting partial evidence."""
from pathlib import Path
import os
import uuid


def _boundary(name: str) -> None:
    """Interruption seam for the local acceptance tests."""


def advance(cycle_root, *, integration_root, store, publication_pin, manifest_pin, accepted_raw_root):
    from src.data import roster_refresh_integration as bridge
    from src.data import roster_hosted_candidate as candidate
    from src.data.roster_hosted_store import _safe
    from src.data.roster_refresh_files import publish_new

    cycle_root, accepted_raw_root = _safe(cycle_root), _safe(accepted_raw_root)
    target = _safe(cycle_root / bridge.BASELINE_FILE)
    publication, manifest = bridge._load(publication_pin), bridge._load(manifest_pin)
    generation = str(publication['accepted_generation_id'])
    root = _safe(Path(integration_root) / 'accepted' / f'{generation}-{cycle_root.name}')
    seed = _safe(root / 'seed')

    def exact(path, content):
        path = _safe(path)
        if path.exists():
            bridge._require(path.is_file() and path.read_bytes() == content,
                            f"partial baseline bytes conflict: {path}")
            return bridge.file_pin(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = _safe(cycle_root / 'baseline-write-attempts')
        attempts.mkdir(exist_ok=True)
        temporary = attempts / f'{uuid.uuid4().hex}.tmp'
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        _boundary('before_baseline_file_link')
        publish_new(temporary, path)
        _boundary('after_baseline_file_publish')
        bridge._fsync_directory(path.parent)
        bridge._fsync_directory(attempts)
        return bridge.file_pin(path)

    def save(path, value):
        return exact(path, bridge._json_bytes(value))

    if target.is_file():
        record = bridge._load_path(target)
        bridge._require(record.get('schema') == bridge.SCHEMA and record.get('kind') == bridge.BASELINE_KIND,
                        'completed baseline schema changed')
        bridge._require(record['publication'] == publication_pin, 'baseline belongs to another publication')
        bridge._require(record['root'] == str(root) and record['seed_root'] == str(seed)
                        and record['cycle_id'] == cycle_root.name and record['accepted_generation_id'] == generation,
                        'completed baseline destination changed')
        intent = bridge._load(record['materialization_intent'])
        bridge._require(intent.get('schema') == 1 and intent.get('kind') == 'roster_refresh_baseline_intent_v1',
                        'completed baseline intent schema changed')
        bridge._require(intent['binding'] == {'publication': publication_pin, 'retained_inputs': manifest_pin,
                        'root': record['root'], 'source_raw_root': str(accepted_raw_root)}, 'completed baseline intent changed')
        bridge._require(intent['store_head'] == record['store_head'], 'completed baseline accounting changed')
        checkpoint = bridge._load(record['checkpoint'])
        bridge._require(checkpoint['store_root'] == store.root.as_posix(), 'completed baseline store changed')
        for key in ('seed_inventory', 'accepted_input_manifest', 'accepted_pointer', 'checkpoint'):
            bridge._load(record[key])
        for pin in record['input_pins'].values():
            bridge._load(pin)
        pointer = bridge._load(record['accepted_pointer'])
        inputs = bridge._load(record['accepted_input_manifest'])
        bridge._require(pointer['input_manifest'] == record['accepted_input_manifest']
                        and inputs['input_pins'] == record['input_pins']
                        and bridge._load(record['input_pins']['owners']) == {'schema': 1, 'seed_root': str(seed),
                                                                          'seed_inventory': record['seed_inventory']}
                        and checkpoint['accepted_pointer'] == record['accepted_pointer']
                        and checkpoint['store_head'] == record['store_head'], 'completed baseline reference linkage changed')
        bridge._require(record['input_pins']['membership'] == manifest['input_pins']['membership']
                        and record['input_pins']['release'] == manifest['release'], 'completed baseline input binding changed')
        bridge._require(candidate.tree_inventory(Path(record['seed_root'])) == bridge._load(record['seed_inventory']),
                        'completed baseline seed changed')
        catalog_relative = f'{bridge.ROSTERWATCH_SUBDIR}/{bridge.IDENTITY_CAPTURES_FILE}'
        expected_seed = {name: value for name, value in intent['source_files'].items() if name != catalog_relative}
        bridge._require(bridge._tree_files(seed) == expected_seed, 'completed baseline differs from original intent')
        if catalog_relative in intent['source_files']:
            bridge._require(bridge._identity(Path(record['input_pins']['catalog']['path'])) == intent['source_files'][catalog_relative],
                            'completed baseline catalog differs from original intent')
        else:
            bridge._require(record['input_pins']['catalog'] == manifest['catalog'], 'completed baseline catalog binding changed')
        return {'baseline': bridge.file_pin(target), 'existing': True, **record}

    intent_path = cycle_root / 'accepted-baseline-intent.json'
    binding = {'publication': publication_pin, 'retained_inputs': manifest_pin,
               'root': str(root), 'source_raw_root': str(accepted_raw_root)}
    if intent_path.exists():
        intent = bridge._load_path(intent_path)
        bridge._require(intent.get('schema') == 1 and intent.get('kind') == 'roster_refresh_baseline_intent_v1',
                        'baseline intent schema changed')
        bridge._require(intent.get('binding') == binding, 'baseline intent binding changed')
    else:
        bridge._require(not root.exists(), 'unbound existing baseline destination')
        source_files = {bridge.ROSTER_FILE: bridge._identity(accepted_raw_root / bridge.ROSTER_FILE)}
        state = accepted_raw_root / bridge.ROSTERWATCH_SUBDIR
        if state.is_dir():
            source_files.update({f'{bridge.ROSTERWATCH_SUBDIR}/{name}': identity
                                 for name, identity in bridge._tree_files(state).items()})
        intent = {'schema': 1, 'kind': 'roster_refresh_baseline_intent_v1', 'binding': binding,
                  'source_files': source_files, 'store_head': store.snapshot()['head']}
        # Pin the complete original work before creating its destination. A later
        # store head or changed accepted input cannot silently redefine a retry.
        save(intent_path, intent)
    bridge._require(not root.is_symlink() and not seed.is_symlink(), 'baseline destination symlink')
    root.mkdir(parents=True, exist_ok=True)
    seed.mkdir(exist_ok=True)
    bridge._fsync_directory(root.parent)
    bridge._fsync_directory(root)
    _boundary('after_seed_creation')
    copied = []
    catalog_relative = f'{bridge.ROSTERWATCH_SUBDIR}/{bridge.IDENTITY_CAPTURES_FILE}'
    for relative, identity in sorted(intent['source_files'].items()):
        source = _safe(accepted_raw_root / relative)
        bridge._require(not source.is_symlink() and bridge._identity(source) == identity,
                        f'accepted baseline input changed: {relative}')
        content = source.read_bytes()
        bridge._require({'bytes': len(content), 'sha256': bridge._sha(content)} == identity,
                        f'accepted baseline input changed while reading: {relative}')
        if relative == catalog_relative:
            exact(root / 'identity-catalog.json', content)
        else:
            exact(seed / relative, content)
            copied.append(relative)
        _boundary('after_seed_file')
    expected = {name: value for name, value in intent['source_files'].items() if name != catalog_relative}
    bridge._require(bridge._tree_files(seed) == expected, 'partial baseline contains unexpected files')
    for path in sorted(seed.rglob('*')):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(seed, 0o555)
    inventory = save(root / 'seed-inventory.json', candidate.tree_inventory(seed))
    catalog = bridge.file_pin(root / 'identity-catalog.json') if catalog_relative in intent['source_files'] else manifest['catalog']
    owners = save(root / 'owners.json', {'schema': 1, 'seed_root': str(seed), 'seed_inventory': inventory})
    inputs = {'owners': owners, 'membership': manifest['input_pins']['membership'],
              'catalog': catalog, 'release': manifest['release']}
    input_manifest = save(root / 'accepted-input-manifest.json', {'schema': 1, 'input_pins': inputs})
    head = intent['store_head']
    pointer = save(root / 'accepted-pointer.json', {'schema': 1, 'kind': 'accepted_roster_pointer_v1',
                   'input_manifest': input_manifest, 'store_manifest_sha256': head['manifest_sha256']})
    checkpoint = save(root / 'checkpoint.json', {'schema': 1, 'kind': 'roster_input_checkpoint_v1',
                      'accepted_pointer': pointer, 'store_root': store.root.as_posix(), 'store_head': head})
    _boundary('before_baseline_receipt')
    record = {'schema': bridge.SCHEMA, 'kind': bridge.BASELINE_KIND, 'cycle_id': cycle_root.name,
              'accepted_generation_id': generation, 'root': str(root), 'seed_root': str(seed),
              'seed_files': copied, 'seed_inventory': inventory, 'input_pins': inputs,
              'accepted_input_manifest': input_manifest, 'accepted_pointer': pointer,
              'expected_pointer_sha256': pointer['sha256'], 'expected_release_sha256': manifest['expected_release_sha256'],
              'checkpoint': checkpoint, 'expected_checkpoint_sha256': checkpoint['sha256'], 'store_head': head,
              'adopted_by_trusted_configuration': False, 'publication': publication_pin,
              'materialization_intent': bridge.file_pin(intent_path)}
    pin = save(target, record)
    return {'baseline': pin, 'existing': False, **record}
