"""Bind runtime predictions to a reviewed immutable input generation.

This module only reads saved files. Live observations are separately scoped
read-only inputs; source collection and release installation belong elsewhere.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from src.data.selected_source_inputs import SelectedSourceInputs, build_selected_fight_features


class SelectedRuntimeInputError(ValueError):
    """The selected release cannot safely supply this prediction."""


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pinned_file(root, item):
    if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
        raise SelectedRuntimeInputError('Incomplete selected runtime file pin')
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root):
        raise SelectedRuntimeInputError('Immutable selected runtime file escapes the release root')
    if not path.is_file() or _hash(path) != item['sha256']:
        raise SelectedRuntimeInputError(f'Selected runtime file changed or is absent: {path.name}')
    if item.get('bytes') is not None and path.stat().st_size != item['bytes']:
        raise SelectedRuntimeInputError(f'Selected runtime file size changed: {path.name}')
    return path


@dataclass(frozen=True)
class RuntimeSelectedInputs:
    sources: SelectedSourceInputs
    binding: dict
    manifest_path: Path

    @classmethod
    def from_summary(cls, summary, *, model_result, no_odds_result):
        """Load an explicitly pinned binding; legacy summaries remain unchanged."""
        summary = summary or {}
        raw_path = summary.get('selected_runtime_manifest_path')
        expected_hash = summary.get('selected_runtime_manifest_sha256')
        if not raw_path and not expected_hash:
            if summary.get('selected_candidate_schema_version') is not None:
                raise SelectedRuntimeInputError('Selected candidate has no selected runtime binding')
            return None
        if not raw_path or not expected_hash:
            raise SelectedRuntimeInputError('Selected runtime manifest needs its path and SHA256')
        path = Path(raw_path).resolve()
        try:
            body = path.read_bytes()
            actual_hash = hashlib.sha256(body).hexdigest()
            if actual_hash != expected_hash:
                raise SelectedRuntimeInputError('Selected runtime manifest hash mismatch')
            document = json.loads(body)
            if document.get('schema_version') != 1 or not str(document.get('release_id') or '').strip():
                raise SelectedRuntimeInputError('Unsupported selected runtime manifest')
            if summary.get('selected_release_id') != document['release_id']:
                raise SelectedRuntimeInputError('Selected release identity mismatch')
            root = path.parent
            selection_path = _pinned_file(root, document['selection'])
            processed = document['processed']
            processed_root = (root / processed['directory']).resolve()
            if not processed_root.is_relative_to(root):
                raise SelectedRuntimeInputError('Selected processed directory escapes the release root')
            if set(processed['files']) != {'features.csv', 'fights_cleaned.csv'}:
                raise SelectedRuntimeInputError('Selected processed pair is incomplete')
            for name, item in processed['files'].items():
                if _pinned_file(root, item) != processed_root / name:
                    raise SelectedRuntimeInputError('Selected processed filename mismatch')
            if set(document['models']) != {'primary', 'no_odds', 'logistic'}:
                raise SelectedRuntimeInputError('Selected runtime requires all three model roles')
            models = {role: _pinned_file(root, item) for role, item in document['models'].items()}
            for role, loaded in (('primary', model_result), ('no_odds', no_odds_result)):
                if not isinstance(loaded, dict) or not loaded.get('artifact_path'):
                    raise SelectedRuntimeInputError(f'Selected {role} model is not loaded')
                if Path(loaded['artifact_path']).resolve() != models[role]:
                    raise SelectedRuntimeInputError(f'Selected {role} model role/path mismatch')
            runtime_processed_root = Path(summary['processed_dir']).resolve()
            for name, item in processed['files'].items():
                if _hash(runtime_processed_root / name) != item['sha256']:
                    raise SelectedRuntimeInputError('Runtime lookup generation differs from selected history')
            sources = SelectedSourceInputs.from_manifest(selection_path, processed_data_dir=runtime_processed_root)
            for name, item in sources.files.items():
                if not item.get('absent') and not sources.path(name).resolve().is_relative_to(root):
                    raise SelectedRuntimeInputError(f'Selected source escapes release root: {name}')
            observations = document['observations']
            if set(observations) != {'method_observations', 'rankings_observations', 'line_history'}:
                raise SelectedRuntimeInputError('Explicit runtime observation roots are incomplete')
            from src.config import DATA_DIR
            observation_paths = {}
            for name, value in observations.items():
                expanded = str(value).replace('${DATA_DIR}', str(DATA_DIR))
                if '${' in expanded:
                    raise SelectedRuntimeInputError('Unsupported observation path token')
                if not Path(expanded).is_absolute():
                    raise SelectedRuntimeInputError('Live observation roots must be explicit absolute paths')
                observation_paths[name] = Path(expanded).resolve()
            if any(not value.is_absolute() or value.resolve().is_relative_to(root)
                   for value in observation_paths.values()):
                raise SelectedRuntimeInputError('Live observations must be absolute paths outside the immutable release')
            sources = replace(sources, live_observation_paths=observation_paths)
            sources.verify()
            cutoff = date.fromisoformat(document['approved_history_cutoff']).isoformat()
            if cutoff != summary.get('processed_snapshot_max_event_date'):
                raise SelectedRuntimeInputError('Selected history cutoff differs from the processed bundle')
            binding = {
                'release_id': document['release_id'], 'manifest_sha256': actual_hash,
                'selection_sha256': document['selection']['sha256'],
                'processed_sha256': {name: item['sha256'] for name, item in processed['files'].items()},
                'model_sha256': {role: item['sha256'] for role, item in document['models'].items()},
                'approved_history_cutoff': cutoff,
                'observation_roots': {name: str(value) for name, value in observation_paths.items()},
            }
            from src.data.runtime_identity_bindings import reviewed_bindings_for_runtime
            identity_review = reviewed_bindings_for_runtime(sources, binding)
            if identity_review is not None:
                sources = replace(sources, runtime_identity_review=identity_review)
                binding['runtime_identity_review_sha256'] = identity_review.artifact_sha256
            return cls(sources, binding, path)
        except SelectedRuntimeInputError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SelectedRuntimeInputError(f'Selected runtime binding is invalid: {exc}') from exc

    @contextmanager
    def observations(self):
        self._verify_identity_review()
        with self.sources.activate():
            yield

    def _verify_identity_review(self):
        review = self.sources.runtime_identity_review
        if review is not None:
            try:
                review.verify_binding(self.binding)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SelectedRuntimeInputError(f'Selected live identity generation refused: {exc}') from exc

    def bind_live_contexts(self, contexts):
        """Apply this generation's reviewed identities after shared raw caches."""
        from copy import deepcopy
        review = self.sources.runtime_identity_review
        if review is None:
            raw = list(contexts)
            if any(context.get('runtime_identity_binding') is not None for context in raw):
                raise SelectedRuntimeInputError('Unreviewed generation cannot reuse runtime-bound contexts')
            return deepcopy(raw)
        self._verify_identity_review()
        try:
            return review.bind_contexts(self.sources, contexts)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SelectedRuntimeInputError(f'Selected live identity evidence refused: {exc}') from exc

    def build(self, **fight):
        self._verify_identity_review()
        if fight.get('as_of_date') is not None:
            raise SelectedRuntimeInputError('Live selected runtime does not accept historical replay cutoffs')
        try:
            event_day = date.fromisoformat(str(fight.get('event_date') or ''))
            if event_day <= date.fromisoformat(self.binding['approved_history_cutoff']):
                raise SelectedRuntimeInputError('Live event must follow the selected completed history; use an explicit replay')
            for name, expected in self.binding['processed_sha256'].items():
                if _hash(self.sources.processed_data_dir / name) != expected:
                    raise SelectedRuntimeInputError('Runtime lookup generation changed after release validation')
            fight = {**fight, 'include_provenance': True}
            payload = build_selected_fight_features(self.sources, **fight)
            features, provenance = payload
            for side in ('a', 'b'):
                value = features.get(f'{side}_num_fights')
                if value is None or not math.isfinite(float(value)) or float(value) < 0:
                    raise SelectedRuntimeInputError('Selected fighter history has no observed UFC fight count')
            provenance['selected_runtime_binding'] = dict(self.binding)
            return features, provenance
        except SelectedRuntimeInputError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SelectedRuntimeInputError(f'Selected prediction inputs refused: {exc}') from exc


def verified_selected_provenance(provenance, side):
    """Require adapter binding and the existing successful selected-history status."""
    binding = provenance.get('selected_runtime_binding')
    return (
        isinstance(binding, dict)
        and bool(binding.get('release_id'))
        and all(isinstance(binding.get(key), str) and len(binding[key]) == 64
                for key in ('manifest_sha256', 'selection_sha256'))
        and set(binding.get('processed_sha256') or {}) == {'features.csv', 'fights_cleaned.csv'}
        and set(binding.get('model_sha256') or {}) == {'primary', 'no_odds', 'logistic'}
        and provenance.get(f'fighter_{side}_history_scope') == 'selected_processed'
        and provenance.get(f'fighter_{side}_fight_history_status') == 'selected_history'
    )
