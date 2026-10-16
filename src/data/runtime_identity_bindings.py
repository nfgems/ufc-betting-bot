"""Saved, independently reviewed live-card bindings for one selected generation.

This supplements runtime identity evidence only. It never acquires a source,
changes selected files, mutates shared card rows, or changes historical matching.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.data.name_utils import normalize_person_name, normalize_ufcstats_id


ARTIFACT_PATH = Path(__file__).resolve().parents[2] / 'config/runtime_fighter_bindings_v1.json'
ARTIFACT_SHA256 = '41b67c414cb1c3039d640cfe647523d9f06c8f1b0753ac8f6701c50f143729d1'


class RuntimeIdentityBindingError(ValueError):
    """Reviewed runtime evidence or its selected generation is inconsistent."""


def _require(condition, reason):
    if not condition:
        raise RuntimeIdentityBindingError(reason)


def _athlete_url(value):
    """Normalize only the two official HTTPS hosts and an exact athlete route."""
    try:
        parsed = urlsplit(str(value or '').strip())
        if (parsed.scheme != 'https' or parsed.hostname not in {'ufc.com', 'www.ufc.com'}
                or parsed.username or parsed.password or parsed.port is not None
                or parsed.query or parsed.fragment):
            return None
        path = parsed.path.rstrip('/')
        if not re.fullmatch(r'/athlete/[a-z0-9]+(?:-[a-z0-9]+)*', path):
            return None
        return 'https://www.ufc.com' + path
    except ValueError:
        return None


@dataclass(frozen=True)
class ReviewedRuntimeIdentityBindings:
    artifact_path: Path
    artifact_sha256: str
    selection_sha256: str
    processed_fights_sha256: str
    manifest_sha256: str
    release_id: str

    def document(self):
        """Recheck evidence bytes so process reuse cannot hide evidence changes."""
        raw = self.artifact_path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == self.artifact_sha256,
                 'Runtime identity artifact hash changed')
        data = json.loads(raw)
        _require(data['schema_version'] == 1 and data['scope'] == 'reviewed_selected_runtime_only',
                 'Unsupported runtime identity review')
        _require(data['selection_sha256'] == self.selection_sha256
                 and data['processed_fights_sha256'] == self.processed_fights_sha256
                 and data['manifest_sha256'] == self.manifest_sha256
                 and data['release_id'] == self.release_id,
                 'Runtime identity review belongs to another selected generation')
        evidence_root = self.artifact_path.parent.resolve()
        for item in data['evidence_files']:
            path = (evidence_root / item['path']).resolve()
            _require(path.is_relative_to(evidence_root), 'Runtime evidence path escapes its package')
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
                     'Runtime identity source/review evidence hash changed')
        return data

    def verify_binding(self, binding):
        self.document()
        _require(binding.get('selection_sha256') == self.selection_sha256
                 and binding.get('manifest_sha256') == self.manifest_sha256
                 and binding.get('release_id') == self.release_id
                 and binding.get('processed_sha256', {}).get('fights_cleaned.csv') == self.processed_fights_sha256
                 and binding.get('runtime_identity_review_sha256') == self.artifact_sha256,
                 'Runtime identity binding generation changed')

    def verify_sources(self, sources):
        data = self.document()
        _require(sources.live_observation_paths is not None,
                 'Runtime identity review cannot be used for historical inputs')
        for name, expected in data['selected_sources_sha256'].items():
            _require(sources.files.get(name, {}).get('sha256') == expected,
                     'Runtime identity review selected source generation differs: ' + name)
        return data

    def extend_selected_names(self, names, sources):
        """Extend only the exact accepted alias, anchored to its selected owner."""
        data = self.verify_sources(sources)
        for alias in data['aliases']:
            identity = alias['fighter_id']
            canonical = normalize_person_name(alias['canonical_name'])
            name = normalize_person_name(alias['name'])
            _require(canonical in names.get(identity, set()),
                     'Runtime reviewed alias lacks its canonical selected owner')
            _require(not any(owner != identity and (name in values or canonical in values)
                             for owner, values in names.items()),
                     'Runtime reviewed alias conflicts with another selected owner')
            names[identity].add(name)

    def bind_contexts(self, sources, contexts):
        """Bind copies after raw cache retrieval; retain refusals and original IDs."""
        data = self.verify_sources(sources)
        sources.verify()
        with sources.path('active_roster').open(encoding='utf-8-sig', newline='') as stream:
            roster = list(csv.DictReader(stream))
        owners = {}
        for row in roster:
            url = _athlete_url(row.get('official_athlete_url'))
            if url:
                owners.setdefault(url, []).append(row)
        reviewed = {}
        for record in data['bindings']:
            url = _athlete_url(record['athlete_url'])
            _require(url and url not in reviewed, 'Duplicate or invalid reviewed athlete route')
            reviewed[url] = record
        withheld = {row['athlete_url']: row['reason'] for row in data['withheld']}
        bound = copy.deepcopy(list(contexts))
        with sources.activate():
            for context in bound:
                prior = context.get('runtime_identity_binding')
                if prior is not None:
                    _require(isinstance(prior, dict)
                             and prior.get('selection_sha256') == self.selection_sha256
                             and prior.get('manifest_sha256') == self.manifest_sha256
                             and prior.get('release_id') == self.release_id
                             and prior.get('review_sha256') == self.artifact_sha256,
                             'Cached context carries another runtime identity generation')
                    raise RuntimeIdentityBindingError('Context already carries runtime identity binding; original raw context required')
                provenance = {'selection_sha256': self.selection_sha256,
                              'manifest_sha256': self.manifest_sha256, 'release_id': self.release_id,
                              'review_sha256': self.artifact_sha256, 'sides': {}}
                for side in ('a', 'b'):
                    field = f'fighter_{side}_id'
                    original_id = context.get(field)
                    name = context.get(f'fighter_{side}', '')
                    raw_url = context.get(f'fighter_{side}_athlete_url')
                    result = {'submitted_name': name, 'submitted_id': original_id,
                              'submitted_athlete_url': raw_url}
                    reason = None
                    identity = normalize_ufcstats_id(original_id)
                    if raw_url:
                        url = _athlete_url(raw_url)
                        rows = owners.get(url, [])
                        record = reviewed.get(url)
                        # A separately reviewed current route may differ from the
                        # exact selected roster route. Only an explicit anchor to
                        # one selected owner can bridge that absence. Existing
                        # current-route owners always take precedence, including
                        # conflicting or ambiguous owners.
                        if url and not rows and record and 'selected_owner' in record:
                            anchor = record['selected_owner']
                            anchored_rows = [row for row in roster
                                if normalize_ufcstats_id(row.get('ufcstats_url')) == record['fighter_id']]
                            if (isinstance(anchor, dict)
                                    and set(anchor) == {'athlete_url', 'name', 'fighter_id'}
                                    and anchor['fighter_id'] == record['fighter_id']
                                    and len(anchored_rows) == 1
                                    and anchored_rows[0].get('official_name') == anchor['name'] == record['name']
                                    and anchored_rows[0].get('official_athlete_url', '') == anchor['athlete_url']):
                                rows = anchored_rows
                                result['reviewed_selected_owner'] = copy.deepcopy(anchor)
                            else:
                                reason = 'reviewed_selected_owner_conflict'
                        if not url:
                            reason = 'invalid_official_athlete_url'
                        elif reason is not None:
                            pass
                        elif len(rows) != 1:
                            reason = 'selected_route_owner_missing_or_ambiguous'
                        elif url in withheld:
                            reason = withheld[url]
                        elif record is not None:
                            row = rows[0]
                            expected_id = record['fighter_id']
                            if (normalize_ufcstats_id(row.get('ufcstats_url')) != expected_id
                                    or normalize_person_name(row.get('official_name')) != normalize_person_name(record['name'])):
                                reason = 'reviewed_route_selected_owner_conflict'
                            elif original_id not in (None, '') and identity != expected_id:
                                reason = 'submitted_id_conflicts_with_reviewed_route'
                            else:
                                identity = expected_id
                                result['source_evidence_sha256'] = record['evidence_sha256']
                        else:
                            # Preserve the pre-existing positive trust rule for
                            # other selected owners; a legacy valid flag is insufficient.
                            from src.data.ufc_active_roster import _official_url_identity_positively_trusted
                            row = rows[0]
                            expected_id = normalize_ufcstats_id(row.get('ufcstats_url'))
                            if not expected_id or not _official_url_identity_positively_trusted(row):
                                reason = 'no_reviewed_official_route_binding'
                            elif original_id not in (None, '') and identity != expected_id:
                                reason = 'submitted_id_conflicts_with_selected_route'
                            else:
                                identity = expected_id
                    elif not identity:
                        reason = 'missing_bound_id_and_official_route'
                    if reason is None:
                        try:
                            sources.validate_fighter(name, identity)
                        except ValueError as exc:
                            reason = 'selected_name_id_refused: ' + str(exc)
                    if reason:
                        context[field] = None
                        result.update(status='withheld', reason=reason)
                    else:
                        context[field] = identity
                        result.update(status='bound', fighter_id=identity)
                    provenance['sides'][side] = result
                if context.get('fighter_a_id') and context.get('fighter_a_id') == context.get('fighter_b_id'):
                    for side in ('a', 'b'):
                        context[f'fighter_{side}_id'] = None
                        provenance['sides'][side].update(status='withheld', reason='same_identity_on_both_sides')
                context['runtime_identity_binding'] = provenance
        return bound


def reviewed_bindings_for_runtime(sources, binding):
    """An unreviewed generation receives no supplemental route or alias trust."""
    raw = ARTIFACT_PATH.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256,
             'Runtime identity artifact hash changed')
    data = json.loads(raw)
    if binding.get('selection_sha256') != data['selection_sha256']:
        return None
    review = ReviewedRuntimeIdentityBindings(
        ARTIFACT_PATH, ARTIFACT_SHA256, binding['selection_sha256'],
        binding['processed_sha256']['fights_cleaned.csv'], binding['manifest_sha256'], binding['release_id'])
    review.verify_sources(sources)
    return review
