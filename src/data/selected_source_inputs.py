"""Explicit saved inputs shared by historical features and fresh predictions.

The opt-in context carries file choices through existing readers. It does not
change configured production paths, collect data, or load a model.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

_ACTIVE = ContextVar('selected_feature_sources', default=None)


def active_sources():
    state = _ACTIVE.get()
    return None if state is None else state[0]


def selected_path(name, default):
    sources = active_sources()
    return Path(default) if sources is None else sources.path(name)


def selected_cache(name, loader):
    """Keep source-specific reusable reads within one explicit invocation."""
    state = _ACTIVE.get()
    if state is None:
        return loader()
    cache = state[1]
    if name not in cache:
        cache[name] = loader()
    return cache[name]


@dataclass(frozen=True)
class SelectedSourceInputs:
    files: dict[str, dict]
    processed_data_dir: Path
    missing_data_dir: Path
    live_observation_paths: dict[str, Path] | None = None
    runtime_identity_review: object | None = None

    @property
    def historical_market_replay(self):
        return self.live_observation_paths is None

    @classmethod
    def from_manifest(cls, path, *, processed_data_dir):
        document = json.loads(Path(path).read_bytes())
        files = {
            name: {**item, **({'path': str((Path(path).parent / item['path']).resolve())}
                             if item.get('path') and not Path(item['path']).is_absolute() else {})}
            for name, item in document['files'].items()
        }
        selected = cls(files, Path(processed_data_dir),
                       Path(path).parent / 'explicitly-absent-inputs')
        selected.verify()
        return selected

    def path(self, name):
        if self.live_observation_paths is not None and name in self.live_observation_paths:
            return self.live_observation_paths[name]
        return self._selected_path(name)

    def _selected_path(self, name):
        if name not in self.files:
            raise ValueError(f'Selected source is not specified: {name}')
        item = self.files[name]
        if item.get('absent') is True:
            path = self.missing_data_dir / name
            if path.exists():
                raise ValueError(f'Input recorded as absent now exists: {name}')
            return path
        return Path(item['path'])

    def verify(self):
        if self.live_observation_paths is not None:
            if set(self.live_observation_paths) != {'method_observations', 'rankings_observations', 'line_history'}:
                raise ValueError('Live observations require exactly the three explicit observation roots')
            if any(not Path(path).is_absolute() for path in self.live_observation_paths.values()):
                raise ValueError('Live observation roots must be absolute')
        for name, item in self.files.items():
            path = self._selected_path(name)
            if item.get('absent') is True:
                continue
            if item.get('directory') is True:
                if not path.is_dir():
                    raise FileNotFoundError(path)
                expected = item['files']
                actual = {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in path.rglob('*') if p.is_file()}
                if actual != expected:
                    raise ValueError(f'Selected source directory changed: {name}')
            else:
                body = path.read_bytes()
                if len(body) != item['bytes'] or hashlib.sha256(body).hexdigest() != item['sha256']:
                    raise ValueError(f'Selected source file changed: {name}')

    @contextmanager
    def activate(self):
        self.verify()
        token = _ACTIVE.set((self, {}))
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def markets(self):
        if not self.historical_market_replay:
            raise ValueError('Accepted historical markets cannot supply live market observations')
        from src.data.accepted_historical_markets import AcceptedHistoricalMarkets, _name
        def load():
            reviewed_names = {}
            if 'identity_aliases' in self.files and not self.files['identity_aliases'].get('absent'):
                aliases = json.loads(self.path('identity_aliases').read_bytes())
                if aliases:
                    import pandas as pd
                    from src.data.name_utils import normalize_ufcstats_id
                    profiles = pd.read_csv(self.path('profiles'), usecols=['name', 'fighter_url'])
                    names_by_id, owners_by_name = {}, {}
                    for row in profiles.itertuples(index=False):
                        identity = normalize_ufcstats_id(row.fighter_url)
                        names_by_id.setdefault(identity, set()).add(str(row.name))
                        owners_by_name.setdefault(_name(row.name), set()).add(identity)
                    for alias in aliases:
                        identity = normalize_ufcstats_id(alias.get('fighter_id'))
                        if not identity or not alias.get('evidence') or not str(alias.get('name') or '').strip():
                            raise ValueError('Selected market alias lacks its exact identity evidence')
                        names = names_by_id.get(identity, set())
                        if len(names) != 1:
                            raise ValueError('Selected market alias has no unique selected profile name')
                        alias_key = _name(alias['name'])
                        if owners_by_name.get(alias_key, set()) - {identity}:
                            raise ValueError('Selected market alias conflicts with another profile identity')
                        canonical = next(iter(names))
                        if alias_key in reviewed_names and reviewed_names[alias_key] != canonical:
                            raise ValueError('Selected market alias has conflicting reviewed identities')
                        reviewed_names[alias_key] = canonical
            return AcceptedHistoricalMarkets(
                moneyline_path=self.path('accepted_moneyline_bridge'),
                moneyline_sha256=self.files['accepted_moneyline_bridge']['sha256'],
                method_odds_path=self.path('accepted_method_odds'),
                method_odds_sha256=self.files['accepted_method_odds']['sha256'],
                reviewed_name_aliases=reviewed_names,
            )
        return selected_cache('accepted_markets', load)

    def validate_fighter(self, name, fighter_id):
        """Require the selected name and ID to describe the same fighter."""
        import pandas as pd
        from src.data.name_utils import (
            REVIEWED_FIGHTER_IDENTITIES, normalize_ufcstats_id, normalize_person_name,
            reviewed_spelling_bindings,
        )
        identity = normalize_ufcstats_id(fighter_id)
        if not identity:
            raise ValueError('Selected feature inputs require a bound UFCStats fighter ID')
        def names_by_id():
            table = pd.read_csv(self.path('profiles'), usecols=['name', 'fighter_url'])
            names = {}
            for row in table.itertuples(index=False):
                key = normalize_ufcstats_id(row.fighter_url)
                if not key:
                    raise ValueError('Selected profile has no valid stable identity')
                names.setdefault(key, set()).add(normalize_person_name(row.name))
            if 'identity_aliases' in self.files:
                aliases = json.loads(self.path('identity_aliases').read_bytes())
                for alias in aliases:
                    if not alias.get('evidence') or not normalize_ufcstats_id(alias.get('fighter_id')):
                        raise ValueError('Selected alias lacks its identity evidence')
                    names.setdefault(alias['fighter_id'], set()).add(normalize_person_name(alias['name']))
            # Reuse only the existing reviewed spellings, anchored to a name
            # and stable ID already present in these selected inputs.
            for key, reviewed in REVIEWED_FIGHTER_IDENTITIES.items():
                tracked = {normalize_person_name(value) for value in reviewed['tracked_names']}
                if not names.get(key, set()).intersection(tracked):
                    continue
                if any(owner != key and values.intersection(tracked)
                       for owner, values in names.items()):
                    raise ValueError('Selected reviewed name conflicts with another fighter identity')
                names[key].update(tracked)
            # Owner decision D10: the reviewed spellings bound to one UFCStats
            # ID, added to that ID's selected owner only when it already
            # carries one of them.
            for key, bound in reviewed_spelling_bindings().items():
                if names.get(key, set()).intersection(bound):
                    names[key].update(bound)
            if self.runtime_identity_review is not None:
                from src.data.runtime_identity_bindings import ReviewedRuntimeIdentityBindings
                if not isinstance(self.runtime_identity_review, ReviewedRuntimeIdentityBindings):
                    raise ValueError('Invalid runtime identity evidence overlay')
                self.runtime_identity_review.extend_selected_names(names, self)
            # Owner decision D10-OLD-GAPS (review finding IS3-3): a reviewed
            # bound spelling under any other selected owner fails closed for the
            # whole invocation, whether or not the bound ID's own row is in
            # this selection.
            for key, bound in reviewed_spelling_bindings().items():
                if any(owner != key and values.intersection(bound)
                       for owner, values in names.items()):
                    raise ValueError('Selected reviewed spelling conflicts with another fighter identity')
            return names
        if self.runtime_identity_review is not None:
            # Even an in-invocation name cache cannot outlive changed evidence.
            from src.data.runtime_identity_bindings import ReviewedRuntimeIdentityBindings
            if not isinstance(self.runtime_identity_review, ReviewedRuntimeIdentityBindings):
                raise ValueError('Invalid runtime identity evidence overlay')
            self.runtime_identity_review.verify_sources(self)
        known = selected_cache('names_by_id', names_by_id)
        if normalize_person_name(name) not in known.get(identity, set()):
            raise ValueError(f'Selected fighter name and ID disagree: {name} / {identity}')


def build_selected_training_rows(sources: SelectedSourceInputs):
    """Build from the explicit accepted pair, profiles, dates and historical markets.

    This returns a table in memory. Publication and model operations remain with
    the caller, under their separate authorization.
    """
    import pandas as pd
    from src.data.accepted_historical_markets import (
        AcceptedMarketSelection, METHOD_FEATURES, TIMING_POLICY,
        merge_accepted_historical_features,
    )
    from src.data.kaggle_loader import load_kaggle_dataset
    from src.data.ufc_refresh import (
        _load_event_metadata_cache, _normalize_name,
        build_pulled_all_plus_legacy_market_training_dataset,
        build_training_rows_from_pulled_data,
    )
    with sources.activate():
        dates = pd.read_csv(sources.path('event_dates'))
        if not {'event_name', 'event_date'}.issubset(dates.columns) or dates.empty:
            raise ValueError('Selected event dates are missing or have an invalid schema')
        dates['event_date'] = pd.to_datetime(dates['event_date'], errors='raise', format='mixed')
        if dates['event_date'].isna().any() or dates['event_name'].isna().any():
            raise ValueError('Selected event dates contain a missing event name or date')
        names = dates['event_name'].map(_normalize_name)
        if dates.assign(event_key=names).groupby('event_key')['event_date'].nunique().gt(1).any():
            raise ValueError('Selected event name has conflicting dates')
        lookup = dict(zip(names, dates['event_date']))
        results = pd.read_csv(sources.path('fight_results'), usecols=['EVENT'])
        missing_dates = set(results['EVENT'].map(_normalize_name)) - set(lookup)
        if missing_dates:
            raise ValueError(f'Selected fights have no saved event date: {sorted(missing_dates)}')
        metadata = _load_event_metadata_cache(sources.path('event_dates'))
        legacy = load_kaggle_dataset(sources.path('legacy_market_dataset'))
        frame = build_training_rows_from_pulled_data(
            fight_results_path=sources.path('fight_results'),
            fight_stats_path=sources.path('fight_stats'),
            scraped_fighters_path=sources.path('profiles'),
            supplemental_profiles_path=sources.path('supplemental_profiles'),
            official_active_roster_path=sources.path('active_roster'),
            profile_integrity_decisions_path=sources.path('profile_decisions'),
            prior_profile_integrity_decisions_path=sources.path('prior_profile_decisions'),
            pierce_fight_identities_path=sources.path('fight_identity_reference'),
            event_dates_cache_path=sources.path('event_dates'), event_date_lookup=lookup,
            event_metadata_lookup=metadata, legacy_df=legacy,
        )
        # Keep the established ordinary moneyline and rankings selection. This
        # reads only the exact pinned historical directory, not nearby archives.
        frame = build_pulled_all_plus_legacy_market_training_dataset(
            legacy, frame, historical_odds_dir=sources.path('historical_odds'),
            rankings_history_path=sources.path('rankings_history'),
            method_odds_history_path=sources.path('accepted_method_odds'),
        )
        markets = sources.markets()
        evidence = []
        for index, row in frame.iterrows():
            selected = markets.for_fight(event_date=row['event_date'],
                                        fighter_a=row['fighter_a'], fighter_b=row['fighter_b'])
            evidence.append(json.dumps(selected.records, sort_keys=True))
            # These 55 bridge rows are already approved historical corrections.
            # Older raw odds are not new evidence against the accepted decision.
            for field, value in selected.raw_moneyline_odds().items():
                frame.at[index, field] = value
            methods = {field: value for field, value in selected.features.items() if field in METHOD_FEATURES}
            if not methods:
                continue
            policy = row.get('method_observation_policy')
            strict_values = ({field: None if pd.isna(row.get(field)) else row[field] for field in METHOD_FEATURES}
                             if pd.notna(policy) else {})
            combined = merge_accepted_historical_features(
                strict_values, AcceptedMarketSelection(methods, selected.records))
            for field in METHOD_FEATURES:
                value = combined.get(field)
                frame.at[index, field] = float('nan') if value is None else value
            # The builder already respects a row's selected method probabilities
            # when this policy is present, including the accepted real blanks.
            frame.at[index, 'method_observation_policy'] = TIMING_POLICY
        frame['accepted_historical_market_records_json'] = evidence
        return frame


def build_selected_features(frame, sources: SelectedSourceInputs):
    import math
    import pandas as pd
    from src.features.build_features import build_features
    with sources.activate():
        result = build_features(frame)
        # A reader or later conversion must not silently erase or change the
        # accepted market inputs. Actual previously missing cells may be filled
        # by a compatible, separately selected observation.
        markets = sources.markets()
        for row in result.to_dict('records'):
            selected = markets.for_fight(event_date=row['event_date'],
                                        fighter_a=row['fighter_a'], fighter_b=row['fighter_b'])
            for field, value in selected.features.items():
                if value is None:
                    continue
                actual = row.get(field)
                if pd.isna(actual) or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-10):
                    raise ValueError(f'Accepted historical market input changed during feature building: {field}')
        return result


def build_selected_fight_features(sources: SelectedSourceInputs, **fight):
    """Use the same selected files for a fresh feature calculation, without collection."""
    from src.data.fighter_lookup import build_fight_features
    forbidden = {'processed_data_dir', 'profile_integrity_decisions_path',
                 'prior_profile_integrity_decisions_path'} & fight.keys()
    if forbidden:
        raise ValueError(f'Selected source paths cannot be overridden: {sorted(forbidden)}')
    if fight.get('prefer_live_refresh') or fight.get('force_fighter_refresh'):
        raise ValueError('Saved-source prediction does not collect replacement observations')
    with sources.activate():
        for side in ('a', 'b'):
            sources.validate_fighter(fight[f'fighter_{side}'], fight.get(f'fighter_{side}_id'))
            if fight.get(f'fighter_{side}_lookup_name'):
                sources.validate_fighter(fight[f'fighter_{side}_lookup_name'], fight.get(f'fighter_{side}_id'))
        return build_fight_features(
            **fight, processed_data_dir=sources.processed_data_dir,
            profile_integrity_decisions_path=sources.path('profile_decisions'),
            prior_profile_integrity_decisions_path=sources.path('prior_profile_decisions'),
        )
