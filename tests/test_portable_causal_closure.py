from __future__ import annotations

import pytest

from portable_regression_support import (
    PortableContractError,
    fight_physical_key,
    strict_date_causal_closure,
)


def _row(
    event_date: str,
    fighter_a: str,
    fighter_b: str,
    *,
    fighter_a_id: object = None,
    fighter_b_id: object = None,
) -> dict[str, object]:
    return {
        "event_date": event_date,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "fighter_a_id": fighter_a_id,
        "fighter_b_id": fighter_b_id,
    }


def test_ordinary_name_identity_closes_mixed_id_history_on_strict_later_dates():
    added = _row(
        "2021-01-01",
        "Seed Fighter",
        "Seed Opponent",
        fighter_a_id="aaaaaaaaaaaaaaaa",
        fighter_b_id="bbbbbbbbbbbbbbbb",
    )
    same_day = _row("2021-01-01", "Seed Fighter", "Same Day Opponent")
    relay = _row(
        "2022-01-01",
        "Seed Fighter",
        "Relay Fighter",
        fighter_b_id="cccccccccccccccc",
    )
    relay_same_day = _row("2022-01-01", "Relay Fighter", "Same Day Target")
    target = _row("2023-01-01", "Relay Fighter", "Target Opponent")
    unrelated = _row("2023-01-01", "Unrelated Fighter", "Unrelated Opponent")
    rows = [added, same_day, relay, relay_same_day, target, unrelated]

    closure = strict_date_causal_closure(rows, {fight_physical_key(added)})

    assert closure == {
        fight_physical_key(relay),
        fight_physical_key(target),
    }
    assert fight_physical_key(same_day) not in closure
    assert fight_physical_key(relay_same_day) not in closure
    assert fight_physical_key(unrelated) not in closure


def test_reviewed_ambiguous_name_stays_id_keyed_and_missing_id_fails_closed():
    added = _row(
        "2021-01-01",
        "Jean Silva",
        "First Opponent",
        fighter_a_id="1111111111111111",
        fighter_b_id="2222222222222222",
    )
    namesake = _row(
        "2022-01-01",
        "Jean Silva",
        "Second Opponent",
        fighter_a_id="3333333333333333",
        fighter_b_id="4444444444444444",
    )
    same_identity = _row(
        "2023-01-01",
        "Jean Silva",
        "Third Opponent",
        fighter_a_id="1111111111111111",
        fighter_b_id="5555555555555555",
    )

    closure = strict_date_causal_closure(
        [added, namesake, same_identity],
        {fight_physical_key(added)},
        ambiguous_names={"Jean Silva"},
    )

    assert fight_physical_key(namesake) not in closure
    assert fight_physical_key(same_identity) in closure

    with pytest.raises(PortableContractError, match="requires a UFCStats ID"):
        strict_date_causal_closure(
            [added, _row("2024-01-01", "Jean Silva", "Unknown Opponent")],
            {fight_physical_key(added)},
            ambiguous_names={"Jean Silva"},
        )
