"""Validated runtime representation of a locked confirmation strategy.

The performance-recovery sweep evaluates S and C only.  A promoted runtime
therefore has to carry the exact confirmed sweep row and explicitly disable
the two live-only execution paths that the sweep does not model: the flat M
tracker and resting near-miss limit orders.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


LOCKED_STRATEGY_FIELDS = (
    "name",
    "s_min_edge",
    "s_blend_weight",
    "s_model_agreement_min_edge",
    "c_min_model_prob",
    "c_min_no_odds_prob",
    "c_share",
    "c_bet_fraction",
    "c_max_bet_fraction",
    "c_min_edge",
    "c_max_decimal_odds",
    "m_enabled",
    "m_min_model_prob",
    "m_min_no_odds_prob",
    "m_bet_fraction",
    "m_share",
)

CONFIRMED_STRATEGY_FIELDS = frozenset(
    {
        "schema_version",
        "strategy_config",
        "strategy_config_sha256",
        "runtime_constraints",
        "shared_constants",
        "shared_constants_sha256",
    }
)
RUNTIME_CONSTRAINTS = {
    "model_tracker_enabled": False,
    "near_miss_orders_enabled": False,
}


def shared_strategy_constants_snapshot() -> dict[str, Any]:
    """Resolve the effective shared sizing/filter constants at call time.

    These module constants shape live S/C selection and sizing but are not
    among the locked strategy fields, so confirmed evidence would not notice
    an ordinary code edit to them (DIR-AUD-P2-019). The confirmed payload
    binds their confirmation-era values and the trading edge revalidates them
    against this same resolver on every confirmed run.
    """
    from dataclasses import asdict

    from src import config
    from src.strategy import value

    return {
        "kelly_fraction": float(config.KELLY_FRACTION),
        "max_bet_fraction": float(config.MAX_BET_FRACTION),
        "stop_loss_fraction": float(config.STOP_LOSS_FRACTION),
        "min_fighter_fights": int(config.MIN_FIGHTER_FIGHTS),
        "require_model_agreement": bool(config.REQUIRE_MODEL_AGREEMENT),
        "conviction_confidence_bonus": float(config.CONVICTION_CONFIDENCE_BONUS),
        "blend_weight_min": float(config.BLEND_WEIGHT_MIN),
        "blend_weight_max": float(config.BLEND_WEIGHT_MAX),
        "newbie_fights_threshold": int(config.NEWBIE_FIGHTS_THRESHOLD),
        "newbie_major_extra_edge": float(config.NEWBIE_MAJOR_EXTRA_EDGE),
        "newbie_feeder_extra_edge": float(config.NEWBIE_FEEDER_EXTRA_EDGE),
        "newbie_regional_extra_edge": float(config.NEWBIE_REGIONAL_EXTRA_EDGE),
        "newbie_size_multiplier": float(config.NEWBIE_SIZE_MULTIPLIER),
        "newbie_skip_zero_fights_regional": bool(
            config.NEWBIE_SKIP_ZERO_FIGHTS_REGIONAL
        ),
        "default_newbie_rule": asdict(value.DEFAULT_NEWBIE_RULE),
    }


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_unit_interval(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"strategy_config.{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"strategy_config.{field} must be finite and within [0, 1]"
        )
    return numeric


def validate_locked_strategy_config(payload: object) -> dict[str, Any]:
    """Validate the exact serialized strategy shape used by confirmation."""
    if not isinstance(payload, dict):
        raise ValueError("strategy_config must be a JSON object")
    if set(payload) != set(LOCKED_STRATEGY_FIELDS):
        raise ValueError(
            "strategy_config fields are not exact: "
            f"missing={sorted(set(LOCKED_STRATEGY_FIELDS) - set(payload))}, "
            f"unexpected={sorted(set(payload) - set(LOCKED_STRATEGY_FIELDS))}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("strategy_config.name must be a non-empty string")

    probability_fields = (
        "s_min_edge",
        "s_blend_weight",
        "s_model_agreement_min_edge",
        "c_min_model_prob",
        "c_min_no_odds_prob",
        "c_share",
        "c_bet_fraction",
        "c_max_bet_fraction",
        "c_min_edge",
        "m_min_model_prob",
        "m_min_no_odds_prob",
        "m_bet_fraction",
        "m_share",
    )
    for field in probability_fields:
        _finite_unit_interval(payload.get(field), field=field)

    if not isinstance(payload.get("m_enabled"), bool):
        raise ValueError("strategy_config.m_enabled must be boolean")
    if payload["m_enabled"] is not False:
        raise ValueError(
            "Promoted strategy_config must keep the unevaluated M tracker disabled"
        )
    max_odds = payload.get("c_max_decimal_odds")
    if max_odds is not None:
        if isinstance(max_odds, bool) or not isinstance(max_odds, (int, float)):
            raise ValueError("strategy_config.c_max_decimal_odds must be numeric or null")
        if not math.isfinite(float(max_odds)) or float(max_odds) <= 1.0:
            raise ValueError(
                "strategy_config.c_max_decimal_odds must be finite and greater than 1"
            )
    if float(payload["c_bet_fraction"]) > float(payload["c_max_bet_fraction"]):
        raise ValueError("strategy_config conviction bet fraction exceeds its cap")
    # M is prohibited and dormant in every confirmed package, so its legacy
    # allocation field cannot reduce the independently evaluated C allocation.
    # A future package that evaluates M must introduce and bind the joint-wallet
    # constraint as part of that new execution contract.
    return dict(payload)


def build_confirmed_strategy_payload(
    strategy_config: object,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Build the immutable bundle section from a PASS result binding."""
    config = validate_locked_strategy_config(strategy_config)
    actual_sha256 = canonical_json_sha256(config)
    if actual_sha256 != str(expected_sha256 or "").lower():
        raise ValueError("Confirmed strategy config does not reproduce its PASS hash")
    shared_constants = shared_strategy_constants_snapshot()
    return {
        "schema_version": 1,
        "strategy_config": config,
        "strategy_config_sha256": actual_sha256,
        "runtime_constraints": dict(RUNTIME_CONSTRAINTS),
        "shared_constants": shared_constants,
        "shared_constants_sha256": canonical_json_sha256(shared_constants),
    }


def validate_confirmed_strategy_payload(payload: object) -> dict[str, Any]:
    """Validate a staged/runtime confirmed-strategy bundle section."""
    if not isinstance(payload, dict) or set(payload) != CONFIRMED_STRATEGY_FIELDS:
        raise ValueError("confirmed_strategy fields are missing or unknown")
    if payload.get("schema_version") != 1:
        raise ValueError("confirmed_strategy.schema_version must equal 1")
    if payload.get("runtime_constraints") != RUNTIME_CONSTRAINTS:
        raise ValueError(
            "confirmed_strategy must disable unevaluated M and near-miss execution"
        )
    config = validate_locked_strategy_config(payload.get("strategy_config"))
    actual_sha256 = canonical_json_sha256(config)
    if payload.get("strategy_config_sha256") != actual_sha256:
        raise ValueError("confirmed_strategy strategy hash does not reproduce")
    shared_constants = payload.get("shared_constants")
    if not isinstance(shared_constants, dict) or set(shared_constants) != set(
        shared_strategy_constants_snapshot()
    ):
        raise ValueError(
            "confirmed_strategy shared strategy constants are missing or unknown"
        )
    if payload.get("shared_constants_sha256") != canonical_json_sha256(
        shared_constants
    ):
        raise ValueError(
            "confirmed_strategy shared-constants hash does not reproduce"
        )
    return {
        "schema_version": 1,
        "strategy_config": config,
        "strategy_config_sha256": actual_sha256,
        "runtime_constraints": dict(RUNTIME_CONSTRAINTS),
        "shared_constants": dict(shared_constants),
        "shared_constants_sha256": canonical_json_sha256(shared_constants),
    }
