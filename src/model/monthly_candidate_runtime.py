"""Production adapter for permissioned monthly UFC candidate-only training."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from src.model.monthly_candidate import (
    CANDIDATE_SPEC_NAME,
    EXPECTED_LOGISTIC_FEATURE_COUNT,
    EXPECTED_NO_ODDS_FEATURE_COUNT,
    EXPECTED_PRIMARY_FEATURE_COUNT,
    MonthlyCandidateError,
    resolve_monthly_candidate_contract,
    run_monthly_candidate,
    _verify_input_receipt,
    _validate_completed_evaluation,
)


ENABLED_ENV = "UFC_MONTHLY_CANDIDATE_ENABLED"
PROMOTION_MONTH_ENV = "UFC_V7_PROMOTION_MONTH_UTC"
CANDIDATE_ROOT_ENV = "UFC_MONTHLY_CANDIDATE_ROOT"
STATE_PATH_ENV = "UFC_MONTHLY_CANDIDATE_STATE_PATH"
EVALUATOR_COMMAND_ENV = "UFC_MONTHLY_CANDIDATE_EVALUATOR_COMMAND_JSON"
EVALUATOR_TIMEOUT_ENV = "UFC_MONTHLY_CANDIDATE_EVALUATOR_TIMEOUT_SECONDS"
PRODUCER_TEMPLATE_ENV = "UFC_MONTHLY_EVALUATOR_TEMPLATE_PATH"
PRODUCER_TEMPLATE_BYTES_ENV = "UFC_MONTHLY_EVALUATOR_TEMPLATE_BYTES"
PRODUCER_TEMPLATE_SHA_ENV = "UFC_MONTHLY_EVALUATOR_TEMPLATE_SHA256"
PRODUCER_INPUT_ROOT_ENV = "UFC_MONTHLY_EVALUATOR_INPUT_ROOT"
PRODUCER_OUTPUT_ROOT_ENV = "UFC_MONTHLY_EVALUATOR_OUTPUT_ROOT"
PUBLICATION_RECEIPT_FIELD = "accepted_data_publication"
MAX_EVALUATOR_TIMEOUT_SECONDS = 7200.0
EVALUATOR_RESULT_SCHEMA_VERSION = 1
REQUIRED_EVALUATION_CHECKS = (
    "fixed_evaluation",
    "production_load",
    "exact_parity",
    "prefight_parity",
    "coverage_report",
    "current_live_comparison",
)


class MonthlyCandidateRuntimeError(RuntimeError):
    """Raised when production candidate automation cannot prove its inputs."""


@dataclass(frozen=True)
class MonthlyCandidateRuntimeConfig:
    promotion_month: str
    candidate_root: Path
    state_path: Path
    live_models_dir: Path
    active_manifest_path: Path
    evaluator_command: tuple[str, ...]
    evaluator_timeout_seconds: float
    monthly_policy: Any = None
    evaluator_template_pin: Any = None
    evaluator_input_root: Path | None = None
    evaluator_output_root: Path | None = None


def _enabled(env: Mapping[str, str]) -> bool:
    return str(env.get(ENABLED_ENV, "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolved_explicit_path(env: Mapping[str, str], name: str) -> Path:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        raise MonthlyCandidateRuntimeError(f"{name} is required when monthly candidates are enabled")
    return Path(raw).resolve(strict=False)


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _parse_command(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = str(env.get(EVALUATOR_COMMAND_ENV, "") or "").strip()
    if not raw:
        raise MonthlyCandidateRuntimeError(
            f"{EVALUATOR_COMMAND_ENV} is required when monthly candidates are enabled"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MonthlyCandidateRuntimeError(
            f"{EVALUATOR_COMMAND_ENV} must be a JSON array of arguments"
        ) from exc
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(value, str) or not value.strip() for value in payload)
    ):
        raise MonthlyCandidateRuntimeError(
            f"{EVALUATOR_COMMAND_ENV} must be a non-empty JSON string array"
        )
    return tuple(payload)


def _parse_timeout(env: Mapping[str, str]) -> float:
    raw = str(env.get(EVALUATOR_TIMEOUT_ENV, "") or "").strip()
    if not raw:
        raise MonthlyCandidateRuntimeError(
            f"{EVALUATOR_TIMEOUT_ENV} is required when monthly candidates are enabled"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise MonthlyCandidateRuntimeError(f"{EVALUATOR_TIMEOUT_ENV} must be numeric") from exc
    if not 1.0 <= value <= MAX_EVALUATOR_TIMEOUT_SECONDS:
        raise MonthlyCandidateRuntimeError(
            f"{EVALUATOR_TIMEOUT_ENV} must be between 1 and "
            f"{MAX_EVALUATOR_TIMEOUT_SECONDS:g} seconds"
        )
    return value


def load_monthly_candidate_runtime_config(
    *,
    active_manifest_path: Path | str,
    live_models_dir: Path | str,
    env: Mapping[str, str] | None = None,
) -> MonthlyCandidateRuntimeConfig | None:
    """Return strict runtime configuration only when separately enabled."""
    source = os.environ if env is None else env
    if not _enabled(source):
        return None
    from src.model.monthly_provisioning import load_configured_monthly_policy
    policy = load_configured_monthly_policy(source)
    promotion_month = (policy.promotion_month_utc if policy is not None
                       else str(source.get(PROMOTION_MONTH_ENV, "") or "").strip())
    try:
        datetime.strptime(promotion_month, "%Y-%m")
    except ValueError as exc:
        raise MonthlyCandidateRuntimeError(
            "Explicit accepted policy or historical v7 promotion month is required") from exc
    if len(promotion_month) != 7:
        raise MonthlyCandidateRuntimeError("Promotion month must use YYYY-MM")

    candidate_root = _resolved_explicit_path(source, CANDIDATE_ROOT_ENV)
    state_path = _resolved_explicit_path(source, STATE_PATH_ENV)
    live_dir = Path(live_models_dir).resolve(strict=False)
    manifest = Path(active_manifest_path).resolve(strict=False)
    if _overlap(candidate_root, live_dir) or _overlap(state_path, live_dir):
        raise MonthlyCandidateRuntimeError(
            "Monthly candidate root/state must be completely isolated from live models"
        )
    if candidate_root not in state_path.parents:
        raise MonthlyCandidateRuntimeError(
            "Monthly candidate state path must be stored inside the isolated candidate root"
        )
    if not live_dir.is_dir():
        raise MonthlyCandidateRuntimeError(f"Live model directory is missing: {live_dir}")
    command = _parse_command(source)
    fields = (PRODUCER_TEMPLATE_ENV, PRODUCER_TEMPLATE_BYTES_ENV, PRODUCER_TEMPLATE_SHA_ENV,
              PRODUCER_INPUT_ROOT_ENV, PRODUCER_OUTPUT_ROOT_ENV)
    values = [str(source.get(field, "") or "").strip() for field in fields]
    template_pin = input_root = output_root = None
    if any(values):
        if not all(values) or not values[1].isdigit():
            raise MonthlyCandidateRuntimeError("All explicit evaluator producer settings are required together")
        from src.model.monthly_evaluator_inputs import validate_pin, safe_path
        template_pin = validate_pin({"path": values[0], "bytes": int(values[1]), "sha256": values[2]}, limit=128 * 1024)
        input_root, output_root = safe_path(values[3]), safe_path(values[4])
        protected = (candidate_root, live_dir, manifest.parent, Path(__file__).resolve().parents[2])
        if _overlap(input_root, output_root) or any(_overlap(path, item) for path in (input_root, output_root) for item in protected):
            raise MonthlyCandidateRuntimeError("Evaluator producer storage overlaps protected paths")
    elif "src.model.monthly_evaluator" in command:
        raise MonthlyCandidateRuntimeError("Concrete evaluator producer requires explicit accepted input settings")
    return MonthlyCandidateRuntimeConfig(
        promotion_month=promotion_month,
        candidate_root=candidate_root,
        state_path=state_path,
        live_models_dir=live_dir,
        active_manifest_path=manifest,
        evaluator_command=command,
        evaluator_timeout_seconds=_parse_timeout(source),
        monthly_policy=policy,
        evaluator_template_pin=template_pin,
        evaluator_input_root=input_root,
        evaluator_output_root=output_root,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hash(payload: Mapping[str, Any], field: str, *, label: str) -> str:
    value = str(payload.get(field) or "").strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MonthlyCandidateRuntimeError(f"{label}.{field} is missing or invalid")
    return value


def _candidate_model_hashes(candidate_dir: Path, *, contract: Mapping[str, Any] | None = None) -> dict[str, str]:
    import re
    spec_name = contract.get("name") if contract is not None else CANDIDATE_SPEC_NAME
    if not isinstance(spec_name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec_name) is None:
        raise MonthlyCandidateRuntimeError("Evaluator artifact spec name is unsafe")
    models_dir = candidate_dir / "models"
    paths = {
        "primary": models_dir / "xgboost_model.pkl",
        "logistic": models_dir / "logistic_model.pkl",
        "no_odds": models_dir / "xgboost_no_odds_model.pkl",
        "spec": models_dir / f"{spec_name}_spec.json",
    }
    missing = [label for label, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise MonthlyCandidateRuntimeError(
            f"Approved evaluator candidate artifacts are missing: {missing}"
        )
    return {label: _sha256(path) for label, path in paths.items()}


def _validate_evaluator_result(
    result: object,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Share the direct candidate's pure semantic evidence decision."""
    from src.model.monthly_evidence import MonthlyEvidenceRefusal, validate_evaluator_report
    try:
        return validate_evaluator_report(result, request=request)
    except MonthlyEvidenceRefusal as exc:
        raise MonthlyCandidateRuntimeError(str(exc)) from exc


def _load_accepted_generation(
    config: MonthlyCandidateRuntimeConfig,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    from src.model.monthly_input import load_monthly_input

    transport = claim.get("monthly_input")
    context = claim.get("monthly_context")
    if not isinstance(context, dict) or any(claim.get(field) != context.get(field)
        for field in ("utc_month", "accepted_update_id", "accepted_at_utc")):
        raise MonthlyCandidateRuntimeError("Monthly attempt conflicts with original selection context")
    loaded = load_monthly_input(transport, context=context)
    if config.monthly_policy is not None and loaded.policy is None:
        raise MonthlyCandidateRuntimeError("Explicit successor runtime refuses historical policy-free input")
    if loaded.policy is None and context["promotion_month_utc"] != config.promotion_month:
        raise MonthlyCandidateRuntimeError("Monthly runtime promotion month conflicts with saved selection")
    return {
        "accepted_generation_id": context["accepted_update_id"],
        "accepted_at": datetime.fromisoformat(context["accepted_at_utc"].replace("Z", "+00:00")),
        "features_path": Path(transport["root"]) / "prepared" / "input.csv",
        "processed_dir": Path(transport["root"]),
        "update_report": loaded.publication["update_report"],
        "monthly_input": transport,
        "monthly_context": context,
        "loaded_input": loaded,
    }



def _subprocess_evaluator(
    config: MonthlyCandidateRuntimeConfig,
    *,
    update_report: Mapping[str, Any],
) -> Callable[..., dict[str, Any]]:
    def evaluate(**kwargs: Any) -> dict[str, Any]:
        from src.model.monthly_evidence import build_evaluator_request, parse_evaluator_json
        from src.model.monthly_candidate import _write_json_atomically
        candidate_dir = Path(kwargs["candidate_dir"]).resolve(strict=False)
        producer = kwargs.get("producer_inputs") is not None
        if producer and (candidate_dir / "runtime_evaluation_request.json").exists():
            raise MonthlyCandidateRuntimeError("Original evaluator dispatch attempt is consumed; retain its evidence")
        def write_request(value, path):
            if producer:
                from src.model.monthly_evaluator_inputs import _exclusive
                from src.model.monthly_evidence import canonical_bytes
                _exclusive(path, canonical_bytes(value))
            else:
                _write_json_atomically(value, path)
        request = build_evaluator_request(candidate_dir=str(candidate_dir),
            input_receipt=kwargs["input_receipt"], contract=kwargs["contract"],
            model_hashes=_candidate_model_hashes(candidate_dir, contract=kwargs["contract"]),
            accepted_update_report=update_report, producer_inputs=kwargs.get("producer_inputs"))
        write_request(request, candidate_dir / "runtime_evaluation_request.json")
        wire = request
        if kwargs.get("producer_inputs") is not None:
            from src.model.monthly_evaluator_inputs import load_evaluator_inputs
            plan = load_evaluator_inputs(kwargs["producer_inputs"], request=request)
            write_request(plan, candidate_dir / "runtime_evaluation_plan.json")
            wire = {"schema": 1, "kind": "monthly_evaluator_producer_command_v1", "request": request,
                    "output_root": kwargs["producer_output_root"]}
            write_request(wire, candidate_dir / "runtime_evaluation_command.json")
        try:
            completed = subprocess.run(
                list(config.evaluator_command),
                input=json.dumps(wire),
                text=True,
                capture_output=True,
                shell=False,
                timeout=config.evaluator_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            from src.data.io_utils import write_bytes_atomically
            for label, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                if value is not None:
                    write_bytes_atomically(value if isinstance(value, bytes) else str(value).encode("utf-8"),
                                           candidate_dir / f"runtime_evaluation_timeout_{label}.txt")
            raise MonthlyCandidateRuntimeError(f"Approved evaluator timed out: {exc}") from exc
        except OSError as exc:
            raise MonthlyCandidateRuntimeError(f"Approved evaluator failed to execute: {exc}") from exc
        # Keep exact unsuccessful or malformed evaluator originals for review.
        from src.data.io_utils import write_bytes_atomically
        write_bytes_atomically(completed.stdout.encode("utf-8"), candidate_dir / "runtime_evaluation_stdout.json")
        write_bytes_atomically(completed.stderr.encode("utf-8"), candidate_dir / "runtime_evaluation_stderr.txt")
        if completed.returncode != 0:
            raise MonthlyCandidateRuntimeError(
                "Approved evaluator failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()[:1000]}"
            )
        try:
            result = parse_evaluator_json(completed.stdout)
        except ValueError as exc:
            raise MonthlyCandidateRuntimeError("Approved evaluator output is not valid JSON") from exc
        if _candidate_model_hashes(candidate_dir, contract=kwargs["contract"]) != request["model_hashes"]:
            raise MonthlyCandidateRuntimeError("Evaluator changed selected candidate artifact identities")
        if kwargs.get("producer_inputs") is not None:
            load_evaluator_inputs(kwargs["producer_inputs"], request=request)
            from src.model.monthly_evaluator import verify_saved_report
            if verify_saved_report(kwargs["producer_output_root"], request=request) != result:
                raise MonthlyCandidateRuntimeError("Evaluator stdout differs from retained producer original")
        return _validate_evaluator_result(result, request=request)

    return evaluate


def _stable_candidate_id(config: MonthlyCandidateRuntimeConfig, month: str) -> str:
    receipt = config.candidate_root / month / "final" / "candidate_receipt.json"
    if not receipt.is_file():
        raise MonthlyCandidateRuntimeError("Completed monthly candidate receipt is missing")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonthlyCandidateRuntimeError("Completed monthly candidate receipt is invalid") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("status") != "complete_awaiting_owner_review"
        or payload.get("month") != month
        or payload.get("owner_review_required") is not True
        or payload.get("auto_promotion_performed") is not False
        or payload.get("live_model_replaced") is not False
    ):
        raise MonthlyCandidateRuntimeError("Completed monthly candidate receipt is unsafe")
    receipt_hash = _sha256(receipt)
    marker = config.candidate_root / month / "completed.json"
    if not marker.is_file():
        raise MonthlyCandidateRuntimeError("Completed monthly candidate marker is missing")
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonthlyCandidateRuntimeError("Completed monthly candidate marker is invalid") from exc
    if type(marker_payload.get("schema_version")) is not int or marker_payload != {
        "schema_version": 1,
        "candidate_receipt": "final/candidate_receipt.json",
        "candidate_receipt_sha256": receipt_hash,
    }:
        raise MonthlyCandidateRuntimeError("Completed monthly candidate marker is unbound")
    policy = payload.get("contract", {}).get("monthly_policy")
    if policy is not None:
        import re
        model_id = policy.get("model_id") if isinstance(policy, dict) else None
        if not isinstance(model_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model_id) is None:
            raise MonthlyCandidateRuntimeError("Completed monthly policy model identity is invalid")
        return f"monthly-{model_id}-{month}-{receipt_hash[:16]}"
    return f"monthly-v7-{month}-{receipt_hash[:16]}"


def _reconcile_completed_monthly_candidates(
    config: MonthlyCandidateRuntimeConfig,
    schedule: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recover the crash window after a candidate receipt but before schedule completion."""
    state = schedule.load_state()
    monthly = state.get("monthly_candidates")
    if not isinstance(monthly, dict):
        raise MonthlyCandidateRuntimeError("Monthly schedule state is invalid")
    recovered: list[dict[str, Any]] = []
    for month, entry in sorted(monthly.items()):
        if not isinstance(entry, dict):
            raise MonthlyCandidateRuntimeError(f"Monthly schedule entry is invalid: {month}")
        source = entry.get("source_update")
        slots = entry.get("slots")
        if not isinstance(source, dict) or not isinstance(slots, dict):
            raise MonthlyCandidateRuntimeError(f"Monthly schedule entry is incomplete: {month}")
        receipt_path = config.candidate_root / month / "final" / "candidate_receipt.json"
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MonthlyCandidateRuntimeError(
                f"Completed monthly candidate receipt is invalid: {month}"
            ) from exc
        if not isinstance(receipt, dict):
            raise MonthlyCandidateRuntimeError(
                f"Completed monthly candidate receipt is invalid: {month}"
            )
        source_generation = str(source.get("accepted_update_id") or "").strip()
        if str(receipt.get("accepted_generation_id") or "").strip() != source_generation:
            raise MonthlyCandidateRuntimeError(
                f"Completed monthly candidate is bound to another accepted generation: {month}"
            )
        transport = source.get("monthly_input")
        context = transport.get("context") if isinstance(transport, dict) else None
        from src.model.monthly_input import load_monthly_input
        loaded = load_monthly_input(transport, context=context)
        if config.monthly_policy is not None and loaded.policy is None:
            raise MonthlyCandidateRuntimeError("Explicit successor runtime refuses historical policy-free recovery")
        if (context.get("utc_month") != month
            or context.get("accepted_update_id") != source_generation
            or context.get("accepted_at_utc") != source.get("accepted_at_utc")
            or receipt.get("monthly_input") != transport):
            raise MonthlyCandidateRuntimeError("Completed candidate does not bind the original monthly selection")
        if loaded.policy is not None:
            contract = resolve_monthly_candidate_contract(loaded.policy)
            if receipt.get("contract") != contract.receipt_payload():
                raise MonthlyCandidateRuntimeError("Completed candidate saved policy contract conflicts")
            saved = _verify_input_receipt(config.candidate_root / month / "input", contract=contract,
                accepted_generation_id=source_generation,
                weekly_update_completed_at=datetime.fromisoformat(context["accepted_at_utc"].replace("Z", "+00:00")),
                monthly_input=transport)
            _validate_completed_evaluation(receipt, candidate_dir=config.candidate_root / month / "final",
                                           input_receipt=saved, contract=contract)
        candidate_id = _stable_candidate_id(config, month)
        for slot in ("tuesday", "wednesday"):
            attempt = slots.get(slot)
            if not isinstance(attempt, dict) or attempt.get("status") != "claimed":
                continue
            claim_id = str(attempt.get("claim_id") or "").strip()
            if not claim_id:
                raise MonthlyCandidateRuntimeError(
                    f"Claimed monthly schedule attempt has no claim ID: {month}/{slot}"
                )
            completion = schedule.complete_monthly_candidate(
                claim_id,
                succeeded=True,
                candidate_id=candidate_id,
                now=now,
            )
            recovered.append(
                {
                    "utc_month": month,
                    "slot": slot,
                    "claim_id": claim_id,
                    "candidate_id": candidate_id,
                    "completion": completion,
                }
            )
    return {
        "changed": bool(recovered),
        "recovered_count": len(recovered),
        "recovered_candidates": recovered,
    }


def build_monthly_candidate_runner(
    config: MonthlyCandidateRuntimeConfig,
    *,
    run_candidate: Callable[..., dict[str, Any]] = run_monthly_candidate,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Map calendar claims onto the existing isolated candidate workflow."""
    def runner(claim: dict[str, Any]) -> dict[str, Any]:
        try:
            accepted = _load_accepted_generation(config, claim)
            evaluator = _subprocess_evaluator(
                config,
                update_report=accepted["update_report"],
            )
            producer_kwargs = {}
            # Once saved in ordinary candidate state, that month's selection
            # governs restart even if current template settings have changed.
            state = {}
            if config.state_path.is_file():
                from src.model.monthly_candidate import _load_json_object
                state = _load_json_object(config.state_path, label="candidate state")
            saved_month = state.get("months", {}).get(accepted["monthly_context"]["utc_month"], {})
            from src.model.monthly_candidate import _completion_receipt
            completed = _completion_receipt(config.candidate_root / accepted["monthly_context"]["utc_month"], repair_marker=False)
            if completed is not None and completed.get("producer_inputs") is not None:
                saved_month = completed
            original = saved_month.get("producer_inputs")
            if original is not None or config.evaluator_template_pin is not None:
                from src.model.monthly_evaluator_inputs import save_evaluator_inputs, load_evaluator_inputs, validate_selection
                if original is None:
                    from src.model import prepared_input as prepared
                    loaded = accepted["loaded_input"]
                    descriptor = prepared._json(prepared._attached(loaded.receipt["acceptance"]))
                    originals = [*prepared._paths(loaded.receipt), *prepared._paths(loaded.selection.references)]
                    protected = [config.candidate_root, config.state_path, config.live_models_dir,
                                 config.active_manifest_path.parent, prepared._path(descriptor["raw_root"]),
                                 *originals, *(path.parent for path in originals)]
                    original = save_evaluator_inputs(template_pin=config.evaluator_template_pin,
                        output_root=config.evaluator_input_root / accepted["monthly_context"]["utc_month"],
                        monthly_input=accepted["monthly_input"],
                        protected_paths=protected)
                selected = load_evaluator_inputs(original)
                validate_selection(selected["template"], accepted["monthly_input"])
                output_root = saved_month.get("producer_output_base") if original is not None and saved_month.get("producer_inputs") is not None else config.evaluator_output_root
                if output_root is None:
                    raise MonthlyCandidateRuntimeError("Saved producer input requires explicit report output root")
                producer_kwargs = {"producer_inputs": original, "producer_output_root": output_root}
            result = run_candidate(
                weekly_update_succeeded=True,
                weekly_update_completed_at=accepted["accepted_at"],
                accepted_generation_id=accepted["accepted_generation_id"],
                accepted_features_path=accepted["features_path"],
                candidate_root=config.candidate_root,
                live_models_dir=config.live_models_dir,
                v7_promotion_month=accepted["monthly_context"]["promotion_month_utc"],
                evaluator=evaluator,
                state_path=config.state_path,
                monthly_input=accepted["monthly_input"],
                monthly_context=accepted["monthly_context"],
                **producer_kwargs,
            )
            if not isinstance(result, dict) or result.get("passed") is not True:
                return {
                    "succeeded": False,
                    "candidate_id": "",
                    "error": str((result or {}).get("error") or "monthly_candidate_failed"),
                    "monthly_result": result,
                }
            month = str(result.get("month") or claim.get("utc_month") or "").strip()
            candidate_id = _stable_candidate_id(config, month)
            return {
                "succeeded": True,
                "candidate_id": candidate_id,
                "monthly_result": result,
                "accepted_generation_id": accepted["accepted_generation_id"],
                "auto_promotion_performed": False,
                "live_model_replaced": False,
                "owner_review_required": True,
            }
        except (MonthlyCandidateError, MonthlyCandidateRuntimeError, OSError, ValueError) as exc:
            return {
                "succeeded": False,
                "candidate_id": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def reconcile_completed(schedule: Any, *, now: datetime | None = None) -> dict[str, Any]:
        return _reconcile_completed_monthly_candidates(
            config,
            schedule,
            now=now,
        )

    setattr(runner, "reconcile_completed", reconcile_completed)
    return runner


def build_monthly_candidate_runner_from_env(
    *,
    active_manifest_path: Path | str,
    live_models_dir: Path | str,
    env: Mapping[str, str] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    config = load_monthly_candidate_runtime_config(
        active_manifest_path=active_manifest_path,
        live_models_dir=live_models_dir,
        env=env,
    )
    return build_monthly_candidate_runner(config) if config is not None else None


__all__ = [
    "CANDIDATE_ROOT_ENV",
    "ENABLED_ENV",
    "EVALUATOR_COMMAND_ENV",
    "EVALUATOR_TIMEOUT_ENV",
    "MonthlyCandidateRuntimeConfig",
    "MonthlyCandidateRuntimeError",
    "PROMOTION_MONTH_ENV",
    "PUBLICATION_RECEIPT_FIELD",
    "STATE_PATH_ENV",
    "build_monthly_candidate_runner",
    "build_monthly_candidate_runner_from_env",
    "load_monthly_candidate_runtime_config",
]
