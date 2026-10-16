"""Persistent data-only UFC calendar worker and prediction/status HTTP service.

This entry point never trains a model, places an order, or starts a betting loop.
The original slot intent is durable before the calendar claim; all subsequent
recovery uses that same intent, producer job, and prediction cache namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.data.io_utils import write_bytes_atomically
from src.data.ufc_update_schedule import (
    ACTION_DATA_UPDATE,
    UFCUpdateSchedule,
    _exclusive_file_lock,
)

logger = logging.getLogger(__name__)


def _now(value=None):
    value = value or datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("decision instant must be timezone aware")
    return value.astimezone(timezone.utc)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_directory(path):
    """Persist each new directory link before writing an original intent."""
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if os.name != "nt":
            descriptor = os.open(directory.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _write(path, value):
    _make_directory(Path(path).parent)
    write_bytes_atomically((json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), path)


def generation_id(pointer):
    return str(pointer.get("generation_id") or pointer.get("job_id") or pointer["processed_dir"])


class RefreshService:
    def __init__(self, *, root, baseline, selected_manifest=None, producer=None, predictor=None):
        self.root = Path(root).resolve()
        self.baseline = Path(baseline).resolve()
        self.selected_manifest = Path(selected_manifest or self.baseline / "selected-source-manifest.json").resolve()
        self.runtime_manifest = self.baseline / "selected-runtime-manifest.json"
        self.data_root = self.root / "data"
        self.state_path = self.root / "service.json"
        self.lock_path = self.root / "service.lock"
        self.status_path = self.root / "status.json"
        self.published_path = self.root / "published.json"
        self._producer_override = producer
        self._predictor_override = predictor
        self.binding = {
            "selected_manifest": "selected-source-manifest.json",
            "selected_manifest_sha256": hashlib.sha256(self.selected_manifest.read_bytes()).hexdigest(),
            "runtime_manifest_sha256": hashlib.sha256(self.runtime_manifest.read_bytes()).hexdigest(),
        }
        _make_directory(self.root)

    def _producer(self):
        if self._producer_override is not None:
            return self._producer_override
        from src.data import incremental_refresh
        return incremental_refresh

    def _predictor(self):
        if self._predictor_override is not None:
            return self._predictor_override
        from src.data.refresh_predictions import run_refresh_predictions
        return run_refresh_predictions

    def _state(self):
        if not self.state_path.exists():
            return {"schema_version": 1, "binding": self.binding, "pending": None}
        state = self._paths(_read(self.state_path))
        if state.get("schema_version") != 1 or state.get("binding") != self.binding:
            raise RuntimeError("Saved refresh service binding conflicts with this startup")
        return state

    def _paths(self, value, *, encode=False):
        """Keep interrupted producer results relocatable with the same bytes."""
        if isinstance(value, dict):
            return {key: self._paths(item, encode=encode) for key, item in value.items()}
        if isinstance(value, list):
            return [self._paths(item, encode=encode) for item in value]
        if isinstance(value, str):
            for token, root in (("${SERVICE_DIR}", self.root), ("${BASELINE_DIR}", self.baseline)):
                prefix = str(root).replace("\\", "/").rstrip("/")
                normalized = value.replace("\\", "/")
                if encode and (normalized.lower() == prefix.lower() or normalized.lower().startswith(prefix.lower() + "/")):
                    return token + normalized[len(prefix):]
                if not encode and (normalized == token or normalized.startswith(token + "/")):
                    return str(root / normalized[len(token):].lstrip("/"))
        return value

    def _save(self, state, instant):
        state["updated_at_utc"] = _now().isoformat()
        _write(self.state_path, self._paths(state, encode=True))

    @staticmethod
    def _completion_time(result, original_now):
        stamp = result.get("finished_at") or result.get("completed_at_utc")
        value = _now(datetime.fromisoformat(stamp)) if stamp else _now()
        if value < original_now:
            raise RuntimeError("Producer completion predates its original decision")
        return value.isoformat()

    def _schedule(self, claim_id=None):
        return UFCUpdateSchedule(
            state_path=self.root / "calendar.json",
            promotion_month_utc="",
            monthly_policy=None,
            timezone_name="America/New_York",
            claim_id_factory=(lambda: claim_id) if claim_id else None,
        )

    def current_snapshot(self):
        return self._producer().load_current_snapshot(
            state_dir=self.data_root, baseline_dir=self.baseline,
            selected_manifest=self.selected_manifest,
        )

    def _verify_pin(self, record):
        path = Path(record["path"]).resolve()
        if not (path.is_relative_to(self.root) or path.is_relative_to(self.baseline)):
            raise RuntimeError("Published artifact escapes its retained namespace")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for body in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(body)
                digest.update(body)
        if size != record.get("bytes") or digest.hexdigest() != record.get("sha256"):
            raise RuntimeError("Published artifact differs from its exact saved pin")

    def _validate_pair(self, pointer, report):
        if report.get("status") != "success" or report.get("accepted_generation") != generation_id(pointer):
            raise RuntimeError("Completed prediction does not match its retained data generation")
        artifacts, models = pointer.get("artifacts", {}), pointer.get("models", {})
        if set(artifacts) != {"fights_cleaned.csv", "features.csv"} or set(models) != {"primary", "no_odds", "logistic"}:
            raise RuntimeError("Published data must retain both artifacts and all original model roles")
        runtime_models = _read(self.runtime_manifest)["models"]
        for records, predictions in ((artifacts, report.get("data_pins", {})), (models, report.get("model_pins", {}))):
            if set(records) != set(predictions):
                raise RuntimeError("Completed prediction is missing original data or model pins")
            for name, pin in records.items():
                self._verify_pin(pin)
                if any(predictions[name].get(key) != pin.get(key) for key in ("bytes", "sha256")):
                    raise RuntimeError("Prediction data/model pins differ from the published snapshot")
                if records is models and any(runtime_models[name].get(key) != pin.get(key) for key in ("bytes", "sha256")):
                    raise RuntimeError("Published snapshot changed the original selected models")
        rows = report.get("predictions")
        if not isinstance(rows, list) or type(report.get("prediction_count")) is not int or report["prediction_count"] != len(rows):
            raise RuntimeError("Completed prediction count is inconsistent")
        for row in rows:
            for a_key, b_key in (("prob_a", "prob_b"), ("no_odds_prob_a", "no_odds_prob_b")):
                a, b = row.get(a_key), row.get(b_key)
                if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (float, int))
                        or not isinstance(b, (float, int)) or not math.isfinite(a) or not math.isfinite(b)
                        or not 0 <= a <= 1 or not 0 <= b <= 1 or not math.isclose(a + b, 1, abs_tol=1e-12)):
                    raise RuntimeError("Completed prediction contains invalid probabilities")

    def published_pair(self):
        """Read only the final data-and-prediction commit, never a candidate."""
        publication = self._paths(_read(self.published_path))
        if publication.get("schema") != 1 or publication.get("binding") != self.binding:
            raise RuntimeError("Published result binding conflicts with this startup")
        pointer, pin = publication["snapshot"], publication["prediction_report"]
        self._verify_pin(pin)
        report = self._paths(_read(pin["path"]))
        if publication.get("generation_id") != generation_id(pointer):
            raise RuntimeError("Published result generation is inconsistent")
        self._validate_pair(pointer, report)
        return publication, pointer, report

    def _publish_pair(self, job_id, pointer, report):
        self._validate_pair(pointer, report)
        path = self.root / "published_reports" / f"{job_id}.json"
        body = (json.dumps(self._paths(report, encode=True), indent=2, allow_nan=False) + "\n").encode()
        _make_directory(path.parent)
        if path.exists():
            if path.read_bytes() != body:
                raise RuntimeError("Original job's immutable prediction report conflicts with recovery")
        else:
            # The service writer lock covers this new immutable report. A kill
            # during its temporary write leaves the prior publication intact.
            write_bytes_atomically(body, path)
        publication = {
            "schema": 1, "binding": self.binding, "job_id": job_id,
            "generation_id": generation_id(pointer), "snapshot": pointer,
            "prediction_report": {"path": str(path), "bytes": len(body),
                                  "sha256": hashlib.sha256(body).hexdigest()},
            "published_at_utc": _now().isoformat(),
        }
        # This is the single served commit point. Its predecessor remains live
        # throughout acquisition, feature building, and prediction failures.
        _write(self.published_path, self._paths(publication, encode=True))
        return report

    def _ensure_original_claim(self, pending, instant):
        if pending["decision"]["action"] == "manual_refresh":
            return None
        schedule = self._schedule(pending["job_id"])
        decision = pending["decision"]
        original_now = _now(datetime.fromisoformat(decision["now_utc"]))
        calendar = schedule.load_state()
        attempt = ((calendar["weekly"].get(decision["iso_week"]) or {}).get("slots") or {}).get(decision["slot"])
        if attempt is None:
            claim = schedule.claim_data_update(now=original_now)
            if (not claim.get("claimed") or claim.get("claim_id") != pending["job_id"]
                    or claim.get("iso_week") != decision["iso_week"]
                    or claim.get("slot") != decision["slot"]
                    or claim.get("due_at_utc") != decision["due_at_utc"]):
                raise RuntimeError("Original slot intent does not reproduce its calendar claim")
            attempt = {"claim_id": claim["claim_id"], "status": "claimed"}
        if attempt.get("claim_id") != pending["job_id"]:
            raise RuntimeError("Original slot is already associated with another job")
        if attempt["status"] == "claimed":
            schedule.protect_data_completion(pending["job_id"], now=instant)
        elif not ((attempt["status"] == "succeeded" and pending.get("phase") in {"data_completed", "published", "calendar_completed"})
                  or (attempt["status"] == "failed" and pending.get("phase") == "failure_completed")):
            raise RuntimeError("Saved calendar outcome conflicts with the original unfinished job")
        return schedule

    def _predict(self, job_id, instant, pointer):
        if _now() < instant:
            raise RuntimeError("Prediction publication cannot precede its original decision")
        if self.published_path.exists():
            publication, published, saved = self.published_pair()
            if publication["job_id"] == job_id:
                if generation_id(published) != generation_id(pointer) or any(
                        published.get(key) != pointer.get(key) for key in ("artifacts", "models")):
                    raise RuntimeError("Original published job conflicts with its saved completion")
                return saved
        report = self._predictor()(
            accepted_pointer=pointer, selected_manifest=self.runtime_manifest,
            state_dir=self.root / "prediction_runs" / job_id,
            decision_at=instant, force_refresh=False,
        )
        if report.get("accepted_generation") != generation_id(self.current_snapshot()):
            raise RuntimeError("Prediction generation differs from the current accepted data")
        current = self.current_snapshot()
        if generation_id(pointer) != generation_id(current):
            raise RuntimeError("Original prediction job is no longer the current completed candidate")
        return self._publish_pair(job_id, current, report)

    def _resume(self, state, instant):
        pending = state["pending"]
        schedule = self._ensure_original_claim(pending, instant)
        original_now = _now(datetime.fromisoformat(pending["decision"]["now_utc"]))
        if pending["phase"] not in {"data_completed", "published", "calendar_completed", "failure_completed"}:
            pending["phase"] = "claimed"
            self._save(state, instant)
            producer = self._producer()
            active = producer.inspect_active_refresh(self.data_root)
            if active is not None and active.get("job_id") != pending["job_id"]:
                raise RuntimeError("Another original producer job requires recovery")
            result = producer.recover_incremental_refresh(
                state_dir=self.data_root, baseline_dir=self.baseline,
                selected_manifest=self.selected_manifest,
            ) if active is not None else None
            if result is None:
                result = producer.run_incremental_refresh(
                    baseline_dir=self.baseline, state_dir=self.data_root,
                    selected_manifest=self.selected_manifest,
                    job_id=pending["job_id"], decision_at=original_now,
                )
            if result.get("job_id") not in (None, pending["job_id"]):
                raise RuntimeError("Recovered completion belongs to another original job")
            if result.get("status") == "failed" and result.get("failure_complete") is True:
                pending["result"] = result
                pending["completed_at_utc"] = self._completion_time(result, original_now)
                pending["phase"] = "failure_completed"
                self._save(state, instant)
            elif result.get("status") != "completed":
                pending["producer_status"] = result.get("status", "held")
                state["last_error"] = "Original source work requires recovery before another claim"
                self._save(state, instant)
                return self._publish_status(state, instant, state_name="held")
            else:
                pending["result"] = result
                pending["completed_at_utc"] = self._completion_time(result, original_now)
                pending["phase"] = "data_completed"
                self._save(state, instant)
        if pending["phase"] == "failure_completed":
            if schedule is not None:
                schedule.complete_data_update(
                    pending["job_id"], succeeded=False,
                    error=str(pending["result"].get("error") or "complete_source_failure"),
                    now=datetime.fromisoformat(pending["completed_at_utc"]),
                )
            state["last_completed_job"] = {
                "job_id": pending["job_id"], "decision": pending["decision"],
                "status": "failed", "completed_at_utc": pending["completed_at_utc"],
            }
            state.setdefault("completed_jobs", {})[pending["job_id"]] = state["last_completed_job"]
            state["pending"] = None
            state["last_error"] = "Source failure completed; accepted data preserved"
            self._save(state, instant)
            return self._publish_status(state, instant)
        self._predict(pending["job_id"], original_now, pending["result"])
        publication = self.published_pair()[0]
        published_at = _now(datetime.fromisoformat(publication["published_at_utc"]))
        if published_at < original_now:
            raise RuntimeError("Published workflow completion predates its original decision")
        if pending["phase"] != "calendar_completed":
            pending["published_at_utc"] = publication["published_at_utc"]
            pending["phase"] = "published"
            self._save(state, instant)
            if schedule is not None:
                schedule.complete_data_update(
                    pending["job_id"], succeeded=True,
                    accepted_update_id=pending["job_id"],
                    accepted_at=published_at, now=published_at,
                )
            pending["phase"] = "calendar_completed"
            self._save(state, instant)
        state["last_completed_job"] = {
            "job_id": pending["job_id"], "decision": pending["decision"],
            "data_completed_at_utc": pending["completed_at_utc"],
            "completed_at_utc": publication["published_at_utc"],
            "new_rows": pending["result"].get("new_rows"),
            "total_rows": pending["result"].get("total_rows"),
            "generation": generation_id(self.current_snapshot()),
        }
        state.setdefault("completed_jobs", {})[pending["job_id"]] = state["last_completed_job"]
        state["pending"] = None
        state["last_error"] = None
        self._save(state, instant)
        return self._publish_status(state, instant, state_name="waiting")

    def _publish_status(self, state, instant, *, state_name="waiting"):
        current = self.current_snapshot()
        served = None
        if self.published_path.exists():
            served = self._paths(_read(self.published_path))
        status = {
            "service": "ufc-refresh-predictions", "state": state_name,
            "updated_at_utc": _now().isoformat(), "decision_at_utc": instant.isoformat(),
            "trading_enabled": False, "monthly_enabled": False,
            "retraining_enabled": False,
            "current_generation": generation_id(current),
            "pending_data_generation": generation_id(current),
            "served_generation": None if served is None else served.get("generation_id"),
            "total_rows": current.get("total_rows") or current.get("artifacts", {}).get("fights_cleaned.csv", {}).get("rows"),
            "schedule": self._schedule().plan_data_update(now=instant),
            "pending_job": None if not state.get("pending") else {
                "job_id": state["pending"]["job_id"], "phase": state["pending"]["phase"],
                "original_decision_at": state["pending"]["decision"]["now_utc"],
            },
            "last_completed_job": state.get("last_completed_job"),
            "last_error": state.get("last_error"),
        }
        _write(self.status_path, status)
        return status

    def tick(self, *, now=None):
        instant = _now(now)
        with _exclusive_file_lock(self.lock_path):
            state = self._state()
            try:
                # Recover the original association even when its old due window
                # has closed, before admitting current data or fresh planning.
                if state.get("pending"):
                    return self._resume(state, instant)
                active = self._producer().inspect_active_refresh(self.data_root)
                if active is not None:
                    raise RuntimeError("Unassociated producer job requires original intent recovery")
                self.current_snapshot()
                decision = self._schedule().plan_data_update(now=instant)
                if decision["action"] != ACTION_DATA_UPDATE:
                    return self._publish_status(state, instant)
                job_id = f"{decision['iso_week']}-{decision['slot']}"
                state["pending"] = {"job_id": job_id, "decision": decision, "phase": "intent"}
                self._save(state, instant)
                return self._resume(state, instant)
            except Exception as exc:
                state["last_error"] = f"{type(exc).__name__}: {exc}"[:1000]
                self._save(state, instant)
                logger.exception("Original refresh job held for recovery")
                # A failed status write must not erase the original exception.
                try:
                    self._publish_status(state, instant, state_name="held")
                except Exception:
                    logger.exception("Unable to publish refresh status")
                raise

    def initialize_predictions(self, *, now=None):
        instant = _now(now)
        with _exclusive_file_lock(self.lock_path):
            state = self._state()
            if state.get("pending"):
                return self._resume(state, instant)
            pointer = self.current_snapshot()
            initial = state.setdefault("initial_prediction", {
                "decision_at_utc": instant.isoformat(), "generation": generation_id(pointer),
            })
            self._save(state, instant)
            original_now = _now(datetime.fromisoformat(initial["decision_at_utc"]))
            report = self._predict("initial", original_now, pointer)
            self._save(state, instant)
            self._publish_status(state, instant)
            return report

    def refresh_now(self, *, job_id, now=None):
        """Run an explicitly requested manual refresh without inventing a slot."""
        instant = _now(now)
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", job_id):
            raise ValueError("manual job ID must be a bounded path-safe identifier")
        if (job_id.casefold() == "initial" or re.fullmatch(r"\d{4}-W\d{2}-(tuesday|wednesday)", job_id, re.IGNORECASE)
                or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", job_id, re.IGNORECASE)):
            raise ValueError("manual job ID is reserved for calendar, startup, or the filesystem")
        with _exclusive_file_lock(self.lock_path):
            state = self._state()
            if state.get("pending"):
                if state["pending"]["job_id"] != job_id:
                    raise RuntimeError("Recover the original pending job before another manual refresh")
            elif job_id in state.get("completed_jobs", {}):
                return self._publish_status(state, instant)
            else:
                if self._producer().inspect_active_refresh(self.data_root) is not None:
                    raise RuntimeError("Recover the original producer before a manual refresh")
                state["pending"] = {
                    "job_id": job_id, "phase": "intent",
                    "decision": {"action": "manual_refresh", "now_utc": instant.isoformat()},
                }
                self._save(state, instant)
            try:
                return self._resume(state, instant)
            except Exception as exc:
                state["last_error"] = f"{type(exc).__name__}: {exc}"[:1000]
                self._save(state, instant)
                try:
                    self._publish_status(state, instant, state_name="held")
                except Exception:
                    logger.exception("Unable to publish manual refresh status")
                raise


def create_app(service, *, worker=None, trading=None):
    import hmac
    from flask import Flask, jsonify, redirect, render_template_string, request, session
    web_root = Path(__file__).resolve().parents[1] / "web"
    app = Flask(__name__, template_folder=str(web_root / "templates"),
                static_folder=str(web_root / "static"))
    token = str(os.getenv("WEB_DASHBOARD_TOKEN", "") or "")
    app.secret_key = hashlib.sha256(("ufc-refresh-session:" + token).encode()).digest() if token else os.urandom(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=True)
    public_dashboard_paths = set()

    @app.before_request
    def authenticate():
        if trading is not None and request.method in {"GET", "HEAD"} and (
            request.path in public_dashboard_paths or request.endpoint == "static"
        ):
            return None
        if token and request.path not in {"/healthz", "/readyz", "/login"}:
            supplied = (request.headers.get("Authorization", "").removeprefix("Bearer ")
                        or request.headers.get("X-Dashboard-Token", ""))
            if not session.get("refresh_authenticated") and not hmac.compare_digest(supplied, token):
                if request.path in {"/", "/predictions"}:
                    return redirect("/login")
                return jsonify({"error": "authentication_required"}), 401

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if trading is not None and request.method == "GET":
            return redirect("/")
        if not token:
            return redirect("/predictions")
        if request.method == "POST" and hmac.compare_digest(request.form.get("token", ""), token):
            session["refresh_authenticated"] = True
            return redirect("/predictions")
        return render_template_string(
            "<!doctype html><meta name='viewport' content='width=device-width'><title>UFC dashboard access</title>"
            "<main style='max-width:440px;margin:60px auto;font:16px system-ui'><h1>UFC predictions</h1>"
            "<form method='post'><label>Dashboard token <input type='password' name='token' autocomplete='current-password' required></label>"
            "<button type='submit'>Open predictions</button></form></main>"
        ), 401 if request.method == "POST" else 200

    def report():
        return service.published_pair()[2]

    def trading_state():
        if trading is None:
            return {"effective_live_mode": "off", "trading_enabled": False, "orders_disabled": True,
                    "paused": False}
        try:
            state = trading.status()
            current = {key: state.get(key) for key in
                       ("effective_live_mode", "trading_enabled", "orders_disabled", "paused")}
            current["effective_live_mode"] = current["effective_live_mode"] or getattr(trading, "mode", "off")
            return current
        except Exception:
            return {"effective_live_mode": getattr(trading, "mode", "off"), "trading_enabled": False,
                    "orders_disabled": getattr(trading, "mode", "off") != "real", "paused": False}

    @app.context_processor
    def dashboard_mode_context():
        state = trading_state()
        return {"dashboard_trading_text": (
            "UFC trading is paused." if state.get("paused") else
            "UFC real trading is enabled under the existing entry and risk rules."
            if state.get("trading_enabled") else "UFC trading is off or waiting for verified startup."
        )}

    @app.get("/healthz")
    def health():
        return jsonify({"alive": True, **trading_state()})

    @app.get("/readyz")
    def ready():
        try:
            _publication, current, saved = service.published_pair()
            if worker is not None and not worker.is_alive():
                raise RuntimeError("Refresh worker is not running")
            components = {"refresh": {"ready": True}}
            state = trading_state()
            if trading is not None:
                components["trading_consumer"] = trading.readiness()
                state = {key: components["trading_consumer"][key] for key in
                         ("effective_live_mode", "trading_enabled", "orders_disabled", "paused")}
            return jsonify({"ready": True, "accepted_generation": saved["accepted_generation"],
                            "total_rows": current.get("total_rows") or current.get("artifacts", {}).get("fights_cleaned.csv", {}).get("rows"),
                            "prediction_count": saved.get("prediction_count", len(saved.get("predictions", []))),
                            **state, "components": components})
        except Exception:
            return jsonify({"ready": False,
                            "reason": "combined_workers_unavailable" if trading is not None else "accepted_predictions_unavailable",
                            **trading_state()}), 503

    if trading is not None:
        @app.get("/api/trading-status")
        def trading_status():
            try:
                return jsonify(trading.status())
            except Exception:
                return jsonify({"state": "starting", "orders_disabled": True,
                                "trading_enabled": False}), 503

        @app.get("/api/trading-predictions")
        def trading_predictions():
            try:
                return jsonify(trading.report())
            except Exception:
                return jsonify({"error": "current_trading_predictions_unavailable"}), 503

        @app.post("/api/trading-control")
        def trading_control():
            if not token:
                return jsonify({"error": "authenticated_trading_control_unavailable"}), 503
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or set(payload) != {"paused"} or type(payload["paused"]) is not bool:
                return jsonify({"error": "paused_must_be_a_boolean"}), 400
            from src.trading_controls import set_paused
            try:
                return jsonify(set_paused(trading.state_dir, payload["paused"]))
            except Exception:
                return jsonify({"error": "trading_control_unavailable"}), 503

    @app.get("/api/refresh-status")
    def status():
        try:
            saved = _read(service.status_path)
            saved["current_generation"] = generation_id(service.current_snapshot())
            saved["pending_data_generation"] = saved["current_generation"]
            saved["served_generation"] = service.published_pair()[0]["generation_id"] if service.published_path.exists() else None
            return jsonify(saved)
        except Exception:
            return jsonify({"state": "starting", "trading_enabled": False}), 503

    @app.get("/api/refresh-predictions" if trading is not None else "/api/predictions")
    def predictions():
        try:
            return jsonify(report())
        except Exception:
            return jsonify({"error": "current_predictions_unavailable"}), 503

    def page():
        try:
            saved = report()
            state = trading_state()
            return render_template_string(
                "<!doctype html><html><meta name='viewport' content='width=device-width'>"
                "<title>UFC predictions</title><style>body{max-width:960px;margin:40px auto;padding:0 20px;font:16px system-ui;color:#172233}"
                "table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px 8px;border-bottom:1px solid #ddd}"
                "small{color:#566377}a{color:#185ca1}</style><body>"
                "<h1>UFC predictions</h1><p>Upcoming fights scored with the saved models and served accepted data. {{trading_text}}</p>"
                "<p><small>Updated {{updated}} · Generation {{generation}}</small></p>"
                "<table><thead><tr><th>Event</th><th>Fight</th><th>Win probability</th></tr></thead><tbody>"
                "{% for row in rows %}<tr><td>{{row.get('event_date','')}}<br><small>{{row.get('event_title',row.get('event_name',row.get('event','')))}}</small></td>"
                "<td>{{row.get('fighter_a','')}}<br>{{row.get('fighter_b','')}}</td>"
                "<td>{% if row.get('prob_a') is number %}{{ '%.1f'|format(100*row['prob_a']) }}%{% else %}Unavailable{% endif %}<br>"
                "{% if row.get('prob_b') is number %}{{ '%.1f'|format(100*row['prob_b']) }}%{% else %}Unavailable{% endif %}</td></tr>"
                "{% else %}<tr><td colspan='3'>No upcoming fights are available in the saved source window.</td></tr>{% endfor %}"
                "</tbody></table><p><a href='/api/predictions'>Prediction data</a> · <a href='/api/refresh-status'>Refresh status</a></p></body></html>",
                generation=saved["accepted_generation"], updated=saved.get("decision_at", ""), rows=saved.get("predictions", []),
                trading_text=("UFC trading is paused." if state.get("paused")
                              else "UFC real trading is enabled under the existing entry and risk rules."
                              if state.get("trading_enabled")
                              else "UFC trading is off or waiting for verified startup."),
            )
        except Exception:
            return "Accepted predictions are being prepared.", 503
    if trading is None:
        app.add_url_rule("/", view_func=page)
        app.add_url_rule("/predictions", view_func=page)
    else:
        from src.web.refresh_dashboard import install_dashboard
        public_dashboard_paths.update(install_dashboard(app, service, trading))
    return app


def serve(service, *, host="0.0.0.0", port=5050, poll_seconds=60):
    stop = threading.Event()

    def work():
        while not stop.is_set():
            try:
                service.tick()
                if not service.published_path.exists():
                    service.initialize_predictions()
            except Exception:
                logger.exception("Refresh worker retained its original state")
            stop.wait(poll_seconds)

    worker = threading.Thread(target=work, name="ufc-refresh-calendar", daemon=True)
    worker.start()
    try:
        create_app(service, worker=worker).run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        stop.set()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "tick", "refresh-now", "initialize-predictions", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5050")))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--now", help="Explicit aware ISO decision instant for a local tick")
    parser.add_argument("--job-id", help="Unique original ID for an explicitly requested manual refresh")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least one second")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = RefreshService(root=args.root, baseline=args.baseline, selected_manifest=args.selected_manifest)
    instant = _now(datetime.fromisoformat(args.now)) if args.now else None
    if args.command == "serve":
        if args.now:
            parser.error("serve uses the actual clock")
        serve(service, host=args.host, port=args.port, poll_seconds=args.poll_seconds)
    elif args.command == "tick":
        print(json.dumps(service.tick(now=instant), indent=2))
    elif args.command == "initialize-predictions":
        print(json.dumps(service.initialize_predictions(now=instant), indent=2))
    elif args.command == "refresh-now":
        if not args.job_id:
            parser.error("refresh-now requires --job-id")
        print(json.dumps(service.refresh_now(job_id=args.job_id, now=instant), indent=2))
    else:
        print(json.dumps(_read(service.status_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
