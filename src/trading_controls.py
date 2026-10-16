"""Durable UFC pause shared by the supervisor and the order executor."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def paused_status(state_dir=None):
    root = state_dir or os.environ.get("UFC_REFRESH_TRADING_STATE_DIR")
    if not root:
        return {"paused": False, "source": "unconfigured"}
    path = Path(root) / "control.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or type(value.get("paused")) is not bool:
            raise ValueError("Invalid trading pause record")
        return value
    except FileNotFoundError:
        return {"paused": False, "source": "no_pause_requested"}
    except (OSError, ValueError):
        return {"paused": True, "source": "unreadable_pause_record"}


def require_unpaused():
    if paused_status().get("paused"):
        raise RuntimeError("UFC real trading is paused")


def set_paused(state_dir, paused):
    if type(paused) is not bool:
        raise ValueError("paused must be a boolean")
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"paused": paused, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "source": "authenticated_control"}
    descriptor, temporary = tempfile.mkstemp(prefix=".control-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "control.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload
