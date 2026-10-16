"""Original local saved-helper association; never a replacement receipt journal."""
from pathlib import Path
import json

from src.data.roster_hosted_store import _read, _require, _safe, file_pin, verify_pin

NAME = "launcher-helper-binding.json"


def validate_helper_binding(configuration_pin, controls, *, journal_root=None):
    from src.data.roster_hosted_preservation import _native

    configuration = json.loads(_read(verify_pin(configuration_pin)["path"]))
    _require(json.loads(_read(verify_pin(configuration["controls"])["path"])) == controls,
             "Helper association needs exact original custody controls")
    root = Path(configuration_pin["path"]).parent
    path = root / NAME
    _native(path, directory=False)
    value = json.loads(_read(path))
    _require(set(value) == {"schema", "kind", "configuration", "controls", "journal"}
             and value["schema"] == 1 and value["kind"] == "original_local_saved_helper_v1"
             and value["configuration"] == configuration_pin
             and value["controls"] == configuration["controls"],
             "Exact original launcher helper association required")
    journal = Path(value["journal"]["path"])
    _require(_native(journal, directory=True) == value["journal"]
             and not journal.stat().st_mode & 0o077,
             "Original private native helper journal changed")
    _require(journal_root is None or _safe(journal_root) == journal,
             "Original helper journal cannot be replaced")
    retention = json.loads(_read(verify_pin(controls["retention_binding"])["path"]))
    others = [root, Path(controls["root"]["path"]), Path(retention["mutable_root"]["path"]),
              Path(retention["directories"]["root"]["path"]),
              *[Path(row["path"]) for row in controls["accepted_roots"]]]
    _require(all(journal != other and journal not in other.parents and other not in journal.parents
                 for other in others), "Helper journal must be disjoint from all role trees")
    return file_pin(path)


def provision_helper_binding(configuration_pin, controls, journal_root):
    """Bind one unused protected journal before the first preparation launch."""
    from src.data.roster_hosted_preservation import _native, _public_original

    configuration = json.loads(_read(verify_pin(configuration_pin)["path"]))
    _require(json.loads(_read(verify_pin(configuration["controls"])["path"])) == controls,
             "Helper association needs exact original custody controls")
    root = Path(configuration_pin["path"]).parent
    path = root / NAME
    journal = Path(_native(journal_root, directory=True)["path"])
    _require(not list(journal.iterdir()) and not journal.stat().st_mode & 0o077,
             "Unused private native helper journal required for initial preparation")
    _require(not path.exists(), "Original helper association already exists; preserve it")
    # Validate disjointness before creating even a control file.
    retention = json.loads(_read(verify_pin(controls["retention_binding"])["path"]))
    others = [root, Path(controls["root"]["path"]), Path(retention["mutable_root"]["path"]),
              Path(retention["directories"]["root"]["path"]),
              *[Path(row["path"]) for row in controls["accepted_roots"]]]
    _require(all(journal != other and journal not in other.parents and other not in journal.parents
                 for other in others), "Helper journal must be disjoint from all role trees")
    _public_original(path, {"schema": 1, "kind": "original_local_saved_helper_v1",
        "configuration": configuration_pin, "controls": configuration["controls"],
        "journal": _native(journal, directory=True)})
    return validate_helper_binding(configuration_pin, controls, journal_root=journal)
