"""Run the complete contained roster candidate workflow with synthetic replies.

No live transport, store creation, clock override, scheduler or activation
option exists. Exact immutable parent/worker/source authority and local reply
objects are verified before consuming a fresh source plan. Every callback reads
only its next pinned local fixture; the contained worker cannot supply targets.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def local_pin(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "Canonical local file required")
    for part in (path, *path.parents):
        require(not part.is_symlink(), "Linked local file forbidden")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o222,
            "Regular immutable unaliased file required")
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def checked(pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}
            and local_pin(pin["path"]) == pin, "Pinned workflow artifact drift")
    return Path(pin["path"]).read_bytes()


def store_binding(plan, source, value):
    path = Path(value).absolute()
    require(path == path.resolve() and path.is_dir(), "Existing canonical store root required")
    for part in (path, *path.parents):
        require(not part.is_symlink(), "Linked store root forbidden")
    require(source.get("store_root") == plan.get("broker_store_root") == str(path),
            "Source and supervisor plans must bind the exact durable store root")
    return path


class SyntheticReplies:
    """A sealed finite local response sequence, without any source transport."""
    def __init__(self, catalog, source, input_pins, *, is_challenge, document_supported, solve, cookie, post_url):
        require(isinstance(catalog, dict) and set(catalog) == {"schema_version", "execution_mode", "responses"}
                and catalog["schema_version"] == 1 and catalog["execution_mode"] == "synthetic",
                "Only an exact synthetic response catalog is supported")
        self.responses = deepcopy(catalog["responses"])
        require(isinstance(self.responses, list) and 0 < len(self.responses) <= source["max_requests"],
                "Finite fixture response sequence required")
        self.position = 0
        for reply in self.responses:
            require(isinstance(reply, dict) and set(reply) == {"method", "url", "body", "status", "headers", "complete", "error", "wire_metadata"},
                    "Exact synthetic reply shape required")
            require(reply["body"] in input_pins, "Fixture body must be in supervisor input pins")
            raw = checked(reply["body"])
            require(len(raw) <= 8388609 and type(reply["complete"]) is bool
                    and (reply["error"] is None or isinstance(reply["error"], str))
                    and isinstance(reply["wire_metadata"], dict), "Invalid bounded synthetic reply")
            require(isinstance(reply["headers"], list) and len(reply["headers"]) <= 100
                    and all(isinstance(h, list) and len(h) == 2 and all(isinstance(v,str) for v in h)
                            for h in reply["headers"]), "Invalid synthetic headers")
        # Simulate the exact broker target sequence before claiming any plan.
        # Terminal source failures may be represented by a proper short prefix;
        # unsupported/extra targets or missing successful steps fail preflight.
        index, rounds = 0, 0
        def take(method, url):
            nonlocal index
            require(index < len(self.responses), "Fixture sequence is incomplete")
            reply = self.responses[index]
            index += 1
            require((reply["method"], reply["url"]) == (method, url), "Fixture target outside exact plan sequence")
            return reply, checked(reply["body"])
        def failed(reply):
            return not reply["complete"] or bool(reply["error"]) or reply["status"] != 200
        def stop():
            require(index == len(self.responses), "Extra fixture targets after terminal failure")
        if source["membership"]:
            reply, raw = take("GET", "https://www.roster.watch/")
            if failed(reply) or is_challenge(raw):
                stop()
                return
        for url in source["documents"]:
            reply, raw = take("GET", url)
            if failed(reply):
                stop()
                return
            if is_challenge(raw):
                if not source["challenge"]["enabled"] or rounds:
                    stop()
                    return
                try:
                    solve(raw, checked(source["challenge"]["template_pin"]))
                except RuntimeError:
                    stop()
                    return
                rounds += 1
                posted, _ = take("POST", post_url)
                if not posted["complete"] or posted["error"] or type(posted["status"]) is not int or not 200 <= posted["status"] < 300:
                    stop()
                    return
                try:
                    cookie(posted["headers"])
                except RuntimeError:
                    stop()
                    return
                reply, raw = take("GET", url)
            if failed(reply) or not document_supported(url, raw, reply["headers"]):
                stop()
                return
        require(index == len(self.responses), "Unplanned extra synthetic replies")

    def __call__(self, method, url, headers, form, plan):
        require(self.position < len(self.responses), "Synthetic reply sequence exhausted")
        reply = self.responses[self.position]
        require((method, url) == (reply["method"], reply["url"]), "Synthetic callback target drift")
        body = checked(reply["body"])
        self.position += 1
        result = {key: deepcopy(reply[key]) for key in ("status", "headers", "complete", "error", "wire_metadata")}
        result["wire_metadata"].update(transport="synthetic", synthetic=True, fixture_pin=reply["body"])
        return dict(result, body=body)


def prepare(args):
    plan_pin, review_pin = local_pin(args.plan), local_pin(args.review)
    require(review_pin["sha256"] == args.review_sha256, "External supervisor review hash mismatch")
    plan, review = json.loads(checked(plan_pin)), json.loads(checked(review_pin))
    require(review.get("status") == "passed" and review.get("blockers") == [] and review.get("plan") == plan_pin,
            "Supervisor review does not bind plan")
    code, inputs = plan["code_pins"], plan["input_pins"]
    for pin in code + inputs:
        checked(pin)
    package = Path(__file__).resolve().parents[1]
    required = [Path(__file__).resolve(), Path(sys.executable).resolve(),
                package / "scripts/supervise_roster_acquisition_linux.py",
                *[package / p for p in ("src/__init__.py", "src/data/__init__.py",
                 "src/data/roster_hosted_store.py", "src/data/roster_hosted_broker.py",
                 "src/data/roster_hosted_mailbox.py", "src/data/ufcstats_identity_access.py")]]
    require(all(local_pin(path) in code for path in required), "Parent interpreter/source must be sealed in worker review")
    source_pin, source_review_pin, fixtures_pin = (plan[key] for key in ("broker_plan", "broker_review", "synthetic_responses"))
    require(all(pin in inputs for pin in (source_pin, source_review_pin, fixtures_pin)),
            "Source plan/review and synthetic catalog must bind supervisor inputs")
    source = json.loads(checked(source_pin))
    source_review = json.loads(checked(source_review_pin))
    require(source.get("execution_mode") == "synthetic" and source.get("publication_requested") is False
            and source_review.get("status") == "passed" and source_review.get("blockers") == []
            and source_review.get("plan") == source_pin, "Exact synthetic source review required")
    store_root = store_binding(plan, source, args.store)
    argv = plan["argv"]
    require(len(argv) == 6 and argv[0] == "--template" and argv[2] == "--template-sha256"
            and argv[4] == "--source-plan-sha256" and argv[5] == source_pin["sha256"], "Exact candidate worker arguments required")
    template_pin = local_pin(argv[1])
    require(template_pin in inputs and template_pin["sha256"] == argv[3]
            and Path(plan["worker"]["path"]) == package / "scripts/prepare_roster_hosted_workflow_worker.py",
            "Exact pinned workflow worker/template required")
    template = json.loads(checked(template_pin))
    require(template.get("acquisition_plan_id") == source.get("plan_id")
            and template.get("publication_requested") is False and template.get("network_requests") == 0,
            "Template source/publication binding mismatch")
    control_path = package / "scripts/supervise_roster_acquisition_linux.py"
    spec = importlib.util.spec_from_file_location("roster_workflow_supervisor", control_path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    require(control.authority(args.plan,args.review,args.review_sha256) == plan, "Supervisor authority drift")
    sys.path.insert(0,str(package))
    from src.data.roster_hosted_store import HostedControlStore
    from src.data.roster_hosted_broker import SourceBroker, is_challenge, _document_supported, _solve, _cookie, POST_URL
    from src.data.roster_hosted_mailbox import ParentMailbox
    replies = SyntheticReplies(json.loads(checked(fixtures_pin)), source, inputs,
        is_challenge=is_challenge, document_supported=_document_supported, solve=_solve, cookie=_cookie, post_url=POST_URL)
    # Existing store only. All reviewed static authority and fixture targets were
    # checked before this irreversible reservation; no implicit reset/recreate.
    store = HostedControlStore(store_root)
    require(store.root == store_root, "Store namespace changed after preflight")
    def verify_all():
        require(control.authority(args.plan,args.review,args.review_sha256) == plan, "Workflow authority drift")
        for pin in inputs:
            checked(pin)
    store.claim_plan(source["plan_id"],source_pin,source_review_pin)
    broker = SourceBroker(store,source_pin,source_review_pin,synthetic_wire=replies,protected_verifier=verify_all)
    return control.supervise(args,broker=ParentMailbox(broker,artifact_roots=plan["broker_artifact_roots"]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","review","review-sha256","control-root","store"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    try:
        return prepare(args)
    except (RuntimeError,OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"failed_synthetic_preparation","error":type(exc).__name__+": "+str(exc),
            "source_requests":0,"publication":False,"preserve_existing_evidence":True}),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
