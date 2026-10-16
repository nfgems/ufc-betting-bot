"""Explicitly offline, publication-disabled candidate entrypoint.

Invoke from the sealed release under the Linux supervisor. No default spec,
source URL, production path, publication flag or scheduler entrypoint exists.
"""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    from src.data.roster_hosted_candidate import prepare_candidate
    report = prepare_candidate(args.spec, args.spec_sha256)
    print(json.dumps({key: report[key] for key in (
        "status", "publication_performed", "network_requests", "current_members",
        "total_owners", "historical_owners", "pending_members")}, sort_keys=True))


if __name__ == "__main__":
    main()
