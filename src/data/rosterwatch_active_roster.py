"""Roster Watch authoritative UFC roster acquisition and reconciliation.

Roster Watch owns *membership*.  UFC and UFCStats records may enrich a member,
but they may neither add nor remove a member in this module.  The mutable state
written here is intended to live inside a candidate raw-data generation so it
is published (or discarded) with that generation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from src.config import (
    PENDING_IDENTITY_MAX_AGE_DAYS,
    PENDING_IDENTITY_MAX_COUNT,
    ROSTERWATCH_IDENTITY_REQUEST_CAP,
    ROSTERWATCH_IDENTITY_CAPTURE_FILENAME,
)
from src.data.io_utils import write_bytes_atomically
from src.data.name_utils import (
    canonical_fighter_name_key,
    normalize_person_name,
    normalize_ufcstats_id,
)


ROSTERWATCH_SCHEMA_VERSION = 1
ROSTERWATCH_MEMBERSHIP_SOURCE = "rosterwatch"
ROSTERWATCH_INITIAL_CUTOFF = "2026-09-01"
ROSTERWATCH_INITIAL_ROW_COUNT = 675
ROSTERWATCH_INITIAL_SHA256 = (
    "8cbb71d74549d852f254600d87fe0856bf7e6d1a45758c63ea29eec6815fea19"
)
ROSTERWATCH_INITIAL_LEGACY_ROW_COUNT = 996
ROSTERWATCH_INITIAL_LEGACY_BYTE_COUNT = 486_738
ROSTERWATCH_INITIAL_LEGACY_SHA256 = (
    "f06aa2440d51aa00981c4621e23f7cdfdbc0188517042eab2002d9ae902015fb"
)
ROSTERWATCH_MINIMUM_ROW_COUNT = 100
ROSTERWATCH_RECURRING_GROWTH_RATIO = 1.35
ROSTERWATCH_RECURRING_SHRINK_RATIO = 0.95
ROSTERWATCH_RECURRING_SHRINK_ABSOLUTE = 50
ROSTERWATCH_REQUEST_TIMEOUT = (10.0, 30.0)

ROSTERWATCH_COLUMNS = (
    "fighter",
    "date_of_birth",
    "tuf",
    "r2ufc",
    "dwcs",
    "pride",
    "sf",
    "wec",
    "bmf",
    "hof",
    "country",
    "gender",
    "weightclass",
    "current_rank",
    "rank_flag",
    "current_p4p",
    "p4p_flag",
    "peak_rank",
    "peak_p4p",
    "bouts",
    "wins",
    "win_types",
    "losses",
    "loss_types",
    "streak",
    "best_streak",
    "bonus",
    "slot",
    "elo",
    "peak_elo",
    "opp_elo",
    "debut",
    "debut_result",
    "prev_bout",
    "prev_result",
    "next_date",
    "next_opp",
)

ROSTERWATCH_REVIEWED_ALIASES = {
    "Muhammad Said": "Muhammad Saidov",
    "Matthieu Duclos": "Matthieu Letho Duclos",
    "Bruno Gustavo da Silva": "Bruno Silva",
    "Juan Martinetti": "Adrián Luna Martinetti",
    "Nikolija Milosevic": "Nina Milošević",
    "Michael Page": "Michael Venom Page",
    "Jose Delgado": "Jose Miguel Delgado",
    "Tina Black": "Valesca Machado",
    "Treston Vines": "Tre’ston Vines",
    "Yi Sak Lee": "YiSak Lee",
    "Ce Liu": "Liu Ce",
    "Jingnan Xiong": "Xiong Jingnan",
    "Taiyilake Nueraji": "Nueraji Taiyilake",
}

ROSTERWATCH_REVIEWED_ALIAS_CARRIERS = {
    "Muhammad Said": "ufcstats:b01a5e0dcd9f231d",
    "Matthieu Duclos": (
        "ufc:https://www.ufc.com/athlete/matthieu-letho-duclos"
    ),
    "Bruno Gustavo da Silva": "ufcstats:294aa73dbf37d281",
    "Juan Martinetti": "ufcstats:e3b30b573e7f0d19",
    "Nikolija Milosevic": "ufcstats:eabece0edc2d62ca",
    "Michael Page": "ufcstats:a67d071163962af8",
    "Jose Delgado": "ufcstats:7d6ceff6747f2de2",
    "Tina Black": "ufcstats:a1a91dbf41a029c4",
    "Treston Vines": "ufcstats:563a1d42bb0e3cef",
    "Yi Sak Lee": "ufcstats:e6666480a830f764",
    "Ce Liu": "ufcstats:a9936818d709e2d7",
    "Jingnan Xiong": "ufcstats:2cdb479a0ef0ee1d",
    "Taiyilake Nueraji": "ufcstats:7498b3ac79a3b948",
}

# Identity-resolution evidence for the Roster Watch additions that lacked a stable
# internal carrier at the September 1, 2026 baseline (u31q1 attempt 01 rejected the
# candidate on exactly seven unresolved carriers).  Every record pins the sealed
# u31r1 network evidence it was read from (plan id and decoded-capture SHA-256), so a
# carrier can never be added by name alone.  The owner's binding rule (2026-09-03)
# is ROSTERWATCH_ADDITION_BINDING_RULE: exact name, matching date of birth, the exact
# Dana White's Contender Series bout (date and opponent) on UFCStats, corroborated on
# the official UFC.com athlete page.  A carrier is derived from a record ONLY when its
# status is ROSTERWATCH_ADDITION_BOUND; every other status contributes no carrier, the
# member stays ``pending_external_carrier``, and identity completeness remains false.
# Membership readiness is independent. Changing these reviewed statuses still
# requires an owner decision recorded here with its evidence.
#
# On 2026-09-03 the owner decided the seven unresolved cases (see
# ROSTERWATCH_OWNER_IDENTITY_DECISIONS_2026_09_03).  Each decision is implemented as
# its own explicit binding basis with a RECORD-SCOPED allowlist of the exact
# (name, DOB) keys it may bind; none of them is a global name, DOB, article or
# alias rule, and none of them relaxes the original owner binding rule, which
# remains available unchanged for future additions.
ROSTERWATCH_ADDITION_BINDING_RULE = (
    "exact_name+matching_dob+ufcstats_contender_series_bout+ufccom_athlete_page"
)
ROSTERWATCH_ADDITION_BOUND = "bound"
ROSTERWATCH_ADDITION_PENDING_OWNER = (
    "ufcstats_bound_official_article_corroborated_pending_owner_decision"
)
ROSTERWATCH_ADDITION_NO_CONTENDER_BOUT = "unbound_no_contender_series_bout"
ROSTERWATCH_ADDITION_DOB_CONFLICT = "unbound_dob_conflict"
ROSTERWATCH_ADDITION_STATUSES = frozenset(
    {
        ROSTERWATCH_ADDITION_BOUND,
        ROSTERWATCH_ADDITION_PENDING_OWNER,
        ROSTERWATCH_ADDITION_NO_CONTENDER_BOUT,
        ROSTERWATCH_ADDITION_DOB_CONFLICT,
    }
)
ROSTERWATCH_ADDITION_BASIS_OWNER_RULE = "owner_binding_rule_2026-09-03"
ROSTERWATCH_ADDITION_BASIS_ACCEPTED_CACHE = "accepted_ufcstats_cache_identity_u31q1"
# The accepted-cache basis is closed: only the identity that u31q1 attempt 01 already
# resolved through the accepted UFCStats profile cache may use it.
ROSTERWATCH_ADDITION_ACCEPTED_CACHE_ALLOWLIST = frozenset({("Guilherme Uriel", "1997-02-18")})
ROSTERWATCH_ADDITION_RULE_ELEMENTS = (
    "exact_name",
    "dob_match",
    "contender_series_bout_on_ufcstats",
    "ufccom_athlete_page_corroboration",
)

# Owner identity decisions of 2026-09-03, recorded verbatim.  They are the only
# authority for the three decision bases below and are cited by every record they bind.
ROSTERWATCH_OWNER_IDENTITY_DECISIONS_2026_09_03: dict[str, str] = {
    "1": (
        "Clancy Jr., Humphrey, Apodaca: bind now on exact name + matching DOB + exact DWCS "
        "10.3 bout on UFCStats, corroborated by the official UFC.com Week 3 results/welcome/"
        "weigh-in articles. The UFC.com athlete page is a FOLLOW-UP, not a blocker: record it "
        "as pending; when a fighter is booked on a card and the event page links the profile, "
        "capture it under a small sealed plan and complete the record; any disagreement must "
        "flag, never silently accept."
    ),
    "2": (
        "Galanti: same as 1, plus 'Nick' is an approved spelling of UFCStats 'Nicholas' for "
        "ufcstats:8988705468d04ef7 only (record-scoped alias)."
    ),
    "3": (
        "Szabova and Andrusca (direct signees, no Contender Series bout): bind on exact name + "
        "matching DOB on UFCStats and on the official UFC.com athlete page (JSON-LD birth date), "
        "with the scheduled debut matching on both sites (2026-10-31 vs Tainara Lisboa; "
        "2026-09-05 vs Nathaniel Wood)."
    ),
    "4": (
        "Gasanov: the UFCStats record 30ba4c2779e68a1e (23-2-0, Russia, welterweight) is him. "
        "His DOB is 1994-11-23 (Roster Watch is correct); record UFCStats' 1994-10-23 as a "
        "source error in the DOB-conflict list, do not overwrite anything."
    ),
}
ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE = "2026-09-03"
ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION = (
    "owner_decision_1_2026-09-03_contender_series_ufcstats_bout_official_articles_athlete_page_follow_up"
)
ROSTERWATCH_ADDITION_BASIS_DIRECT_SIGNEE_DECISION = (
    "owner_decision_3_2026-09-03_direct_signee_ufcstats_and_athlete_page_dob_scheduled_debut"
)
ROSTERWATCH_ADDITION_BASIS_DOB_ADJUDICATED_DECISION = (
    "owner_decision_4_2026-09-03_ufcstats_record_bound_rosterwatch_dob_adjudicated"
)
ROSTERWATCH_ADDITION_BASES = frozenset(
    {
        ROSTERWATCH_ADDITION_BASIS_OWNER_RULE,
        ROSTERWATCH_ADDITION_BASIS_ACCEPTED_CACHE,
        ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION,
        ROSTERWATCH_ADDITION_BASIS_DIRECT_SIGNEE_DECISION,
        ROSTERWATCH_ADDITION_BASIS_DOB_ADJUDICATED_DECISION,
    }
)
# Record-scoped allowlists: a decision basis may bind only the exact (name, DOB) keys
# the owner named.  Any other key using one of these bases fails validation.
ROSTERWATCH_ADDITION_CONTENDER_SERIES_DECISION_ALLOWLIST = frozenset(
    {
        ("Sean Clancy Jr.", "2002-11-25"),
        ("Nick Galanti", "1998-06-18"),
        ("Ronald Humphrey", "2000-09-20"),
        ("Alex Apodaca", "1994-06-30"),
    }
)
ROSTERWATCH_ADDITION_DIRECT_SIGNEE_DECISION_ALLOWLIST = frozenset(
    {("Lucia Szabova", "1997-12-23"), ("Pavel Andrusca", "2002-05-08")}
)
ROSTERWATCH_ADDITION_DOB_ADJUDICATION_ALLOWLIST = frozenset(
    {("Magomedrasul Gasanov", "1994-11-23")}
)
# Decision 2: the only approved given-name spelling variant, scoped to one carrier.
# This is not a Nick/Nicholas rule; no other record may claim it.
ROSTERWATCH_ADDITION_APPROVED_NAME_VARIANTS: dict[tuple[str, str], dict[str, str]] = {
    ("Nick Galanti", "1998-06-18"): {
        "rosterwatch_name": "Nick Galanti",
        "ufcstats_listed_name": "Nicholas Galanti",
        "carrier": "ufcstats:8988705468d04ef7",
        "owner_decision": "2",
        "scope": "ufcstats:8988705468d04ef7 only; not a global Nick/Nicholas rule",
    },
}
ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_PENDING = (
    "pending_until_booked_on_a_card_whose_event_page_links_the_profile"
)
ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_COMPLETE = "complete"
ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_STATUSES = frozenset(
    {ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_PENDING, ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_COMPLETE}
)
_U31R1_ATHLETE_PAGE_FOLLOW_UP_PENDING = {
    "status": ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_PENDING,
    "rule": "owner decision 1 (2026-09-03) follow-up",
    "required_when": (
        "the fighter is booked on a UFC card and the official event page links the athlete profile"
    ),
    "capture_rule": (
        "capture the linked route under a small sealed launcher plan; complete this record with "
        "the route, the athlete-page name, the decoded-capture SHA-256 and the JSON-LD birthDate; "
        "any disagreement with the bound name or DOB must flag (the validator rejects the record) "
        "and is never accepted silently"
    ),
    "athlete_page_url": None,
    "athlete_page_name": None,
    "capture_sha256": None,
    "json_ld_birth_date": None,
    "disagreement": None,
}
_U31R1_UFCSTATS_DWCS_10_3 = {
    "event": "DWCS 10.3",
    "event_date": "2026-08-25",
    "event_url": "http://ufcstats.com/event-details/2015c944226ad2df",
}
_U31R1_UFCCOM_ARTICLES = {
    "kind": "official_news_article",
    "athlete_page_url": None,
    "athlete_page_route_evidenced": False,
    "urls": (
        "https://www.ufc.com/news/dana-whites-contender-series-season-10-week-3-results",
        "https://www.ufc.com/news/welcome-ufc-dwcs-season-10-week-3",
        "https://www.ufc.com/news/official-weigh-in-results-dana-whites-contender-series-season-10",
    ),
    "capture_sha256s": (
        "e608050ba95b51722aca559892810cc842fedc0ccada1d0aaaf1fd830d2fc7e7",
        "fb2b1bcb7f8a0d6609fe026c4179ed4b792f80f485d0e9c094a664948c21a12f",
        "0718e8e6d5f8a3d74a23be4d22dda27d68c98829252373706cad633c76c85fdc",
    ),
    # plan_ids lists every sealed plan that contributed to this block (the three articles
    # plus the post-fight article below); article_captures is the positional attribution
    # of the three corroborating articles read from the terminal seals.
    "plan_ids": ("ufccom-dwcs-week3-01", "ufccom-dwcs-hub-03", "ufccom-dwcs-week3-02"),
    "article_captures": (
        {"url": "https://www.ufc.com/news/dana-whites-contender-series-season-10-week-3-results", "capture_sha256": "e608050ba95b51722aca559892810cc842fedc0ccada1d0aaaf1fd830d2fc7e7", "plan_id": "ufccom-dwcs-week3-01", "ordinal": 2},
        {"url": "https://www.ufc.com/news/welcome-ufc-dwcs-season-10-week-3", "capture_sha256": "fb2b1bcb7f8a0d6609fe026c4179ed4b792f80f485d0e9c094a664948c21a12f", "plan_id": "ufccom-dwcs-week3-01", "ordinal": 1},
        {"url": "https://www.ufc.com/news/official-weigh-in-results-dana-whites-contender-series-season-10", "capture_sha256": "0718e8e6d5f8a3d74a23be4d22dda27d68c98829252373706cad633c76c85fdc", "plan_id": "ufccom-dwcs-hub-03", "ordinal": 1},
    ),
    "post_fight_article_without_athlete_links": {
        "url": "https://www.ufc.com/news/dana-white-contender-series-season-10-week-3-alex-apodaca-bella-mirs-next-move-ronald-humphrey",
        "capture_sha256": "9277d4c584eea6839c8149a300ff373e230e6d5a93c63b33c0da6cb4532c615b",
    },
    "athlete_route_search_exhausted": True,
    "athlete_route_search_scope": (
        "sealed captures: UFC.com events index, Contender Series hub pages 1-3, season weigh-in article, "
        "Week 3 welcome/results/post-fight articles, the six roster.watch fighter pages and the roster page; "
        "immutable u31l1 2026-09-01 all-athletes crawl log and upcoming-card scan; none names an athlete-page or "
        "event-page route for the four Week 3 winners, and no route may be guessed"
    ),
}
# Decision 1/2 bind on THE exact DWCS 10.3 bout and THE three sealed official Week 3
# articles; the validator ties every decision-1 record to these exact identifiers, not
# merely to a well-formed bout or an arbitrary ufc.com article.
ROSTERWATCH_DECISION_1_CONTENDER_SERIES_EVENT: dict[str, str] = dict(_U31R1_UFCSTATS_DWCS_10_3)
ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES: dict[str, tuple[object, ...]] = {
    "urls": tuple(_U31R1_UFCCOM_ARTICLES["urls"]),
    "capture_sha256s": tuple(_U31R1_UFCCOM_ARTICLES["capture_sha256s"]),
    "plan_ids": tuple(_U31R1_UFCCOM_ARTICLES["plan_ids"]),
    "article_captures": tuple(dict(item) for item in _U31R1_UFCCOM_ARTICLES["article_captures"]),
}
# The positional attribution must agree with the url/hash tuples it explains.
assert tuple(item["url"] for item in ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["article_captures"]) == ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["urls"]
assert tuple(item["capture_sha256"] for item in ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["article_captures"]) == ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["capture_sha256s"]
assert {item["plan_id"] for item in ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["article_captures"]} <= set(ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES["plan_ids"])
ROSTERWATCH_REVIEWED_ADDITION_EVIDENCE: dict[tuple[str, str], dict[str, object]] = {
    ("Guilherme Uriel", "1997-02-18"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_ACCEPTED_CACHE,
        "carrier": "ufcstats:4252b09f0c8c4472",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": True,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Guilherme Uriel", "date_of_birth": "1997-02-18", "dwcs": True},
        "ufcstats": {
            "id": "4252b09f0c8c4472",
            "url": "http://ufcstats.com/fighter-details/4252b09f0c8c4472",
            "listed_name": "Guilherme Uriel",
            "dob": "1997-02-18",
            "source": "accepted C:/Temp/uv7b5/data/raw/ufc_fighters_scraped.csv row (DOB 'Feb 18, 1997', 6-2-0)",
        },
        "contender_series_bout": {
            **_U31R1_UFCSTATS_DWCS_10_3,
            "opponent": "Mario Piazzon",
            "result": "win",
            "method": "SUB Standing Guillotine Choke",
            "round": "1",
            "time": "0:50",
            "source": "official UFC.com Week 3 results article ('Guilherme Uriel defeats Mario Piazzon via submission (standing guillotine choke) - Round 1, 0:50') and weigh-in article ('Mario Piazzon (238) vs Guilherme Uriel (241.5)'); his UFCStats fighter page was not captured in u31r1, so this element is article-derived, which the accepted-cache basis does not require",
        },
        "ufccom": {"kind": "official_news_article", "athlete_page_url": None, "athlete_page_route_evidenced": False, "capture_sha256s": ("0718e8e6d5f8a3d74a23be4d22dda27d68c98829252373706cad633c76c85fdc",)},
        "note": "Resolved in u31q1 attempt 01 through the pre-existing accepted UFCStats cache identity; retained unchanged.",
    },
    ("Sean Clancy Jr.", "2002-11-25"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION,
        "owner_decision": {"decision": "1", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:c02862a5f6d3fbd0",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": True,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Sean Clancy Jr.", "date_of_birth": "2002-11-25", "dwcs": True, "fighter_page_sha256": "319e92b0eaed5c05c3cdf6f41b79c3aa2be0cb2def935528a38ca7f76703d420"},
        "ufcstats": {
            "id": "c02862a5f6d3fbd0",
            "url": "http://ufcstats.com/fighter-details/c02862a5f6d3fbd0",
            "listed_name": "Sean Clancy Jr.",
            "nickname": "The One",
            "dob": "2002-11-25",
            "record": "9-0-0",
            "listing_capture_sha256": "34d461c308da0ccecd837584f0c00e7cb184fe3e10647eb9f844e262bad0b1ba",
            "fighter_page_sha256": "275729a65d837b8d0f3e283ddb894af51aa273bb123ef935d85b3bc2ae1160b7",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": {**_U31R1_UFCSTATS_DWCS_10_3, "opponent": "Gary Balletto", "opponent_ufcstats_id": "623f2eed9e1495af", "result": "win", "method": "KO/TKO Elbows", "round": "2", "time": "3:54"},
        "ufccom": {**_U31R1_UFCCOM_ARTICLES, "corroborating_text": "Sean Clancy Jr. defeats Gary Balletto via TKO (ground strikes) - Round 2, 3:54 (Week 3 results, August 25, 2026)"},
        "athlete_page_follow_up": dict(_U31R1_ATHLETE_PAGE_FOLLOW_UP_PENDING),
        "note": "Bound by owner decision 1 (2026-09-03): exact name, matching DOB and the exact DWCS 10.3 bout on UFCStats, corroborated by the official UFC.com Week 3 results/welcome/weigh-in articles. The official UFC.com athlete page is a pending follow-up, not a blocker: no route was evidenced on any sealed capture and none may be guessed.",
    },
    ("Nick Galanti", "1998-06-18"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION,
        "owner_decision": {"decision": "2", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "approved_name_variant": dict(ROSTERWATCH_ADDITION_APPROVED_NAME_VARIANTS[("Nick Galanti", "1998-06-18")]),
        "carrier": "ufcstats:8988705468d04ef7",
        "rule_elements": {
            "exact_name": False,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": True,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Nick Galanti", "date_of_birth": "1998-06-18", "dwcs": True, "fighter_page_sha256": "41500717606c6b6229f4c4201b023dc265abf00f1dafc2110c9c569a4d7dc63a"},
        "ufcstats": {
            "id": "8988705468d04ef7",
            "url": "http://ufcstats.com/fighter-details/8988705468d04ef7",
            "listed_name": "Nicholas Galanti",
            "dob": "1998-06-18",
            "record": "8-1-0",
            "listing_capture_sha256": "ba952bab42f5827132e8bafdd8e8864d2f913789850e7514e1ff0cd35cdbb6fd",
            "fighter_page_sha256": "375280b5eff60761ca9ee29f9c50a80661abe77639e59cffd8c4ca6843bfc5d0",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": {**_U31R1_UFCSTATS_DWCS_10_3, "opponent": "Carlos Petruzzella", "opponent_ufcstats_id": "2e1f3640443272ad", "result": "win", "method": "KO/TKO Punch", "round": "1", "time": "0:35"},
        "ufccom": {**_U31R1_UFCCOM_ARTICLES, "corroborating_text": "Nick Galanti defeats Carlos Petruzzella via knockout (punches) - Round 1, 0:35 (Week 3 results, August 25, 2026)"},
        "athlete_page_follow_up": dict(_U31R1_ATHLETE_PAGE_FOLLOW_UP_PENDING),
        "note": "Bound by owner decision 2 (2026-09-03): decision 1 plus the record-scoped approval of 'Nick' as the spelling of UFCStats 'Nicholas' for ufcstats:8988705468d04ef7 only. UFCStats lists 'Nicholas Galanti'; Roster Watch and the official UFC.com articles spell 'Nick Galanti'; DOB and the exact DWCS 10.3 bout match. The athlete page is a pending follow-up.",
    },
    ("Ronald Humphrey", "2000-09-20"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION,
        "owner_decision": {"decision": "1", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:15e0d2df09025bda",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": True,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Ronald Humphrey", "date_of_birth": "2000-09-20", "dwcs": True, "fighter_page_sha256": "b262270f26c7b9eef0455f2fd4288678b89512a7524d62eeb5ccede084044fe0"},
        "ufcstats": {
            "id": "15e0d2df09025bda",
            "url": "http://ufcstats.com/fighter-details/15e0d2df09025bda",
            "listed_name": "Ronald Humphrey",
            "nickname": "The Career Ender",
            "dob": "2000-09-20",
            "record": "5-0-0",
            "excluded_same_surname_listing_rows": (
                {
                    "listed_name": "Abongo Humphrey",
                    "ufcstats_id": "cb1b4d237b0079e6",
                    "listing": "H listing: 5'11, 205 lbs, Orthodox, 7-3-0 (no nickname)",
                    "accepted_cache_dob": "1977-05-27",
                    "accepted_cache_source": "C:/Temp/uv7b5/data/raw/ufc_fighters_scraped.csv row 'Abongo Humphrey' (DOB 'May 27, 1977')",
                    "relationship": "official Week 3 results article names Ronald as 'the fighting son of former Strikeforce athlete Ron \"Abongo\" Humphrey'",
                    "exclusion_basis": "different first name, DOB 1977-05-27 (accepted cache) versus Roster Watch 2000-09-20, 205-lb light heavyweight record 7-3-0, and no DWCS 10.3 bout",
                },
            ),
            "listing_capture_sha256": "a5502f4e84c3b3818e22ddcb4a55845079dfb289793b3b8c6901b3c937bf9c31",
            "fighter_page_sha256": "f15fbe57efff3417887991b69797b1e2847c251098d496f8c2ab32b66da53ff2",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": {**_U31R1_UFCSTATS_DWCS_10_3, "opponent": "Alexis Miranda", "opponent_ufcstats_id": "d458266c796643e2", "result": "win", "method": "SUB Rear Naked Choke", "round": "1", "time": "4:03"},
        "ufccom": {**_U31R1_UFCCOM_ARTICLES, "corroborating_text": "Ronald Humphrey defeats Alexis Miranda via submission (rear-naked choke) - Round 1, 4:03 (Week 3 results, August 25, 2026)"},
        "athlete_page_follow_up": dict(_U31R1_ATHLETE_PAGE_FOLLOW_UP_PENDING),
        "note": "Bound by owner decision 1 (2026-09-03): exact name, matching DOB and the exact DWCS 10.3 bout on UFCStats, corroborated by the official UFC.com Week 3 articles; the same-surname Abongo Humphrey listing row stays explicitly excluded. The athlete page is a pending follow-up, not a blocker.",
    },
    ("Alex Apodaca", "1994-06-30"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION,
        "owner_decision": {"decision": "1", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:14a48a86b2b57f0e",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": True,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Alex Apodaca", "date_of_birth": "1994-06-30", "dwcs": True, "fighter_page_sha256": "8435183d44a7d858cab4395ecb7d3adb850d67fe629cbaf58f8c754bef86627e"},
        "ufcstats": {
            "id": "14a48a86b2b57f0e",
            "url": "http://ufcstats.com/fighter-details/14a48a86b2b57f0e",
            "listed_name": "Alex Apodaca",
            "dob": "1994-06-30",
            "record": "4-1-0",
            "listing_capture_sha256": "d7d57e225cc77fd0a516ed2778bb62dc0fe0ee3cff121e5f7724f1d50544b5ac",
            "fighter_page_sha256": "daa8af0d626cb46e4cc8624b3946edb78773bbcaee0adccb786d929a8b17aeb6",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": {**_U31R1_UFCSTATS_DWCS_10_3, "opponent": "Bella Mir", "opponent_ufcstats_id": "9ab21ab88230ce48", "result": "win", "method": "U-DEC", "round": "3", "time": "5:00"},
        "ufccom": {**_U31R1_UFCCOM_ARTICLES, "corroborating_text": "Alex Apodaca defeats Bella Mir via unanimous decision (29-28, 29-28, 29-28) (Week 3 results, August 25, 2026)"},
        "athlete_page_follow_up": dict(_U31R1_ATHLETE_PAGE_FOLLOW_UP_PENDING),
        "note": "Bound by owner decision 1 (2026-09-03): exact name, matching DOB and the exact DWCS 10.3 bout on UFCStats, corroborated by the official UFC.com Week 3 articles. The athlete page is a pending follow-up, not a blocker.",
    },
    ("Lucia Szabova", "1997-12-23"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_DIRECT_SIGNEE_DECISION,
        "owner_decision": {"decision": "3", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:184f6ec4ee712e9f",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": False,
            "ufccom_athlete_page_corroboration": True,
        },
        "rosterwatch": {"name": "Lucia Szabova", "date_of_birth": "1997-12-23", "dwcs": False, "fighter_page_sha256": "0aa98e07b5c0d55f9a5e4f425e8468f07d62781ef09146e36832b02a3a4ff81f", "next_bout": "2026-10-31 vs Tainara Lisboa", "next_date": "2026-10-31", "next_opp": "Tainara Lisboa", "opponent_row_next_opp": "Lucia Szabova", "opponent_row_source": "Roster Watch 2026-09-01 row 'Tainara Lisboa' (next_date 2026-10-31, next_opp 'Lucia Szabova')"},
        "scheduled_debut": {
            "event_date": "2026-10-31",
            "opponent": "Tainara Lisboa",
            "opponent_ufcstats_id": "2c5f5749569eef66",
            "ufcstats_event_url": "http://ufcstats.com/event-details/0a66de3a797cf73a",
            "sources_agreeing_on_date": ("rosterwatch", "rosterwatch_opponent_row", "ufcstats_next_bout_row", "ufccom_athlete_page_octagon_debut"),
            "sources_agreeing_on_opponent": ("rosterwatch", "rosterwatch_opponent_row", "ufcstats_next_bout_row"),
            "ufccom_athlete_page_names_opponent": False,
            "ufccom_athlete_page_names_opponent_note": "the sealed athlete page (434e22f7...) carries 'Octagon Debut Oct. 31, 2026' but no opponent or card block; the opponent is agreed by Roster Watch (both rows) and the UFCStats 'next' row",
        },
        "ufcstats": {
            "id": "184f6ec4ee712e9f",
            "url": "http://ufcstats.com/fighter-details/184f6ec4ee712e9f",
            "listed_name": "Lucia Szabova",
            "nickname": "Silent Killer",
            "dob": "1997-12-23",
            "record": "12-0-0",
            "bouts_listed": 0,
            "next_bout_row": {"flag": "next", "opponent": "Tainara Lisboa", "opponent_ufcstats_id": "2c5f5749569eef66", "event": "UFC Fight Night", "event_date": "2026-10-31", "event_url": "http://ufcstats.com/event-details/0a66de3a797cf73a"},
            "listing_capture_sha256": "692c64eec862b3e7e8ba8671838d2a7accd8719a35929be90315acb9216ba718",
            "fighter_page_sha256": "67d833deafe2ee19248bb9927d6dcea80cd6a2898da7ffe87070295f23d8f036",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": None,
        "ufccom": {"kind": "athlete_page", "athlete_page_url": "https://www.ufc.com/athlete/lucia-szabova", "athlete_page_route_evidenced": True, "route_evidence": "immutable C:/Temp/u31l1/runtime/refresh-logs/candidate-refresh.log (2026-09-01 crawl)", "capture_sha256s": ("434e22f70a7f3673f0f1406406e63f917efa6aa5d45eb23bda4b5e80c58100f2",), "plan_ids": ("ufccom-athletes-01",), "json_ld_birth_date": "1997-12-23", "octagon_debut_text": "Oct. 31, 2026", "octagon_debut_date": "2026-10-31", "bio": {"status": "Active", "age": "28", "weight": "125.00", "division": "Women's Flyweight Division"}, "corroborating_text": "Lucia Szabova | Active | Women's Flyweight Division | 0-0-0 | Age 28 | Octagon Debut Oct. 31, 2026"},
        "note": "Bound by owner decision 3 (2026-09-03): direct signee with no Contender Series bout; exact name and matching DOB on Roster Watch, UFCStats and the official UFC.com athlete page (JSON-LD birthDate 1997-12-23), and the scheduled debut 2026-10-31 vs Tainara Lisboa matches on both sites (UFCStats 'next' row and the athlete page's Octagon Debut date; the opponent is named by Roster Watch and UFCStats). Division differs (UFC.com women's flyweight versus Roster Watch/UFCStats 135 lb); that is a data-quality note, not an identity conflict.",
    },
    ("Pavel Andrusca", "2002-05-08"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_DIRECT_SIGNEE_DECISION,
        "owner_decision": {"decision": "3", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:93f9c19ae41d2a97",
        "rule_elements": {
            "exact_name": True,
            "dob_match": True,
            "contender_series_bout_on_ufcstats": False,
            "ufccom_athlete_page_corroboration": True,
        },
        "rosterwatch": {"name": "Pavel Andrusca", "date_of_birth": "2002-05-08", "dwcs": False, "fighter_page_sha256": "31c55c564de2798c7b41a45ddfbd6d4e5a2f8bec685988230425a90aff1c3808", "next_bout": "2026-09-05 vs Nathaniel Wood", "next_date": "2026-09-05", "next_opp": "Nathaniel Wood", "opponent_row_next_opp": "Pavel Andrusca", "opponent_row_source": "Roster Watch 2026-09-01 row 'Nathaniel Wood' (next_date 2026-09-05, next_opp 'Pavel Andrusca')"},
        "scheduled_debut": {
            "event_date": "2026-09-05",
            "opponent": "Nathaniel Wood",
            "opponent_ufcstats_id": "329e403448756217",
            "ufcstats_event_url": "http://ufcstats.com/event-details/2144954270be834d",
            "sources_agreeing_on_date": ("rosterwatch", "rosterwatch_opponent_row", "ufcstats_next_bout_row", "ufccom_athlete_page_octagon_debut", "ufccom_athlete_page_fight_card"),
            "sources_agreeing_on_opponent": ("rosterwatch", "rosterwatch_opponent_row", "ufcstats_next_bout_row", "ufccom_athlete_page_fight_card"),
            "ufccom_athlete_page_names_opponent": True,
            "ufccom_athlete_page_names_opponent_note": "the sealed athlete page (bec173df...) carries the card block 'Wood vs Andrusca Sep. 5, 2026' and the text 'UFC Fight Night (9/5/26) Andrusca makes his UFC debut against Nathaniel Wood'",
        },
        "ufcstats": {
            "id": "93f9c19ae41d2a97",
            "url": "http://ufcstats.com/fighter-details/93f9c19ae41d2a97",
            "listed_name": "Pavel Andrusca",
            "dob": "2002-05-08",
            "record": "8-0-0",
            "bouts_listed": 0,
            "next_bout_row": {"flag": "next", "opponent": "Nathaniel Wood", "opponent_ufcstats_id": "329e403448756217", "event": "UFC Fight Night: Hooker vs. Parnasse", "event_date": "2026-09-05", "event_url": "http://ufcstats.com/event-details/2144954270be834d"},
            "listing_capture_sha256": "d7d57e225cc77fd0a516ed2778bb62dc0fe0ee3cff121e5f7724f1d50544b5ac",
            "fighter_page_sha256": "5286f1a28a0911b48dc93c4b322400ee969aa4f5b490b49fb1b3a3d0ffd690ca",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": None,
        "ufccom": {"kind": "athlete_page", "athlete_page_url": "https://www.ufc.com/athlete/pavel-andrusca", "athlete_page_route_evidenced": True, "route_evidence": "immutable C:/Temp/u31l1/candidate/single-refresh-evidence/candidate-run-result.json (2026-09-01 upcoming-card scan)", "capture_sha256s": ("bec173df617ba7559d12c8eebef881b916e956b63ad3d85a56fc4b16b8858790",), "plan_ids": ("ufccom-athletes-01",), "json_ld_birth_date": "2002-05-08", "octagon_debut_text": "Sep. 5, 2026", "octagon_debut_date": "2026-09-05", "fight_card_text": "Wood vs Andrusca Sep. 5, 2026; UFC Fight Night (9/5/26) Andrusca makes his UFC debut against Nathaniel Wood", "bio": {"status": "Active", "place_of_birth": "Moldova", "age": "24", "height": "67.00", "weight": "145.00", "reach": "66.50", "division": "Featherweight Division"}, "corroborating_text": "Pavel Andrusca | Active | Featherweight Division | 8-0-0 | Age 24 | Moldova | Octagon Debut Sep. 5, 2026"},
        "note": "Bound by owner decision 3 (2026-09-03): direct signee with no Contender Series bout; exact name, matching DOB and record on Roster Watch, UFCStats and the official UFC.com athlete page (JSON-LD birthDate 2002-05-08), and the scheduled debut 2026-09-05 vs Nathaniel Wood matches on both sites (UFCStats 'next' row and the athlete page's Octagon Debut date and 'Wood vs Andrusca' card block).",
    },
    ("Magomedrasul Gasanov", "1994-11-23"): {
        "status": ROSTERWATCH_ADDITION_BOUND,
        "binding_basis": ROSTERWATCH_ADDITION_BASIS_DOB_ADJUDICATED_DECISION,
        "owner_decision": {"decision": "4", "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE},
        "carrier": "ufcstats:30ba4c2779e68a1e",
        "rule_elements": {
            "exact_name": True,
            "dob_match": False,
            "contender_series_bout_on_ufcstats": False,
            "ufccom_athlete_page_corroboration": False,
        },
        "rosterwatch": {"name": "Magomedrasul Gasanov", "date_of_birth": "1994-11-23", "dwcs": False, "fighter_page_sha256": None, "country": "Russia", "gender": "NA", "weightclass": ""},
        "ufcstats": {
            "id": "30ba4c2779e68a1e",
            "url": "http://ufcstats.com/fighter-details/30ba4c2779e68a1e",
            "listed_name": "Magomedrasul Gasanov",
            "dob": "1994-10-23",
            "record": "23-2-0",
            "weight": "170 lbs.",
            "height": "--",
            "reach": "--",
            "stance": "",
            "bouts_listed": 0,
            "zero_bout_layout": True,
            "listing_row_cells": ("Magomedrasul", "Gasanov", "", "--", "170 lbs.", "--", "", "23", "2", "0", ""),
            "listing_capture_sha256": "ba952bab42f5827132e8bafdd8e8864d2f913789850e7514e1ff0cd35cdbb6fd",
            "fighter_page_sha256": "6e72bc9300351c77471eb52ebfcdd98ff92830cae024f4d1261f6a10c98020e9",
            "plan_ids": ("ufcstats-listing-02", "ufcstats-fighters-01"),
        },
        "contender_series_bout": None,
        "ufccom": {"kind": None, "athlete_page_url": None, "athlete_page_route_evidenced": False, "capture_sha256s": ()},
        "dob_adjudication": {
            "rosterwatch_dob": "1994-11-23",
            "ufcstats_dob": "1994-10-23",
            "owner_dob": "1994-11-23",
            "disposition": "ufcstats_source_error",
            "ufcstats_dob_overwritten": False,
            "conflict_list_entry": ("1994-11-23", "1994-10-23"),
            "identity_anchors": {
                "ufcstats_record": "23-2-0",
                "ufcstats_weight": "170 lbs.",
                "weight_class_implied_by_ufcstats_weight": "welterweight (170 lb limit)",
                "rosterwatch_country": "Russia",
                "owner_characterization": "23-2-0, Russia, welterweight",
                "sources": "record and 170 lb weight from the sealed UFCStats listing (ba952bab...) and fighter page (6e72bc93...); country from the Roster Watch September 1 row; UFCStats shows no country and Roster Watch shows no weight class",
            },
        },
        "note": "Bound by owner decision 4 (2026-09-03): the only same-name UFCStats record 30ba4c2779e68a1e (23-2-0, 170 lbs) is him; his DOB is 1994-11-23 (Roster Watch is correct) and UFCStats' 1994-10-23 is recorded as a source error in ROSTERWATCH_INITIAL_DOB_CONFLICTS and ROSTERWATCH_DOB_CONFLICT_ADJUDICATIONS. Nothing is overwritten: the UFCStats DOB stays recorded exactly as captured and dob_match stays False.",
    },
}


def _bound_addition_carriers(
    evidence: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[tuple[str, str], str]:
    return {
        key: str(record["carrier"])
        for key, record in evidence.items()
        if record.get("status") == ROSTERWATCH_ADDITION_BOUND
    }


ROSTERWATCH_REVIEWED_ADDITION_CARRIERS = _bound_addition_carriers(
    ROSTERWATCH_REVIEWED_ADDITION_EVIDENCE
)

# Future rename/reactivation bindings belong here only after source-specific
# review.  Keys include both the exact new Roster Watch name and DOB; there is
# intentionally no name-only or DOB-only inference.
ROSTERWATCH_REVIEWED_SOURCE_OWNER_TRANSITIONS: dict[tuple[str, str], str] = {}

# These are evidence conflicts, not matching rules.  Roster Watch DOB values
# remain source attributes and must not overwrite profile/history attributes.
ROSTERWATCH_INITIAL_DOB_CONFLICTS = {
    "Ethyn Ewing": ("1998-02-16", "1996-02-19"),
    "Gokhan Saricam": ("1991-02-26", "1991-02-28"),
    "Javier Reyes": ("1993-10-10", "1993-10-13"),
    "Marco Tulio": ("1994-08-30", "1994-06-13"),
    "Jefferson Nascimento": ("2001-02-18", "1998-11-19"),
    "Zhu Kangjie": ("1996-01-15", "1996-01-16"),
    "Hector Santiago": ("1993-03-23", "1993-03-21"),
    "Gianni Vazquez": ("1994-07-13", "1994-07-17"),
    "Ding Meng": ("1994-09-26", "1994-09-27"),
    "Yi Sak Lee": ("2000-01-11", "2000-01-01"),
    "Terrance Chatman": ("1995-09-14", "1997-05-05"),
    # Owner decision 4 (2026-09-03): UFCStats 1994-10-23 is a source error; the
    # conflicting value stays recorded here and in the sealed evidence, unchanged.
    "Magomedrasul Gasanov": ("1994-11-23", "1994-10-23"),
}

# Owner adjudications of DOB conflicts.  The conflicting profile value stays
# recorded (nothing is overwritten); the adjudication names which source is in
# error and which value the owner declared correct.
ROSTERWATCH_DOB_CONFLICT_ADJUDICATIONS: dict[str, dict[str, str]] = {
    "Magomedrasul Gasanov": {
        "rosterwatch_dob": "1994-11-23",
        "ufcstats_dob": "1994-10-23",
        "owner_dob": "1994-11-23",
        "disposition": "ufcstats_source_error",
        "ufcstats_id": "30ba4c2779e68a1e",
        "owner_decision": "4",
        "date": ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE,
    },
}

ROSTERWATCH_IDENTITY_REVIEW_FILENAME = "rosterwatch_identity_review.json"
ROSTERWATCH_LEARNED_ALIASES_FILENAME = "rosterwatch_learned_aliases.json"
ROSTERWATCH_STATE_FILENAME = "rosterwatch_state.json"
ROSTERWATCH_SOURCE_LEDGER_FILENAME = "rosterwatch_membership_ledger.csv"
ROSTERWATCH_LEGACY_LEDGER_FILENAME = "rosterwatch_legacy_reconciliation_ledger.csv"
ROSTERWATCH_RECURRING_LEDGER_FILENAME = "rosterwatch_recurring_reconciliation_ledger.csv"
ROSTERWATCH_OWNER_REGISTRY_FILENAME = "rosterwatch_owner_registry.csv"
ROSTERWATCH_DELTA_LOG_FILENAME = "rosterwatch_delta_log.jsonl"


class RosterWatchError(ValueError):
    """Fail-closed Roster Watch acquisition/reconciliation error."""


_CARRIER_RE = re.compile(r"^(ufcstats:[0-9a-f]{16}|ufc:https://www\.ufc\.com/athlete/[a-z0-9-]+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UFCSTATS_EVENT_URL_RE = re.compile(r"^http://ufcstats\.com/event-details/[0-9a-f]{16}$")
_UFCSTATS_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_UFCCOM_ATHLETE_URL_RE = re.compile(r"^https://www\.ufc\.com/athlete/[a-z0-9-]+$")


def _require_owner_decision(
    key: tuple[str, str], record: Mapping[str, object], allowed: frozenset[str]
) -> str:
    decision = record.get("owner_decision")
    # The last disjunct is defensive (every caller's ``allowed`` set is a subset of the
    # recorded decisions); it keeps the citation honest if a caller is ever widened.
    if (
        not isinstance(decision, Mapping)
        or str(decision.get("decision")) not in allowed
        or str(decision.get("date")) != ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE
        or str(decision.get("decision")) not in ROSTERWATCH_OWNER_IDENTITY_DECISIONS_2026_09_03
    ):
        raise RosterWatchError(
            f"Reviewed addition {key!r} cites no owner decision in {sorted(allowed)!r} dated {ROSTERWATCH_OWNER_IDENTITY_DECISION_DATE}"
        )
    return str(decision["decision"])


def _athlete_page_follow_up_complete(
    key: tuple[str, str], follow_up: object
) -> bool:
    """Validate a decision-1 athlete-page follow-up; True only when complete and agreeing."""
    if not isinstance(follow_up, Mapping):
        raise RosterWatchError(f"Reviewed addition {key!r} athlete-page follow-up must be a record")
    status = str(follow_up.get("status") or "")
    if status not in ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_STATUSES:
        raise RosterWatchError(f"Reviewed addition {key!r} athlete-page follow-up has unknown status {status!r}")
    disagreement = follow_up.get("disagreement")
    if disagreement not in (None, ""):
        raise RosterWatchError(
            f"Reviewed addition {key!r} athlete-page follow-up records a disagreement that must be resolved by the owner, never accepted silently: {disagreement!r}"
        )
    page_fields = ("athlete_page_url", "athlete_page_name", "capture_sha256", "json_ld_birth_date")
    if status == ROSTERWATCH_ATHLETE_PAGE_FOLLOW_UP_PENDING:
        if any(follow_up.get(field) not in (None, "") for field in page_fields):
            raise RosterWatchError(
                f"Reviewed addition {key!r} pending athlete-page follow-up must not carry page evidence"
            )
        return False
    if not _UFCCOM_ATHLETE_URL_RE.match(str(follow_up.get("athlete_page_url") or "")):
        raise RosterWatchError(f"Reviewed addition {key!r} completed athlete-page follow-up lacks an evidenced UFC.com athlete route")
    if not _SHA256_RE.match(str(follow_up.get("capture_sha256") or "")):
        raise RosterWatchError(f"Reviewed addition {key!r} completed athlete-page follow-up lacks a full capture SHA-256")
    if str(follow_up.get("json_ld_birth_date") or "") != key[1]:
        raise RosterWatchError(
            f"Reviewed addition {key!r} athlete-page follow-up DOB disagrees with the bound DOB: {follow_up.get('json_ld_birth_date')!r}"
        )
    if str(follow_up.get("athlete_page_name") or "") != key[0]:
        raise RosterWatchError(
            f"Reviewed addition {key!r} athlete-page follow-up name disagrees with the bound name: {follow_up.get('athlete_page_name')!r}"
        )
    return True


def validate_reviewed_addition_evidence(
    evidence: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    carriers: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, object]:
    """Fail closed unless every reviewed addition carrier is derived from a bound record.

    A record may be ``bound`` only under one explicit basis with its complete evidence:

    * the owner binding rule (all four elements, unchanged);
    * the closed accepted-cache basis (Guilherme Uriel only);
    * owner decision 1/2 (2026-09-03): record-scoped Contender Series winners with
      matching DOB, the exact UFCStats DWCS bout, official UFC.com article
      corroboration and a pending-or-complete athlete-page follow-up; decision 2
      additionally approves one record-scoped given-name spelling variant;
    * owner decision 3: record-scoped direct signees with exact name and DOB on
      UFCStats and the official athlete page (JSON-LD birthDate) and the scheduled
      debut agreeing on both sites;
    * owner decision 4: the record-scoped DOB adjudication for Magomedrasul Gasanov,
      whose UFCStats DOB stays recorded unchanged as a source error.

    Every carrier must be well formed, every capture hash must be a full SHA-256, an
    athlete-page follow-up that disagrees with the bound name or DOB is rejected
    rather than accepted, and the carrier map must equal exactly the bound records;
    a non-bound record contributes no carrier.
    """
    evidence = (
        ROSTERWATCH_REVIEWED_ADDITION_EVIDENCE if evidence is None else evidence
    )
    carriers = (
        ROSTERWATCH_REVIEWED_ADDITION_CARRIERS if carriers is None else carriers
    )
    counts: dict[str, int] = {}
    bases: dict[str, int] = {}
    decisions_applied: set[str] = set()
    seen_carriers: dict[str, tuple[str, str]] = {}
    for key, record in evidence.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not str(key[0]).strip()
            or not _valid_iso_date(str(key[1]))
        ):
            raise RosterWatchError(f"Reviewed addition key must be (name, ISO DOB): {key!r}")
        status = str(record.get("status") or "")
        if status not in ROSTERWATCH_ADDITION_STATUSES:
            raise RosterWatchError(f"Reviewed addition {key!r} has unknown status {status!r}")
        carrier = str(record.get("carrier") or "")
        if not _CARRIER_RE.match(carrier):
            raise RosterWatchError(f"Reviewed addition {key!r} has a malformed carrier {carrier!r}")
        if carrier in seen_carriers:
            raise RosterWatchError(
                f"Reviewed addition carrier {carrier!r} is claimed by both {seen_carriers[carrier]!r} and {key!r}"
            )
        seen_carriers[carrier] = key
        elements = record.get("rule_elements")
        if not isinstance(elements, Mapping) or set(elements) != set(ROSTERWATCH_ADDITION_RULE_ELEMENTS):
            raise RosterWatchError(f"Reviewed addition {key!r} must declare every rule element")
        if any(not isinstance(elements[name], bool) for name in ROSTERWATCH_ADDITION_RULE_ELEMENTS):
            raise RosterWatchError(f"Reviewed addition {key!r} rule elements must be booleans")
        rw = record.get("rosterwatch")
        stats = record.get("ufcstats")
        if not isinstance(rw, Mapping) or not isinstance(stats, Mapping):
            raise RosterWatchError(f"Reviewed addition {key!r} lacks rosterwatch/ufcstats evidence")
        if (str(rw.get("name")), str(rw.get("date_of_birth"))) != key:
            raise RosterWatchError(f"Reviewed addition {key!r} evidence name/DOB differs from its key")
        dob_match = _valid_iso_date(str(stats.get("dob") or "")) and str(stats.get("dob")) == key[1]
        if bool(elements["dob_match"]) != dob_match:
            raise RosterWatchError(f"Reviewed addition {key!r} dob_match flag contradicts its evidence")
        exact_name = str(stats.get("listed_name") or "") == key[0]
        if bool(elements["exact_name"]) != exact_name:
            raise RosterWatchError(f"Reviewed addition {key!r} exact_name flag contradicts its evidence")
        bout = record.get("contender_series_bout")
        has_bout = isinstance(bout, Mapping) and _valid_iso_date(str(bout.get("event_date") or "")) and bool(str(bout.get("opponent") or "").strip())
        if bool(elements["contender_series_bout_on_ufcstats"]) != has_bout:
            raise RosterWatchError(f"Reviewed addition {key!r} contender-series flag contradicts its evidence")
        ufccom = record.get("ufccom")
        athlete_page_block = (
            isinstance(ufccom, Mapping)
            and ufccom.get("kind") == "athlete_page"
            and bool(_UFCCOM_ATHLETE_URL_RE.match(str(ufccom.get("athlete_page_url") or "")))
            and ufccom.get("athlete_page_route_evidenced") is True
            and any(_SHA256_RE.match(str(item)) for item in (ufccom.get("capture_sha256s") or ()))
        )
        follow_up = record.get("athlete_page_follow_up")
        follow_up_complete = (
            _athlete_page_follow_up_complete(key, follow_up) if follow_up is not None else False
        )
        athlete_page = athlete_page_block or follow_up_complete
        if bool(elements["ufccom_athlete_page_corroboration"]) != athlete_page:
            raise RosterWatchError(f"Reviewed addition {key!r} athlete-page flag contradicts its evidence")
        for holder in (rw, stats, ufccom if isinstance(ufccom, Mapping) else {}, follow_up if isinstance(follow_up, Mapping) else {}):
            for field, value in holder.items():
                if field.endswith("sha256") and value is not None and not _SHA256_RE.match(str(value)):
                    raise RosterWatchError(f"Reviewed addition {key!r} has a malformed capture hash in {field!r}")
                if field.endswith("sha256s") and any(not _SHA256_RE.match(str(item)) for item in value):
                    raise RosterWatchError(f"Reviewed addition {key!r} has a malformed capture hash in {field!r}")
        if carrier.startswith("ufcstats:") and carrier.split(":", 1)[1] != str(stats.get("id") or ""):
            raise RosterWatchError(f"Reviewed addition {key!r} carrier does not match its UFCStats evidence")
        basis = str(record.get("binding_basis") or "")
        if basis not in ROSTERWATCH_ADDITION_BASES:
            raise RosterWatchError(f"Reviewed addition {key!r} has unknown binding basis {basis!r}")
        ufcstats_bout = has_bout and bool(_UFCSTATS_EVENT_URL_RE.match(str(bout.get("event_url") or ""))) and bool(_UFCSTATS_ID_RE.match(str(bout.get("opponent_ufcstats_id") or "")))
        official_articles = (
            isinstance(ufccom, Mapping)
            and ufccom.get("kind") == "official_news_article"
            and any(_SHA256_RE.match(str(item)) for item in (ufccom.get("capture_sha256s") or ()))
            and any(str(item).startswith("https://www.ufc.com/") for item in (ufccom.get("urls") or ()))
            and ufccom.get("athlete_page_route_evidenced") is False
        )
        variant_recorded = record.get("approved_name_variant")
        if status == ROSTERWATCH_ADDITION_BOUND:
            if basis == ROSTERWATCH_ADDITION_BASIS_OWNER_RULE:
                if not all(elements[name] for name in ROSTERWATCH_ADDITION_RULE_ELEMENTS):
                    raise RosterWatchError(f"Reviewed addition {key!r} is bound without every rule element")
                if not ufcstats_bout:
                    raise RosterWatchError(f"Reviewed addition {key!r} is bound without a UFCStats event route and opponent id for its Contender Series bout")
                if variant_recorded is not None:
                    raise RosterWatchError(f"Reviewed addition {key!r} claims a name variant outside decision 2")
            elif basis == ROSTERWATCH_ADDITION_BASIS_ACCEPTED_CACHE:
                if key not in ROSTERWATCH_ADDITION_ACCEPTED_CACHE_ALLOWLIST:
                    raise RosterWatchError(f"Reviewed addition {key!r} may not use the closed accepted-cache basis")
                if not (elements["exact_name"] and elements["dob_match"]):
                    raise RosterWatchError(f"Reviewed addition {key!r} accepted-cache binding lacks exact name and DOB")
                if variant_recorded is not None:
                    raise RosterWatchError(f"Reviewed addition {key!r} claims a name variant outside decision 2")
            elif basis == ROSTERWATCH_ADDITION_BASIS_CONTENDER_SERIES_DECISION:
                if key not in ROSTERWATCH_ADDITION_CONTENDER_SERIES_DECISION_ALLOWLIST:
                    raise RosterWatchError(f"Reviewed addition {key!r} is not in the owner's record-scoped decision-1 allowlist")
                decision = _require_owner_decision(key, record, frozenset({"1", "2"}))
                if not (elements["dob_match"] and elements["contender_series_bout_on_ufcstats"] and ufcstats_bout):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires a matching DOB and the exact UFCStats Contender Series bout")
                if not official_articles:
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires official UFC.com article corroboration with captured evidence")
                exact_event = ROSTERWATCH_DECISION_1_CONTENDER_SERIES_EVENT
                if (
                    str(bout.get("event")) != exact_event["event"]
                    or str(bout.get("event_date")) != exact_event["event_date"]
                    or str(bout.get("event_url")) != exact_event["event_url"]
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires the exact DWCS 10.3 bout ({exact_event['event_date']}, {exact_event['event_url']})")
                exact_articles = ROSTERWATCH_DECISION_1_OFFICIAL_ARTICLES
                if (
                    tuple(ufccom.get("urls") or ()) != exact_articles["urls"]
                    or tuple(ufccom.get("capture_sha256s") or ()) != exact_articles["capture_sha256s"]
                    or tuple(ufccom.get("plan_ids") or ()) != exact_articles["plan_ids"]
                    or tuple(dict(item) for item in (ufccom.get("article_captures") or ())) != exact_articles["article_captures"]
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires exactly the three sealed official UFC.com Week 3 articles (results, welcome, weigh-in) with their captured hashes and plan attribution")
                if not str(ufccom.get("corroborating_text") or "").startswith(f"{key[0]} defeats {bout.get('opponent')}"):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires the article corroboration to name this fighter defeating the recorded opponent")
                if follow_up is None:
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-1 binding requires an athlete-page follow-up record")
                if elements["exact_name"]:
                    if variant_recorded is not None or decision != "1":
                        raise RosterWatchError(f"Reviewed addition {key!r} claims a name variant it does not need")
                else:
                    approved = ROSTERWATCH_ADDITION_APPROVED_NAME_VARIANTS.get(key)
                    if (
                        decision != "2"
                        or approved is None
                        or not isinstance(variant_recorded, Mapping)
                        or dict(variant_recorded) != dict(approved)
                        or approved["carrier"] != carrier
                        or approved["rosterwatch_name"] != key[0]
                        or str(stats.get("listed_name") or "") != approved["ufcstats_listed_name"]
                    ):
                        raise RosterWatchError(f"Reviewed addition {key!r} has a name variant that is not the owner's record-scoped decision-2 approval")
            elif basis == ROSTERWATCH_ADDITION_BASIS_DIRECT_SIGNEE_DECISION:
                if key not in ROSTERWATCH_ADDITION_DIRECT_SIGNEE_DECISION_ALLOWLIST:
                    raise RosterWatchError(f"Reviewed addition {key!r} is not in the owner's record-scoped decision-3 allowlist")
                _require_owner_decision(key, record, frozenset({"3"}))
                if variant_recorded is not None:
                    raise RosterWatchError(f"Reviewed addition {key!r} claims a name variant outside decision 2")
                if not (elements["exact_name"] and elements["dob_match"] and not elements["contender_series_bout_on_ufcstats"] and athlete_page_block):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-3 binding requires exact name, matching DOB, no Contender Series bout and an evidenced official athlete page")
                if str(ufccom.get("json_ld_birth_date") or "") != key[1]:
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-3 binding requires the athlete page JSON-LD birthDate to match the bound DOB")
                debut = record.get("scheduled_debut")
                next_row = stats.get("next_bout_row")
                if not isinstance(debut, Mapping) or not isinstance(next_row, Mapping):
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-3 binding requires a scheduled debut and the UFCStats next-bout row")
                debut_date = str(debut.get("event_date") or "")
                opponent = str(debut.get("opponent") or "")
                if not _valid_iso_date(debut_date) or not opponent:
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut lacks a date or opponent")
                if (
                    str(next_row.get("flag")) != "next"
                    or str(next_row.get("event_date")) != debut_date
                    or str(next_row.get("opponent")) != opponent
                    or str(ufccom.get("octagon_debut_date")) != debut_date
                    or str(rw.get("next_date")) != debut_date
                    or str(rw.get("next_opp")) != opponent
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut does not match on both sites (UFCStats next row, UFC.com Octagon Debut) and Roster Watch")
                if (
                    not _UFCSTATS_EVENT_URL_RE.match(str(debut.get("ufcstats_event_url") or ""))
                    or str(debut.get("ufcstats_event_url")) != str(next_row.get("event_url"))
                    or not _UFCSTATS_ID_RE.match(str(debut.get("opponent_ufcstats_id") or ""))
                    or str(debut.get("opponent_ufcstats_id")) != str(next_row.get("opponent_ufcstats_id"))
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut lacks the exact UFCStats event route and opponent id")
                if str(rw.get("opponent_row_next_opp") or "") != key[0]:
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut is not reciprocated by the opponent's Roster Watch row")
                names_opponent = debut.get("ufccom_athlete_page_names_opponent")
                if not isinstance(names_opponent, bool):
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut must state whether the athlete page names the opponent")
                card_text = str(ufccom.get("fight_card_text") or "")
                if names_opponent != (bool(card_text) and opponent in card_text):
                    raise RosterWatchError(f"Reviewed addition {key!r} scheduled debut opponent-on-athlete-page flag contradicts the captured page text")
            elif basis == ROSTERWATCH_ADDITION_BASIS_DOB_ADJUDICATED_DECISION:
                if key not in ROSTERWATCH_ADDITION_DOB_ADJUDICATION_ALLOWLIST:
                    raise RosterWatchError(f"Reviewed addition {key!r} is not in the owner's record-scoped decision-4 allowlist")
                _require_owner_decision(key, record, frozenset({"4"}))
                if variant_recorded is not None:
                    raise RosterWatchError(f"Reviewed addition {key!r} claims a name variant outside decision 2")
                if not elements["exact_name"] or elements["dob_match"]:
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-4 binding requires the exact name and the recorded UFCStats DOB to remain the conflicting value; nothing may be overwritten")
                adjudication = record.get("dob_adjudication")
                registry = ROSTERWATCH_DOB_CONFLICT_ADJUDICATIONS.get(key[0])
                if not isinstance(adjudication, Mapping) or registry is None:
                    raise RosterWatchError(f"Reviewed addition {key!r} decision-4 binding requires a DOB adjudication record and its registry entry")
                ufcstats_dob = str(stats.get("dob") or "")
                if (
                    str(adjudication.get("rosterwatch_dob")) != key[1]
                    or str(adjudication.get("owner_dob")) != key[1]
                    or str(adjudication.get("ufcstats_dob")) != ufcstats_dob
                    or ufcstats_dob == key[1]
                    or str(adjudication.get("disposition")) != "ufcstats_source_error"
                    or adjudication.get("ufcstats_dob_overwritten") is not False
                    or tuple(adjudication.get("conflict_list_entry") or ()) != (key[1], ufcstats_dob)
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} DOB adjudication does not record the Roster Watch DOB as correct and the unchanged UFCStats DOB as a source error")
                if (
                    str(registry.get("rosterwatch_dob")) != key[1]
                    or str(registry.get("ufcstats_dob")) != ufcstats_dob
                    or str(registry.get("disposition")) != "ufcstats_source_error"
                    or str(registry.get("ufcstats_id")) != str(stats.get("id"))
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} DOB adjudication registry entry disagrees with the evidence")
                if ROSTERWATCH_INITIAL_DOB_CONFLICTS.get(key[0]) != (key[1], ufcstats_dob):
                    raise RosterWatchError(f"Reviewed addition {key!r} DOB conflict must be recorded in the DOB-conflict list as (Roster Watch, UFCStats)")
                anchors = adjudication.get("identity_anchors")
                if (
                    not isinstance(anchors, Mapping)
                    or str(anchors.get("ufcstats_record")) != str(stats.get("record") or "")
                    or str(anchors.get("ufcstats_weight")) != str(stats.get("weight") or "")
                    or str(anchors.get("rosterwatch_country")) != str(rw.get("country") or "")
                ):
                    raise RosterWatchError(f"Reviewed addition {key!r} DOB adjudication identity anchors do not match the UFCStats record, weight and Roster Watch country")
            else:  # pragma: no cover - unreachable: every member of ROSTERWATCH_ADDITION_BASES has a branch above
                raise RosterWatchError(f"Reviewed addition {key!r} has unknown binding basis {basis!r}")
            bases[basis] = bases.get(basis, 0) + 1
            if isinstance(record.get("owner_decision"), Mapping):
                decisions_applied.add(str(record["owner_decision"].get("decision")))
        elif status == ROSTERWATCH_ADDITION_PENDING_OWNER and not (elements["dob_match"] and ufcstats_bout):
            raise RosterWatchError(f"Reviewed addition {key!r} is pending an owner decision without a matching DOB and UFCStats Contender Series bout")
        elif status == ROSTERWATCH_ADDITION_DOB_CONFLICT and elements["dob_match"]:
            raise RosterWatchError(f"Reviewed addition {key!r} claims a DOB conflict but DOBs match")
        elif status == ROSTERWATCH_ADDITION_NO_CONTENDER_BOUT and elements["contender_series_bout_on_ufcstats"]:
            raise RosterWatchError(f"Reviewed addition {key!r} claims no Contender Series bout but one is recorded")
        counts[status] = counts.get(status, 0) + 1
    expected = _bound_addition_carriers(evidence)
    if dict(carriers) != expected:
        raise RosterWatchError(
            "Reviewed addition carriers must equal exactly the bound evidence records: "
            f"carriers={sorted(carriers)!r} bound={sorted(expected)!r}"
        )
    return {
        "binding_rule": ROSTERWATCH_ADDITION_BINDING_RULE,
        "records": len(evidence),
        "bound": len(expected),
        "counts_by_status": dict(sorted(counts.items())),
        "counts_by_basis": dict(sorted(bases.items())),
        "owner_decisions_applied": sorted(decisions_applied),
        "unresolved": len(evidence) - len(expected),
    }


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_row_bytes(row: Mapping[str, object], columns: Sequence[str]) -> bytes:
    payload = {column: _text(row.get(column)) for column in columns}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def parse_rosterwatch_csv(
    content: bytes,
    *,
    expected_sha256: str | None = None,
    expected_row_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Strictly parse one complete Roster Watch CSV payload."""
    if not isinstance(content, bytes) or not content:
        raise RosterWatchError("Roster Watch payload is empty")
    digest = _sha256(content)
    if expected_sha256 and digest != expected_sha256.casefold():
        raise RosterWatchError(
            f"Roster Watch SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    if content.startswith(b"\xef\xbb\xbf"):
        raise RosterWatchError("Roster Watch payload unexpectedly contains a UTF-8 BOM")
    if b"\x00" in content:
        raise RosterWatchError("Roster Watch payload contains NUL bytes")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RosterWatchError("Roster Watch payload is not valid UTF-8") from exc

    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise RosterWatchError(f"Roster Watch CSV parse failed: {exc}") from exc
    if not rows:
        raise RosterWatchError("Roster Watch CSV has no header")
    if tuple(rows[0]) != ROSTERWATCH_COLUMNS:
        raise RosterWatchError("Roster Watch CSV header/schema mismatch")
    data_rows = rows[1:]
    if expected_row_count is not None and len(data_rows) != expected_row_count:
        raise RosterWatchError(
            "Roster Watch row-count mismatch: "
            f"expected {expected_row_count}, got {len(data_rows)}"
        )
    if len(data_rows) < ROSTERWATCH_MINIMUM_ROW_COUNT and expected_row_count is None:
        raise RosterWatchError(
            f"Roster Watch payload is implausibly small ({len(data_rows)} rows)"
        )

    width = len(ROSTERWATCH_COLUMNS)
    exact_names: set[str] = set()
    normalized_names: set[str] = set()
    validated: list[list[str]] = []
    for index, values in enumerate(data_rows, start=2):
        if len(values) != width:
            raise RosterWatchError(
                f"Roster Watch CSV line {index} has {len(values)} fields, expected {width}"
            )
        fighter = values[0].strip()
        dob = values[1].strip()
        if not fighter:
            raise RosterWatchError(f"Roster Watch CSV line {index} has a blank fighter")
        if not _valid_iso_date(dob):
            raise RosterWatchError(
                f"Roster Watch CSV line {index} has invalid date_of_birth {dob!r}"
            )
        normalized = normalize_person_name(fighter)
        if not normalized:
            raise RosterWatchError(
                f"Roster Watch CSV line {index} has an empty normalized fighter key"
            )
        if fighter in exact_names or normalized in normalized_names:
            raise RosterWatchError(
                f"Roster Watch CSV contains duplicate fighter identity {fighter!r}"
            )
        exact_names.add(fighter)
        normalized_names.add(normalized)
        validated.append([value.strip() for value in values])

    frame = pd.DataFrame(validated, columns=ROSTERWATCH_COLUMNS, dtype="object")
    frame.attrs.update(
        {
            "schema_version": ROSTERWATCH_SCHEMA_VERSION,
            "source_sha256": digest,
            "row_count": len(frame),
        }
    )
    metadata = {
        "schema_version": ROSTERWATCH_SCHEMA_VERSION,
        "source_sha256": digest,
        "byte_count": len(content),
        "row_count": len(frame),
        "column_count": width,
        "encoding": "utf-8",
        "bom": False,
        "header_valid": True,
        "field_width_valid": True,
        "names_unique": True,
        "dates_valid": True,
        "status": "passed",
    }
    return frame, metadata


def _canonical_ufc_url(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    host = (parsed.hostname or "").casefold().rstrip(".")
    path = parsed.path.rstrip("/")
    if parsed.scheme.casefold() != "https" or host not in {"ufc.com", "www.ufc.com"}:
        return ""
    if not re.fullmatch(r"/athlete/[^/?#]+", path, flags=re.IGNORECASE):
        return ""
    if parsed.query or parsed.fragment:
        return ""
    return f"https://www.ufc.com{path.casefold()}"


def stable_owner_key(row: Mapping[str, object]) -> str:
    """Return a stable reviewed carrier key, never a name-only external ID."""
    explicit = _text(row.get("internal_owner_id") or row.get("stable_owner_key"))
    if explicit:
        return explicit
    stats_id = normalize_ufcstats_id(row.get("stable_ufcstats_id"))
    if not stats_id:
        stats_id = normalize_ufcstats_id(row.get("ufcstats_url"))
    if not stats_id:
        stats_id = normalize_ufcstats_id(row.get("ufcstats_id"))
    if stats_id:
        return f"ufcstats:{stats_id}"
    official_url = _canonical_ufc_url(
        row.get("canonical_athlete_url") or row.get("official_athlete_url")
    )
    if official_url:
        return f"ufc:{official_url}"
    raise RosterWatchError("A stable owner requires a reviewed UFCStats ID or UFC URL")


def _legacy_names(row: Mapping[str, object]) -> tuple[str, ...]:
    # The accepted roster's preferred official spelling is the crosswalk
    # surface.  Profile/UFCStats display names are preserved as evidence, not
    # silently promoted into broad aliases.
    official_name = _text(row.get("official_name"))
    return (official_name,) if official_name else ()


def _provisional_owner_id(
    *, source_sha256: str, row_ordinal: int, fighter: str, dob: str
) -> str:
    # The source hash and row ordinal are accepted for an auditable call site,
    # but deliberately excluded from the durable identity preimage: row order
    # and snapshot bytes may change without changing the person.
    del source_sha256, row_ordinal
    payload = (
        "rw-v1\0" + normalize_person_name(fighter) + "\0" + dob
    ).encode("utf-8")
    return f"rosterwatch:v1:{_sha256(payload)}"


def reconcile_rosterwatch_baseline(
    source_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
) -> dict[str, object]:
    """Reconcile authoritative membership against immutable legacy provenance."""
    if "fighter" not in source_df or "date_of_birth" not in source_df:
        raise RosterWatchError("Roster Watch frame lacks fighter/date_of_birth")
    if "official_name" not in legacy_df:
        raise RosterWatchError("Legacy roster lacks official_name")

    legacy_rows: list[dict[str, object]] = []
    owner_rows: dict[str, list[int]] = defaultdict(list)
    owner_overrides = dict(
        legacy_df.attrs.get("rosterwatch_owner_by_row") or {}
    )
    for ordinal, (_, series) in enumerate(legacy_df.iterrows(), start=1):
        row = {str(key): value for key, value in series.items()}
        owner = _text(owner_overrides.get(ordinal)) or stable_owner_key(row)
        names = _legacy_names(row)
        record = {
            "ordinal": ordinal,
            "owner": owner,
            "names": names,
            "row": row,
        }
        legacy_rows.append(record)
        owner_rows[owner].append(ordinal)

    exact_index: dict[str, set[str]] = defaultdict(set)
    normalized_index: dict[str, set[str]] = defaultdict(set)
    canonical_index: dict[str, set[str]] = defaultdict(set)
    exact_physical: dict[str, set[int]] = defaultdict(set)
    for record in legacy_rows:
        for name in record["names"]:
            exact_index[name].add(record["owner"])
            exact_physical[name].add(record["ordinal"])
            normalized_index[normalize_person_name(name)].add(record["owner"])
            canonical_index[canonical_fighter_name_key(name)].add(record["owner"])

    source_sha = _text(source_df.attrs.get("source_sha256"))
    require_source_owner_continuity = bool(
        source_df.attrs.get("require_explicit_owner_for_existing_name")
    )
    source_ledger: list[dict[str, object]] = []
    ambiguities: list[dict[str, object]] = []
    matched_owners: set[str] = set()
    match_counts: dict[str, int] = defaultdict(int)

    for ordinal, (_, series) in enumerate(source_df.iterrows(), start=1):
        row = {str(key): _text(value) for key, value in series.items()}
        fighter = row["fighter"]
        explicit_source_owner = _text(
            series.get("internal_owner_id") or series.get("stable_owner_key")
        )
        candidates: set[str] = set()
        method = ""
        physical_count = 0

        if not require_source_owner_continuity or explicit_source_owner:
            candidates = set(exact_index.get(fighter, set()))
            physical_count = len(exact_physical.get(fighter, set()))
            if candidates:
                method = (
                    "duplicate_owner_collapse"
                    if len(candidates) == 1 and physical_count > 1
                    else "exact"
                )
            if not candidates:
                candidates = set(
                    normalized_index.get(normalize_person_name(fighter), set())
                )
                if candidates:
                    method = "normalized"
            if not candidates:
                candidates = set(
                    canonical_index.get(canonical_fighter_name_key(fighter), set())
                )
                if candidates:
                    method = "canonical"
            if not candidates and fighter in ROSTERWATCH_REVIEWED_ALIASES:
                legacy_name = ROSTERWATCH_REVIEWED_ALIASES[fighter]
                candidates = set(exact_index.get(legacy_name, set()))
                expected_carrier = ROSTERWATCH_REVIEWED_ALIAS_CARRIERS[fighter]
                if candidates != {expected_carrier}:
                    raise RosterWatchError(
                        f"Reviewed alias carrier drift for {fighter!r}: {sorted(candidates)!r}"
                    )
                method = "reviewed_alias"
        if (
            explicit_source_owner
            and candidates
            and candidates != {explicit_source_owner}
        ):
            ambiguities.append(
                {
                    "rw_row_ordinal": ordinal,
                    "fighter": fighter,
                    "candidate_owners": sorted(candidates | {explicit_source_owner}),
                    "match_method": "source_owner_conflicts_with_name_crosswalk",
                }
            )
            continue
        if not candidates and explicit_source_owner in owner_rows:
            candidates = {explicit_source_owner}
            method = "source_identity_continuity"

        if len(candidates) > 1:
            ambiguities.append(
                {
                    "rw_row_ordinal": ordinal,
                    "fighter": fighter,
                    "candidate_owners": sorted(candidates),
                    "match_method": method,
                }
            )
            continue

        if candidates:
            owner = next(iter(candidates))
            if owner in matched_owners:
                ambiguities.append(
                    {
                        "rw_row_ordinal": ordinal,
                        "fighter": fighter,
                        "candidate_owners": [owner],
                        "match_method": "duplicate_source_owner",
                    }
                )
                continue
            carrier_status = (
                "pending_external_carrier"
                if owner.startswith("rosterwatch:")
                else "resolved"
            )
            matched_owners.add(owner)
        else:
            owner = explicit_source_owner or ROSTERWATCH_REVIEWED_ADDITION_CARRIERS.get(
                (fighter, row["date_of_birth"]), ""
            )
            carrier_status = (
                "pending_external_carrier"
                if owner.startswith("rosterwatch:")
                else "resolved"
                if owner
                else "pending_external_carrier"
            )
            if not owner:
                owner = _provisional_owner_id(
                    source_sha256=source_sha,
                    row_ordinal=ordinal,
                    fighter=fighter,
                    dob=row["date_of_birth"],
                )
            method = "addition"

        legacy_ordinals = owner_rows.get(owner, [])
        conflict = ROSTERWATCH_INITIAL_DOB_CONFLICTS.get(fighter)
        preferred_carrier = owner
        reviewed_carrier = ROSTERWATCH_REVIEWED_ADDITION_CARRIERS.get(
            (fighter, row["date_of_birth"])
        )
        if owner.startswith("rosterwatch:") and reviewed_carrier:
            carrier_status = "resolved"
            preferred_carrier = reviewed_carrier
        source_ledger.append(
            {
                "rw_row_ordinal": ordinal,
                "rw_csv_line": ordinal + 1,
                "rw_row_sha256": _sha256(_canonical_row_bytes(row, ROSTERWATCH_COLUMNS)),
                "rw_fighter_raw": fighter,
                "rw_dob_raw": row["date_of_birth"],
                "normalized_name": normalize_person_name(fighter),
                "project_canonical_key": canonical_fighter_name_key(fighter),
                "internal_owner_id": owner,
                "preferred_carrier_type": preferred_carrier.split(":", 1)[0],
                "preferred_carrier_value": preferred_carrier.split(":", 1)[1],
                "carrier_resolution_status": carrier_status,
                "legacy_owner_key": owner if legacy_ordinals else "",
                "legacy_primary_row_ordinal": legacy_ordinals[0] if legacy_ordinals else "",
                "all_legacy_row_ordinals": ";".join(map(str, legacy_ordinals)),
                "match_method": method,
                "identity_conflict_flags": (
                    f"dob_source={conflict[0]};dob_profile={conflict[1]}"
                    if conflict
                    else ""
                ),
                "current_at_cutoff": True,
                "materialized_output_row": ordinal,
            }
        )
        match_counts[method] += 1

    if ambiguities:
        failed_gates = {
            "rosterwatch_source_pin_proven": bool(
                source_df.attrs.get("source_sha256")
            ),
            "rosterwatch_membership_scope_owner_accepted": True,
            "rosterwatch_schema_and_count_proven": len(source_df) > 0,
            "rosterwatch_identity_crosswalk_complete": False,
            "legacy_identity_crosswalk_complete": False,
            "all_675_members_materialized_once": False,
            "zero_duplicate_stable_owners": False,
            "all_996_legacy_rows_reconciled": False,
            "historical_data_preserved": True,
            "authoritative_baseline_candidate_proven": False,
        }
        return {
            "schema_version": ROSTERWATCH_SCHEMA_VERSION,
            "status": "failed",
            "gates": failed_gates,
            "counts": {
                "source_rows": len(source_df),
                "legacy_rows": len(legacy_df),
            },
            "source_ledger": source_ledger,
            "legacy_ledger": [],
            "identity_conflicts": ambiguities,
            "production_ready": False,
            "identity_complete": False,
        }

    represented_owners = {
        row["internal_owner_id"]
        for row in source_ledger
        if row["match_method"] != "addition"
    }
    source_by_owner = {
        row["internal_owner_id"]: row for row in source_ledger
    }
    legacy_ledger: list[dict[str, object]] = []
    primary_seen: set[str] = set()
    noncurrent_owners = set(owner_rows) - represented_owners
    for record in legacy_rows:
        owner = record["owner"]
        is_primary = owner not in primary_seen
        primary_seen.add(owner)
        matched = source_by_owner.get(owner)
        combat_sport = _text(record["row"].get("combat_sport")).casefold()
        if matched and is_primary:
            disposition = "current_primary"
        elif matched:
            disposition = "current_duplicate_representation"
        elif combat_sport and combat_sport != "mma":
            disposition = "non_mma_evidence"
        else:
            disposition = "non_current_mma_history"
        legacy_ledger.append(
            {
                "legacy_row_ordinal": record["ordinal"],
                "legacy_row_sha256": _sha256(
                    _canonical_row_bytes(record["row"], tuple(legacy_df.columns))
                ),
                "official_name": _text(record["row"].get("official_name")),
                "official_athlete_url": _text(
                    record["row"].get("official_athlete_url")
                ),
                "ufcstats_url": _text(record["row"].get("ufcstats_url")),
                "stable_legacy_owner_key": owner,
                "is_primary_representation": is_primary,
                "duplicate_of_row": "" if is_primary else owner_rows[owner][0],
                "matched_rw_row": matched["rw_row_ordinal"] if matched else "",
                "disposition": disposition,
                "preserved_as_history": True,
            }
        )

    counts = {
        "source_rows": len(source_df),
        "represented": len(represented_owners),
        "additions": sum(row["match_method"] == "addition" for row in source_ledger),
        "legacy_physical_rows": len(legacy_df),
        "legacy_logical_owners": len(owner_rows),
        "legacy_duplicate_rows": len(legacy_df) - len(owner_rows),
        "noncurrent_logical_owners": len(noncurrent_owners),
        "unresolved_addition_carriers": sum(
            row["carrier_resolution_status"] != "resolved"
            for row in source_ledger
        ),
        "identity_conflicts": sum(
            bool(row["identity_conflict_flags"]) for row in source_ledger
        ),
        "match_methods": dict(sorted(match_counts.items())),
    }
    counts.update(
        {
            "legacy_rows": counts["legacy_physical_rows"],
            "matched": counts["represented"],
            "added": counts["additions"],
            "non_current": counts["noncurrent_logical_owners"],
            "duplicate_collapses": counts["legacy_duplicate_rows"],
            "unresolved": counts["unresolved_addition_carriers"],
        }
    )
    gates = {
        "rosterwatch_source_pin_proven": bool(source_df.attrs.get("source_sha256")),
        "rosterwatch_membership_scope_owner_accepted": True,
        "rosterwatch_schema_and_count_proven": len(source_df) > 0,
        "rosterwatch_identity_crosswalk_complete": len(source_ledger)
        == len(source_df),
        "legacy_identity_crosswalk_complete": True,
        "all_675_members_materialized_once": len(source_ledger) == len(source_df),
        "zero_duplicate_stable_owners": len(
            {row["internal_owner_id"] for row in source_ledger}
        )
        == len(source_ledger),
        "all_996_legacy_rows_reconciled": len(legacy_ledger) == len(legacy_df),
        "historical_data_preserved": all(
            row["preserved_as_history"] for row in legacy_ledger
        ),
        "source_rows_represented_once": len(source_ledger) == len(source_df),
        "all_legacy_rows_reconciled": len(legacy_ledger) == len(legacy_df),
        "no_identity_ambiguity": True,
        "stable_owner_unique": len({row["internal_owner_id"] for row in source_ledger})
        == len(source_ledger),
    }
    if len(source_df) == ROSTERWATCH_INITIAL_ROW_COUNT and len(legacy_df) == 996:
        gates.update(
            {
                "initial_source_rows_exact": counts["source_rows"] == 675,
                "initial_represented_exact": counts["represented"] == 667,
                "initial_additions_exact": counts["additions"] == 8,
                "initial_noncurrent_exact": counts["noncurrent_logical_owners"] == 328,
                "initial_duplicate_exact": counts["legacy_duplicate_rows"] == 1,
                "initial_arithmetic_exact": (
                    counts["legacy_physical_rows"]
                    - counts["legacy_duplicate_rows"]
                    - counts["noncurrent_logical_owners"]
                    + counts["additions"]
                    == counts["source_rows"]
                ),
            }
        )
    gates["authoritative_baseline_candidate_proven"] = all(gates.values())
    status = "passed" if all(gates.values()) else "failed"
    return {
        "schema_version": ROSTERWATCH_SCHEMA_VERSION,
        "status": status,
        "gates": gates,
        "counts": counts,
        "source_ledger": source_ledger,
        "legacy_ledger": legacy_ledger,
        "identity_conflicts": [
            row for row in source_ledger if row["identity_conflict_flags"]
        ],
        "production_ready": status == "passed",
        "identity_complete": counts["unresolved_addition_carriers"] == 0,
    }


def _ledger_owner_map(
    frame: pd.DataFrame,
    ledger: pd.DataFrame | Sequence[Mapping[str, object]] | None,
) -> dict[int, str]:
    records: list[Mapping[str, object]] = []
    if isinstance(ledger, pd.DataFrame):
        records = ledger.to_dict(orient="records")
    elif ledger is not None:
        records = list(ledger)
    result: dict[int, str] = {}
    allow_positional = len(records) == len(frame)
    for offset, record in enumerate(records, start=1):
        ordinal_text = _text(record.get("rw_row_ordinal"))
        if not ordinal_text.isdigit() and not allow_positional:
            continue
        ordinal = int(ordinal_text) if ordinal_text.isdigit() else offset
        owner = _text(record.get("internal_owner_id") or record.get("stable_owner_key"))
        if owner:
            result[ordinal] = owner
    for ordinal, (_, row) in enumerate(frame.iterrows(), start=1):
        if ordinal in result:
            continue
        owner = _text(row.get("internal_owner_id") or row.get("stable_owner_key"))
        if not owner:
            owner = "rosterwatch:row:" + _sha256(
                f"{normalize_person_name(row.get('fighter'))}\0{_text(row.get('date_of_birth'))}".encode(
                    "utf-8"
                )
            )
        result[ordinal] = owner
    return result


def calculate_rosterwatch_delta(
    previous_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    previous_ledger: pd.DataFrame | Sequence[Mapping[str, object]] | None = None,
    current_ledger: pd.DataFrame | Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Calculate mutually exclusive membership changes by stable owner."""
    previous_owners = _ledger_owner_map(previous_df, previous_ledger)
    current_owners = _ledger_owner_map(current_df, current_ledger)

    def indexed(
        frame: pd.DataFrame, owners: Mapping[int, str]
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for ordinal, (_, row) in enumerate(frame.iterrows(), start=1):
            owner = owners[ordinal]
            if owner in result:
                raise RosterWatchError(f"Duplicate stable owner in snapshot: {owner}")
            result[owner] = {
                "internal_owner_id": owner,
                "fighter": _text(row.get("fighter")),
                "date_of_birth": _text(row.get("date_of_birth")),
                "rw_row_ordinal": ordinal,
            }
        return result

    before = indexed(previous_df, previous_owners)
    after = indexed(current_df, current_owners)
    unchanged: list[dict[str, object]] = []
    renames: list[dict[str, object]] = []
    for owner in sorted(set(before) & set(after)):
        old = before[owner]
        new = after[owner]
        if old["fighter"] == new["fighter"]:
            unchanged.append(new)
        else:
            renames.append(
                {
                    "internal_owner_id": owner,
                    "previous_fighter": old["fighter"],
                    "fighter": new["fighter"],
                    "date_of_birth": new["date_of_birth"],
                }
            )

    historical: dict[str, Mapping[str, object]] = {}
    for ledger_records in (previous_ledger, current_ledger):
        if isinstance(ledger_records, pd.DataFrame):
            values = ledger_records.to_dict(orient="records")
        else:
            values = list(ledger_records or [])
        for record in values:
            owner = _text(record.get("internal_owner_id") or record.get("stable_owner_key"))
            if owner:
                historical.setdefault(owner, record)

    additions: list[dict[str, object]] = []
    reactivations: list[dict[str, object]] = []
    for owner in sorted(set(after) - set(before)):
        record = after[owner]
        history = historical.get(owner, {})
        was_seen = bool(history) and str(
            history.get("is_current", "")
        ).strip().casefold() in {"false", "0", "no"}
        was_seen = was_seen or str(history.get("ever_seen") or "").strip().casefold() in {
            "true",
            "1",
            "yes",
        } or str(history.get("reactivated") or "").strip().casefold() in {
            "true",
            "1",
            "yes",
        }
        (reactivations if was_seen else additions).append(record)
    removals = [before[owner] for owner in sorted(set(before) - set(after))]
    categories = {
        "unchanged": unchanged,
        "additions": additions,
        "removals": removals,
        "renames": renames,
        "reactivations": reactivations,
    }
    return {
        "schema_version": ROSTERWATCH_SCHEMA_VERSION,
        **categories,
        "counts": {key: len(value) for key, value in categories.items()},
    }


def _baseline_migration_delta(
    reconciliation: Mapping[str, object],
) -> dict[str, object]:
    """Describe the protected 996-to-675 migration without calling it recurring."""
    source_ledger = list(reconciliation.get("source_ledger") or [])
    legacy_ledger = list(reconciliation.get("legacy_ledger") or [])
    unchanged = [
        {
            "internal_owner_id": row["internal_owner_id"],
            "fighter": row["rw_fighter_raw"],
            "rw_row_ordinal": row["rw_row_ordinal"],
        }
        for row in source_ledger
        if row.get("match_method") != "addition"
    ]
    additions = [
        {
            "internal_owner_id": row["internal_owner_id"],
            "fighter": row["rw_fighter_raw"],
            "rw_row_ordinal": row["rw_row_ordinal"],
        }
        for row in source_ledger
        if row.get("match_method") == "addition"
    ]
    removals = [
        {
            "internal_owner_id": row["stable_legacy_owner_key"],
            "fighter": row["official_name"],
            "legacy_row_ordinal": row["legacy_row_ordinal"],
        }
        for row in legacy_ledger
        if row.get("is_primary_representation") is True
        and str(row.get("disposition") or "").startswith("non_")
    ]
    categories = {
        "unchanged": unchanged,
        "additions": additions,
        "removals": removals,
        "renames": [],
        "reactivations": [],
    }
    counts = {key: len(value) for key, value in categories.items()}
    counts["duplicate_legacy_representations"] = int(
        reconciliation.get("counts", {}).get("legacy_duplicate_rows") or 0
    )
    return {
        "schema_version": ROSTERWATCH_SCHEMA_VERSION,
        "kind": "baseline_migration",
        **categories,
        "counts": counts,
    }


def materialize_rosterwatch_roster(
    source_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
    reconciliation: Mapping[str, object],
) -> pd.DataFrame:
    """Build the operational roster while preserving profile/history fields."""
    if reconciliation.get("status") != "passed":
        raise RosterWatchError("Cannot materialize a failed reconciliation")
    source_ledger = list(reconciliation.get("source_ledger") or [])
    if len(source_ledger) != len(source_df):
        raise RosterWatchError("Source ledger does not cover the authoritative roster")
    output_rows: list[dict[str, object]] = []
    columns = list(legacy_df.columns)
    generation_id = hashlib.sha256(
        "\n".join(row["internal_owner_id"] for row in source_ledger).encode("utf-8")
    ).hexdigest()[:32]
    for ledger_row in source_ledger:
        primary = _text(ledger_row.get("legacy_primary_row_ordinal"))
        identity_complete = ledger_row.get("carrier_resolution_status") == "resolved"
        if primary and identity_complete:
            row = legacy_df.iloc[int(primary) - 1].to_dict()
        else:
            source_row = source_df.iloc[int(ledger_row["rw_row_ordinal"]) - 1]
            row = {column: "" for column in columns}
            fighter = _text(source_row.get("fighter"))
            if "official_name" in row:
                row["official_name"] = fighter
            if "slug_name" in row:
                row["slug_name"] = normalize_person_name(fighter)
            if "profile_status" in row:
                row["profile_status"] = "status_unknown"
            if "combat_sport" in row:
                row["combat_sport"] = "mma"
            if "combat_sport_reason" in row:
                row["combat_sport_reason"] = "rosterwatch_authoritative_membership"
            if "coverage_eligible" in row:
                row["coverage_eligible"] = False
            carrier_type = _text(ledger_row.get("preferred_carrier_type"))
            carrier_value = _text(ledger_row.get("preferred_carrier_value"))
            if carrier_type == "ufcstats" and carrier_value:
                stats_id = carrier_value
                if "ufcstats_name" in row:
                    # The UFCStats-listed name comes from the reviewed evidence record
                    # when one exists (decision 2 lists "Nicholas Galanti" for Roster
                    # Watch "Nick Galanti"); otherwise the source name is used.
                    reviewed = ROSTERWATCH_REVIEWED_ADDITION_EVIDENCE.get(
                        (fighter, _text(source_row.get("date_of_birth")))
                    )
                    reviewed_stats = (
                        reviewed.get("ufcstats") if isinstance(reviewed, Mapping) else None
                    )
                    listed_name = (
                        _text(reviewed_stats.get("listed_name"))
                        if isinstance(reviewed_stats, Mapping)
                        and f"ufcstats:{_text(reviewed_stats.get('id'))}" == f"ufcstats:{stats_id}"
                        else ""
                    )
                    row["ufcstats_name"] = listed_name or fighter
                if "ufcstats_url" in row:
                    row["ufcstats_url"] = (
                        f"http://ufcstats.com/fighter-details/{stats_id}"
                    )
                if "ufcstats_resolution" in row:
                    row["ufcstats_resolution"] = "rosterwatch_reviewed_carrier"
            elif carrier_type == "ufc" and carrier_value:
                if "official_athlete_url" in row:
                    row["official_athlete_url"] = carrier_value
                if "canonical_athlete_url" in row:
                    row["canonical_athlete_url"] = carrier_value
        binding = _text(ledger_row.get("identity_binding_json"))
        if binding:
            # A previously pending legacy row still needs its new carrier.
            # Reparse evidence here too: direct callers cannot inject carrier
            # values by adding an unverified JSON field to a ledger.
            from src.data.rosterwatch_identity_matching import validate_identity_decision

            decision = validate_identity_decision(json.loads(binding), {
                "internal_owner_id": ledger_row["internal_owner_id"],
                "current_source_name": ledger_row["rw_fighter_raw"],
                "current_source_dob": ledger_row["rw_dob_raw"],
            }, occupied_carriers=reconciliation.get("identity_occupied_carriers"))
            names = decision.get("source_names") or {}
            official_page = _text(decision.get("official_athlete_url"))
            if official_page:
                for column in ("official_athlete_url", "canonical_athlete_url"):
                    if column in row:
                        row[column] = official_page
            if decision["carrier_type"] == "ufcstats":
                if "ufcstats_url" in row:
                    row["ufcstats_url"] = f"http://ufcstats.com/fighter-details/{decision['carrier_value']}"
                if "ufcstats_name" in row:
                    row["ufcstats_name"] = _text(names.get("ufcstats"))
                if "ufcstats_resolution" in row:
                    row["ufcstats_resolution"] = "rosterwatch_automatic_evidence"
            elif decision["carrier_type"] == "ufc":
                for column in ("official_athlete_url", "canonical_athlete_url"):
                    if column in row:
                        row[column] = decision["carrier_value"]
                if "official_name" in row and names.get("ufc"):
                    row["official_name"] = names["ufc"]
        if "active_roster_status_reason" in row:
            row["active_roster_status_reason"] = "rosterwatch_authoritative_current"
        if "active_roster_live_present" in row:
            row["active_roster_live_present"] = True
        if "active_roster_retained_from_previous" in row:
            row["active_roster_retained_from_previous"] = False
        if "active_roster_retained_at_utc" in row:
            row["active_roster_retained_at_utc"] = ""
        if "active_roster_first_missing_at_utc" in row:
            row["active_roster_first_missing_at_utc"] = ""
        if "active_roster_consecutive_missing_syncs" in row:
            row["active_roster_consecutive_missing_syncs"] = 0
        if "active_roster_missing_from_live_reason" in row:
            row["active_roster_missing_from_live_reason"] = ""
        if "active_roster_sync_generation_id" in row:
            row["active_roster_sync_generation_id"] = generation_id
        output_rows.append(row)
    frame = pd.DataFrame(output_rows, columns=columns)
    frame.attrs.update(
        {
            "sync_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
            "membership_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
            "sync_complete": reconciliation.get("production_ready") is True,
            "production_ready": reconciliation.get("production_ready") is True,
            "identity_complete": reconciliation.get("identity_complete") is True,
            "sync_completeness_reason": (
                "complete"
                if reconciliation.get("production_ready") is True
                else "membership_reconciliation_failed"
            ),
            "sync_generation_id": generation_id,
            "reconciliation": dict(reconciliation),
        }
    )
    return frame


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _ledger_csv_bytes(records: Sequence[Mapping[str, object]], evidence_store=None) -> bytes:
    if evidence_store is not None:
        records = evidence_store.compact_records(records)
    return pd.DataFrame(list(records)).to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )


def _write_immutable_snapshot(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        if path.is_file() and path.read_bytes() == content:
            return
        raise RosterWatchError(f"Immutable snapshot collision at {path}")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    try:
        path.chmod(0o444)
    except OSError:
        pass



def _identity_json_list(record: Mapping[str, object], key: str) -> list:
    value = record.get(key)
    if value is None or value == "":
        return []
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, list):
        raise RosterWatchError(f"Invalid Roster Watch identity evidence: {key}")
    return decoded


def _identity_json(records: object) -> str:
    return json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_current_identity(record: Mapping[str, object]) -> bool:
    return _text(record.get("is_current")).casefold() in {"true", "1"}


def _prepare_pending_identities(
    owner_registry: Sequence[dict[str, object]], *, source_cutoff: str
) -> None:
    for record in owner_registry:
        if not _is_current_identity(record):
            continue
        if record.get("external_carrier_status") == "resolved":
            continue
        pending_since = _text(record.get("pending_since"))
        if not pending_since:
            first_seen = _text(record.get("first_seen_cutoff"))
            pending_since = first_seen if _valid_iso_date(first_seen) else source_cutoff
        if not _valid_iso_date(pending_since) or pending_since > source_cutoff:
            raise RosterWatchError("Pending identity has an invalid first-seen cutoff")
        record["pending_since"] = pending_since
        record.setdefault("last_identity_check", "")


def load_rosterwatch_identity_provider(state_root: Path | str):
    """Load only candidate-local captured responses, without acquiring sources."""
    root = Path(state_root).resolve()
    manifest = root / ROSTERWATCH_IDENTITY_CAPTURE_FILENAME
    if not manifest.exists():
        return None
    if manifest.is_symlink() or not manifest.is_file():
        raise RosterWatchError("Identity capture manifest must be a regular candidate file")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RosterWatchError("Identity capture manifest schema is invalid")
    captures = payload.get("captures")
    if not isinstance(captures, list):
        raise RosterWatchError("Identity capture manifest lacks captured responses")
    # Inline captured bytes make the candidate portable and avoid silently
    # following a manifest-supplied filesystem path outside its generation.
    for capture in captures:
        if not isinstance(capture, dict) or not any(
            key in capture for key in ("body_text", "body_base64")
        ) or any(key in capture for key in ("capture", "decoded_path", "raw_path", "path")):
            raise RosterWatchError("Identity manifests require embedded captured bytes")
    from src.data.rosterwatch_identity_matching import CapturedIdentityProvider

    return CapturedIdentityProvider(captures)


def _claim_identity_page(occupied: dict[str, str], carrier: str, owner: str) -> None:
    if not owner or not _CARRIER_RE.fullmatch(carrier):
        raise RosterWatchError("Resolved identity registry contains an invalid carrier")
    if carrier in occupied and occupied[carrier] != owner:
        raise RosterWatchError("An external identity page is assigned to two owners")
    occupied[carrier] = owner


def _occupied_identity_carriers(
    records: Sequence[Mapping[str, object]], known: Mapping[str, str] | None = None,
) -> dict[str, str]:
    occupied = dict(known or {})
    for record in records:
        if record.get("external_carrier_status") != "resolved":
            continue
        owner = _text(record.get("internal_owner_id"))
        carrier = f"{_text(record.get('preferred_carrier_type'))}:{_text(record.get('preferred_carrier_value'))}"
        _claim_identity_page(occupied, carrier, owner)
        encoded = _text(record.get("identity_binding_json"))
        if encoded:
            # Replay below verifies these fields. Claiming them first rejects
            # competing learned assignments regardless of registry row order.
            secondary = _text(json.loads(encoded).get("official_athlete_url"))
            if secondary:
                _claim_identity_page(occupied, "ufc:" + secondary, owner)
    return occupied


def _validated_identity_bindings(
    records: Sequence[Mapping[str, object]],
    known: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Revalidate bytes and replay every learned carrier against current owners.

    Only plain document extraction is shared within this complete pass. A new
    provider and full decision replay still run for every owner; a later pass
    (including the separate aliases pass without ``known``) starts cold.
    """
    occupied = _occupied_identity_carriers(records, known)
    bindings = {}
    parse_cache = None
    try:
        for record in records:
            encoded = _text(record.get("identity_binding_json"))
            if not encoded:
                continue
            from src.data.rosterwatch_identity_matching import (
                _CaptureParseCache, validate_identity_decision,
            )

            if parse_cache is None:
                parse_cache = _CaptureParseCache()
            decision = validate_identity_decision(
                json.loads(encoded), record, occupied_carriers=occupied,
                _parsed_capture_cache=parse_cache,
            )
            carrier = f"{decision.get('carrier_type')}:{decision.get('carrier_value')}"
            if decision.get("status") != "resolved" or not _CARRIER_RE.fullmatch(carrier):
                raise RosterWatchError("Learned identity evidence does not establish a carrier")
            if record.get("external_carrier_status") != "resolved" or carrier != (
                f"{record.get('preferred_carrier_type')}:{record.get('preferred_carrier_value')}"
            ):
                raise RosterWatchError("Learned identity evidence disagrees with registry carrier")
            owner = _text(record.get("internal_owner_id"))
            if occupied.get(carrier, owner) != owner:
                raise RosterWatchError("Learned identity page belongs to another owner")
            occupied[carrier] = owner
            bindings[owner] = decision
    finally:
        if parse_cache is not None:
            parse_cache.clear()
    return bindings


def _apply_identity_bindings(
    source_ledger: Sequence[dict[str, object]],
    owner_registry: Sequence[Mapping[str, object]],
    known: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    bindings = _validated_identity_bindings(owner_registry, known)
    for row in source_ledger:
        decision = bindings.get(_text(row.get("internal_owner_id")))
        if decision is None:
            continue
        row.update({
            "carrier_resolution_status": "resolved",
            "preferred_carrier_type": decision["carrier_type"],
            "preferred_carrier_value": decision["carrier_value"],
            "identity_binding_json": _identity_json(decision),
        })
    return bindings


def _identity_contexts(
    source_df: pd.DataFrame, ledger: Sequence[Mapping[str, object]], *, snapshot_sha256: str
) -> dict:
    contexts = {}
    for row in ledger:
        source = source_df.iloc[int(row["rw_row_ordinal"]) - 1]
        contexts[_text(row.get("internal_owner_id"))] = {
            key: _text(source.get(key))
            for key in ("fighter", "date_of_birth", "next_date", "next_opp", "wins", "losses", "bouts",
                        "prev_bout", "prev_result", "debut", "debut_result")
        } | {"snapshot_sha256": snapshot_sha256}
    return contexts


def _confirmed_identity_profiles(
    source_ledger: Sequence[Mapping[str, object]], roster: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Carry official URLs through exact ledger ownership, never name joins."""
    profiles = {}
    for record in source_ledger:
        if record.get("carrier_resolution_status") != "resolved":
            continue
        row = roster.iloc[int(record["rw_row_ordinal"]) - 1]
        profiles[_text(record.get("internal_owner_id"))] = {
            key: _text(row.get(key)) for key in (
                "official_name", "ufcstats_name", "ufcstats_url",
                "official_athlete_url", "canonical_athlete_url",
            )
        }
    return profiles


def _confirmed_identity_opponents(
    owner_registry: Sequence[Mapping[str, object]],
    confirmed_profiles: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Persist only the opponent fields consumed by captured identity matching.

    Full owner rows contain earlier decisions and their captured bodies. Passing
    those histories into the next decision recursively duplicates evidence.
    Existing persisted decisions still replay their original evidence unchanged.
    """
    keys = (
        "external_carrier_status", "internal_owner_id", "current_source_name",
        "rw_fighter", "rw_fighter_raw", "fighter", "official_name", "current_name",
        "ufcstats_url", "preferred_carrier_type", "preferred_carrier_value",
        "official_athlete_url", "canonical_athlete_url",
    )
    opponents = []
    for row in owner_registry:
        if not _is_current_identity(row) or row.get("external_carrier_status") != "resolved":
            continue
        profile = (confirmed_profiles or {}).get(_text(row.get("internal_owner_id"))) or {}
        opponent = {key: row[key] for key in keys if key in row}
        opponent.update({key: profile[key] for key in keys if key in profile})
        opponents.append(opponent)
    return opponents


def _known_identity_pages(
    legacy_ledger_path: Path, source_ledger: Sequence[Mapping[str, object]], roster: pd.DataFrame,
) -> dict[str, str]:
    """Index current and historical accepted source pages by exact ledger owner."""
    known = {}
    rows = []
    if legacy_ledger_path.is_file():
        for row in pd.read_csv(legacy_ledger_path, dtype="object", keep_default_na=False).to_dict(orient="records"):
            rows.append((_text(row.get("stable_legacy_owner_key")), row))
    rows.extend(_confirmed_identity_profiles(source_ledger, roster).items())
    for owner, row in rows:
        stats_id = normalize_ufcstats_id(row.get("ufcstats_url"))
        if stats_id:
            _claim_identity_page(known, "ufcstats:" + stats_id, owner)
        for key in ("official_athlete_url", "canonical_athlete_url"):
            official = _canonical_ufc_url(row.get(key))
            if official:
                _claim_identity_page(known, "ufc:" + official, owner)
    return known


def _identity_alias_payload(records: Sequence[Mapping[str, object]]) -> dict:
    aliases = []
    for owner, decision in _validated_identity_bindings(records).items():
        for alias in decision.get("learned_aliases") or []:
            record = {"internal_owner_id": owner, "alias": alias, "decision": decision}
            if record not in aliases:
                aliases.append(record)
    aliases.sort(key=_identity_json)
    return {"schema_version": 1, "aliases": aliases}


def _write_identity_aliases(records: Sequence[Mapping[str, object]], path: Path, evidence_store=None) -> str:
    payload = _identity_alias_payload(records)
    if not payload["aliases"] and not path.exists():
        return ""
    if evidence_store is not None:
        for alias in payload["aliases"]:
            alias["decision"] = {"identity_evidence_reference": evidence_store.put_bytes(
                _identity_json(alias["decision"]).encode("utf-8"))}
    content = _json_bytes(payload)
    if not path.is_file() or path.read_bytes() != content:
        write_bytes_atomically(content, path)
    return _sha256(content)


def _verify_identity_aliases(
    records: Sequence[Mapping[str, object]], path: Path, known: Mapping[str, str] | None = None,
) -> None:
    _validated_identity_bindings(records, known)
    expected = _identity_alias_payload(records)
    if not path.exists():
        if expected["aliases"]:
            raise RosterWatchError("Learned identity aliases are absent from candidate state")
        return
    if path.is_symlink() or not path.is_file():
        raise RosterWatchError("Learned aliases disagree with replayed identity evidence")
    retained = json.loads(path.read_text(encoding="utf-8"))
    from src.data.rosterwatch_identity_evidence import IdentityEvidenceStore

    evidence_store = IdentityEvidenceStore.for_state(path.parent)
    if evidence_store is not None:
        for alias in retained.get("aliases", []):
            decision = alias.get("decision")
            if isinstance(decision, dict) and set(decision) == {"identity_evidence_reference"}:
                alias["decision"] = json.loads(evidence_store.get_bytes(decision["identity_evidence_reference"]))
    if retained != expected:
        raise RosterWatchError("Learned aliases disagree with replayed identity evidence")


def recheck_rosterwatch_identities(
    owner_registry: Sequence[dict[str, object]],
    *,
    checker: Callable[[Mapping[str, object], str], Sequence[Mapping[str, object]]] | None = None,
    request_cap: int | None = None,
    checked_at: str | None = None,
    identity_provider=None,
    source_contexts: Mapping[str, Mapping[str, object]] | None = None,
    confirmed_profiles: Mapping[str, Mapping[str, object]] | None = None,
    occupied_carriers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Bound source checks and replay verified automatic decisions without HTTP.

    Each callback represents one source check and consumes one slot *before*
    invocation. Per-source timestamps rotate a member's sources when the cap
    stops midway. Legacy callback pages remain unbound; the capture provider
    additionally makes deterministic decisions that are independently reparsed.
    """
    cap = ROSTERWATCH_IDENTITY_REQUEST_CAP if request_cap is None else request_cap
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise RosterWatchError("Identity request cap must be a nonnegative integer")
    pending = [
        row for row in owner_registry
        if _is_current_identity(row) and row.get("external_carrier_status") != "resolved"
    ]
    pending.sort(key=lambda row: (
        _text(row.get("last_identity_check")),
        _text(row.get("pending_since")),
        _text(row.get("internal_owner_id")),
    ))
    report = {
        "mode": "offline_scaffold", "request_cap": cap, "checks_performed": 0,
        "checked_members": [], "deferred_members": [], "errors": [],
        "network_requests": 0,
    }
    if checker is not None and identity_provider is not None:
        raise RosterWatchError("Configure one identity source provider")
    if checker is None and identity_provider is None:
        report["deferred_members"] = [_text(row.get("internal_owner_id")) for row in pending]
        report["reason"] = "automatic_lookup_deferred_to_phase_2"
        return report
    if identity_provider is not None:
        report["mode"] = "captured_automatic_matching"
        report["cap_unit"] = "captured_source_checks_no_http_sends"
        report["requested_urls"] = []
    timestamp = checked_at or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
    except (TypeError, ValueError) as exc:
        raise RosterWatchError("Identity check timestamp must include its timezone") from exc
    for record in pending:
        owner = _text(record.get("internal_owner_id"))
        source_checks = {
            _text(item.get("site")): _text(item.get("checked_at"))
            for item in _identity_json_list(record, "identity_source_checks_json")
        }
        pages = _identity_json_list(record, "identity_candidate_pages_json")
        reasons = _identity_json_list(record, "identity_check_reasons_json")
        checked = False
        lookup_reports = []
        confirmed = _confirmed_identity_opponents(owner_registry, confirmed_profiles)
        context = dict((source_contexts or {}).get(owner) or {})
        for site in sorted(("ufcstats", "ufc"), key=lambda site: (source_checks.get(site, ""), ("ufcstats", "ufc").index(site))):
            if report["checks_performed"] >= cap:
                break
            report["checks_performed"] += 1
            checked = True
            source_checks[site] = timestamp
            try:
                if identity_provider is None:
                    observations = checker(dict(record), site)
                else:
                    lookup = identity_provider.lookup(
                        dict(record), site, source_context=context, confirmed_opponents=confirmed
                    )
                    # CapturedIdentityProvider has no transport. Live acquisition
                    # must happen in the independently bounded capture launcher.
                    if lookup.get("request_count") != 0:
                        raise RosterWatchError("Captured identity lookup attempted network acquisition")
                    lookup_reports.append(lookup)
                    for url in lookup.get("requested_urls") or []:
                        if url not in report["requested_urls"]:
                            report["requested_urls"].append(url)
                    observations = lookup.get("candidate_pages", [])
                    reasons.extend(reason for reason in lookup.get("reason_codes", []) if reason not in reasons)
                if not isinstance(observations, (list, tuple)):
                    raise ValueError("offline checker must return candidate-page records")
                for observation in observations:
                    if not isinstance(observation, Mapping):
                        raise ValueError("invalid candidate-page evidence")
                    page = dict(observation) if identity_provider is not None else {
                        "site": site,
                        "page": _text(observation.get("page")),
                        "name": _text(observation.get("name")),
                        "date_of_birth": _text(observation.get("date_of_birth")),
                        "evidence": observation.get("evidence") or {},
                        "reason_not_bound": _text(observation.get("reason_not_bound"))
                            or "automatic_binding_deferred_to_phase_2",
                    }
                    if identity_provider is not None:
                        page["phase2_candidate"] = True
                    if page not in pages:
                        pages.append(page)
            except Exception as exc:
                reason = f"{site}_identity_check_failed: {type(exc).__name__}: {exc}"
                if reason not in reasons:
                    reasons.append(reason)
                report["errors"].append({"internal_owner_id": owner, "reason": reason})
        if checked:
            record["last_identity_check"] = timestamp
            record["identity_source_checks_json"] = _identity_json([
                {"site": site, "checked_at": when} for site, when in sorted(source_checks.items())
            ])
            record["identity_candidate_pages_json"] = _identity_json(pages)
            record["identity_check_reasons_json"] = _identity_json(reasons)
            if identity_provider is not None:
                from src.data.rosterwatch_identity_matching import validate_identity_decision

                decision = identity_provider.decide(
                    dict(record), lookup_reports, source_context=context,
                    confirmed_opponents=confirmed,
                    occupied_carriers=_occupied_identity_carriers(owner_registry, occupied_carriers),
                )
                record["identity_decision_json"] = _identity_json(decision)
                record["identity_check_reasons_json"] = _identity_json(decision.get("reason_codes") or [])
                if decision.get("status") == "resolved":
                    decision = validate_identity_decision(
                        decision, record, occupied_carriers=_occupied_identity_carriers(owner_registry, occupied_carriers)
                    )
                    record.update({
                        "external_carrier_status": "resolved",
                        "preferred_carrier_type": decision["carrier_type"],
                        "preferred_carrier_value": decision["carrier_value"],
                        "identity_binding_json": _identity_json(decision),
                        "identity_check_reasons_json": _identity_json(decision.get("soft_flags") or []),
                    })
            report["checked_members"].append(owner)
        else:
            report["deferred_members"].append(owner)
    return report


def build_rosterwatch_identity_review(
    source_df: pd.DataFrame,
    source_ledger: Sequence[Mapping[str, object]],
    owner_registry: Sequence[Mapping[str, object]],
    *,
    source_cutoff: str,
    max_age_days: int | None = None,
    max_count: int | None = None,
    newly_confirmed: int = 0,
    as_of: date | None = None,
) -> dict[str, object]:
    """Render current membership identity evidence without making identity decisions."""
    age_limit = PENDING_IDENTITY_MAX_AGE_DAYS if max_age_days is None else max_age_days
    count_limit = PENDING_IDENTITY_MAX_COUNT if max_count is None else max_count
    if not _valid_iso_date(source_cutoff):
        raise RosterWatchError("Identity review cutoff must be YYYY-MM-DD")
    for limit in (age_limit, count_limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise RosterWatchError("Identity visibility limits must be nonnegative integers")
    reference_date = as_of or datetime.now(timezone.utc).date()
    registry = {_text(row.get("internal_owner_id")): row for row in owner_registry}
    rows = []
    for ledger_row in source_ledger:
        owner = _text(ledger_row.get("internal_owner_id"))
        complete = ledger_row.get("carrier_resolution_status") == "resolved"
        conflict = _text(ledger_row.get("identity_conflict_flags"))
        record = registry.get(owner, {})
        binding = json.loads(_text(record.get("identity_binding_json")) or "{}")
        soft_flags = list(binding.get("soft_flags") or [])
        name_variant = bool(binding.get("learned_aliases"))
        if complete and not conflict and not soft_flags and not name_variant:
            continue
        source = source_df.iloc[int(ledger_row["rw_row_ordinal"]) - 1]
        pending_since = _text(record.get("pending_since")) if not complete else ""
        if not complete and not pending_since:
            pending_since = source_cutoff
        days_pending = (
            max((reference_date - date.fromisoformat(pending_since)).days, 0)
            if pending_since else 0
        )
        def source_text(key: str) -> str:
            value = _text(source.get(key))
            return "" if value.casefold() in {"na", "nan", "none", "null", "--"} else value
        next_date, opponent = source_text("next_date"), source_text("next_opp")
        booked = bool(next_date)
        pages = _identity_json_list(record, "identity_candidate_pages_json")
        reasons = _identity_json_list(record, "identity_check_reasons_json")
        flags = list(soft_flags)
        if name_variant:
            flags.append("confirmed_name_variant")
        if conflict:
            flags.append("dob_conflict")
            reasons = [*reasons, conflict]
        seen_stats = any(
            page.get("site") == "ufcstats" and page.get("page")
            and (not page.get("phase2_candidate") or page.get("identity_target_proven") is True)
            for page in pages
        )
        if not complete and seen_stats:
            flags.append("seen_ufcstats_page_unbound")
        if not complete and days_pending > age_limit:
            flags.append("aged_pending")
        if not complete:
            reasons = reasons or ["automatic_lookup_deferred_to_phase_2"]
        rows.append({
            "internal_owner_id": owner,
            "fighter": _text(ledger_row.get("rw_fighter_raw")),
            "date_of_birth": _text(ledger_row.get("rw_dob_raw")),
            "weightclass": source_text("weightclass"),
            "next_date": next_date, "next_opp": opponent, "booked": booked,
            "identity_complete": complete,
            "carrier_resolution_status": _text(ledger_row.get("carrier_resolution_status")),
            "pending_since": pending_since,
            "last_identity_check": _text(record.get("last_identity_check")),
            "days_pending": days_pending,
            "status": (
                "conflicted" if complete and conflict else "confirmed" if complete else "booked_unconfirmed" if booked
                else "aged_pending" if "aged_pending" in flags else "pending_unbooked"
            ),
            "candidate_pages": pages, "reasons": reasons,
            "what_clears": (
                "Review the recorded DOB conflict; the existing carrier remains confirmed."
                if complete and conflict else "Identity confirmed from captured source evidence."
                if complete else "Confirm this fighter's external page using independently supported identity evidence."
            ),
            "flags": flags,
        })
    rows.sort(key=lambda row: (not row["booked"], -row["days_pending"], row["fighter"]))
    pending = [row for row in rows if not row["identity_complete"]]
    counts = {
        "pending": len(pending), "newly_confirmed": newly_confirmed,
        "aged": sum("aged_pending" in row["flags"] for row in pending),
        "booked": sum(row["booked"] for row in pending),
        "conflicted": sum("dob_conflict" in row["flags"] for row in rows),
        "seen_ufcstats_page_defect": sum("seen_ufcstats_page_unbound" in row["flags"] for row in pending),
        "pending_count_exceeded": len(pending) > count_limit,
    }
    return {
        "schema_version": 1, "source_cutoff": source_cutoff, "rows": rows, "counts": counts,
        # Ordinary resolved members are absent from the review rows. Keep explicit
        # current-ledger evidence so alerts can recover only after confirmation.
        "resolved_owner_ids": sorted({
            _text(row.get("internal_owner_id"))
            for row in source_ledger
            if row.get("carrier_resolution_status") == "resolved"
            and _text(row.get("internal_owner_id"))
        }),
        "thresholds": {"pending_age_days": age_limit, "pending_count": count_limit},
    }


def _write_identity_review(review: Mapping[str, object], path: Path) -> dict[str, object]:
    content = _json_bytes(review)
    if not path.is_file() or path.read_bytes() != content:
        write_bytes_atomically(content, path)
    return {
        "identity_review": dict(review),
        "identity_review_path": str(path.resolve()),
        "identity_review_sha256": _sha256(content),
    }


def _read_previous_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RosterWatchError(f"Roster Watch state is unreadable: {exc}") from exc
    if state.get("schema_version") != ROSTERWATCH_SCHEMA_VERSION:
        raise RosterWatchError("Roster Watch state schema is not version 1")
    if state.get("status") != "passed":
        raise RosterWatchError("Roster Watch accepted state status is not passed")
    if state.get("membership_source") != ROSTERWATCH_MEMBERSHIP_SOURCE:
        raise RosterWatchError(
            "Roster Watch accepted state has the wrong membership source"
        )
    if (
        _text(state.get("initial_legacy_snapshot_sha256"))
        != ROSTERWATCH_INITIAL_LEGACY_SHA256
        or int(state.get("initial_legacy_snapshot_rows") or -1)
        != ROSTERWATCH_INITIAL_LEGACY_ROW_COUNT
        or int(state.get("initial_legacy_snapshot_bytes") or -1)
        != ROSTERWATCH_INITIAL_LEGACY_BYTE_COUNT
    ):
        raise RosterWatchError(
            "Roster Watch state is not bound to the exact initial legacy baseline"
        )
    state_root = state_path.parent.resolve()
    snapshot_root = (state_root / "snapshots").resolve()

    def bound_snapshot(value: object, label: str) -> Path:
        candidate = Path(_text(value))
        if not candidate.is_absolute():
            candidate = state_root / candidate
        candidate = candidate.resolve()
        if snapshot_root not in candidate.parents or candidate.is_symlink():
            raise RosterWatchError(
                f"Roster Watch state {label} escapes the immutable snapshot root"
            )
        return candidate

    snapshot = bound_snapshot(state.get("snapshot_path"), "snapshot")
    expected = _text(state.get("snapshot_sha256"))
    if not snapshot.is_file() or _sha256(snapshot.read_bytes()) != expected:
        raise RosterWatchError("Roster Watch state snapshot/hash binding is invalid")
    legacy_snapshot = bound_snapshot(
        state.get("initial_legacy_snapshot_path"), "initial legacy snapshot"
    )
    if (
        not legacy_snapshot.is_file()
        or _sha256(legacy_snapshot.read_bytes())
        != _text(state.get("initial_legacy_snapshot_sha256"))
    ):
        raise RosterWatchError(
            "Roster Watch state initial legacy snapshot/hash binding is invalid"
        )
    bound_files = {
        ROSTERWATCH_SOURCE_LEDGER_FILENAME: "source_ledger_sha256",
        ROSTERWATCH_LEGACY_LEDGER_FILENAME: "legacy_ledger_sha256",
        ROSTERWATCH_RECURRING_LEDGER_FILENAME: "recurring_ledger_sha256",
        ROSTERWATCH_OWNER_REGISTRY_FILENAME: "owner_registry_sha256",
        ROSTERWATCH_DELTA_LOG_FILENAME: "delta_log_sha256",
    }
    aliases_path = state_path.parent / ROSTERWATCH_LEARNED_ALIASES_FILENAME
    if aliases_path.exists() or state.get("learned_aliases_sha256"):
        bound_files[ROSTERWATCH_LEARNED_ALIASES_FILENAME] = "learned_aliases_sha256"
    for filename, state_key in bound_files.items():
        expected_file_hash = _text(state.get(state_key))
        bound_path = state_path.parent / filename
        if (
            not expected_file_hash
            or not bound_path.is_file()
            or _sha256(bound_path.read_bytes()) != expected_file_hash
        ):
            raise RosterWatchError(
                f"Roster Watch state binding is invalid for {filename}"
            )
    state["_resolved_snapshot_path"] = str(snapshot)
    state["_resolved_initial_legacy_snapshot_path"] = str(legacy_snapshot)
    return state


def _restore_mutable_files(backups: Mapping[Path, bytes | None]) -> None:
    for path, previous in backups.items():
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            write_bytes_atomically(previous, path)


def _failed_fallback(
    *,
    output_path: Path,
    previous_output: bytes | None,
    error: Exception,
    request_count: int,
) -> pd.DataFrame:
    if previous_output is None:
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(io.BytesIO(previous_output))
    frame.attrs.update(
        {
            "sync_source": "cached_rosterwatch",
            "membership_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
            "sync_complete": False,
            "production_ready": False,
            "identity_complete": False,
            "sync_fallback_used": previous_output is not None,
            "sync_completeness_reason": "rosterwatch_acquisition_or_validation_failed",
            "sync_error": f"{type(error).__name__}: {error}",
            "output_path": str(output_path),
            "request_count": request_count,
        }
    )
    return frame


def sync_rosterwatch_active_roster(
    *,
    output_path: Path | str,
    identity_audit_path: Path | str | None = None,
    source_path: Path | str | None = None,
    source_url: str | None = None,
    source_cutoff: str | None = None,
    state_root: Path | str | None = None,
    allow_initial_migration: bool = False,
    session: requests.Session | None = None,
    source_site: bool | None = None,
    identity_review_path: Path | str | None = None,
    identity_checker: Callable[[Mapping[str, object], str], Sequence[Mapping[str, object]]] | None = None,
    identity_request_cap: int | None = None,
    identity_provider=None,
    shared_identity_evidence: bool = False,
) -> pd.DataFrame:
    """Attempt one fresh Roster Watch sync, retaining prior bytes on any failure.

    Acquisition modes, exactly one per call:

    * ``source_site`` (the plan of record for every recurring refresh): scrape
      the public roster.watch current-roster page through
      :mod:`src.data.rosterwatch_site`, preserve the raw page immutably, and
      derive the 37-column membership CSV from it.  This is the default when
      neither an owner-supplied file nor a URL is configured.
    * ``source_path``: an optional owner-supplied Roster Watch CSV override for
      one specific dated snapshot.  Nothing scheduled may depend on it.
    * ``source_url``: a fixed HTTPS CSV address (legacy; not the plan of record).

    There are no retries and no alternate membership provider.  A redirect or a
    bot challenge is rejected.  ``identity_audit_path`` is accepted only for
    call-site compatibility; the Roster Watch ledgers are written under
    ``state_root``.
    """
    del identity_audit_path
    output = Path(output_path)
    previous_output = output.read_bytes() if output.is_file() else None
    root = (
        Path(state_root) if state_root is not None else output.parent / "rosterwatch"
    ).resolve()
    state_path = root / ROSTERWATCH_STATE_FILENAME
    review_path = (
        Path(identity_review_path).resolve()
        if identity_review_path is not None else root / ROSTERWATCH_IDENTITY_REVIEW_FILENAME
    )
    source_ledger_path = root / ROSTERWATCH_SOURCE_LEDGER_FILENAME
    legacy_ledger_path = root / ROSTERWATCH_LEGACY_LEDGER_FILENAME
    recurring_ledger_path = root / ROSTERWATCH_RECURRING_LEDGER_FILENAME
    owner_registry_path = root / ROSTERWATCH_OWNER_REGISTRY_FILENAME
    delta_path = root / ROSTERWATCH_DELTA_LOG_FILENAME
    aliases_path = root / ROSTERWATCH_LEARNED_ALIASES_FILENAME
    from src.data.rosterwatch_identity_evidence import STORE_DIRECTORY

    if root not in review_path.parents or review_path in {
        output.resolve(), state_path, source_ledger_path, legacy_ledger_path,
        recurring_ledger_path, owner_registry_path, delta_path, aliases_path,
        root / ROSTERWATCH_IDENTITY_CAPTURE_FILENAME,
    } or (root / "snapshots") in review_path.parents or (root / STORE_DIRECTORY) in review_path.parents:
        raise RosterWatchError("Identity review path must be a separate candidate-state artifact")
    mutable_paths = (
        output,
        state_path,
        source_ledger_path,
        legacy_ledger_path,
        recurring_ledger_path,
        owner_registry_path,
        delta_path,
        review_path,
        aliases_path,
    )
    backups = {
        path: path.read_bytes() if path.is_file() else None for path in mutable_paths
    }
    request_count = 0
    site_acquisition: dict[str, object] | None = None
    try:
        from src.data.rosterwatch_identity_evidence import IdentityEvidenceStore

        if type(shared_identity_evidence) is not bool:
            raise RosterWatchError("Shared identity evidence must be explicitly boolean")
        evidence_store = IdentityEvidenceStore.for_state(root, create=shared_identity_evidence)
        if evidence_store is not None:
            evidence_store.archive_originals((owner_registry_path, source_ledger_path, aliases_path))

        def identity_records(path):
            records = pd.read_csv(path, dtype="object", keep_default_na=False).to_dict(orient="records")
            if evidence_store is not None:
                return evidence_store.resolve_records(records)
            from src.data.rosterwatch_identity_evidence import resolve_records

            return resolve_records(records, root)

        if identity_provider is None and identity_checker is None:
            identity_provider = load_rosterwatch_identity_provider(root)
        configured_path = source_path or os.getenv("UFC_ROSTERWATCH_SOURCE_PATH")
        configured_url = source_url or os.getenv("UFC_ROSTERWATCH_SOURCE_URL")
        if source_site is None:
            site_env = _text(os.getenv("UFC_ROSTERWATCH_SOURCE_SITE")).casefold()
            if site_env in {"0", "false", "no", "off"}:
                configured_site = False
            elif site_env in {"1", "true", "yes", "on"}:
                configured_site = True
            else:
                # The website is the plan of record: default to it whenever no
                # explicit owner override file or legacy URL is configured.
                configured_site = not configured_path and not configured_url
        else:
            configured_site = bool(source_site)
        cutoff = source_cutoff or os.getenv("UFC_ROSTERWATCH_SOURCE_CUTOFF")
        selected_modes = int(bool(configured_path)) + int(bool(configured_url)) + int(configured_site)
        if selected_modes != 1:
            raise RosterWatchError(
                "Configure exactly one Roster Watch source: the public website "
                "(default), an owner-supplied CSV path override, or a legacy HTTPS CSV URL"
            )
        if not configured_site and (not cutoff or not _valid_iso_date(str(cutoff))):
            raise RosterWatchError("Roster Watch source cutoff must be YYYY-MM-DD")

        acquired_at = datetime.now(timezone.utc).isoformat()
        if configured_site:
            from src.data.rosterwatch_site import acquire_rosterwatch_site_roster

            # Exactly one page request is attempted; count it before the wire so a
            # rejected or interrupted acquisition still reports the request honestly.
            request_count = 1
            site_acquisition = acquire_rosterwatch_site_roster(
                capture_root=root / "snapshots" / "site",
                session=session,
            )
            request_count = int(site_acquisition["request_count"])
            content = bytes(site_acquisition["csv_bytes"])
            if not cutoff:
                cutoff = str(site_acquisition["retrieved_date_utc"])
            if not _valid_iso_date(str(cutoff)):
                raise RosterWatchError("Roster Watch site cutoff must be YYYY-MM-DD")
            capture = site_acquisition["capture"]
            source_description = (
                f"{capture['url']} retrieved {capture['retrieved_at_utc']} "
                f"page sha256 {capture['sha256']} ({capture['byte_count']} bytes)"
            )
            acquisition = "site_scrape"
        elif configured_path:
            source = Path(str(configured_path))
            if not source.is_file():
                raise RosterWatchError(f"Roster Watch source file is absent: {source}")
            content = source.read_bytes()
            source_description = str(source.resolve())
            acquisition = "local_ingest"
        else:
            parsed = urlparse(str(configured_url))
            if parsed.scheme.casefold() != "https" or not parsed.hostname:
                raise RosterWatchError("Roster Watch URL must be absolute HTTPS")
            client = session or requests.Session()
            request_count = 1
            response = client.get(
                str(configured_url),
                timeout=ROSTERWATCH_REQUEST_TIMEOUT,
                allow_redirects=False,
                headers={"User-Agent": "ufc-betting-bot-rosterwatch/1"},
            )
            if 300 <= int(response.status_code) < 400:
                raise RosterWatchError("Roster Watch acquisition redirected")
            if int(response.status_code) != 200:
                raise RosterWatchError(
                    f"Roster Watch acquisition returned HTTP {response.status_code}"
                )
            content = bytes(response.content)
            source_description = (
                f"https://{parsed.hostname.casefold()}{parsed.path or '/'}"
            )
            acquisition = "https"

        expected_sha = (
            ROSTERWATCH_INITIAL_SHA256
            if str(cutoff) == ROSTERWATCH_INITIAL_CUTOFF
            else None
        )
        expected_rows = (
            ROSTERWATCH_INITIAL_ROW_COUNT
            if str(cutoff) == ROSTERWATCH_INITIAL_CUTOFF
            else None
        )
        source_df, parse_metadata = parse_rosterwatch_csv(
            content,
            expected_sha256=expected_sha,
            expected_row_count=expected_rows,
        )
        digest = str(parse_metadata["source_sha256"])
        snapshot_path = root / "snapshots" / f"{cutoff}--{digest}.csv"
        _write_immutable_snapshot(snapshot_path, content)

        previous_state = _read_previous_state(state_path)
        if previous_state is None and not allow_initial_migration:
            raise RosterWatchError(
                "No accepted Roster Watch state; initial migration authorization required"
            )
        if previous_state is None and (
            str(cutoff) != ROSTERWATCH_INITIAL_CUTOFF
            or digest != ROSTERWATCH_INITIAL_SHA256
            or len(source_df) != ROSTERWATCH_INITIAL_ROW_COUNT
            or _sha256(previous_output or b"") != ROSTERWATCH_INITIAL_LEGACY_SHA256
            or len(previous_output or b"") != ROSTERWATCH_INITIAL_LEGACY_BYTE_COUNT
        ):
            raise RosterWatchError(
                "Initial migration must use the exact owner-approved source and legacy baselines"
            )
        if previous_output is None:
            raise RosterWatchError("Previous accepted operational roster is absent")
        if previous_state is not None and _sha256(previous_output) != _text(
            previous_state.get("operational_roster_sha256")
        ):
            raise RosterWatchError(
                "Accepted operational roster does not match Roster Watch state"
            )
        if previous_state is not None and (
            str(cutoff) == _text(previous_state.get("source_cutoff"))
            and digest == _text(previous_state.get("snapshot_sha256"))
        ):
            retained = pd.read_csv(
                io.BytesIO(previous_output), dtype="object", keep_default_na=False
            )
            retained_ledger = identity_records(source_ledger_path)
            if len(retained_ledger) != len(retained):
                raise RosterWatchError(
                    "Accepted Roster Watch ledger/output row coverage mismatch"
                )
            unchanged = [
                {
                    "internal_owner_id": _text(row.get("internal_owner_id")),
                    "fighter": _text(row.get("rw_fighter_raw")),
                    "date_of_birth": _text(row.get("rw_dob_raw")),
                    "rw_row_ordinal": int(_text(row.get("rw_row_ordinal"))),
                }
                for row in retained_ledger
            ]
            unresolved = int(
                (previous_state.get("reconciliation_counts") or {}).get(
                    "unresolved_addition_carriers", 0
                )
            )
            retained_registry = identity_records(owner_registry_path)
            known_identity_pages = (
                _known_identity_pages(legacy_ledger_path, retained_ledger, retained)
                if identity_provider is not None or any(_text(row.get("identity_binding_json")) for row in retained_registry)
                else {}
            )
            _verify_identity_aliases(retained_registry, aliases_path, known_identity_pages)
            if unresolved:
                safe_pending_ledger = [
                    dict(row, legacy_primary_row_ordinal=row["rw_row_ordinal"])
                    for row in retained_ledger
                ]
                safe_pending = materialize_rosterwatch_roster(
                    source_df, retained,
                    {"status": "passed", "source_ledger": safe_pending_ledger,
                     "production_ready": True, "identity_complete": False,
                     "identity_occupied_carriers": _occupied_identity_carriers(retained_registry, known_identity_pages)},
                )
                for ordinal, ledger_row in enumerate(retained_ledger):
                    if ledger_row.get("carrier_resolution_status") == "resolved":
                        continue
                    if any(
                        _text(retained.iloc[ordinal].get(column))
                        != _text(safe_pending.iloc[ordinal].get(column))
                        for column in retained.columns
                    ):
                        raise RosterWatchError(
                            "Pending Roster Watch member contains unconfirmed carrier or page-derived data"
                        )
            _prepare_pending_identities(retained_registry, source_cutoff=str(cutoff))
            old_resolved = {row["internal_owner_id"] for row in retained_registry
                            if row.get("external_carrier_status") == "resolved"}
            recheck = recheck_rosterwatch_identities(
                retained_registry, checker=identity_checker, request_cap=identity_request_cap,
                identity_provider=identity_provider,
                source_contexts=_identity_contexts(source_df, retained_ledger, snapshot_sha256=digest),
                confirmed_profiles=_confirmed_identity_profiles(retained_ledger, retained),
                occupied_carriers=known_identity_pages,
            )
            bindings = _apply_identity_bindings(retained_ledger, retained_registry, known_identity_pages)
            newly_confirmed = sum(owner not in old_resolved for owner in bindings)
            unresolved = sum(row.get("carrier_resolution_status") != "resolved" for row in retained_ledger)
            if bindings:
                retained_materialization_ledger = [
                    dict(row, legacy_primary_row_ordinal=row["rw_row_ordinal"])
                    for row in retained_ledger
                ]
                retained = materialize_rosterwatch_roster(
                    source_df, retained,
                    {"status": "passed", "source_ledger": retained_materialization_ledger,
                     "production_ready": True, "identity_complete": unresolved == 0,
                     "identity_occupied_carriers": _occupied_identity_carriers(retained_registry, known_identity_pages)},
                )
            review = build_rosterwatch_identity_review(
                source_df, retained_ledger, retained_registry, source_cutoff=str(cutoff),
                newly_confirmed=newly_confirmed,
            )
            membership_ready = previous_state.get("reconciliation_status") == "passed"
            # A pending state created under the old identity gate remains a
            # complete membership observation. Persist only its identity evidence.
            if unresolved or bindings:
                registry_bytes = _ledger_csv_bytes(retained_registry, evidence_store)
                next_state = {
                    key: value for key, value in previous_state.items()
                    if not key.startswith("_resolved_")
                }
                next_state.update({
                    "identity_complete": unresolved == 0, "production_ready": membership_ready,
                    "pending_identities": [row for row in review["rows"] if not row["identity_complete"]],
                    "owner_registry_sha256": _sha256(registry_bytes),
                })
                if bindings:
                    roster_bytes = _frame_csv_bytes(retained)
                    ledger_bytes = _ledger_csv_bytes(retained_ledger, evidence_store)
                    counts = dict(next_state.get("reconciliation_counts") or {})
                    counts["unresolved_addition_carriers"] = unresolved
                    next_state.update({
                        "reconciliation_counts": counts,
                        "operational_roster_sha256": _sha256(roster_bytes),
                        "source_ledger_sha256": _sha256(ledger_bytes),
                    })
                    if roster_bytes != backups[output]:
                        write_bytes_atomically(roster_bytes, output)
                    if ledger_bytes != backups[source_ledger_path]:
                        write_bytes_atomically(ledger_bytes, source_ledger_path)
                    alias_hash = _write_identity_aliases(retained_registry, aliases_path, evidence_store)
                    if alias_hash:
                        next_state["learned_aliases_sha256"] = alias_hash
                if registry_bytes != backups[owner_registry_path]:
                    write_bytes_atomically(registry_bytes, owner_registry_path)
                if _json_bytes(next_state) != backups[state_path]:
                    write_bytes_atomically(_json_bytes(next_state), state_path)
            review_attrs = _write_identity_review(review, review_path)
            retained.attrs.update(
                {
                    **review_attrs,
                    "identity_recheck": recheck,
                    "production_ready": membership_ready,
                    "identity_complete": unresolved == 0,
                    "sync_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
                    "membership_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
                    "sync_complete": membership_ready,
                    "sync_fallback_used": False,
                    "sync_completeness_reason": (
                        "complete" if membership_ready else "membership_reconciliation_failed"
                    ),
                    "snapshot_path": str(snapshot_path.resolve()),
                    "snapshot_sha256": digest,
                    "snapshot_bytes": len(content),
                    "source_cutoff": str(cutoff),
                    "request_count": request_count,
                    "acquisition_complete": True,
                    "validation_complete": True,
                    "reconciliation_complete": True,
                    "delta": {
                        "schema_version": ROSTERWATCH_SCHEMA_VERSION,
                        "kind": "idempotent_same_snapshot",
                        "unchanged": unchanged,
                        "additions": [],
                        "removals": [],
                        "renames": [],
                        "reactivations": [],
                        "counts": {
                            "unchanged": len(unchanged),
                            "additions": 0,
                            "removals": 0,
                            "renames": 0,
                            "reactivations": 0,
                        },
                    },
                    "unresolved_identity_carriers": unresolved,
                    "rosterwatch_state_path": str(state_path),
                    "idempotent_snapshot": True,
                }
            )
            return retained
        legacy_df = pd.read_csv(
            io.BytesIO(previous_output), dtype="object", keep_default_na=False
        )
        if previous_state is None and len(legacy_df) != ROSTERWATCH_INITIAL_LEGACY_ROW_COUNT:
            raise RosterWatchError(
                "Initial migration legacy roster row-count pin failed"
            )
        initial_legacy_snapshot_path = (
            root
            / "snapshots"
            / (
                "legacy-accepted-2026-09-01--"
                f"{ROSTERWATCH_INITIAL_LEGACY_SHA256}.csv"
            )
        )
        if previous_state is None:
            _write_immutable_snapshot(
                initial_legacy_snapshot_path, previous_output
            )
        else:
            initial_legacy_snapshot_path = Path(
                str(previous_state["_resolved_initial_legacy_snapshot_path"])
            )
        previous_ledger: Sequence[Mapping[str, object]] | None = None
        previous_registry: list[dict[str, object]] = []
        known_identity_pages: dict[str, str] = {}
        previous_df = pd.DataFrame(columns=ROSTERWATCH_COLUMNS)
        if previous_state is not None:
            previous_cutoff = _text(previous_state.get("source_cutoff"))
            if not _valid_iso_date(previous_cutoff):
                raise RosterWatchError("Accepted Roster Watch cutoff is invalid")
            if str(cutoff) < previous_cutoff:
                raise RosterWatchError("Roster Watch source cutoff moved backwards")
            if (
                str(cutoff) == previous_cutoff
                and digest != _text(previous_state.get("snapshot_sha256"))
            ):
                raise RosterWatchError(
                    "Roster Watch supplied different bytes for an accepted cutoff"
                )
            if not source_ledger_path.is_file():
                raise RosterWatchError(
                    "Accepted Roster Watch state lacks its membership ledger"
                )
            previous_ledger = identity_records(source_ledger_path)
            if len(previous_ledger) != len(legacy_df):
                raise RosterWatchError(
                    "Accepted Roster Watch ledger/output row coverage mismatch"
                )
            legacy_df.attrs["rosterwatch_owner_by_row"] = {
                ordinal: _text(record.get("internal_owner_id"))
                for ordinal, record in enumerate(previous_ledger, start=1)
            }
            previous_bytes = Path(
                str(previous_state["_resolved_snapshot_path"])
            ).read_bytes()
            previous_df, _ = parse_rosterwatch_csv(previous_bytes)
            if len(previous_df) != len(previous_ledger):
                raise RosterWatchError(
                    "Accepted Roster Watch snapshot/ledger row coverage mismatch"
                )
            if not owner_registry_path.is_file():
                raise RosterWatchError(
                    "Accepted Roster Watch state lacks its owner registry"
                )
            previous_registry = identity_records(owner_registry_path)
            if identity_provider is not None or any(_text(row.get("identity_binding_json")) for row in previous_registry):
                known_identity_pages = _known_identity_pages(legacy_ledger_path, previous_ledger, legacy_df)
            _verify_identity_aliases(previous_registry, aliases_path, known_identity_pages)
            registry_owners = {
                _text(record.get("internal_owner_id"))
                for record in previous_registry
            }
            if "" in registry_owners or len(registry_owners) != len(previous_registry):
                raise RosterWatchError("Roster Watch owner registry is not unique")

            # Carry durable source identities forward only through exact
            # source-name+DOB continuity or an explicit reviewed transition.
            by_name_dob: dict[tuple[str, str], set[str]] = defaultdict(set)
            for record in previous_registry:
                owner = _text(record.get("internal_owner_id"))
                name = _text(record.get("current_source_name"))
                dob = _text(record.get("current_source_dob"))
                if owner and name and dob:
                    by_name_dob[(normalize_person_name(name), dob)].add(owner)
            carried_owners: list[str] = []
            for _, source_row in source_df.iterrows():
                fighter = _text(source_row.get("fighter"))
                dob = _text(source_row.get("date_of_birth"))
                candidates = by_name_dob.get((normalize_person_name(fighter), dob), set())
                reviewed_owner = ROSTERWATCH_REVIEWED_SOURCE_OWNER_TRANSITIONS.get(
                    (fighter, dob), ""
                )
                if reviewed_owner:
                    if reviewed_owner not in registry_owners:
                        raise RosterWatchError(
                            f"Reviewed source transition owner is absent: {fighter!r}"
                        )
                    if candidates and candidates != {reviewed_owner}:
                        raise RosterWatchError(
                            f"Reviewed source transition conflicts with continuity: {fighter!r}"
                        )
                    candidates = {reviewed_owner}
                carried_owners.append(
                    next(iter(candidates)) if len(candidates) == 1 else ""
                )
            source_df = source_df.copy()
            source_df.attrs.update(parse_metadata)
            source_df.attrs["require_explicit_owner_for_existing_name"] = True
            source_df["internal_owner_id"] = carried_owners
        reconciliation = reconcile_rosterwatch_baseline(source_df, legacy_df)
        if reconciliation["status"] != "passed":
            raise RosterWatchError("Roster Watch reconciliation gates failed")
        if previous_registry:
            restored_bindings = _apply_identity_bindings(reconciliation["source_ledger"], previous_registry, known_identity_pages)
            if restored_bindings:
                reconciliation["identity_occupied_carriers"] = _occupied_identity_carriers(previous_registry, known_identity_pages)
            unresolved = sum(row.get("carrier_resolution_status") != "resolved"
                             for row in reconciliation["source_ledger"])
            reconciliation["counts"]["unresolved_addition_carriers"] = unresolved
            reconciliation["identity_complete"] = unresolved == 0

        prior_registry_owners = {
            _text(record.get("internal_owner_id")) for record in previous_registry
        }
        prior_current_owners = {
            _text(record.get("internal_owner_id"))
            for record in (previous_ledger or [])
        }
        for record in reconciliation["source_ledger"]:
            owner = _text(record.get("internal_owner_id"))
            record["ever_seen"] = owner in prior_registry_owners
            record["was_current_previous_snapshot"] = owner in prior_current_owners
        delta = (
            _baseline_migration_delta(reconciliation)
            if previous_state is None
            else calculate_rosterwatch_delta(
                previous_df,
                source_df,
                previous_ledger=previous_ledger,
                current_ledger=reconciliation["source_ledger"],
            )
        )

        previous_count = len(previous_df) if previous_state is not None else len(legacy_df)
        if previous_state is not None:
            current_count = len(source_df)
            if current_count > previous_count * ROSTERWATCH_RECURRING_GROWTH_RATIO:
                raise RosterWatchError("Roster Watch recurring growth guard failed")
            if (
                current_count < previous_count * ROSTERWATCH_RECURRING_SHRINK_RATIO
                and previous_count - current_count >= ROSTERWATCH_RECURRING_SHRINK_ABSOLUTE
            ):
                raise RosterWatchError("Roster Watch recurring shrink guard failed")

        candidate = materialize_rosterwatch_roster(
            source_df, legacy_df, reconciliation
        )
        if identity_provider is not None:
            for carrier, owner in _known_identity_pages(legacy_ledger_path, reconciliation["source_ledger"], candidate).items():
                _claim_identity_page(known_identity_pages, carrier, owner)
        registry_by_owner = {
            _text(record.get("internal_owner_id")): dict(record)
            for record in previous_registry
        }
        if previous_state is None:
            for legacy_record in reconciliation["legacy_ledger"]:
                if legacy_record.get("is_primary_representation") is not True:
                    continue
                owner = _text(legacy_record.get("stable_legacy_owner_key"))
                carrier_type, carrier_value = owner.split(":", 1)
                registry_by_owner[owner] = {
                    "internal_owner_id": owner,
                    "first_seen_cutoff": "legacy_accepted_before_2026-09-01",
                    "last_seen_cutoff": "",
                    "membership_status_at_initial_cutoff": "noncurrent",
                    "is_current": False,
                    "current_source_name": _text(
                        legacy_record.get("official_name")
                    ),
                    "current_source_dob": "",
                    "external_carrier_status": "resolved",
                    "preferred_carrier_type": carrier_type,
                    "preferred_carrier_value": carrier_value,
                }
        for record in registry_by_owner.values():
            record["is_current"] = False
        for ledger_record in reconciliation["source_ledger"]:
            owner = _text(ledger_record.get("internal_owner_id"))
            registry_record = registry_by_owner.setdefault(
                owner,
                {
                    "internal_owner_id": owner,
                    "first_seen_cutoff": str(cutoff),
                },
            )
            if previous_state is None:
                registry_record["membership_status_at_initial_cutoff"] = "current"
            registry_record.update(
                {
                    "last_seen_cutoff": str(cutoff),
                    "is_current": True,
                    "current_source_name": _text(
                        ledger_record.get("rw_fighter_raw")
                    ),
                    "current_source_dob": _text(ledger_record.get("rw_dob_raw")),
                    "external_carrier_status": _text(
                        ledger_record.get("carrier_resolution_status")
                    ),
                    "preferred_carrier_type": _text(
                        ledger_record.get("preferred_carrier_type")
                    ),
                    "preferred_carrier_value": _text(
                        ledger_record.get("preferred_carrier_value")
                    ),
                }
            )
        owner_registry = [
            registry_by_owner[owner] for owner in sorted(registry_by_owner)
        ]
        _prepare_pending_identities(owner_registry, source_cutoff=str(cutoff))
        recheck = recheck_rosterwatch_identities(
            owner_registry, checker=identity_checker, request_cap=identity_request_cap,
            identity_provider=identity_provider,
            source_contexts=_identity_contexts(source_df, reconciliation["source_ledger"], snapshot_sha256=digest),
            confirmed_profiles=_confirmed_identity_profiles(reconciliation["source_ledger"], candidate),
            occupied_carriers=known_identity_pages,
        )
        bindings = _apply_identity_bindings(reconciliation["source_ledger"], owner_registry, known_identity_pages)
        if bindings:
            unresolved = sum(row.get("carrier_resolution_status") != "resolved"
                             for row in reconciliation["source_ledger"])
            reconciliation["counts"]["unresolved_addition_carriers"] = unresolved
            reconciliation["identity_complete"] = unresolved == 0
            reconciliation["identity_occupied_carriers"] = _occupied_identity_carriers(owner_registry, known_identity_pages)
            candidate = materialize_rosterwatch_roster(source_df, legacy_df, reconciliation)
        old_resolved = {
            _text(record.get("internal_owner_id")) for record in previous_registry
            if record.get("external_carrier_status") == "resolved"
        }
        old_pending = {_text(record.get("internal_owner_id")) for record in previous_registry
                       if record.get("external_carrier_status") != "resolved"}
        newly_confirmed = sum(
            (record["internal_owner_id"] in old_pending or (
                bool(_text(record.get("identity_binding_json")))
                and record["internal_owner_id"] not in old_resolved
            ))
            and record.get("external_carrier_status") == "resolved"
            and _is_current_identity(record)
            for record in owner_registry
        )
        review = build_rosterwatch_identity_review(
            source_df, reconciliation["source_ledger"], owner_registry,
            source_cutoff=str(cutoff), newly_confirmed=newly_confirmed,
        )
        owner_registry_bytes = _ledger_csv_bytes(owner_registry, evidence_store)
        candidate_bytes = _frame_csv_bytes(candidate)
        source_ledger_bytes = _ledger_csv_bytes(reconciliation["source_ledger"], evidence_store)
        recurring_ledger_bytes = _ledger_csv_bytes(reconciliation["legacy_ledger"])
        baseline_legacy_ledger_bytes = (
            backups[legacy_ledger_path]
            if previous_state is not None
            else recurring_ledger_bytes
        )
        if baseline_legacy_ledger_bytes is None:
            raise RosterWatchError("Initial legacy reconciliation ledger is absent")
        state_record = {
            "schema_version": ROSTERWATCH_SCHEMA_VERSION,
            "status": "passed",
            "membership_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
            "source_description": source_description,
            "acquisition": acquisition,
            "acquired_at_utc": acquired_at,
            "source_cutoff": str(cutoff),
            "snapshot_path": str(Path("snapshots") / snapshot_path.name),
            "snapshot_sha256": digest,
            "snapshot_bytes": len(content),
            "snapshot_rows": len(source_df),
            "initial_legacy_snapshot_path": str(
                initial_legacy_snapshot_path.relative_to(root)
            ),
            "initial_legacy_snapshot_sha256": ROSTERWATCH_INITIAL_LEGACY_SHA256,
            "initial_legacy_snapshot_bytes": len(
                initial_legacy_snapshot_path.read_bytes()
            ),
            "initial_legacy_snapshot_rows": ROSTERWATCH_INITIAL_LEGACY_ROW_COUNT,
            "request_count": request_count,
            "parse": parse_metadata,
            "site_acquisition": (
                {
                    "url": site_acquisition["capture"]["url"],
                    "requested_at_utc": site_acquisition["capture"]["requested_at_utc"],
                    "retrieved_at_utc": site_acquisition["capture"]["retrieved_at_utc"],
                    "status_code": site_acquisition["capture"]["status_code"],
                    "page_bytes": site_acquisition["capture"]["byte_count"],
                    "page_sha256": site_acquisition["capture"]["sha256"],
                    "page_path": str(
                        Path("snapshots")
                        / "site"
                        / Path(str(site_acquisition["capture"]["page_path"])).name
                    ),
                    "metadata_path": str(
                        Path("snapshots")
                        / "site"
                        / Path(str(site_acquisition["capture"]["metadata_path"])).name
                    ),
                    "user_agent": site_acquisition["capture"]["user_agent"],
                    "redirects_followed": site_acquisition["capture"]["redirects_followed"],
                    "retries": site_acquisition["capture"]["retries"],
                    "completeness": site_acquisition["parse"],
                    "derived_csv_sha256": site_acquisition["csv_sha256"],
                }
                if site_acquisition is not None
                else None
            ),
            "reconciliation_status": reconciliation["status"],
            "reconciliation_counts": reconciliation["counts"],
            "production_ready": reconciliation["production_ready"],
            "identity_complete": reconciliation["identity_complete"],
            "pending_identities": [row for row in review["rows"] if not row["identity_complete"]],
            "previous_snapshot_sha256": (
                previous_state.get("snapshot_sha256") if previous_state else None
            ),
            "delta_counts": delta["counts"],
            "owner_registry_rows": len(owner_registry),
            "owner_registry_sha256": _sha256(owner_registry_bytes),
            "operational_roster_sha256": _sha256(candidate_bytes),
            "source_ledger_sha256": _sha256(source_ledger_bytes),
            "legacy_ledger_sha256": _sha256(baseline_legacy_ledger_bytes),
            "recurring_ledger_sha256": _sha256(recurring_ledger_bytes),
        }
        delta_record = {
            "schema_version": ROSTERWATCH_SCHEMA_VERSION,
            "recorded_at_utc": acquired_at,
            "previous_snapshot_sha256": state_record["previous_snapshot_sha256"],
            "snapshot_sha256": digest,
            "source_cutoff": str(cutoff),
            "delta": delta,
        }
        previous_delta = backups[delta_path] or b""
        new_delta = previous_delta + (
            json.dumps(delta_record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        state_record["delta_log_sha256"] = _sha256(new_delta)

        # Pointer is committed last.  Any error restores every mutable byte.
        write_bytes_atomically(candidate_bytes, output)
        write_bytes_atomically(source_ledger_bytes, source_ledger_path)
        if previous_state is None:
            write_bytes_atomically(
                baseline_legacy_ledger_bytes, legacy_ledger_path
            )
        write_bytes_atomically(recurring_ledger_bytes, recurring_ledger_path)
        write_bytes_atomically(owner_registry_bytes, owner_registry_path)
        write_bytes_atomically(new_delta, delta_path)
        review_attrs = _write_identity_review(review, review_path)
        alias_hash = _write_identity_aliases(owner_registry, aliases_path, evidence_store)
        if alias_hash:
            state_record["learned_aliases_sha256"] = alias_hash
        write_bytes_atomically(_json_bytes(state_record), state_path)

        candidate.attrs.update(
            {
                "sync_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
                "membership_source": ROSTERWATCH_MEMBERSHIP_SOURCE,
                "sync_complete": reconciliation["production_ready"] is True,
                "production_ready": reconciliation["production_ready"] is True,
                "identity_complete": reconciliation["identity_complete"] is True,
                "identity_recheck": recheck,
                **review_attrs,
                "sync_fallback_used": False,
                "snapshot_path": str(snapshot_path.resolve()),
                "snapshot_sha256": digest,
                "snapshot_bytes": len(content),
                "source_cutoff": str(cutoff),
                "request_count": request_count,
                "acquisition_complete": True,
                "validation_complete": True,
                "reconciliation_complete": True,
                "acquisition": acquisition,
                "site_acquisition": state_record["site_acquisition"],
                "delta": delta,
                "unresolved_identity_carriers": reconciliation["counts"][
                    "unresolved_addition_carriers"
                ],
                "rosterwatch_state_path": str(state_path),
            }
        )
        return candidate
    except Exception as exc:
        try:
            _restore_mutable_files(backups)
        except Exception as rollback_exc:
            raise RosterWatchError(
                f"Roster Watch sync failed ({exc}); rollback also failed ({rollback_exc})"
            ) from exc
        failed = _failed_fallback(
            output_path=output,
            previous_output=previous_output,
            error=exc,
            request_count=request_count,
        )
        # This is a candidate diagnostic, never a replacement membership claim.
        # Preserve the last review's rows visibly as stale after acquisition failure.
        previous_review = {}
        try:
            if backups[review_path]:
                previous_review = json.loads(backups[review_path].decode("utf-8"))
            if not isinstance(previous_review, dict):
                previous_review = {}
        except (ValueError, UnicodeDecodeError):
            previous_review = {}
        failure_review = {
            **previous_review,
            "schema_version": 1,
            "source_cutoff": previous_review.get("source_cutoff", ""),
            "rows": previous_review.get("rows", []),
            "counts": previous_review.get("counts", {}),
            "status": "stale" if previous_review else "unavailable",
            "membership_current": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            failed.attrs.update(_write_identity_review(failure_review, review_path))
        except Exception as review_exc:
            failed.attrs["identity_review_write_error"] = f"{type(review_exc).__name__}: {review_exc}"
        return failed


def merge_event_safety_net(
    roster_df: pd.DataFrame,
    event_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return transient backfill targets without changing roster membership."""
    base = roster_df.copy()
    base.attrs = dict(roster_df.attrs)
    base.attrs["event_safety_net_rows"] = int(len(event_rows))
    base.attrs["event_safety_net_membership_mutations"] = 0
    return base


# Fail fast at import: every reviewed addition carrier must be derived from a bound
# evidence record (see validate_reviewed_addition_evidence).
validate_reviewed_addition_evidence()
