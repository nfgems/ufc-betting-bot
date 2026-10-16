"""Fail-closed provenance for the approved targeted identity follow-up.

This registry records search anchors, identity edges, and explicit holds.  It is
not a candidate mutation input and is intentionally not imported by roster
resolution or publication code.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from src.data.profile_integrity import source_entity_id
from src.data.reviewed_roster_identity import load_reviewed_roster_identity_registry
from src.data.reviewed_roster_recovery import load_reviewed_roster_recovery_registry


REVIEWED_ROSTER_IDENTITY_FOLLOWUP_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ufc_roster_identity_followup_v1.json"
)

_EXPECTED_REGISTRY_SHA256 = (
    "a44a3ad1647c7d81b8336dc14fea8048e083a527f0ae568e047aa4c8ff7475f7"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "candidate_effect",
        "official_search_anchors",
        "official_internal_identity_edges",
        "external_identity_bindings",
        "approved_unavailable_source_bindings",
        "unbound_candidate_holds",
        "preserved_source_identity_rejections",
        "permanent_search_false_positive_rejections",
        "non_mma_policy",
    }
)
_SCOPE = {
    "scope_id": "canonical-118-targeted-identity-followup-2026-08-30",
    "approved_as_of_date": "2026-08-30",
    "approved_decision_ids": [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "U30-EMILIO-SHERDOG",
        "U30-EMILIO-HEIGHT-5FT4",
        "U30-FELIX-ESPN",
        "U30-FENG-ESPN",
        "U30-JOSE-FIGHTDX",
        "U30-JOSE-WIKIPEDIA-NAMESAKE",
    ],
    "identity_worklist_sha256": (
        "414c78e71d69b1c9378015b6e4280b8b00ee207e5fd05762c0eeb80e718e82b7"
    ),
    "stop_review_sha256": (
        "2df4dc51b2b224f7c739abcee34fabd78184553b7f731edefed6b5ddc74456a2"
    ),
    "stop_review_markdown_sha256": (
        "5b4ff5cf3ae90b62f29aadcc697f63c52a98ee063e1cd9eb24f66e3b50d3a84c"
    ),
    "stop_proof_sha256": (
        "571093c7c12febdd20b516d73ddad90a43e7e1f6276826a3b2e7c8fbc3ec82d7"
    ),
    "independent_validation_sha256": (
        "d07617b8b6c718907af9101f43e97d5f23f4d9d522c0786ed6eb52764724db9d"
    ),
    "unavailable_worklist_sha256": (
        "e27d2c1788c3a6d4545d5a1ef489aad88d02e4a99f44fc040f560995d3db8bb9"
    ),
    "unavailable_research_manifest_sha256": (
        "a951712c13caf163146ef521d6f9f6776edf50fb9e3f68cf82b0c0d0e73b76e4"
    ),
    "unavailable_research_validation_sha256": (
        "90420cd9f6df7908106a76ef2091918f58d1de1b2dc997bcba9c25518f12626c"
    ),
    "remaining_unavailable_research_manifest_sha256": (
        "3e4bb1f046b2d2375822cecf9c5640c7bd589a5076fab0f9cfaafa12e6b7435d"
    ),
    "stable_worklist_cells": 55,
    "stable_worklist_fighters": 17,
    "active_research_cells": 50,
    "active_research_fighters": 14,
}
_CANDIDATE_EFFECT = {
    "immutable_baseline_candidate_id": "u55p1",
    "fresh_review_candidate_id": "u30h1",
    "field_values_authorized": [
        "D005:athlete:emilio-cuellar:height:162.56cm:fightdx:emilio-cuellar"
    ],
    "modern_ufcstats_bindings_authorized": [],
    "new_non_mma_exclusions_authorized": [],
    "roster_deletions_authorized": [],
    "runtime_behavior_changed": True,
    "publication_authorized": False,
}

_ANCHOR_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "official_node_id",
        "official_uuid",
        "dob",
        "mma_id",
        "legacy_numeric_fightmetric_id",
        "decision_ids",
        "evidence",
        "search_use_only",
        "profile_value_import_allowed",
    }
)
_EVIDENCE_KEYS = frozenset(
    {"evidence_grade", "artifact_sha256", "recorded_at_utc"}
)
_EXPECTED_ANCHORS = {
    "Fabio Agu": (118470, "1988-02-10", "", "3555", ("R1",)),
    "Dakota Weigher": (131058, "1990-04-23", "", "", ("R5",)),
    "Jesse Mariotti": (131061, "1997-01-11", "", "", ("R5",)),
    "Jordan Jackson": (131056, "1999-08-06", "", "", ("R5", "R6")),
    "Keifer Roberts": (131059, "1992-08-11", "", "", ("R5",)),
    "Michael Bonnette": (130703, "1988-09-21", "", "", ("R4", "R5")),
    "Montrel Talbert": (130706, "1992-07-01", "", "", ("R5",)),
    "Martin Mishtaku": (131060, "1987-05-21", "", "", ("R4", "R5")),
    "Timothy Thomas": (128385, "1989-01-01", "", "", ("R4", "R5")),
    "Harrison Garcia": (132237, "", "175", "", ("R5",)),
}
_TRANSCRIBED_LEAD_NAMES = {
    "Fabio Agu",
    "Dakota Weigher",
    "Jesse Mariotti",
    "Harrison Garcia",
}

_INTERNAL_EDGE_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "official_node_id",
        "official_uuid",
        "athlete_stat_uuid",
        "legacy_numeric_fightmetric_id",
        "career_fights_fightid",
        "relationship_url",
        "athlete_stat_url",
        "queue_indices",
        "requested_fields",
        "capture_sha256s",
        "decision_id",
        "edge_recording_authorized",
        "modern_ufcstats_binding_authorized",
        "profile_value_import_allowed",
        "non_mma_exclusion_authorized",
        "roster_deletion_authorized",
    }
)
_EXTERNAL_BINDING_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "official_node_id",
        "source",
        "source_entity_ids",
        "source_url",
        "identity_anchors",
        "queue_indices",
        "requested_fields",
        "capture_sha256s",
        "decision_id",
        "edge_recording_authorized",
        "modern_ufcstats_binding_authorized",
        "profile_value_import_allowed",
        "non_mma_exclusion_authorized",
        "roster_deletion_authorized",
    }
)
_IDENTITY_ANCHOR_KEYS = frozenset(
    {"kind", "official_raw", "external_raw", "normalized"}
)
_APPROVED_UNAVAILABLE_SOURCE_BINDINGS = [
    {
        "official_name": "Emilio Cuellar",
        "official_athlete_url": "https://www.ufc.com/athlete/emilio-cuellar",
        "official_source_entity_id": "athlete:emilio-cuellar",
        "official_node_id": 118293,
        "canonical_ufcstats_id": "3aeb69b4a13ff0e8",
        "canonical_ufcstats_url": (
            "http://ufcstats.com/fighter-details/3aeb69b4a13ff0e8"
        ),
        "source": "sherdog",
        "source_entity_id": "sherdog:275203",
        "source_url": "https://www.sherdog.com/fighter/Emilio-Cuellar-275203",
        "identity_anchors": [
            {
                "opponent": "Jorge Calvo",
                "date": "2022-03-18",
                "organization": "LUX 021",
                "normalized": "2022-03-18|lux-021|jorge-calvo",
            },
            {
                "opponent": "Luis Ivan Rodriguez",
                "date": "2022-11-18",
                "organization": "LUX 028",
                "normalized": (
                    "2022-11-18|lux-028|luis-ivan-rodriguez"
                ),
            },
        ],
        "corroborating_external_only_observations": [
            {
                "opponent": "Enrique Gonzalez",
                "date": "2023-12-01",
                "organization": "LUX 038",
                "normalized": "2023-12-01|lux-038|enrique-gonzalez",
                "identity_bridge_claimed": False,
            },
        ],
        "requested_field_results": [
            {
                "queue_index": 24,
                "field": "reach",
                "raw_value": "",
                "normalized_value": None,
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            },
            {
                "queue_index": 25,
                "field": "stance",
                "raw_value": "",
                "normalized_value": "",
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            },
        ],
        "nonrequested_values_withheld": [
            {
                "field": "dob",
                "raw_value": "Dec 17, 1995",
                "normalized_value": "1995-12-17",
                "existing_state": "accepted_current_future_d004",
                "disposition": "already_present_no_new_import",
                "import_allowed": False,
            },
            {
                "field": "height",
                "raw_value": "5'4\" / 162.56 cm",
                "normalized_value": 64.0,
                "existing_state": "accepted_current_future_d005",
                "disposition": (
                    "owner_approved_corroboration_selected_via_fightdx"
                ),
                "import_allowed": False,
            },
            {
                "field": "weight",
                "raw_value": "125 lbs / 56.7 kg",
                "normalized_value": 125.0,
                "existing_state": "ufcstats_scraped_125_lb",
                "disposition": "already_present_no_new_import",
                "import_allowed": False,
            },
        ],
        "evidence": {
            "retrieval_date_utc": "2026-08-30",
            "exact_request_timestamp_available": False,
            "raw_http_capture_available": False,
            "search_result_transcript_sha256": (
                "671c7b8463c75f0dedd7dfaf5a07aff4a07832ba0f7089ad68bdd46b9a0dd74f"
            ),
            "route_ledger_sha256": (
                "07d3535c554a7a25663b7db64ee1331f6e98051dd31d554baa0eaebe20cb2c83"
            ),
            "official_html_capture_sha256": (
                "583fdce41bf9cee3045816c3e3d08224408eb826147c3eacad105e34dd2d5e8d"
            ),
            "ufcstats_capture_sha256": (
                "c76969c1303f4199f6ea85d723982d5f331e6114362f886ec02c9b8d7267d45e"
            ),
            "fightdx_capture_sha256": (
                "5f74b3eaf78b3bbf33f303d8646d64ed72085c3032b4092f4fe6282da6f2c85f"
            ),
            "tapology_capture_sha256": (
                "9f0fecc028374481e4cf090ab087da99cca22a1d041f5193ce0b768cfa477f50"
            ),
            "independent_validation_sha256": (
                "90420cd9f6df7908106a76ef2091918f58d1de1b2dc997bcba9c25518f12626c"
            ),
        },
        "decision_id": "U30-EMILIO-SHERDOG",
        "owner_approved_at_date": "2026-08-30",
        "edge_recording_authorized": True,
        "profile_value_import_allowed": False,
        "non_mma_exclusion_authorized": False,
        "roster_deletion_authorized": False,
        "candidate_effect_authorized": False,
        "runtime_consumption_authorized": False,
    },
    {
        "official_name": "Felix Klinkhammer",
        "official_athlete_url": "https://www.ufc.com/athlete/felix-klinkhammer",
        "official_source_entity_id": "athlete:felix-klinkhammer",
        "canonical_ufcstats_id": "8276f28f1ad087de",
        "canonical_ufcstats_url": (
            "http://ufcstats.com/fighter-details/8276f28f1ad087de"
        ),
        "known_source_entity_ids": ["tapology:120890", "sherdog:213423"],
        "source": "espn",
        "source_entity_id": "espn:5078427",
        "source_url": (
            "https://www.espn.com/mma/fighter/_/id/5078427/felix-klinkhammer"
        ),
        "identity_anchors": [
            {
                "source_opponent": "Wissame Akhmouch",
                "matched_opponent": "Wissem Akhmouch",
                "date": "2022-04-16",
                "organization": "Ares FC 5",
                "normalized": "2022-04-16|ares-fc-5|wissem-akhmouch",
            },
            {
                "source_opponent": "Karim Rabei",
                "matched_opponent": "Karim Rabei",
                "date": "2021-12-11",
                "organization": "Ares FC 2",
                "normalized": "2021-12-11|ares-fc-2|karim-rabei",
            },
            {
                "source_opponent": "Anatoly Tarasenko",
                "matched_opponent": "Anatoly Tarasenko",
                "date": "2021-07-24",
                "organization": "Oktagon MMA 26",
                "normalized": (
                    "2021-07-24|oktagon-mma-26|anatoly-tarasenko"
                ),
            },
            {
                "source_opponent": "Laid Zerhouni",
                "matched_opponent": "Laid Zerhouni",
                "date": "2019-09-28",
                "organization": "Contenders Norwich 27",
                "normalized": (
                    "2019-09-28|contenders-norwich-27|laid-zerhouni"
                ),
            },
        ],
        "requested_field_results": [
            {
                "queue_index": 37,
                "field": "reach",
                "raw_value": "",
                "normalized_value": None,
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            },
            {
                "queue_index": 38,
                "field": "stance",
                "raw_value": "-",
                "normalized_value": "",
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            },
        ],
        "nonrequested_values_withheld": [
            {
                "field": "height",
                "raw_value": "6'0\"",
                "normalized_value_cm": 182.88,
                "existing_state": "ufcstats_scraped_6ft0",
                "disposition": "already_present_no_new_import",
                "import_allowed": False,
            }
        ],
        "evidence": {
            "retrieval_date_utc": "2026-08-30",
            "exact_request_timestamp_available": False,
            "http_status_available": False,
            "raw_http_capture_available": False,
            "capture_kind": "human_transcribed_web_search_tool_output",
            "evidence_manifest_sha256": (
                "21f293cb748a718e5cbeff10e4bb99c377ddf66a3c055f87db2558dd63b98622"
            ),
            "search_result_transcript_sha256": (
                "796e6004734629cb693de988d01272e1c21245f71968177e74d21e7fadab7812"
            ),
            "route_ledger_sha256": (
                "f3265cf519d18f06737657b38c2661b3496183a6540a0d48e933a0718d20d20d"
            ),
            "finding_report_sha256": (
                "31b98aa74270a2f05e7f14d81c1472a93ef1d87911d11c6d6141d8ac54e019ae"
            ),
            "sherdog_capture_sha256": (
                "345db09f9be27925cfd5bc6d5ba9d4967b750335c3b86216e3ff45748ff7269a"
            ),
        },
        "decision_id": "U30-FELIX-ESPN",
        "owner_approved_at_date": "2026-08-30",
        "edge_recording_authorized": True,
        "profile_value_import_allowed": False,
        "non_mma_exclusion_authorized": False,
        "roster_deletion_authorized": False,
        "candidate_effect_authorized": False,
        "runtime_consumption_authorized": False,
    },
    {
        "official_name": "Feng Pengchao",
        "official_athlete_url": "https://www.ufc.com/athlete/feng-pengchao",
        "official_source_entity_id": "athlete:feng-pengchao",
        "canonical_ufcstats_id": "86597694317e5e1b",
        "canonical_ufcstats_url": (
            "http://ufcstats.com/fighter-details/86597694317e5e1b"
        ),
        "known_source_entity_ids": ["tapology:230675", "sherdog:336611"],
        "source": "espn",
        "source_entity_id": "espn:5073905",
        "source_url": (
            "https://www.espn.com/mma/fighter/_/id/5073905/feng-pengchao"
        ),
        "source_profile_context": {
            "name": "Feng Pengchao",
            "nickname": "Winged Tiger",
            "country": "China",
            "division": "Featherweight",
            "record": "0-2-0",
            "record_disposition": (
                "incomplete_not_imported_not_used_as_identity_anchor"
            ),
        },
        "identity_anchors": [
            {
                "source_opponent": "Yibugele",
                "matched_opponent": "Yibugele",
                "date": "2021-04-30",
                "organization": "WLF: W.A.R.S. 52",
                "result": "loss by TKO (Retirement), round 2 at 5:00",
                "normalized": "2021-04-30|wlf-wars-52|yibugele",
            },
            {
                "source_opponent": "A. Jinensibieke",
                "matched_opponent": "Jinensibieke Asikeerbai",
                "date": "2020-08-21",
                "organization": "WLF: W.A.R.S. 46",
                "result": (
                    "loss by unanimous decision, round 3 at 5:00"
                ),
                "normalized": (
                    "2020-08-21|wlf-wars-46|jinensibieke-asikeerbai"
                ),
            },
        ],
        "requested_field_results": [
            {
                "queue_index": 39,
                "field": "stance",
                "raw_value": "",
                "normalized_value": "",
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            }
        ],
        "nonrequested_values_withheld": [
            {
                "field": "reach",
                "raw_value": "68\"",
                "normalized_value_inches": 68.0,
                "normalized_value_cm": 172.72,
                "existing_state": "accepted_current_future_d010",
                "disposition": (
                    "rounding_compatible_noncarrier_corroboration_"
                    "keep_tapology_d010"
                ),
                "existing_carrier": {
                    "cell_id": "D010",
                    "source": "tapology",
                    "source_entity_id": "tapology:230675",
                    "source_url": (
                        "https://www.tapology.com/fightcenter/fighters/"
                        "230675-feng-peng-zhao-winged-tiger"
                    ),
                    "raw_value": "68.1\" (173cm)",
                    "normalized_value_cm": 172.974,
                },
                "exact_delta_inches": 0.1,
                "exact_delta_cm": 0.254,
                "genuine_conflict": False,
                "import_allowed": False,
                "replacement_allowed": False,
            }
        ],
        "evidence": {
            "retrieval_date_utc": "2026-08-30",
            "exact_request_timestamp_available": False,
            "http_status_available": False,
            "raw_http_capture_available": False,
            "capture_kind": "human_transcribed_web_search_tool_output",
            "evidence_manifest_sha256": (
                "a138b4a58101ac4ba8c288183f921edffbc53fc521c5f4cbcf9810da3a67983b"
            ),
            "search_result_transcript_sha256": (
                "c86595ddcc4c6dc95d17dd17916f209779aa111f7e8d05e24d720ceefef17725"
            ),
            "route_ledger_sha256": (
                "d9d5479ecb88720ca449dfd0d6595fe230b48016955086fd8d3e91cf6cf0e9ed"
            ),
            "finding_report_sha256": (
                "1caaae1c2c5f84e662dcf2b27b193ea2a828ffccebc44e01f6dc8c3f47094cd3"
            ),
            "sherdog_capture_sha256": (
                "a3c30be972072a1835d7c04efc82f1c1124cbc484c57f6da374c0e7553e16c45"
            ),
            "tapology_capture_sha256": (
                "97566ecd9ddd2337104c61cc6cf882c57a36956024585bd0565f84c99739c8a7"
            ),
            "owner_decision_sha256": (
                "da4c2f43d9929de4689188b64ef3f91cb0553686074299db51c1e7592a918a4f"
            ),
        },
        "decision_id": "U30-FENG-ESPN",
        "owner_approved_at_date": "2026-08-30",
        "edge_recording_authorized": True,
        "profile_value_import_allowed": False,
        "non_mma_exclusion_authorized": False,
        "roster_deletion_authorized": False,
        "candidate_effect_authorized": False,
        "runtime_consumption_authorized": False,
        "historical_backfill_allowed": False,
    },
    {
        "official_name": "Jose Augusto",
        "official_athlete_url": "https://www.ufc.com/athlete/jose-augusto",
        "official_source_entity_id": "athlete:jose-augusto",
        "canonical_ufcstats_id": "4d3b08ab127c7585",
        "canonical_ufcstats_url": (
            "http://ufcstats.com/fighter-details/4d3b08ab127c7585"
        ),
        "known_source_entity_ids": [
            "espn:4684239",
            "sherdog:111573",
            "tapology:133511",
            "pfl:jose-augusto",
        ],
        "source": "fightdx",
        "source_entity_id": "fightdx:jose-augusto-azevedo",
        "source_url": "https://fightdx.com/person/jose-augusto-azevedo",
        "prior_url_only_context": {
            "url_was_previously_visible_as_opponent_link": True,
            "prior_capture_path": (
                "C:/Temp/u118r1/sources/batch-06/raw/"
                "simon-biyong-fightdx-initial--565532344f185256.html"
            ),
            "prior_capture_sha256": (
                "565532344f1852569066aaac1366cc9525ead87cfdad58b9d061dc72707333bf"
            ),
            "prior_capture_retrieved_at_utc": "2026-08-29T17:21:54.701965Z",
            "jose_profile_previously_fetched": False,
            "jose_to_fightdx_binding_previously_registered": False,
            "jose_field_evidence_previously_used": False,
            "disposition": (
                "url_known_only_as_opponent_link_new_jose_to_fightdx_edge"
            ),
        },
        "source_profile_context": {
            "name": "Jose Augusto Azevedo",
            "record": "7-3-1-0",
            "record_disposition": (
                "stale_search_index_context_not_imported_not_used_for_roster_status"
            ),
        },
        "identity_anchors": [
            {
                "source_opponent": "Jonathan Wilson",
                "matched_opponent": "Jonathan Wilson",
                "date": "2021-04-02",
                "organization": "Bellator 255",
                "result": (
                    "win by arm-triangle submission, round 1 at 4:58"
                ),
                "normalized": (
                    "2021-04-02|bellator-255|jonathan-wilson"
                ),
            },
            {
                "source_opponent": "Anthony Johnson",
                "matched_opponent": "Anthony Johnson",
                "date": "2021-05-07",
                "organization": "Bellator 258",
                "result": "loss by punch knockout, round 2 at 1:30",
                "normalized": (
                    "2021-05-07|bellator-258|anthony-johnson"
                ),
            },
            {
                "source_opponent": "Alex Polizzi",
                "matched_opponent": "Alex Polizzi",
                "date": "2022-03-12",
                "organization": "Bellator 276",
                "result": (
                    "loss by rear-naked-choke submission, round 3 at 0:49"
                ),
                "normalized": "2022-03-12|bellator-276|alex-polizzi",
            },
            {
                "source_opponent": "Simon Biyong",
                "matched_opponent": "Simon Biyong",
                "date": "2023-05-12",
                "organization": "Bellator 296",
                "result": (
                    "win by unanimous decision, round 3 at 5:00"
                ),
                "normalized": "2023-05-12|bellator-296|simon-biyong",
            },
            {
                "source_opponent": "Richard Jacobi",
                "matched_opponent": "Richard Jacobi",
                "date": "2023-11-03",
                "organization": "LFA 171",
                "result": "win by decision, round 3 at 5:00",
                "normalized": "2023-11-03|lfa-171|richard-jacobi",
            },
            {
                "source_opponent": "Edivan Santos",
                "matched_opponent": "Edivan Mineiro Santos",
                "date": "2024-11-22",
                "organization": "LFA 197",
                "result": "win by punch TKO, round 2 at 3:20",
                "normalized": (
                    "2024-11-22|lfa-197|edivan-mineiro-santos"
                ),
            },
        ],
        "requested_field_results": [
            {
                "queue_index": 56,
                "field": "stance",
                "raw_value": "",
                "normalized_value": "",
                "decision": "keep_unresolved_blank",
                "import_allowed": False,
            }
        ],
        "nonrequested_values_withheld": [
            {
                "cell_id": "D017",
                "field": "reach",
                "source": "espn",
                "source_entity_id": "espn:4684239",
                "raw_value": "73\"",
                "normalized_value_inches": 73.0,
                "normalized_value_cm": 185.42,
                "existing_state": "accepted_current_future_d017",
                "disposition": "existing_carrier_unchanged_no_new_import",
                "import_allowed": False,
                "replacement_allowed": False,
            }
        ],
        "evidence": {
            "retrieval_date_utc": "2026-08-30",
            "exact_request_timestamp_available": False,
            "http_status_available": False,
            "raw_http_capture_available": False,
            "capture_kind": "human_transcribed_web_search_tool_output",
            "evidence_manifest_sha256": (
                "347e6e37b2311df584b9d43328e777502ece6b096079a0acffaa8c187de03656"
            ),
            "search_result_transcript_sha256": (
                "4cafa8b18ffdd61228e554288218f0edf8d3545d5febf09b660be015ad637efc"
            ),
            "route_ledger_sha256": (
                "5a0fc7f0f7a8cf11281bbe3c2d0bb830675d7c5a898ec3cc6ce5e83160506e25"
            ),
            "finding_report_sha256": (
                "38673701ee5603400b21e1f88bbd86676247a4c4cff00de4cae7962032fbaffc"
            ),
            "owner_review_sha256": (
                "3a33526f3f8a03c98c2604d91accdb38b31fcb2a7c7ee792792923f665076fe7"
            ),
            "independent_audit_sha256": (
                "9809a0449a58702f55ed2346eb6052d7234b1e22bae3feef37a11ad4bbc30c19"
            ),
            "sherdog_capture_sha256": (
                "cfe0b04ba02e0797911dfb40837a681adcf11b40b295da784d36433bf3478273"
            ),
            "simon_fightdx_capture_sha256": (
                "565532344f1852569066aaac1366cc9525ead87cfdad58b9d061dc72707333bf"
            ),
            "owner_decision_sha256": (
                "2ec36cdeeed038950e8189571738c372ec0dcbdb2f5f437882d547777350375e"
            ),
        },
        "decision_id": "U30-JOSE-FIGHTDX",
        "owner_approved_at_date": "2026-08-30",
        "edge_recording_authorized": True,
        "profile_value_import_allowed": False,
        "non_mma_exclusion_authorized": False,
        "roster_deletion_authorized": False,
        "candidate_effect_authorized": False,
        "runtime_consumption_authorized": False,
        "historical_backfill_allowed": False,
        "roster_status_use_allowed": False,
    },
]
_HOLD_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "candidate_source_entities",
        "queue_indices",
        "requested_fields",
        "identity_evidence",
        "missing_bridge",
        "conditional_values_withheld",
        "evidence_sha256s",
        "decision_id",
        "automatic_binding_allowed",
        "profile_value_import_allowed",
        "permanent_exclusion_authorized",
        "roster_deletion_authorized",
    }
)
_CANDIDATE_ENTITY_KEYS = frozenset(
    {"source", "source_entity_id", "source_url"}
)
_EXPECTED_HOLD_CELLS = {
    "Fabio Agu": (32, 33, 34, 35),
    "Martin Mishtaku": (75, 76, 77),
    "Michael Bonnette": (82, 83, 84),
    "Timothy Thomas": (107, 108, 109),
    "Vineesh Subrahmanyan": (110, 111, 112, 113, 114),
}
_EXPECTED_HOLD_ENTITY_IDS = {
    "Fabio Agu": {
        "ufcstats:501821d7fb7b95c1",
        "espn:4334743",
        "tapology:103527",
        "sherdog:132429",
    },
    "Michael Bonnette": {"tapology:22573"},
    "Martin Mishtaku": {"tapology:427291"},
    "Timothy Thomas": {"tapology:7574"},
    "Vineesh Subrahmanyan": {"tapology:485663"},
}
_REJECTION_KEYS = frozenset(
    {
        "official_name",
        "official_athlete_url",
        "official_source_entity_id",
        "source",
        "source_entity_id",
        "source_url",
        "official_dob",
        "candidate_dob",
        "rejection_basis",
        "decision_id",
    }
)
_EXPECTED_REJECTION_PAIRS = {
    ("athlete:jordan-jackson", "tapology:21470"),
    ("athlete:jordan-jackson", "sherdog:400887"),
    ("athlete:leonardo-de-oliveira", "tapology:67533"),
    ("athlete:leonardo-de-oliveira", "sherdog:84049"),
}
_SEARCH_FALSE_POSITIVE_REJECTION_KEYS = frozenset(
    {
        "official_name",
        "official_full_name",
        "official_athlete_url",
        "official_source_entity_id",
        "canonical_ufcstats_id",
        "canonical_ufcstats_url",
        "canonical_sherdog_source_entity_id",
        "canonical_dob",
        "canonical_nickname",
        "route_id",
        "search_query",
        "source",
        "result_page_title",
        "result_page_url",
        "result_page_source_entity_id",
        "mentioned_person_name",
        "mentioned_person_source_entity_id",
        "raw_context",
        "identity_comparison",
        "requested_field_result",
        "evidence",
        "decision_id",
        "owner_approved_at_date",
        "permanent_search_false_positive_rejection",
        "different_person",
        "bind_result_page_to_canonical_fighter",
        "bind_mentioned_person_to_canonical_fighter",
        "profile_value_import_allowed",
        "candidate_effect_authorized",
        "runtime_consumption_authorized",
        "historical_backfill_allowed",
        "roster_status_use_allowed",
        "roster_deletion_authorized",
        "non_mma_exclusion_authorized",
        "publication_authorized",
    }
)
_SEARCH_FALSE_POSITIVE_CONTEXT_KEYS = frozenset(
    {
        "narrative_date",
        "table_date",
        "event",
        "opponent",
        "result_from_page_owner_perspective",
        "round",
        "location",
    }
)
_SEARCH_FALSE_POSITIVE_IDENTITY_KEYS = frozenset(
    {
        "full_name_mismatch",
        "stable_id_bridge_found",
        "matching_bout_bridge_found",
        "joao_zaiden_or_colina_fight_present_in_canonical_record",
        "canonical_2014_bouts",
    }
)
_SEARCH_FALSE_POSITIVE_FIELD_KEYS = frozenset(
    {"queue_index", "field", "raw_value", "normalized_value", "decision"}
)
_SEARCH_FALSE_POSITIVE_EVIDENCE_KEYS = frozenset(
    {
        "retrieval_date_utc",
        "capture_kind",
        "raw_http_capture_available",
        "http_status_available",
        "exact_request_timestamp_available",
        "evidence_manifest_sha256",
        "search_result_transcript_sha256",
        "route_ledger_sha256",
        "failure_ledger_sha256",
        "owner_decision_sha256",
        "independent_u21_audit_sha256",
        "prior_route_identity_audit_sha256",
        "remaining_route_identity_audit_sha256",
        "registry_candidate_identity_audit_sha256",
        "finding_report_sha256",
        "canonical_sherdog_capture_sha256",
    }
)
_POLICY_KEYS = frozenset(
    {
        "permanent_exclusion_selection_key",
        "exact_permanent_exclusions",
        "unbound_power_slap_candidate_entity_ids",
        "broad_name_blacklist_allowed",
        "unbound_candidate_permanent_exclusion_allowed",
        "missing_information_authorizes_deletion",
    }
)
_POLICY_EXCLUSION_KEYS = frozenset(
    {"official_name", "official_athlete_url", "official_source_entity_id"}
)
_EXPECTED_PERMANENT_EXCLUSION_ENTITIES = {
    "athlete:cory-corbin",
    "athlete:dallas-marron",
    "athlete:frank-holland",
}
_EXPECTED_UNBOUND_POWER_SLAP_ENTITIES = {
    "tapology:22573",
    "tapology:427291",
    "tapology:7574",
    "tapology:485663",
}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _require_exact_keys(
    value: Mapping[str, object], keys: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != keys:
        raise ValueError(
            f"{label} has invalid keys "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )


def _require_strings(
    value: object, label: str, *, minimum: int = 1
) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} strings")
    items = [_text(item) for item in value]
    if any(not item for item in items) or len(items) != len(set(items)):
        raise ValueError(f"{label} must contain unique nonblank strings")
    return items


def _require_sha256s(value: object, label: str) -> list[str]:
    items = _require_strings(value, label)
    if any(_SHA256_RE.fullmatch(item) is None for item in items):
        raise ValueError(f"{label} contains an invalid SHA-256")
    return items


def _validate_official_entity(record: Mapping[str, object], label: str) -> str:
    entity = source_entity_id("ufc", record.get("official_athlete_url"))
    if not entity or entity != record.get("official_source_entity_id"):
        raise ValueError(f"{label} has an inconsistent official UFC entity")
    return entity


def _validate_authorization_flags(record: Mapping[str, object], label: str) -> None:
    for field in (
        "modern_ufcstats_binding_authorized",
        "profile_value_import_allowed",
        "non_mma_exclusion_authorized",
        "roster_deletion_authorized",
    ):
        if record.get(field) is not False:
            raise ValueError(f"{label} unexpectedly authorizes {field}")


def _validate_registry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Reviewed identity follow-up registry must be an object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "identity follow-up registry")
    if payload.get("schema_version") != 1:
        raise ValueError("Identity follow-up registry schema_version must be 1")
    if payload.get("scope") != _SCOPE:
        raise ValueError("Identity follow-up scope or evidence pins drifted")
    if payload.get("candidate_effect") != _CANDIDATE_EFFECT:
        raise ValueError("Identity follow-up candidate-effect boundary drifted")

    anchors = payload.get("official_search_anchors")
    if not isinstance(anchors, list) or len(anchors) != 10:
        raise ValueError("Identity follow-up must contain exactly 10 search anchors")
    seen_anchor_entities: set[str] = set()
    anchor_names: set[str] = set()
    for position, record in enumerate(anchors, start=1):
        label = f"identity follow-up search anchor {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _ANCHOR_KEYS, label)
        entity = _validate_official_entity(record, label)
        if entity in seen_anchor_entities:
            raise ValueError(f"{label} duplicates an official entity")
        seen_anchor_entities.add(entity)
        name = _text(record.get("official_name"))
        expected = _EXPECTED_ANCHORS.get(name)
        observed = (
            record.get("official_node_id"),
            _text(record.get("dob")),
            _text(record.get("mma_id")),
            _text(record.get("legacy_numeric_fightmetric_id")),
            tuple(record.get("decision_ids") or ()),
        )
        if expected is None or observed != expected:
            raise ValueError(f"{label} does not match an approved anchor")
        anchor_names.add(name)
        uuid = _text(record.get("official_uuid")).casefold()
        if uuid and _UUID_RE.fullmatch(uuid) is None:
            raise ValueError(f"{label} has an invalid official UUID")
        dob = _text(record.get("dob"))
        if dob and _DATE_RE.fullmatch(dob) is None:
            raise ValueError(f"{label} has an invalid DOB")
        if record.get("search_use_only") is not True:
            raise ValueError(f"{label} is not search-only")
        if record.get("profile_value_import_allowed") is not False:
            raise ValueError(f"{label} authorizes a profile value")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{label} has no evidence records")
        grades: set[str] = set()
        for evidence_position, item in enumerate(evidence, start=1):
            evidence_label = f"{label}.evidence[{evidence_position}]"
            if not isinstance(item, dict):
                raise ValueError(f"{evidence_label} must be an object")
            _require_exact_keys(item, _EVIDENCE_KEYS, evidence_label)
            grade = _text(item.get("evidence_grade"))
            if grade not in {
                "captured_official_html",
                "captured_official_jsonapi",
                "transcribed_no_raw_capture",
            }:
                raise ValueError(f"{evidence_label} has an invalid evidence grade")
            grades.add(grade)
            if _SHA256_RE.fullmatch(_text(item.get("artifact_sha256"))) is None:
                raise ValueError(f"{evidence_label} has an invalid SHA-256")
            if not _text(item.get("recorded_at_utc")):
                raise ValueError(f"{evidence_label} has no capture time")
        if name in _TRANSCRIBED_LEAD_NAMES and "transcribed_no_raw_capture" not in grades:
            raise ValueError(f"{label} silently upgrades a transcribed search lead")
    if anchor_names != set(_EXPECTED_ANCHORS):
        raise ValueError("Identity follow-up search-anchor set drifted")

    internal_edges = payload.get("official_internal_identity_edges")
    if not isinstance(internal_edges, list) or len(internal_edges) != 1:
        raise ValueError("Identity follow-up must contain one official internal edge")
    internal = internal_edges[0]
    if not isinstance(internal, dict):
        raise ValueError("Official internal identity edge must be an object")
    _require_exact_keys(internal, _INTERNAL_EDGE_KEYS, "official internal edge")
    _validate_official_entity(internal, "official internal edge")
    if (
        internal.get("official_name") != "Leonardo De Oliveira"
        or internal.get("official_node_id") != 118341
        or internal.get("athlete_stat_uuid")
        != "e567a99f-09dc-4435-9350-7c53c4bf191d"
        or internal.get("legacy_numeric_fightmetric_id") != "3165"
        or internal.get("career_fights_fightid") != "7490"
        or internal.get("queue_indices") != [67, 68]
        or internal.get("requested_fields") != ["reach", "stance"]
        or internal.get("decision_id") != "R2"
        or internal.get("edge_recording_authorized") is not True
    ):
        raise ValueError("Leonardo official internal identity edge drifted")
    _require_sha256s(internal.get("capture_sha256s"), "Leonardo captures")
    _validate_authorization_flags(internal, "Leonardo official internal edge")
    if re.fullmatch(r"[0-9a-f]{16}", internal["legacy_numeric_fightmetric_id"]):
        raise ValueError("Leonardo legacy numeric ID was mislabeled as modern UFCStats")

    external_bindings = payload.get("external_identity_bindings")
    if not isinstance(external_bindings, list) or len(external_bindings) != 1:
        raise ValueError("Identity follow-up must contain one external identity edge")
    external = external_bindings[0]
    if not isinstance(external, dict):
        raise ValueError("External identity edge must be an object")
    _require_exact_keys(external, _EXTERNAL_BINDING_KEYS, "external identity edge")
    _validate_official_entity(external, "external identity edge")
    if (
        external.get("official_name") != "Sean Mcinerney"
        or external.get("official_node_id") != 131076
        or external.get("source") != "wikipedia"
        or external.get("source_entity_ids")
        != [
            "page-title:Sean_McInerney_(stunt_performer)",
            "wikidata:Q138337036",
        ]
        or external.get("queue_indices") != [99, 100, 101, 102]
        or external.get("requested_fields")
        != ["height", "reach", "stance", "weight"]
        or external.get("decision_id") != "R3"
        or external.get("edge_recording_authorized") is not True
    ):
        raise ValueError("Sean external identity edge drifted")
    anchors_by_kind: dict[str, dict[str, object]] = {}
    for position, item in enumerate(external.get("identity_anchors") or (), start=1):
        label = f"Sean identity anchor {position}"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(item, _IDENTITY_ANCHOR_KEYS, label)
        kind = _text(item.get("kind"))
        if not kind or kind in anchors_by_kind or not _text(item.get("normalized")):
            raise ValueError(f"{label} is invalid or duplicated")
        anchors_by_kind[kind] = item
    if set(anchors_by_kind) != {"dob", "nickname", "full_name"}:
        raise ValueError("Sean external identity anchors drifted")
    if (
        anchors_by_kind["dob"]["normalized"] != "1986-10-21"
        or anchors_by_kind["nickname"]["normalized"] != "poopies"
    ):
        raise ValueError("Sean non-name identity evidence drifted")
    _require_sha256s(external.get("capture_sha256s"), "Sean captures")
    _validate_authorization_flags(external, "Sean external identity edge")

    unavailable_bindings = payload.get("approved_unavailable_source_bindings")
    if unavailable_bindings != _APPROVED_UNAVAILABLE_SOURCE_BINDINGS:
        raise ValueError("Approved unavailable-source binding drifted")
    unavailable_by_name = {
        _text(record.get("official_name")): record for record in unavailable_bindings
    }
    if set(unavailable_by_name) != {
        "Emilio Cuellar",
        "Felix Klinkhammer",
        "Feng Pengchao",
        "Jose Augusto",
    }:
        raise ValueError("Approved unavailable-source binding set drifted")
    for name, record in unavailable_by_name.items():
        label = f"{name} unavailable-source binding"
        _validate_official_entity(record, label)
        if record.get("edge_recording_authorized") is not True:
            raise ValueError(f"{label} is not owner-authorized")
        if any(
            result["import_allowed"] is not False
            or result["decision"] != "keep_unresolved_blank"
            for result in record["requested_field_results"]
        ):
            raise ValueError(f"{label} authorized a requested field import")
        for flag in (
            "profile_value_import_allowed",
            "non_mma_exclusion_authorized",
            "roster_deletion_authorized",
            "candidate_effect_authorized",
            "runtime_consumption_authorized",
        ):
            if record[flag] is not False:
                raise ValueError(f"{label} authorizes {flag}")

    emilio = unavailable_by_name["Emilio Cuellar"]
    active_identity_bindings = load_reviewed_roster_identity_registry()[
        "identity_bindings"
    ]
    active_emilio = next(
        (
            record
            for record in active_identity_bindings
            if record["official_source_entity_id"] == "athlete:emilio-cuellar"
        ),
        None,
    )
    if (
        active_emilio is None
        or active_emilio["stable_ufcstats_id"] != "3aeb69b4a13ff0e8"
        or set(active_emilio["mapped_source_entity_ids"])
        != {"tapology:183405", "fightdx:emilio-cuellar"}
    ):
        raise ValueError("Emilio canonical identity binding drifted")
    active_emilio_height = next(
        (
            record
            for record in load_reviewed_roster_identity_registry()[
                "field_decisions"
            ]
            if record["cell_id"] == "D005"
        ),
        None,
    )
    if (
        active_emilio_height is None
        or active_emilio_height["official_source_entity_id"]
        != "athlete:emilio-cuellar"
        or active_emilio_height["classification"] != "conflicting"
        or active_emilio_height["decision_status"]
        != "accepted_current_future"
        or active_emilio_height["normalized_value"] != 162.56
        or active_emilio_height["observation_selector"]
        != {
            "carrier": "supplement",
            "source": "fightdx",
            "source_entity_id": "fightdx:emilio-cuellar",
            "source_url": "https://fightdx.com/person/emilio-cuellar",
            "source_name": "Emilio Cuellar",
            "field": "height",
            "raw_value": "5'4\" (1.62m)",
        }
    ):
        raise ValueError("Emilio owner-selected height disposition drifted")
    if any(
        result["import_allowed"] is not False
        or result["decision"] != "keep_unresolved_blank"
        for result in emilio["requested_field_results"]
    ):
        raise ValueError("Emilio requested blank fields were authorized for import")
    if any(
        result["import_allowed"] is not False
        for result in emilio["nonrequested_values_withheld"]
    ):
        raise ValueError("Emilio nonrequested values were authorized for import")
    for flag in (
        "profile_value_import_allowed",
        "non_mma_exclusion_authorized",
        "roster_deletion_authorized",
        "candidate_effect_authorized",
        "runtime_consumption_authorized",
    ):
        if emilio[flag] is not False:
            raise ValueError(f"Emilio unavailable-source binding authorizes {flag}")

    felix = unavailable_by_name["Felix Klinkhammer"]
    if (
        felix.get("canonical_ufcstats_id") != "8276f28f1ad087de"
        or felix.get("source") != "espn"
        or felix.get("source_entity_id") != "espn:5078427"
        or felix.get("source_url")
        != "https://www.espn.com/mma/fighter/_/id/5078427/felix-klinkhammer"
        or felix.get("known_source_entity_ids")
        != ["tapology:120890", "sherdog:213423"]
        or felix.get("decision_id") != "U30-FELIX-ESPN"
    ):
        raise ValueError("Felix ESPN provenance binding drifted")
    felix_results = {
        (result["queue_index"], result["field"]): (
            result["raw_value"],
            result["normalized_value"],
        )
        for result in felix["requested_field_results"]
    }
    if felix_results != {(37, "reach"): ("", None), (38, "stance"): ("-", "")}:
        raise ValueError("Felix requested blank field results drifted")
    if felix.get("nonrequested_values_withheld") != [
        {
            "field": "height",
            "raw_value": "6'0\"",
            "normalized_value_cm": 182.88,
            "existing_state": "ufcstats_scraped_6ft0",
            "disposition": "already_present_no_new_import",
            "import_allowed": False,
        }
    ]:
        raise ValueError("Felix nonrequested existing height disposition drifted")
    felix_anchors = {
        (anchor["matched_opponent"], anchor["date"], anchor["organization"])
        for anchor in felix["identity_anchors"]
    }
    if felix_anchors != {
        ("Wissem Akhmouch", "2022-04-16", "Ares FC 5"),
        ("Karim Rabei", "2021-12-11", "Ares FC 2"),
        ("Anatoly Tarasenko", "2021-07-24", "Oktagon MMA 26"),
        ("Laid Zerhouni", "2019-09-28", "Contenders Norwich 27"),
    }:
        raise ValueError("Felix ESPN identity anchors drifted")
    if any(
        "espn:5078427" in binding.get("mapped_source_entity_ids", ())
        for binding in active_identity_bindings
    ):
        raise ValueError("Felix ESPN provenance edge became an active runtime mapping")

    feng = unavailable_by_name["Feng Pengchao"]
    if (
        feng.get("canonical_ufcstats_id") != "86597694317e5e1b"
        or feng.get("source") != "espn"
        or feng.get("source_entity_id") != "espn:5073905"
        or feng.get("source_url")
        != "https://www.espn.com/mma/fighter/_/id/5073905/feng-pengchao"
        or feng.get("known_source_entity_ids")
        != ["tapology:230675", "sherdog:336611"]
        or feng.get("decision_id") != "U30-FENG-ESPN"
    ):
        raise ValueError("Feng ESPN provenance binding drifted")
    if feng.get("source_profile_context") != {
        "name": "Feng Pengchao",
        "nickname": "Winged Tiger",
        "country": "China",
        "division": "Featherweight",
        "record": "0-2-0",
        "record_disposition": (
            "incomplete_not_imported_not_used_as_identity_anchor"
        ),
    }:
        raise ValueError("Feng ESPN profile context disposition drifted")
    feng_results = {
        (result["queue_index"], result["field"]): (
            result["raw_value"],
            result["normalized_value"],
        )
        for result in feng["requested_field_results"]
    }
    if feng_results != {(39, "stance"): ("", "")}:
        raise ValueError("Feng requested blank stance disposition drifted")
    feng_anchors = {
        (anchor["matched_opponent"], anchor["date"], anchor["organization"])
        for anchor in feng["identity_anchors"]
    }
    if feng_anchors != {
        ("Yibugele", "2021-04-30", "WLF: W.A.R.S. 52"),
        (
            "Jinensibieke Asikeerbai",
            "2020-08-21",
            "WLF: W.A.R.S. 46",
        ),
    }:
        raise ValueError("Feng ESPN identity anchors drifted")
    feng_corroboration = feng.get("nonrequested_values_withheld")
    if not isinstance(feng_corroboration, list) or len(feng_corroboration) != 1:
        raise ValueError("Feng reach corroboration record drifted")
    corroboration = feng_corroboration[0]
    if (
        corroboration.get("field") != "reach"
        or corroboration.get("raw_value") != "68\""
        or corroboration.get("normalized_value_inches") != 68.0
        or corroboration.get("normalized_value_cm") != 172.72
        or corroboration.get("existing_state")
        != "accepted_current_future_d010"
        or corroboration.get("disposition")
        != (
            "rounding_compatible_noncarrier_corroboration_"
            "keep_tapology_d010"
        )
        or corroboration.get("existing_carrier")
        != {
            "cell_id": "D010",
            "source": "tapology",
            "source_entity_id": "tapology:230675",
            "source_url": (
                "https://www.tapology.com/fightcenter/fighters/"
                "230675-feng-peng-zhao-winged-tiger"
            ),
            "raw_value": "68.1\" (173cm)",
            "normalized_value_cm": 172.974,
        }
        or corroboration.get("exact_delta_inches") != 0.1
        or corroboration.get("exact_delta_cm") != 0.254
        or corroboration.get("genuine_conflict") is not False
        or corroboration.get("import_allowed") is not False
        or corroboration.get("replacement_allowed") is not False
    ):
        raise ValueError("Feng rounding-compatible reach corroboration drifted")
    if feng.get("historical_backfill_allowed") is not False:
        raise ValueError("Feng ESPN provenance binding authorizes historical backfill")
    active_feng = next(
        (
            record
            for record in active_identity_bindings
            if record["official_source_entity_id"] == "athlete:feng-pengchao"
        ),
        None,
    )
    if (
        active_feng is None
        or active_feng["stable_ufcstats_id"] != "86597694317e5e1b"
        or set(active_feng["mapped_source_entity_ids"])
        != {"tapology:230675", "sherdog:336611"}
    ):
        raise ValueError("Feng active identity binding drifted")
    active_feng_fields = {
        record["cell_id"]: record
        for record in load_reviewed_roster_identity_registry()["field_decisions"]
        if record["official_source_entity_id"] == "athlete:feng-pengchao"
    }
    if (
        active_feng_fields.get("D009", {}).get("decision_status")
        != "conflicting_field_hold"
        or active_feng_fields.get("D009", {}).get("normalized_value") is not None
        or active_feng_fields.get("D010", {}).get("decision_status")
        != "accepted_current_future"
        or active_feng_fields.get("D010", {}).get("normalized_value") != 172.974
        or active_feng_fields.get("D010", {}).get("observation_selector")
        != {
            "carrier": "supplement",
            "source": "tapology",
            "source_entity_id": "tapology:230675",
            "source_url": (
                "https://www.tapology.com/fightcenter/fighters/"
                "230675-feng-peng-zhao-winged-tiger"
            ),
            "source_name": "Pengchao Feng",
            "field": "reach",
            "raw_value": "68.1\" (173cm)",
        }
    ):
        raise ValueError("Feng active D009/D010 field dispositions drifted")
    if any(
        "espn:5073905" in binding.get("mapped_source_entity_ids", ())
        for binding in active_identity_bindings
    ):
        raise ValueError("Feng ESPN provenance edge became an active runtime mapping")

    jose = unavailable_by_name["Jose Augusto"]
    if (
        jose.get("canonical_ufcstats_id") != "4d3b08ab127c7585"
        or jose.get("source") != "fightdx"
        or jose.get("source_entity_id") != "fightdx:jose-augusto-azevedo"
        or jose.get("source_url")
        != "https://fightdx.com/person/jose-augusto-azevedo"
        or jose.get("known_source_entity_ids")
        != [
            "espn:4684239",
            "sherdog:111573",
            "tapology:133511",
            "pfl:jose-augusto",
        ]
        or jose.get("decision_id") != "U30-JOSE-FIGHTDX"
    ):
        raise ValueError("Jose FightDX provenance binding drifted")
    if jose.get("source_profile_context") != {
        "name": "Jose Augusto Azevedo",
        "record": "7-3-1-0",
        "record_disposition": (
            "stale_search_index_context_not_imported_not_used_for_roster_status"
        ),
    }:
        raise ValueError("Jose stale FightDX profile context disposition drifted")
    if jose.get("prior_url_only_context") != {
        "url_was_previously_visible_as_opponent_link": True,
        "prior_capture_path": (
            "C:/Temp/u118r1/sources/batch-06/raw/"
            "simon-biyong-fightdx-initial--565532344f185256.html"
        ),
        "prior_capture_sha256": (
            "565532344f1852569066aaac1366cc9525ead87cfdad58b9d061dc72707333bf"
        ),
        "prior_capture_retrieved_at_utc": "2026-08-29T17:21:54.701965Z",
        "jose_profile_previously_fetched": False,
        "jose_to_fightdx_binding_previously_registered": False,
        "jose_field_evidence_previously_used": False,
        "disposition": (
            "url_known_only_as_opponent_link_new_jose_to_fightdx_edge"
        ),
    }:
        raise ValueError("Jose prior FightDX opponent-link context drifted")
    jose_results = {
        (result["queue_index"], result["field"]): (
            result["raw_value"],
            result["normalized_value"],
        )
        for result in jose["requested_field_results"]
    }
    if jose_results != {(56, "stance"): ("", "")}:
        raise ValueError("Jose requested blank stance disposition drifted")
    jose_anchors = {
        (anchor["matched_opponent"], anchor["date"], anchor["organization"])
        for anchor in jose["identity_anchors"]
    }
    if jose_anchors != {
        ("Jonathan Wilson", "2021-04-02", "Bellator 255"),
        ("Anthony Johnson", "2021-05-07", "Bellator 258"),
        ("Alex Polizzi", "2022-03-12", "Bellator 276"),
        ("Simon Biyong", "2023-05-12", "Bellator 296"),
        ("Richard Jacobi", "2023-11-03", "LFA 171"),
        ("Edivan Mineiro Santos", "2024-11-22", "LFA 197"),
    }:
        raise ValueError("Jose FightDX identity anchors drifted")
    if jose.get("nonrequested_values_withheld") != [
        {
            "cell_id": "D017",
            "field": "reach",
            "source": "espn",
            "source_entity_id": "espn:4684239",
            "raw_value": "73\"",
            "normalized_value_inches": 73.0,
            "normalized_value_cm": 185.42,
            "existing_state": "accepted_current_future_d017",
            "disposition": "existing_carrier_unchanged_no_new_import",
            "import_allowed": False,
            "replacement_allowed": False,
        }
    ]:
        raise ValueError("Jose existing reach-carrier disposition drifted")
    if (
        jose.get("historical_backfill_allowed") is not False
        or jose.get("roster_status_use_allowed") is not False
    ):
        raise ValueError("Jose FightDX edge authorizes history or roster-status use")
    active_jose = next(
        (
            record
            for record in active_identity_bindings
            if record["official_source_entity_id"] == "athlete:jose-augusto"
        ),
        None,
    )
    if (
        active_jose is None
        or active_jose["stable_ufcstats_id"] != "4d3b08ab127c7585"
        or set(active_jose["mapped_source_entity_ids"])
        != {
            "espn:4684239",
            "sherdog:111573",
            "tapology:133511",
            "pfl:jose-augusto",
        }
    ):
        raise ValueError("Jose active identity binding drifted")
    active_jose_fields = [
        record
        for record in load_reviewed_roster_identity_registry()["field_decisions"]
        if record["official_source_entity_id"] == "athlete:jose-augusto"
    ]
    if len(active_jose_fields) != 1:
        raise ValueError("Jose active field-decision set drifted")
    active_jose_reach = active_jose_fields[0]
    if (
        active_jose_reach.get("cell_id") != "D017"
        or active_jose_reach.get("audit_field") != "reach"
        or active_jose_reach.get("profile_field") != "reach"
        or active_jose_reach.get("classification")
        != "verified alias/path miss"
        or active_jose_reach.get("decision_status") != "accepted_current_future"
        or active_jose_reach.get("normalized_value") != 185.42
        or active_jose_reach.get("normalized_unit") != "cm"
        or active_jose_reach.get("scope") != "current_and_future"
        or active_jose_reach.get("historical_backfill_allowed") is not False
        or active_jose_reach.get("automatic_release_allowed") is not False
        or active_jose_reach.get("observation_selector")
        != {
            "carrier": "supplement",
            "source": "espn",
            "source_entity_id": "espn:4684239",
            "source_url": (
                "https://www.espn.com/mma/fighter/_/id/4684239/jose-augusto"
            ),
            "source_name": "Jose Augusto",
            "field": "reach",
            "raw_value": "73\"",
        }
    ):
        raise ValueError("Jose active D017 reach carrier drifted")
    if any(
        "fightdx:jose-augusto-azevedo"
        in binding.get("mapped_source_entity_ids", ())
        for binding in active_identity_bindings
    ):
        raise ValueError("Jose FightDX provenance edge became an active mapping")

    holds = payload.get("unbound_candidate_holds")
    if not isinstance(holds, list) or len(holds) != 5:
        raise ValueError("Identity follow-up must contain exactly five candidate holds")
    hold_names: set[str] = set()
    held_official_entities: set[str] = set()
    for position, record in enumerate(holds, start=1):
        label = f"identity follow-up candidate hold {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _HOLD_KEYS, label)
        official_entity = _validate_official_entity(record, label)
        if official_entity in held_official_entities:
            raise ValueError(f"{label} duplicates an official entity")
        held_official_entities.add(official_entity)
        name = _text(record.get("official_name"))
        if tuple(record.get("queue_indices") or ()) != _EXPECTED_HOLD_CELLS.get(name):
            raise ValueError(f"{label} queue cells drifted")
        hold_names.add(name)
        candidates = record.get("candidate_source_entities")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{label} has no candidate source entities")
        candidate_ids: set[str] = set()
        for candidate_position, candidate in enumerate(candidates, start=1):
            candidate_label = f"{label}.candidate[{candidate_position}]"
            if not isinstance(candidate, dict):
                raise ValueError(f"{candidate_label} must be an object")
            _require_exact_keys(candidate, _CANDIDATE_ENTITY_KEYS, candidate_label)
            candidate_id = _text(candidate.get("source_entity_id"))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"{candidate_label} is blank or duplicated")
            candidate_ids.add(candidate_id)
            if not _text(candidate.get("source_url")):
                raise ValueError(f"{candidate_label} has no URL")
        if candidate_ids != _EXPECTED_HOLD_ENTITY_IDS.get(name):
            raise ValueError(f"{label} candidate entity set drifted")
        _require_strings(record.get("requested_fields"), f"{label}.requested_fields")
        _require_sha256s(record.get("evidence_sha256s"), f"{label}.evidence_sha256s")
        if not _text(record.get("identity_evidence")) or not _text(
            record.get("missing_bridge")
        ):
            raise ValueError(f"{label} lacks its identity rationale or missing bridge")
        if not isinstance(record.get("conditional_values_withheld"), list):
            raise ValueError(f"{label} withheld values must be a list")
        for field in (
            "automatic_binding_allowed",
            "profile_value_import_allowed",
            "permanent_exclusion_authorized",
            "roster_deletion_authorized",
        ):
            if record.get(field) is not False:
                raise ValueError(f"{label} unexpectedly authorizes {field}")
        if record.get("decision_id") not in {"R1", "R4"}:
            raise ValueError(f"{label} has the wrong owner decision")
    if hold_names != set(_EXPECTED_HOLD_CELLS):
        raise ValueError("Identity follow-up candidate-hold set drifted")

    rejections = payload.get("preserved_source_identity_rejections")
    if not isinstance(rejections, list) or len(rejections) != 4:
        raise ValueError("Identity follow-up must preserve exactly four rejections")
    rejection_pairs: set[tuple[str, str]] = set()
    for position, record in enumerate(rejections, start=1):
        label = f"preserved source identity rejection {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _REJECTION_KEYS, label)
        official_entity = _validate_official_entity(record, label)
        pair = (official_entity, _text(record.get("source_entity_id")))
        if pair in rejection_pairs:
            raise ValueError(f"{label} duplicates a rejection pair")
        rejection_pairs.add(pair)
        if record.get("decision_id") != "R6" or not _text(record.get("source_url")):
            raise ValueError(f"{label} has invalid decision provenance")
        if record.get("rejection_basis") == "exact_dob_mismatch":
            if (
                _DATE_RE.fullmatch(_text(record.get("official_dob"))) is None
                or _DATE_RE.fullmatch(_text(record.get("candidate_dob"))) is None
                or record.get("official_dob") == record.get("candidate_dob")
            ):
                raise ValueError(f"{label} has no exact DOB mismatch")
    if rejection_pairs != _EXPECTED_REJECTION_PAIRS:
        raise ValueError("Identity follow-up preserved-rejection set drifted")

    search_false_positives = payload.get(
        "permanent_search_false_positive_rejections"
    )
    if not isinstance(search_false_positives, list) or len(search_false_positives) != 1:
        raise ValueError(
            "Identity follow-up must preserve exactly one search false positive"
        )
    false_positive = search_false_positives[0]
    if not isinstance(false_positive, dict):
        raise ValueError("Permanent search false positive must be an object")
    _require_exact_keys(
        false_positive,
        _SEARCH_FALSE_POSITIVE_REJECTION_KEYS,
        "permanent search false positive",
    )
    _validate_official_entity(false_positive, "permanent search false positive")
    if (
        false_positive.get("official_name") != "Jose Augusto"
        or false_positive.get("official_full_name")
        != "Jose Augusto Azevedo Barros"
        or false_positive.get("canonical_ufcstats_id")
        != "4d3b08ab127c7585"
        or false_positive.get("canonical_ufcstats_url")
        != "http://ufcstats.com/fighter-details/4d3b08ab127c7585"
        or false_positive.get("canonical_sherdog_source_entity_id")
        != "sherdog:111573"
        or false_positive.get("canonical_dob") != "1992-03-28"
        or false_positive.get("canonical_nickname") != "Gugu"
    ):
        raise ValueError("Permanent Jose search false-positive target drifted")
    if (
        false_positive.get("route_id") != "U21"
        or false_positive.get("search_query")
        != 'site:wikipedia.org "Jose Augusto" stance MMA'
        or false_positive.get("source") != "wikipedia"
        or false_positive.get("result_page_title") != "Joao Zaiden"
        or false_positive.get("result_page_url")
        != "https://pt.wikipedia.org/wiki/Jo%C3%A3o_Zaiden"
        or false_positive.get("result_page_source_entity_id")
        != "wikipedia:page-title:Joao_Zaiden"
        or false_positive.get("mentioned_person_name") != "Jose Augusto Lima"
        or false_positive.get("mentioned_person_source_entity_id") != ""
    ):
        raise ValueError("Permanent Jose search false-positive result drifted")

    context = false_positive.get("raw_context")
    if not isinstance(context, dict):
        raise ValueError("Permanent Jose search false-positive context must be an object")
    _require_exact_keys(
        context,
        _SEARCH_FALSE_POSITIVE_CONTEXT_KEYS,
        "permanent Jose search false-positive context",
    )
    if context != {
        "narrative_date": "2014-09-13",
        "table_date": "",
        "event": "Colina Fight",
        "opponent": "Joao Zaiden",
        "result_from_page_owner_perspective": "win_by_triangle_submission",
        "round": 1,
        "location": "Colina",
    }:
        raise ValueError("Permanent Jose search false-positive context drifted")

    comparison = false_positive.get("identity_comparison")
    if not isinstance(comparison, dict):
        raise ValueError(
            "Permanent Jose search false-positive identity comparison must be an object"
        )
    _require_exact_keys(
        comparison,
        _SEARCH_FALSE_POSITIVE_IDENTITY_KEYS,
        "permanent Jose search false-positive identity comparison",
    )
    if comparison != {
        "full_name_mismatch": (
            "Jose Augusto Lima versus Jose Augusto Azevedo Barros"
        ),
        "stable_id_bridge_found": False,
        "matching_bout_bridge_found": False,
        "joao_zaiden_or_colina_fight_present_in_canonical_record": False,
        "canonical_2014_bouts": [
            "2014-02-15|Bokum Fight Championship 2|Jhon Lourenco",
            "2014-04-26|Bokum Fight Championship 3|Jackson Goncalves",
        ],
    }:
        raise ValueError("Permanent Jose search false-positive identity proof drifted")

    field_result = false_positive.get("requested_field_result")
    if not isinstance(field_result, dict):
        raise ValueError(
            "Permanent Jose search false-positive field result must be an object"
        )
    _require_exact_keys(
        field_result,
        _SEARCH_FALSE_POSITIVE_FIELD_KEYS,
        "permanent Jose search false-positive field result",
    )
    if field_result != {
        "queue_index": 56,
        "field": "stance",
        "raw_value": "",
        "normalized_value": "",
        "decision": "keep_unresolved_blank",
    }:
        raise ValueError("Permanent Jose q56 blank disposition drifted")

    evidence = false_positive.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Permanent Jose search false-positive evidence must be an object")
    _require_exact_keys(
        evidence,
        _SEARCH_FALSE_POSITIVE_EVIDENCE_KEYS,
        "permanent Jose search false-positive evidence",
    )
    if evidence != {
        "retrieval_date_utc": "2026-08-30",
        "capture_kind": "human_transcribed_web_search_tool_output",
        "raw_http_capture_available": False,
        "http_status_available": False,
        "exact_request_timestamp_available": False,
        "evidence_manifest_sha256": (
            "3e4bb1f046b2d2375822cecf9c5640c7bd589a5076fab0f9cfaafa12e6b7435d"
        ),
        "search_result_transcript_sha256": (
            "2c87285246d750f0522c8de243041a85e613213afe7169f013aebcb548f81745"
        ),
        "route_ledger_sha256": (
            "03ffdc8d4ad20842b0032dd2b0e385ecaccafae1ec7063d151b9f2b8a15fe071"
        ),
        "failure_ledger_sha256": (
            "94616fc56fca25f36eced56382b8180a780a4232946be8787c4e087e88ee6b37"
        ),
        "owner_decision_sha256": (
            "65d5e47f9bde82bfb64907c9d5d848101c7136b1493a120a51c85ccae1d0096e"
        ),
        "independent_u21_audit_sha256": (
            "add509635a8bc9bde176261835f43a8371a130bb302124161c4d46970adffeae"
        ),
        "prior_route_identity_audit_sha256": (
            "35ebb51bddae465fdb9ab60ae9ac51d24bd5b45bb08ad804f748c339ddeb11ad"
        ),
        "remaining_route_identity_audit_sha256": (
            "d76969aa73fe880a7df6982c18e86207b6077f68c3ba3c1e313f03ac4b0a21a0"
        ),
        "registry_candidate_identity_audit_sha256": (
            "9b6680568ee0d9387ae24ede76c653f11abce3eaf46e7ff631d2d1979f95c245"
        ),
        "finding_report_sha256": (
            "bba4c6697026689f3b9e4bad63b63e4f690dd1133344eaefb66f5bc340ed1a97"
        ),
        "canonical_sherdog_capture_sha256": (
            "cfe0b04ba02e0797911dfb40837a681adcf11b40b295da784d36433bf3478273"
        ),
    }:
        raise ValueError("Permanent Jose search false-positive evidence drifted")

    if (
        false_positive.get("decision_id") != "U30-JOSE-WIKIPEDIA-NAMESAKE"
        or false_positive.get("owner_approved_at_date") != "2026-08-30"
        or false_positive.get("permanent_search_false_positive_rejection")
        is not True
        or false_positive.get("different_person") is not True
    ):
        raise ValueError("Permanent Jose search false-positive decision drifted")
    for field in (
        "bind_result_page_to_canonical_fighter",
        "bind_mentioned_person_to_canonical_fighter",
        "profile_value_import_allowed",
        "candidate_effect_authorized",
        "runtime_consumption_authorized",
        "historical_backfill_allowed",
        "roster_status_use_allowed",
        "roster_deletion_authorized",
        "non_mma_exclusion_authorized",
        "publication_authorized",
    ):
        if false_positive.get(field) is not False:
            raise ValueError(
                f"Permanent Jose search false positive unexpectedly authorizes {field}"
            )

    false_positive_entity = false_positive["result_page_source_entity_id"]
    if any(
        false_positive_entity in binding.get("mapped_source_entity_ids", ())
        for binding in active_identity_bindings
    ):
        raise ValueError("Jose namesake page became an active identity mapping")
    active_field_decisions = load_reviewed_roster_identity_registry()[
        "field_decisions"
    ]
    for record in active_field_decisions:
        selector = record.get("observation_selector") or {}
        if (
            selector.get("source_entity_id") == false_positive_entity
            or selector.get("source_url") == false_positive["result_page_url"]
            or selector.get("source_name") == false_positive["mentioned_person_name"]
        ):
            raise ValueError("Jose namesake evidence became an active field carrier")

    policy = payload.get("non_mma_policy")
    if not isinstance(policy, dict):
        raise ValueError("Identity follow-up non-MMA policy must be an object")
    _require_exact_keys(policy, _POLICY_KEYS, "identity follow-up non-MMA policy")
    if (
        policy.get("permanent_exclusion_selection_key")
        != "official_source_entity_id"
        or policy.get("broad_name_blacklist_allowed") is not False
        or policy.get("unbound_candidate_permanent_exclusion_allowed") is not False
        or policy.get("missing_information_authorizes_deletion") is not False
    ):
        raise ValueError("Identity follow-up non-MMA policy drifted")
    policy_exclusions = policy.get("exact_permanent_exclusions")
    if not isinstance(policy_exclusions, list) or len(policy_exclusions) != 3:
        raise ValueError("Identity follow-up must pin exactly three permanent exclusions")
    policy_entities: set[str] = set()
    for position, record in enumerate(policy_exclusions, start=1):
        label = f"identity follow-up exact non-MMA exclusion {position}"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(record, _POLICY_EXCLUSION_KEYS, label)
        policy_entities.add(_validate_official_entity(record, label))
    if policy_entities != _EXPECTED_PERMANENT_EXCLUSION_ENTITIES:
        raise ValueError("Identity follow-up exact non-MMA exclusion set drifted")
    unbound_power_slap_entities = set(
        _require_strings(
            policy.get("unbound_power_slap_candidate_entity_ids"),
            "unbound Power Slap candidate IDs",
        )
    )
    if unbound_power_slap_entities != _EXPECTED_UNBOUND_POWER_SLAP_ENTITIES:
        raise ValueError("Unbound Power Slap candidate set drifted")
    if policy_entities.intersection(held_official_entities):
        raise ValueError("An unbound candidate was promoted to a permanent exclusion")

    active_exclusions = load_reviewed_roster_recovery_registry()[
        "non_mma_identity_exclusions"
    ]
    active_exclusion_entities = {
        record["official_source_entity_id"] for record in active_exclusions
    }
    if active_exclusion_entities != policy_entities:
        raise ValueError("Active non-MMA exclusions conflict with follow-up policy")

    active_rejections = load_reviewed_roster_identity_registry()[
        "source_identity_rejections"
    ]
    active_rejection_by_pair = {
        (record["official_source_entity_id"], record["source_entity_id"]): record
        for record in active_rejections
    }
    for record in rejections:
        pair = (record["official_source_entity_id"], record["source_entity_id"])
        active = active_rejection_by_pair.get(pair)
        if (
            active is None
            or active["source_url"] != record["source_url"]
            or active["official_athlete_url"] != record["official_athlete_url"]
        ):
            raise ValueError(f"Active wrong-person rejection drifted for {pair}")

    return payload


@lru_cache(maxsize=None)
def load_reviewed_roster_identity_followup_registry(
    path: Path = REVIEWED_ROSTER_IDENTITY_FOLLOWUP_REGISTRY_PATH,
) -> dict[str, object]:
    registry_path = Path(path)
    raw = registry_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != _EXPECTED_REGISTRY_SHA256:
        raise ValueError(
            "Reviewed identity follow-up registry SHA-256 drifted: "
            f"expected {_EXPECTED_REGISTRY_SHA256}, observed {actual_sha256}"
        )
    return _validate_registry(json.loads(raw.decode("utf-8")))


def _record_for_official_entity(
    section: str, row: Mapping[str, object]
) -> dict[str, object] | None:
    entity = source_entity_id("ufc", row.get("official_athlete_url"))
    if not entity:
        return None
    matches = [
        record
        for record in load_reviewed_roster_identity_followup_registry()[section]
        if record["official_source_entity_id"] == entity
    ]
    if len(matches) > 1:
        raise ValueError(f"Identity follow-up is ambiguous for UFC entity {entity}")
    return dict(matches[0]) if matches else None


def reviewed_followup_search_anchor(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an exact official-entity search anchor; names never select it."""

    return _record_for_official_entity("official_search_anchors", row)


def reviewed_followup_official_internal_identity_edge(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an approved official-internal provenance edge, never a profile value."""

    return _record_for_official_entity("official_internal_identity_edges", row)


def reviewed_followup_external_identity_binding(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an approved external identity edge without authorizing sport or fields."""

    return _record_for_official_entity("external_identity_bindings", row)


def reviewed_followup_unbound_candidate_hold(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an exact unresolved candidate hold; names never select it."""

    return _record_for_official_entity("unbound_candidate_holds", row)


def reviewed_followup_unavailable_source_binding(
    row: Mapping[str, object],
) -> dict[str, object] | None:
    """Return an owner-approved provenance edge with all profile imports disabled."""

    return _record_for_official_entity(
        "approved_unavailable_source_bindings", row
    )


__all__ = [
    "REVIEWED_ROSTER_IDENTITY_FOLLOWUP_REGISTRY_PATH",
    "load_reviewed_roster_identity_followup_registry",
    "reviewed_followup_external_identity_binding",
    "reviewed_followup_official_internal_identity_edge",
    "reviewed_followup_search_anchor",
    "reviewed_followup_unbound_candidate_hold",
    "reviewed_followup_unavailable_source_binding",
]
