from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import run_generator_claim_campaign as campaign
from tools.generator_claim_validator import canonical_bytes, file_sha256
from tools.run_generator_cohort import (
    STAGE_SUFFIXES,
    load_declaration,
    verify_stage_membership,
)


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = (
    ROOT
    / "tests/fixtures/multires/B2-GENERATED-COHORT-FRESH-DECLARATION.json"
)
FAILURE_71427 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71427-FAILURE.json"
)
DECLARATION_71427 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71427-DECLARATION.json"
)
FAILURE_71428 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71428-FAILURE.json"
)
DECLARATION_71428 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71428-DECLARATION.json"
)
FAILURE_71429 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71429-FAILURE.json"
)
DECLARATION_71429 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71429-DECLARATION.json"
)
FAILURE_71430 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71430-FAILURE.json"
)
DECLARATION_71430 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71430-DECLARATION.json"
)
FAILURE_71431 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71431-FAILURE.json"
)
DECLARATION_71431 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71431-DECLARATION.json"
)
FAILURE_71432 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71432-FAILURE.json"
)
DECLARATION_71432 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71432-DECLARATION.json"
)
FAILURE_71433 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71433-FAILURE.json"
)
DECLARATION_71433 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71433-DECLARATION.json"
)
FAILURE_71434 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71434-FAILURE.json"
)
DECLARATION_71434 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71434-DECLARATION.json"
)
FAILURE_71435 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71435-FAILURE.json"
)
DECLARATION_71435 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71435-DECLARATION.json"
)
FAILURE_71436 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71436-FAILURE.json"
)
DECLARATION_71436 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71436-DECLARATION.json"
)
FAILURE_71437 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71437-FAILURE.json"
)
DECLARATION_71437 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71437-DECLARATION.json"
)
FAILURE_71438 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71438-FAILURE.json"
)
DECLARATION_71438 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json"
)
FAILURE_71439 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71439-FAILURE.json"
)
DECLARATION_71439 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71439-DECLARATION.json"
)
SHADOW_71430 = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71430-SHADOW-8CE1E75.json"
)
NONZERO = "ab" * 32


def test_campaign_v2_schema_and_operator_contract_are_exact() -> None:
    schema_path = ROOT / "schemas/q2-generator-claim-campaign-v2.schema.json"
    schema = json.loads(schema_path.read_text())
    contract = (
        ROOT / "docs/multires/B2-C-GENERATOR-CLAIM-CONTRACT.md"
    ).read_text()

    assert schema["$id"] == "urn:q2-ml:q2-generator-claim-campaign-v2"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema"]["const"] == campaign.CAMPAIGN_SCHEMA
    assert schema["properties"]["expected_count"]["const"] == 28
    assert schema["properties"]["map_count"]["const"] == 28
    assert schema["$defs"]["materialized_membership"]["allOf"][1][
        "properties"
    ]["expected_file_count"]["const"] == 196
    assert schema["$defs"]["claims_membership"]["allOf"][1][
        "properties"
    ]["expected_file_count"]["const"] == 224
    assert schema["$defs"]["analysis_membership"]["allOf"][1][
        "properties"
    ]["expected_file_count"]["const"] == 224
    assert not (
        ROOT / "schemas/q2-generator-claim-campaign-v1.schema.json"
    ).exists()

    required_contract = (
        "q2-generator-claim-campaign-v2",
        "run_generator_claim_campaign.py prepare",
        "--declaration docs/multires/B2-GENERATED-COHORT-DECLARATION.json",
        "--materialized-dir",
        "--claims-dir",
        "run_generator_claim_campaign.py validate",
        "--analysis-dir",
        "different authority",
        "atomically renames",
        "O_CREAT | O_EXCL",
        "no salvage",
        "b2g26_final_71426",
        "b2g26_final_71427",
        "non-admissible",
        "permanently non-admissible",
        "168 of 196",
        "b2g26_final_71428",
        "0/28",
        "b2g26_final_71429",
        "b2g26_final_71430",
        "b2g26_final_71431",
        "b2g26_final_71432",
        "b2g26_final_71433",
        "b2g26_final_71434",
        "b2g26_final_71435",
        "b2g26_final_71436",
        "b2g26_final_71437",
        "b2g26_final_71438",
        "b2g26_final_71439",
        "b2g26_final_71440",
        "b2g26_final_71441",
        "b2g26_final_71442",
        "b2g26_final_71443",
        "B2-GENERATED-COHORT-71437-FAILURE.json",
        "B2-GENERATED-COHORT-71438-FAILURE.json",
        "B2-GENERATED-COHORT-71439-FAILURE.json",
        "B2-GENERATED-COHORT-71440-FAILURE.json",
        "B2-GENERATED-COHORT-71441-FAILURE.json",
        "B2-GENERATED-COHORT-71443-FAILURE.json",
        "B2-GENERATED-COHORT-71444-FAILURE.json",
        "B2-GENERATED-COHORT-71446-FAILURE.json",
        "lane-wall static blockers",
        "fresh replacement cohort",
        "Fresh replacement cohort `b2g26_final_71442` is explicitly authorized",
        "Fresh replacement cohort `b2g26_final_71443` was explicitly authorized",
        "Cohort 71443 is permanently retired",
        "Cohort 71444 is permanently retired",
        "b2q26_3b17223_71625100",
        "351baccaabf405e0ef240c1def18e4ede796ff417e73230524e9f0f9b0c0491b",
        "58295d227ddd3694a0ddae5af46e2bbc98cc60dbe6b6751b4e42df01c06b1cd6",
        "3b17223ab32e20152aead1eb32a79e239d6f4d8a",
        "fa2b106d19dbb115e6acd4c344b3820b3013464a",
        "b2g26_final_71445",
        "B2-GENERATED-COHORT-71445-DECLARATION.json",
        "ffa5b9ccfee0340f1bad533a23fedd103a08d14d125149d1516a2326fb8a091b",
        "71445000..71445003",
        "71445600..71445603",
        "B2-GENERATED-COHORT-71445-FAILURE.json",
        "Cohort 71445 is permanently retired",
        "d134ddd35bb6e93f1fffa71d2b6176d402ba70c2d4242b2f55b6be40efd651af",
        "cf87d90e7f7d40a9baae7e5bf54c27491f26d4a28531830f4a5cc79e4add1db7",
        "2167bfdef17cf247e329e5761dc7e44d3c22d34f5a3181faea5b8c2f737ee8a3",
        "could not place a unique lava-rim reward",
        "b2q26_a05ddb7_71626100",
        "69e2b1979feae22c706839dc24f8923b60e34d5b623c8f03b0e5ebb51181a549",
        "a05ddb7037774c1b246a6b13972b228570acb8ef",
        "01c27fc60da4ae6f2aedd6138c50dabfcd866525",
        "fb71a121d05dc02ad4d634f537abb331ed7d4ea29da0e5c3199afe8c0b442001",
        "b2g26_final_71446",
        "B2-GENERATED-COHORT-71446-DECLARATION.json",
        "58d52bd958249a70bf8115ab1c442fb6888a6d69b290a636303986f69acb658f",
        "71446000..71446003",
        "71446600..71446603",
        "Cohort 71446 is permanently retired",
        "4b26c670ed54585787505cf7dfbb35bdc1830fdfbd42585a16d0484622ea306f",
        "oracle batch timeout must be finite and in (0, 60]",
        "not a 3,600-second runtime timeout",
        "b2q26_275d4fa_71623700",
        "09bd298d87739515d468f432219eefcad01e8586a87a71339f5121900a6f57c5",
        "eb99e08e5934d281556b0b6584ab23fe236adb8fce81f1cc7045229b368b9a25",
        "275d4fa646ccf2c64ba8628cd4aa8b21644fa90d",
        "7bd808b2194a44b80dc64fb88c700209d4657e9a",
        "b2g26_final_71444",
        "B2-GENERATED-COHORT-71444-DECLARATION.json",
        "da27e96b3fe8c3719a7ff1593e37b4ac768f53a36f38c877566af495a6b551bf",
        "71444000..71444003",
        "71444600..71444603",
        "Exactly one immutable/no-retry final producer attempt is authorized",
        "The first source-generation invocation consumes the sole authorization",
        "final producer lane itself is strictly sequential",
        "Tests never overlap compilation",
        "out-of-order-test-runtime-preflight",
        "unterminated string literal",
        "Full-cold producer closure",
        "tools/atlas_cold_worker.py",
        ".analysis.manifest.json",
        "80 units",
        "62-unit",
        "reserved interior",
        "spawn-bearing source component",
        "monotonic ceiling normalization",
        "one-chunk hurt-boundary",
        "grounded origins at Z 24.03125",
        "first source-freeze attempt",
        "version 2",
        "room-edge fallback",
    )
    assert all(fragment in contract for fragment in required_contract)
    assert "--glob 'b2claim_*.map'" not in contract
    assert "--expected-count 20" not in contract


def test_71427_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71427.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71427"
    assert failure["declaration"]["sha256"] == (
        "78acab13ecb7ad4a365b7f8cb64318650"
        "d07fd88edd4471c4b7f8acb1a36cd1a"
    )
    assert failure["status"] == "permanently-failed-first-materialization"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71427-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71427.read_bytes()).hexdigest(),
    }

    first_map = failure["failure"]["first_map"]
    assert first_map == {
        "map": "b2g26_open_71427000",
        "ordinal": 0,
        "seed": 71427000,
        "style": "open",
    }
    transcript = failure["failure"]["operator_transcript"]
    assert transcript["durably_retained"] is False
    assert transcript["exit_code"] == 1
    assert transcript["proved_unique_geometries"] == 2
    assert transcript["required_unique_geometries"] == 6
    assert transcript["rejections"] == {
        "anchor_not_exactly_attachable": 28,
        "duplicate_proven_geometry": 2,
        "measured_landing_outside_desired_l1": 400,
        "source_not_spawn_reachable": 80,
    }
    assert transcript["stdout_text"] == ""
    assert transcript["stderr_text"] == (
        "hook materialization failed: compiled hook preflight proved 2/6 "
        "unique geometries; rejections={'anchor_not_exactly_attachable': 28, "
        "'duplicate_proven_geometry': 2, "
        "'measured_landing_outside_desired_l1': 400, "
        "'source_not_spawn_reachable': 80}"
    )
    durable = failure["failure"]["durable_process_capture"]
    assert durable["stderr_retained"] is False
    assert durable["exit_code_retained"] is False
    assert failure["evidence"]["failed_materialization_stdout"] == {
        "path_from_artifact_root": (
            "generated-final-71427-d83391aa/"
            "failed-materialization-open-71427000/logs/"
            "b2g26_open_71427000.materialize.json"
        ),
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": 0,
    }

    membership = failure["evidence"]["materialized_membership"]
    assert membership["expected_file_count"] == 196
    assert membership["actual_file_count"] == 168
    assert membership["missing_attestation_count"] == 28
    assert membership["unexpected_file_count"] == 0
    assert membership["passed"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["materialized_stage_published"] is False
    assert admission["claims_stage_published"] is False
    assert admission["analysis_stage_published"] is False
    assert admission["salvage_allowed"] is False
    assert admission["retry_under_same_declaration_allowed"] is False
    assert admission["regeneration_under_same_declaration_allowed"] is False
    assert admission["replacement_declaration_status"] == (
        "pending-implementation-fix"
    )


def test_71428_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71428.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71428"
    assert failure["status"] == "permanently-failed-analysis"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71428-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71428.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["classified_map_count"] == 28
    assert sum(failure["failure"]["classification"].values()) == 28
    assert failure["evidence"]["failed_analysis_campaign"] == {
        "diagnostics_directory": "analysis-diagnostics-v3",
        "expected_map_count": 28,
        "pass_count": 0,
        "passed": False,
        "path_from_artifact_root": "reports/generated-atlas-campaign-v3.json",
        "published": False,
        "sha256": "f74680dca9386920b51b9b38c6b8b6f147369c249a06fefa719eccf658ee9bc1",
    }
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed",
    ):
        assert admission[key] is False


def test_71429_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71429.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71429"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71429-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71429.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "first-source-freeze-validation"
    assert failure["failure"]["first_failure"] == {
        "map": "b2g26_towers_71429101",
        "message": (
            "route 0 endpoint room 2 is unreachable from start room 0 "
            "through source room edges"
        ),
        "ordinal": 5,
        "seed": 71429101,
        "style": "towers",
    }
    assert failure["failure"]["affected_map_count"] == 7
    assert failure["evidence"]["primary_population"][
        "content_manifest_sha256"
    ] == failure["evidence"]["cold_population"]["content_manifest_sha256"]
    assert failure["evidence"]["primary_population"]["actual_file_count"] == 140
    assert failure["evidence"]["cold_population"]["actual_file_count"] == 140
    assert failure["evidence"]["source_freeze_report"]["published"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is False
    assert admission["compiled_stage_published"] is False
    assert admission["salvage_allowed"] is False
    assert admission["retry_under_same_declaration_allowed"] is False
    assert admission["regeneration_under_same_declaration_allowed"] is False


def test_71430_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71430.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71430"
    assert failure["status"] == "permanently-failed-analysis"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71430-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71430.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "exact-compiled-atlas-analysis"
    assert failure["failure"]["classified_map_count"] == 28
    assert failure["failure"]["classification"] == {
        "hurt_zero_retained_sparse_evidence": 28,
    }
    failed = failure["evidence"]["failed_analysis_campaign"]
    assert failed["pass_count"] == 0
    assert failed["expected_map_count"] == 28
    assert failed["passed"] is False
    assert failed["published"] is False
    assert failed["sha256"] == (
        "11b8c6725856938bd8d708590e09532f"
        "594e4b2f9fd03a380bac4525e0abaeb2"
    )
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is True
    assert admission["compiled_stage_published"] is True
    assert admission["materialized_stage_published"] is True
    assert admission["claims_stage_published"] is True
    assert admission["analysis_stage_published"] is False
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed",
    ):
        assert admission[key] is False


def test_71430_shadow_record_is_canonical_failed_closed_and_replacement_ready() -> None:
    payload = SHADOW_71430.read_bytes()
    shadow = json.loads(payload)
    assert payload == canonical_bytes(shadow)

    assert shadow["schema"] == "q2-b2-generated-cohort-shadow-v1"
    assert shadow["cohort_id"] == "b2g26_final_71430"
    assert shadow["status"] == "diagnostic-failed-closed-replacement-ready"
    assert shadow["implementation"]["repository_commit"] == (
        "8ce1e75d00f7e9a9605506a94e717fb6f5f31f84"
    )
    final = shadow["evidence"]["final_shadow"]
    assert final["pass_count"] == final["cold_parity_pass_count"] == 27
    assert final["expected_map_count"] == 28
    assert final["published"] is False
    assert final["sha256"] == (
        "4600fa43d77a8b3f37eadef98f55dca2"
        "9863a3a5a24132b30366a326cdfa40bb"
    )
    assert shadow["remaining_failure"]["map"] == (
        "b2g26_arena_open_71430402"
    )
    admission = shadow["admission"]
    assert admission["fresh_declaration"] == "b2g26_final_71431"
    assert admission["analysis_stage_published"] is False
    assert admission["retired_artifact_reuse_allowed"] is False
    assert admission["retired_cohort_admissible"] is False
    assert admission["shadow_passing_subset_allowed"] is False


def test_71431_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71431.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71431"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71431-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71431.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "first-source-freeze-validation"
    assert failure["failure"]["affected_map_count"] == 1
    assert failure["failure"]["first_failure"] == {
        "map": "b2g26_arena_vertical_71431503",
        "message": (
            "source/static validation failed: objective tower brush_195 and "
            "room ceiling brush_17 leave an unsafe 80-unit horizontal sandwich"
        ),
        "ordinal": 23,
        "seed": 71431503,
        "style": "arena_vertical",
    }
    assert failure["evidence"]["primary_population"] == {
        "actual_file_count": 140,
        "content_manifest_sha256": (
            "cd485a92755ef5a6ff9a5549d2013c94"
            "2a8f98598d558b3e32e3c4e71a5df17f"
        ),
        "expected_file_count": 140,
        "path_from_artifact_root": "source",
        "total_bytes": 10858611,
    }
    assert (
        failure["evidence"]["primary_population"]["content_manifest_sha256"]
        == failure["evidence"]["cold_population"]["content_manifest_sha256"]
    )
    assert failure["evidence"]["validation_summary"] == {
        "all_primary_cold_file_bytes_match": True,
        "expected_map_count": 28,
        "route_contract_pass_count_in_complete_archived_scan": 28,
        "static_pass_count_in_complete_archived_scan": 27,
        "unique_layout_count": 28,
    }
    assert failure["evidence"]["source_freeze_report"]["published"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is False
    assert admission["compiled_stage_published"] is False
    assert admission["materialized_stage_published"] is False
    assert admission["claims_stage_published"] is False
    assert admission["analysis_stage_published"] is False
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed",
    ):
        assert admission[key] is False


def test_71432_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71432.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71432"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71432-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71432.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "first-source-freeze-validation"
    assert failure["failure"]["affected_map_count"] == 1
    assert failure["failure"]["first_failure"] == {
        "map": "b2g26_towers_71432101",
        "message": (
            "source/static validation failed: one of four promised "
            "corner-pocket interiors was erased by the later objective tower"
        ),
        "ordinal": 5,
        "seed": 71432101,
        "style": "towers",
    }
    assert failure["failure"]["geometry"] == {
        "corner_bounds": [728, 1240, 824, 1336],
        "corner_id": "corner_0",
        "corner_safe_probe_count_after_objective": 0,
        "objective_bounds": [724, 1261, 0, 852, 1389, 256],
    }
    for population in ("primary_population", "cold_population"):
        evidence = failure["evidence"][population]
        assert evidence["actual_file_count"] == 140
        assert evidence["expected_file_count"] == 140
        assert evidence["total_bytes"] == 10961177
        assert evidence["stage_membership_sha256"] == (
            "de6beb99445e16954adf635105aec6698"
            "aa0a4931d0051e18f78b84cf7c2556d"
        )
    assert failure["evidence"]["validation_summary"] == {
        "all_primary_cold_file_bytes_match": True,
        "expected_map_count": 28,
        "route_contract_pass_count_in_complete_archived_scan": 28,
        "static_pass_count_in_complete_archived_scan": 27,
        "unique_layout_count": 28,
    }
    assert failure["evidence"]["source_freeze_report"]["published"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is False
    assert admission["compiled_stage_published"] is False
    assert admission["materialized_stage_published"] is False
    assert admission["claims_stage_published"] is False
    assert admission["analysis_stage_published"] is False
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed",
    ):
        assert admission[key] is False


def test_71433_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71433.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71433"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71433-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71433.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "primary-generation-route-contract"
    assert failure["failure"]["first_failure"] == {
        "map": "b2g26_arena_vertical_71433501",
        "message": "no spawn component has two reachable survival items",
        "ordinal": 21,
        "seed": 71433501,
        "style": "arena_vertical",
    }
    assert failure["failure"]["source_component_diagnostic"] == {
        "component_count": 6,
        "spawn_component": {
            "item_count": 7,
            "offense_item_count": 6,
            "spawn_count": 8,
            "survival_or_value_item_count": 1,
        },
    }
    primary = failure["evidence"]["primary_population"]
    assert primary == {
        "actual_file_count": 108,
        "complete_member_count": 21,
        "expected_file_count": 140,
        "missing_file_count": 32,
        "partial_member_file_count": 3,
        "path_from_artifact_root": "source",
        "stage_membership_sha256": (
            "a12354fddddede20e8dca957ee99871a"
            "1a739e78a0bea39e4ade6c7b916231fc"
        ),
        "total_bytes": 8284181,
    }
    cold = failure["evidence"]["cold_population"]
    assert cold["actual_file_count"] == 0
    assert cold["missing_file_count"] == 140
    assert cold["total_bytes"] == 0
    assert failure["evidence"]["source_freeze_report"]["published"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["source_stage_published"] is False
    assert admission["compiled_stage_published"] is False
    assert admission["materialized_stage_published"] is False
    assert admission["claims_stage_published"] is False
    assert admission["analysis_stage_published"] is False
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed",
    ):
        assert admission[key] is False


def test_71434_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71434.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71434"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71434-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71434.read_bytes()).hexdigest(),
    }
    assert failure["failure"]["phase"] == "primary-generation-nontermination"
    assert failure["failure"]["first_failure"] == {
        "map": "b2g26_arena_vertical_71434500",
        "message": (
            "primary generation did not terminate after approximately 30 "
            "CPU-minutes and was interrupted inside "
            "unsafe_horizontal_sandwiches"
        ),
        "ordinal": 20,
        "seed": 71434500,
        "style": "arena_vertical",
    }
    assert failure["failure"]["publication_contract"] == {
        "failed_member_file_count": 0,
        "last_complete_ordinal": 19,
        "route_atomic_publication_preserved": True,
    }
    primary = failure["evidence"]["primary_population"]
    assert primary == {
        "actual_file_count": 100,
        "complete_member_count": 20,
        "expected_file_count": 140,
        "missing_file_count": 40,
        "partial_member_file_count": 0,
        "path_from_artifact_root": "source",
        "stage_membership_sha256": (
            "15c6a580ec445e8ca8e3079f17e9f52d"
            "354c51642d4f7731c8248e699bf2295f"
        ),
        "total_bytes": 7580131,
    }
    cold = failure["evidence"]["cold_population"]
    assert cold["actual_file_count"] == 0
    assert cold["complete_member_count"] == 0
    assert cold["missing_file_count"] == 140
    assert cold["partial_member_file_count"] == 0
    assert cold["stage_membership_sha256"] == (
        "a591cca19562899e9c3a8ce9157c9617"
        "81351f84f537780dc297728ae992c0ea"
    )
    assert cold["total_bytes"] == 0
    assert failure["evidence"]["failed_member"]["files"] == {}
    assert failure["evidence"]["source_freeze_report"]["published"] is False
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    for key in (
        "older_population_reuse_allowed", "passing_subset_allowed",
        "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retry_under_same_declaration_allowed",
        "salvage_allowed", "source_stage_published",
        "compiled_stage_published", "materialized_stage_published",
        "claims_stage_published", "analysis_stage_published",
    ):
        assert admission[key] is False


def test_71435_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71435.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71435"
    assert failure["status"] == "permanently-failed-compiled-validation"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71435-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71435.read_bytes()).hexdigest(),
    }
    assert failure["evidence"]["implementation"] == {
        "atlas_analyzer_authority_file_count": 29,
        "atlas_analyzer_authority_sha256": (
            "1bd8727da427bc4af79983072852bdafe"
            "8675eacb3a70ae769ec0c679cb2f9b9"
        ),
        "generator_sha256": (
            "5eba7670e21c05087920880e1bf983bd"
            "2a533824c41124a6ef996ad6c529b559"
        ),
        "git_clean": True,
        "repository_commit": "1477938f29908a593e563316599d98e42164406a",
        "repository_tree": "6064be90709082a27e09b6f218d80e7ddd5c8011",
        "routes_sha256": (
            "4ceb7bd5b6d5233e75218ab40dd5b16d"
            "ac0f23aba14f04894fec26b698f531d7"
        ),
    }
    assert failure["evidence"]["stage_results"] == {
        "analysis": {
            "actual_file_count": 224,
            "expected_file_count": 224,
            "map_count": 28,
            "pass_count": 28,
            "passed": True,
        },
        "claims": {
            "actual_file_count": 224,
            "expected_file_count": 224,
            "map_count": 28,
            "pass_count": 28,
            "passed": True,
        },
        "compiled": {
            "actual_file_count": 168,
            "expected_file_count": 168,
            "map_count": 28,
            "static_pass_count": 28,
            "passed": True,
        },
        "compiled_validation": {
            "failure_count": 114,
            "map_count": 28,
            "pass_count": 0,
            "passed": False,
        },
        "materialized": {
            "actual_file_count": 196,
            "expected_file_count": 196,
            "map_count": 28,
            "passed": True,
        },
        "source": {
            "cold_file_count": 140,
            "map_count": 28,
            "passed": True,
            "primary_file_count": 140,
            "route_contract_pass_count": 28,
            "unique_layout_count": 28,
        },
    }
    assert failure["failure"] == {
        "classification_counts": {
            "dark_spawn_region_failures": 9,
            "route_cost_failures": 12,
            "spawn_connectivity_failures": 65,
            "spawn_light_region_domain_failures": 28,
        },
        "phase": "compiled-generated-validation",
        "root_causes": {
            "lighting_darkness": (
                "source lighting measured floor-plus-one samples with "
                "horizontal radius and simplified overhead occlusion, while "
                "compiled promotion measured spawn-eye origins with "
                "three-dimensional radius and exact MASK_SHOT collision; "
                "nine maps therefore retained real compiled darkness"
            ),
            "lighting_region_domain": (
                "the validator compared the count of authored floor-light "
                "tiles with the analyzer count of strongly connected spawn "
                "navigation regions; all 28 compiled BSPs retained the exact "
                "authored _ml_region tag set, so this comparison joined "
                "unrelated domains"
            ),
            "route_cost": (
                "source route selection retained only standing-component "
                "membership and Euclidean endpoint distance, discarding "
                "obstacle detour distance; Atlas correctly measured compiled "
                "16-unit graph geodesics and twelve route totals exceeded "
                "both frozen mismatch thresholds"
            ),
            "spawn_connectivity": (
                "distributed spawn placement optimized separation and local "
                "clearance without requiring all eight origins to share one "
                "conservative standing component; nine maps remained split "
                "under compiled directed reachability"
            ),
        },
        "validation_summary": {
            "compiled_route_claim_count": 1487,
            "compiled_route_claims_connected_finite_oracle": 1487,
            "dark_navigation_region_count": 10,
            "darkness_map_count": 9,
            "route_failure_map_count": 6,
            "route_summary_count": 112,
            "source_floor_light_region_count": 720,
            "spawn_connectivity_map_count": 9,
            "spawn_navigation_region_count": 41,
        },
    }
    assert failure["admission"] == {
        "analysis_stage_published": True,
        "claims_stage_published": True,
        "compiled_stage_published": True,
        "materialized_stage_published": True,
        "older_population_reuse_allowed": False,
        "passing_subset_allowed": False,
        "permanently_non_admissible": True,
        "regeneration_under_same_declaration_allowed": False,
        "replacement_declaration_status": "pending-implementation-fix",
        "replacement_member_allowed": False,
        "retry_under_same_declaration_allowed": False,
        "salvage_allowed": False,
        "source_stage_published": True,
    }


def test_71436_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71436.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71436"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71436-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71436.read_bytes()).hexdigest(),
    }
    assert failure["evidence"]["implementation"] == {
        "atlas_analyzer_authority_file_count": 29,
        "atlas_analyzer_authority_sha256": (
            "73d558112b4659c8508e2c848361d5df"
            "ba032ee9d616acd379dce73bb0fa08a4"
        ),
        "generator_sha256": (
            "58d5d7e6517f158e1bf5039dd3bd43b"
            "9fbe53d34630bd0841f20291aad68b424"
        ),
        "git_clean": True,
        "repository_commit": "d28322ca520c898f1eef056cf50e8aaa87755154",
        "repository_tree": "ee88a9153b582cbd997e5d33d79ce58dacf5e74c",
        "routes_sha256": (
            "406b552eb195f6f0fd6a75b689c5ee2"
            "df141b158d7118502c07698eeddae86d7"
        ),
    }
    expected_failure = {
        "grid": 5,
        "map": "b2g26_pits_71436301",
        "message": (
            "could not place 8 clear, separated, map-spanning deathmatch "
            "spawns in one source standing component"
        ),
        "ordinal": 13,
        "seed": 71436301,
        "style": "pits",
    }
    assert failure["failure"]["first_failure"] == expected_failure
    assert failure["evidence"]["failed_member"] == {
        "files": {},
        "grid": 5,
        "map": expected_failure["map"],
        "missing_suffixes": [
            ".map", ".json", ".meta.json", ".lattice.json", ".routes.json",
        ],
        "ordinal": expected_failure["ordinal"],
        "seed": expected_failure["seed"],
        "style": expected_failure["style"],
    }
    assert failure["operator_transcript"] == {
        "exception": "RuntimeError",
        "exit_code": 1,
        "stack_terminal": "maps/generator.py:_place_combat_spawns",
        "stderr_terminal": expected_failure["message"],
    }
    assert failure["evidence"]["primary_population"] == {
        "actual_file_count": 65,
        "complete_member_count": 13,
        "complete_ordinal_range": [0, 12],
        "expected_file_count": 140,
        "last_complete_map": "b2g26_pits_71436300",
        "missing_file_count": 75,
        "partial_member_file_count": 0,
        "path_from_artifact_root": "source",
        "stage_membership_sha256": (
            "d0cd2e1ca8c177184a5bfa1507157e5a"
            "727694920daf14b04ade9bb96753b658"
        ),
        "total_bytes": 4872994,
    }
    assert failure["evidence"]["prior_population_reuse_audit"] == {
        "cohorts": [
            f"b2g26_final_{cohort}" for cohort in range(71429, 71436)
        ],
        "content_sha256_overlap_count": 0,
        "filename_overlap_count": 0,
        "no_content_or_filename_reuse": True,
    }
    assert failure["failure"]["spawn_search_diagnostic"] == {
        "component_count": 10,
        "component_examples": [
            {"component": 2, "span_x": 832, "span_y": 1344},
            {"component": 5, "span_x": 1856, "span_y": 576},
            {"component": 7, "span_x": 1344, "span_y": 448},
            {"component": 8, "span_x": 832, "span_y": 832},
        ],
        "legal_candidate_count": 638,
        "required_map_span": [1024, 1024],
        "single_component_meeting_required_map_span": False,
    }
    assert failure["evidence"]["cold_population"] == {
        "actual_file_count": 0,
        "complete_member_count": 0,
        "expected_file_count": 140,
        "generation_begun": False,
        "missing_file_count": 140,
        "partial_member_file_count": 0,
        "path_from_artifact_root": "source-cold",
        "stage_membership_sha256": (
            "539d0ec7de7365fd4411aac747eb310f1"
            "2eeb2c726d7d30264fb889f59379e01"
        ),
        "total_bytes": 0,
    }
    assert failure["failure"]["phase"] == "primary-generation-spawn-contract"
    assert failure["failure"]["root_cause"] == (
        "final source geometry partitioned 638 legal spawn candidates across "
        "10 standing components, and every component bounding box failed the "
        "required 1024-by-1024 map span; the placer correctly refused to split "
        "starts or weaken admission constraints"
    )
    assert failure["failure"]["publication_contract"] == {
        "cold_generation_begun": False,
        "failed_member_file_count": 0,
        "last_complete_ordinal": 12,
        "source_freeze_report_created": False,
    }
    assert failure["evidence"]["source_freeze_report"] == {
        "exists_at_capture": False,
        "path_from_b2_artifact_root": (
            "generated-final-71436-73d55811-report.json"
        ),
        "published": False,
    }
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["replacement_declaration_status"] == (
        "pending-implementation-fix"
    )
    for key in (
        "source_stage_published", "compiled_stage_published",
        "materialized_stage_published", "claims_stage_published",
        "analysis_stage_published", "older_population_reuse_allowed",
        "passing_subset_allowed", "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retired_artifact_reuse_allowed",
        "retry_under_same_declaration_allowed", "salvage_allowed",
        "substitution_allowed",
    ):
        assert admission[key] is False


def test_71437_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71437.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71437"
    assert failure["status"] == "permanently-failed-first-source-freeze"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71437-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71437.read_bytes()).hexdigest(),
    }
    assert failure["evidence"]["implementation"] == {
        "atlas_analyzer_authority_file_count": 29,
        "atlas_analyzer_authority_sha256": (
            "73d558112b4659c8508e2c848361d5df"
            "ba032ee9d616acd379dce73bb0fa08a4"
        ),
        "generator_sha256": (
            "32d05457fd51108ae67d36584ff369207"
            "aa583b2d7a02c675a0f5f74c9383c22"
        ),
        "git_clean": True,
        "repository_commit": "a87f66c025ea7449d8a09dca517e7a9b6943b2e0",
        "repository_tree": "4a6b414f19c575b9e7cbdf3ffd4de40314b8e65e",
        "routes_sha256": (
            "406b552eb195f6f0fd6a75b689c5ee2"
            "df141b158d7118502c07698eeddae86d7"
        ),
    }
    message = (
        "could not place 8 clear, separated, map-spanning deathmatch spawns "
        "in one source standing component; legal_candidates=487, "
        "component_bounds=[0:count=28,span=64x832; "
        "1:count=151,span=1984x832; 2:count=12,span=64x576; "
        "3:count=95,span=448x1344; 4:count=1,span=0x0; "
        "5:count=12,span=768x64; 6:count=160,span=1472x960; "
        "7:count=20,span=64x576; 8:count=8,span=64x192]"
    )
    expected_failure = {
        "grid": 5,
        "map": "b2g26_canyon_71437202",
        "message": message,
        "ordinal": 10,
        "seed": 71437202,
        "style": "canyon",
    }
    assert failure["failure"]["first_failure"] == expected_failure
    assert failure["evidence"]["failed_member"] == {
        "files": {},
        "grid": 5,
        "map": expected_failure["map"],
        "missing_suffixes": [
            ".map", ".json", ".meta.json", ".lattice.json", ".routes.json",
        ],
        "ordinal": expected_failure["ordinal"],
        "seed": expected_failure["seed"],
        "style": expected_failure["style"],
    }
    assert failure["operator_transcript"] == {
        "exception": "RuntimeError",
        "exit_code": 1,
        "stack_terminal": "maps/generator.py:_place_combat_spawns",
        "stderr_terminal": message,
    }
    assert failure["evidence"]["primary_population"] == {
        "actual_file_count": 50,
        "complete_member_count": 10,
        "complete_ordinal_range": [0, 9],
        "expected_file_count": 140,
        "last_complete_map": "b2g26_canyon_71437201",
        "missing_file_count": 90,
        "partial_member_file_count": 0,
        "path_from_artifact_root": "source",
        "stage_membership_sha256": (
            "8d1d138f367c70c7924f8a592453e501"
            "c3e99118a3a0934244960d090499611c"
        ),
        "total_bytes": 3715165,
    }
    assert failure["evidence"]["cold_population"] == {
        "actual_file_count": 0,
        "complete_member_count": 0,
        "expected_file_count": 140,
        "generation_begun": False,
        "missing_file_count": 140,
        "partial_member_file_count": 0,
        "path_from_artifact_root": "source-cold",
        "stage_membership_sha256": (
            "bf7a444c21a1c041ac3d16a456db305"
            "a9e36c7c719ce83e6643433e6ce6c337d"
        ),
        "total_bytes": 0,
    }
    assert failure["evidence"]["prior_population_reuse_audit"] == {
        "cohorts": [
            f"b2g26_final_{cohort}" for cohort in range(71429, 71437)
        ],
        "content_sha256_overlap_count": 0,
        "filename_overlap_count": 0,
        "no_content_or_filename_reuse": True,
    }
    assert failure["failure"]["spawn_search_diagnostic"] == {
        "component_bounds": [
            {"component": 0, "count": 28, "span_x": 64, "span_y": 832},
            {"component": 1, "count": 151, "span_x": 1984, "span_y": 832},
            {"component": 2, "count": 12, "span_x": 64, "span_y": 576},
            {"component": 3, "count": 95, "span_x": 448, "span_y": 1344},
            {"component": 4, "count": 1, "span_x": 0, "span_y": 0},
            {"component": 5, "count": 12, "span_x": 768, "span_y": 64},
            {"component": 6, "count": 160, "span_x": 1472, "span_y": 960},
            {"component": 7, "count": 20, "span_x": 64, "span_y": 576},
            {"component": 8, "count": 8, "span_x": 64, "span_y": 192},
        ],
        "component_count": 9,
        "legal_candidate_count": 487,
        "required_map_span": [1024, 1024],
        "single_component_meeting_required_map_span": False,
    }
    assert failure["failure"]["blocker_forensics"] == {
        "decisive_blocker_class": "lane-wall static blockers",
        "method": (
            "deterministic in-memory stage replay at the captured clean "
            "implementation; no member artifact was published"
        ),
        "post_lane_walls": {
            "component_count": 8,
            "legal_candidate_count": 603,
            "single_component_meeting_required_map_span": False,
            "widest_components": [
                {"component": 1, "count": 199, "span_x": 1984, "span_y": 960},
                {"component": 5, "count": 192, "span_x": 1472, "span_y": 960},
            ],
        },
        "pre_lane_walls": {
            "component_count": 5,
            "legal_candidate_count": 776,
            "map_spanning_component": {
                "component": 1,
                "count": 524,
                "span_x": 1984,
                "span_y": 1856,
            },
            "single_component_meeting_required_map_span": True,
        },
    }
    assert failure["failure"]["phase"] == "primary-generation-spawn-contract"
    assert failure["failure"]["root_cause"] == (
        "spawn-arena floor normalization initially left a 2880-by-1856 "
        "candidate component and hallway/tower blockers still left a "
        "1984-by-1856 component, but lane-wall static blockers were the first "
        "stage to split the remaining map-spanning component into subdomains "
        "whose widest spans were 1984-by-960 and 1472-by-960; later cover, "
        "corner, and objective blockers reduced the final pool to 487 "
        "candidates across 9 components, every component failed the required "
        "1024-by-1024 span, and the placer correctly refused split starts or "
        "relaxed admission"
    )
    assert failure["failure"]["publication_contract"] == {
        "cold_generation_begun": False,
        "failed_member_file_count": 0,
        "last_complete_ordinal": 9,
        "source_freeze_report_created": False,
    }
    assert failure["evidence"]["source_freeze_report"] == {
        "exists_at_capture": False,
        "path_from_b2_artifact_root": (
            "generated-final-71437-73d55811-report.json"
        ),
        "published": False,
    }
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["replacement_declaration_status"] == (
        "pending-implementation-fix"
    )
    for key in (
        "source_stage_published", "compiled_stage_published",
        "materialized_stage_published", "claims_stage_published",
        "analysis_stage_published", "older_population_reuse_allowed",
        "passing_subset_allowed", "regeneration_under_same_declaration_allowed",
        "replacement_member_allowed", "retired_artifact_reuse_allowed",
        "retry_under_same_declaration_allowed", "salvage_allowed",
        "substitution_allowed",
    ):
        assert admission[key] is False


def test_71438_failure_record_is_canonical_exact_and_no_salvage() -> None:
    payload = FAILURE_71438.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)
    assert hashlib.sha256(payload).hexdigest() == (
        "29c997e49e197688e15e62864cea62897"
        "c22b69e59297b8258c800dc79e93103"
    )

    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71438"
    assert failure["status"] == "permanently-failed-first-compile"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json",
        "sha256": hashlib.sha256(DECLARATION_71438.read_bytes()).hexdigest(),
    }
    assert failure["evidence"]["implementation"] == {
        "atlas_analyzer_authority_file_count": 29,
        "atlas_analyzer_authority_sha256": (
            "73d558112b4659c8508e2c848361d5df"
            "ba032ee9d616acd379dce73bb0fa08a4"
        ),
        "generator_sha256": (
            "dd2d52b3b4fb466f66ab0993a4ecae94"
            "a16389f4bc69653a974f8edf3a4546ed"
        ),
        "git_clean": True,
        "repository_commit": "0fab17225210738a98ff64c362a1be492aa9c290",
        "repository_tree": "164fbd135e2d3ab297c4a0fa28cb5e9556a59f09",
        "routes_sha256": (
            "406b552eb195f6f0fd6a75b689c5ee2"
            "df141b158d7118502c07698eeddae86d7"
        ),
    }
    assert failure["evidence"]["source_freeze"] == {
        "all_file_bytes_match": True,
        "atlas_admissible": False,
        "bundle_admissible": False,
        "cold_file_count": 140,
        "declaration_sha256": hashlib.sha256(
            DECLARATION_71438.read_bytes()
        ).hexdigest(),
        "exists_at_capture": True,
        "map_count": 28,
        "passed": True,
        "path_from_b2_artifact_root": (
            "generated-final-71438-73d55811-report.json"
        ),
        "primary_file_count": 140,
        "report_sha256": (
            "4507d9b4528308fcf1fd05e7a0dba572"
            "37b3a1049fd07f673cb660d58061b34f"
        ),
        "route_contract_pass_count": 28,
        "source_total_bytes": 10861198,
        "spawn_origin_binding_pass_count": 28,
        "status": "source-frozen-pre-compile",
        "unique_layout_count": 28,
    }
    release = failure["evidence"]["release_build_provenance"]
    assert release == {
        "all_three_release_builds_succeeded": True,
        "atlas_run": False,
        "copy_run_reuse_or_substitution_permitted": False,
        "dyn_evidence_run": False,
        "explicitly_non_admissible": True,
        "path": (
            "/home/raymondj/multires-artifacts/atlas-v1/B2/"
            "generated-final-71438-73d55811/"
            "release-build-provenance-0fab17225210.json"
        ),
        "sha256": (
            "362769453e063fb23cd6076f01596ac15"
            "8693e024a8ab00950b548053720ba83"
        ),
        "size_bytes": 5163,
        "source_repository_commit": (
            "0fab17225210738a98ff64c362a1be492aa9c290"
        ),
        "source_repository_tree": (
            "164fbd135e2d3ab297c4a0fa28cb5e9556a59f09"
        ),
        "wsl_touched": False,
    }
    wsl = failure["evidence"]["wsl_compile"]
    assert wsl["host"] == "DESKTOP-RTX2080"
    assert wsl["wsl_root"] == (
        "/home/raymond/q2-multires-isolated/B2/"
        "b2g26_final_71438-73d55811"
    )
    assert wsl["q2tool"] == {
        "path": (
            "/home/raymond/q2-rollout/q2-ml-bot/maps/"
            "q2tools/bin/q2tool"
        ),
        "sha256": (
            "a13dd3095ff56ca668e94c8992c915be"
            "669f9404162c9c87ded3b922316b26f0"
        ),
    }
    assert wsl["asset_isolation"] == {
        "pak0_path": (
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/assets/baseq2/pak0.pak"
        ),
        "pak0_sha256": (
            "1ce99eb11e7e251ccdf690858effba798"
            "36dbe5e32a4083ad00a13ecda491679"
        ),
        "supplied_basedir": (
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/assets"
        ),
    }
    assert wsl["source_transfer"] == {
        "all_file_bytes_match": True,
        "compiled_precompile_file_count": 140,
        "local_source_path": (
            "/home/raymondj/multires-artifacts/atlas-v1/B2/"
            "generated-final-71438-73d55811/source"
        ),
        "relative_sha256_manifest_sha256": (
            "5ce26dfe990a197ea49ae966552216ce8"
            "1edc38af94c0e08b3deb5a4ecafaf88"
        ),
        "source_file_count": 140,
        "source_total_bytes": 10861198,
        "wsl_compiled_path": (
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/compiled"
        ),
        "wsl_source_path": (
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/source"
        ),
    }
    assert wsl["command_log"] == {
        "path_from_wsl_root": (
            "logs/b2g26_arena_lanes_71438600.command.txt"
        ),
        "sha256": (
            "2deb5b11b77d7db7ca77f52381014f50"
            "3f143900d2f00bfb79d8b28bb4dc0e11"
        ),
        "size_bytes": 288,
    }
    assert wsl["compile_log"] == {
        "path_from_wsl_root": (
            "logs/b2g26_arena_lanes_71438600.compile.log"
        ),
        "sha256": (
            "f7ed18bf6ab26ce5f5027453dacfc2b5"
            "25059d60fd521d7b2d55d2a69f23e068"
        ),
        "size_bytes": 3590,
    }
    assert wsl["compiled_residual"] == {
        "actual_file_count": 142,
        "bsp_count": 1,
        "complete_member_count": 0,
        "expected_file_count": 168,
        "missing_bsp_count": 27,
        "partial_member_count": 1,
        "prt_count": 1,
        "source_file_count": 140,
        "total_bytes": 11658404,
        "unattempted_member_count": 27,
        "unexpected_file_count": 1,
        "unexpected_files": [{
            "path_from_wsl_root": (
                "compiled/b2g26_arena_lanes_71438600.prt"
            ),
            "sha256": (
                "c4f71915693f7573b433875014d1ef0e"
                "a119f70398d277c2d6b8d1569ef7d600"
            ),
            "size_bytes": 213866,
        }],
        "written_bsps": [{
            "complete": False,
            "path_from_wsl_root": (
                "compiled/b2g26_arena_lanes_71438600.bsp"
            ),
            "sha256": (
                "4a1feab912c6205378c80d9f5a9fae5"
                "19c2b8491f5b440226cfbf9fb24d369cd"
            ),
            "size_bytes": 583340,
        }],
    }
    assert failure["failure"]["first_failure"] == {
        "grid": 5,
        "map": "b2g26_arena_lanes_71438600",
        "message": "unable to load pics/colormap.pcx",
        "ordinal": 24,
        "phase": "rad",
        "seed": 71438600,
        "style": "arena_lanes",
    }
    assert failure["failure"]["publication_contract"] == {
        "atlas_run": False,
        "compiled_membership_report_created": False,
        "compiled_stage_copied_back": False,
        "compiled_static_campaign_run": False,
        "dyn_run": False,
        "later_stage_run": False,
        "maps_attempted": 1,
        "maps_compiled_completely": 0,
        "maps_declared": 28,
    }
    assert failure["operator_transcript"] == {
        "command": (
            "/home/raymond/q2-rollout/q2-ml-bot/maps/q2tools/bin/q2tool "
            "-bsp -vis -fast -rad -bounce 0 -threads 1 -basedir "
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/assets "
            "/home/raymond/q2-multires-isolated/B2/"
            "b2g26_final_71438-73d55811/compiled/"
            "b2g26_arena_lanes_71438600.map"
        ),
        "failure_branch_entered": True,
        "q2tool_exit_code_retained": False,
        "terminal_log_text": "unable to load pics/colormap.pcx",
        "wrapper_exit_code": 1,
        "wrapper_post_negation_status": 0,
    }
    admission = failure["admission"]
    assert admission["permanently_non_admissible"] is True
    assert admission["replacement_declaration_status"] == "not-authorized"
    for key, value in admission.items():
        if key not in {"permanently_non_admissible", "replacement_declaration_status"}:
            assert value is False, key


def test_71439_failure_record_is_canonical_exact_and_no_replacement() -> None:
    payload = FAILURE_71439.read_bytes()
    failure = json.loads(payload)
    assert payload == canonical_bytes(failure)
    assert hashlib.sha256(payload).hexdigest() == (
        "9d0507b6124ec84da7564e9b6b2a7dd5"
        "004a497ea4bc1e00079752dd07a588a7"
    )

    declaration_sha256 = hashlib.sha256(
        DECLARATION_71439.read_bytes()
    ).hexdigest()
    assert failure["schema"] == "q2-b2-generated-cohort-failure-v1"
    assert failure["cohort_id"] == "b2g26_final_71439"
    assert failure["status"] == "permanently-failed-first-materialization"
    assert failure["declaration"] == {
        "path": "docs/multires/B2-GENERATED-COHORT-71439-DECLARATION.json",
        "sha256": declaration_sha256,
    }
    assert failure["admission"] == {
        "analysis_stage_published": False,
        "claims_stage_published": False,
        "compiled_stage_published": True,
        "dyn_execution_allowed": False,
        "materialized_stage_published": False,
        "older_population_reuse_allowed": False,
        "passing_subset_allowed": False,
        "permanently_non_admissible": True,
        "regeneration_under_same_declaration_allowed": False,
        "release_artifact_copy_allowed": False,
        "release_artifact_execution_allowed": False,
        "release_artifact_reuse_allowed": False,
        "replacement_declaration_status": "not-authorized",
        "replacement_member_allowed": False,
        "retired_artifact_reuse_allowed": False,
        "retry_under_same_declaration_allowed": False,
        "salvage_allowed": False,
        "source_stage_published": True,
        "substitution_allowed": False,
    }

    evidence = failure["evidence"]
    assert evidence["implementation"] == {
        "atlas_analyzer_authority_file_count": 29,
        "atlas_analyzer_authority_sha256": (
            "73d558112b4659c8508e2c848361d5df"
            "ba032ee9d616acd379dce73bb0fa08a4"
        ),
        "generator_sha256": (
            "dd2d52b3b4fb466f66ab0993a4ecae94"
            "a16389f4bc69653a974f8edf3a4546ed"
        ),
        "git_clean": True,
        "repository_commit": "3568b18d8373f3a965f2bdd106ca2b64e0c16fd7",
        "repository_tree": "22a685fa93e775c10c77761ac96e5152edc686b0",
        "routes_sha256": (
            "406b552eb195f6f0fd6a75b689c5ee2"
            "df141b158d7118502c07698eeddae86d7"
        ),
    }
    assert evidence["source_freeze"] == {
        "all_file_bytes_match": True,
        "atlas_admissible": False,
        "bundle_admissible": False,
        "cold_file_count": 140,
        "declaration_sha256": declaration_sha256,
        "exists_at_capture": True,
        "map_count": 28,
        "passed": True,
        "path_from_b2_artifact_root": (
            "generated-final-71439-73d55811-report.json"
        ),
        "primary_file_count": 140,
        "report_sha256": (
            "fbcbca7c134c2d2595ab98cfe939f615"
            "b226cab4a5e28e836f824d41e4f76255"
        ),
        "report_size_bytes": 137234,
        "route_contract_pass_count": 28,
        "source_total_bytes": 10911304,
        "spawn_origin_binding_pass_count": 28,
        "status": "source-frozen-pre-compile",
        "unique_layout_count": 28,
    }

    compile_evidence = evidence["wsl_compile"]
    assert compile_evidence["host"] == "DESKTOP-RTX2080"
    assert compile_evidence["report"] == {
        "passed": True,
        "path_from_wsl_root": "reports/compiled.json",
        "sha256": (
            "fc6435e81ac1d10f8a32602169df68cc"
            "34103c4b64a2cdbcf96be55260a3733d"
        ),
        "size_bytes": 82970,
        "status": "compiled-stage-published-non-admissible",
    }
    assert compile_evidence["compiled_publication"] == {
        "actual_file_count": 168,
        "expected_file_count": 168,
        "expected_map_count": 28,
        "map_pass_count": 28,
        "published": True,
        "relative_sha256_manifest_sha256": (
            "93e4166e0a0e64a70f0d30a6b5bce927"
            "95c3e33bffe8b91bdc8d6e8301c2e9d9"
        ),
        "total_bytes": 59581688,
    }

    materialization = evidence["wsl_materialization"]
    assert materialization["report"] == {
        "attempted_count": 1,
        "not_attempted_count": 27,
        "pass_count": 0,
        "path_from_wsl_root": "reports/materialized.json",
        "sha256": (
            "b171b2ee4ab02f8b960684544e49471dc"
            "fc5e11cdef105687a77938e1dcafe69"
        ),
        "size_bytes": 6851,
    }
    assert materialization["logs"] == {
        "stderr": {
            "path_from_wsl_root": (
                "logs/materialize/"
                "0000-b2g26_open_71439000.materialize.stderr.log"
            ),
            "sha256": (
                "2b97e7f8c13cc822a4f26d31119aa026"
                "6178f000fbaa502a9c07936791f09dbc"
            ),
            "size_bytes": 421,
        },
        "stdout": {
            "path_from_wsl_root": (
                "logs/materialize/"
                "0000-b2g26_open_71439000.materialize.stdout.json"
            ),
            "sha256": (
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
            "size_bytes": 0,
        },
    }
    assert materialization["materialized_residual"] == {
        "expected_file_count": 196,
        "materialized_publish_path_exists": False,
        "materialized_staging_file_count": 168,
        "materialized_staging_matches_compiled": True,
        "materialized_staging_relative_sha256_manifest_sha256": (
            "93e4166e0a0e64a70f0d30a6b5bce927"
            "95c3e33bffe8b91bdc8d6e8301c2e9d9"
        ),
        "materialized_staging_total_bytes": 59581688,
    }
    assert materialization["failed_runtime"]["interpreter"] == {
        "path": "/usr/bin/python3.10",
        "sha256": (
            "7d51cd6b48b521277f5caa4610a82126"
            "e315fa2be4df069823a8b1eeb5bd4a86"
        ),
        "size_bytes": 5917224,
        "version": "Python 3.10.12",
    }
    assert materialization["failed_runtime"]["atlas_analyzer"] == {
        "path_from_producer_snapshot": "harness/atlas_analyzer.py",
        "sha256": (
            "5f6923ede36e1aa1cacd3aa87973c927"
            "463e1b04a48cf071b2223b1bb8ec22bb"
        ),
        "size_bytes": 253841,
    }
    assert materialization["producer_snapshot"] == {
        "path": "/home/raymond/q2-multires-isolated/B2/producer-3568b18d8373",
        "relative_sha256_manifest_file_count": 277,
        "relative_sha256_manifest_sha256": (
            "461f6b93fadd74cd0f90cdb9194265e10"
            "23f726f2cadb251ebb9723d4b3c9c2a"
        ),
    }
    assert materialization["root_relative_sha256_manifest"] == {
        "file_count": 539,
        "sha256": (
            "a08c9c8d77f2c8ec1f0cf48352b557ce"
            "2d356ee9d3d8ce6250fca5fc88cdf618"
        ),
    }

    assert failure["failure"]["first_failure"] == {
        "grid": 5,
        "map": "b2g26_open_71439000",
        "message": "materializer exited 1 before hook materialization",
        "ordinal": 0,
        "phase": "map-materialization",
        "seed": 71439000,
        "style": "open",
    }
    root_cause = failure["failure"]["root_cause"]
    assert "harness/atlas_analyzer.py" in root_cause
    assert "line 5404" in root_cause
    assert "before any hook materialization" in root_cause
    assert "No retry, repair, reuse, salvage, or substitution" in root_cause
    transcript = failure["operator_transcript"]
    assert transcript["first_map_returncode"] == 1
    assert transcript["wrapper_exit_code"] == 1
    assert transcript["stdout_empty"] is True
    assert transcript["stderr_terminal"] == (
        "SyntaxError: unterminated string literal (detected at line 5404)"
    )
    assert "transcript_sha256" not in transcript


def write_stage(directory: Path, stage: str) -> dict:
    declaration, _digest = load_declaration(DECLARATION)
    directory.mkdir()
    for row in declaration["maps"]:
        for suffix in STAGE_SUFFIXES[stage]:
            payload: bytes
            if suffix == ".analysis.manifest.json":
                payload = canonical_bytes({
                    "identity": {
                        "atlas_sha256": hashlib.sha256(
                            row["map"].encode("ascii")
                        ).hexdigest()
                    }
                })
            else:
                payload = (
                    f"{stage}:{row['ordinal']}:{row['map']}:{suffix}\n".encode()
                )
            (directory / f"{row['map']}{suffix}").write_bytes(payload)
    return declaration


def fake_claims(map_path: Path) -> dict:
    return {
        "schema": "test-generator-claims",
        "map": map_path.stem,
    }


def test_prepare_requires_exact_materialized_membership_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    declaration = write_stage(materialized, "materialized")
    missing = f"{declaration['maps'][0]['map']}.bsp"
    (materialized / missing).unlink()
    replacement = "b2g26_replacement_99999999.bsp"
    (materialized / replacement).write_bytes(b"replacement")
    called = []
    monkeypatch.setattr(
        campaign, "build_generator_claims", lambda path: called.append(path)
    )

    report = campaign.prepare_claims(
        DECLARATION, materialized, tmp_path / "claims"
    )

    assert report["passed"] is False
    assert report["pass_count"] == 0
    assert called == []
    assert not (tmp_path / "claims").exists()
    assert f"materialized-stage: missing file {missing}" in report["failures"]
    assert f"materialized-stage: unexpected file {replacement}" in report["failures"]
    assert [row["ordinal"] for row in report["maps"]] == list(range(28))


def test_prepare_builds_all_before_atomic_exact_claims_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    declaration = write_stage(materialized, "materialized")
    calls: list[str] = []

    def build(path: Path) -> dict:
        calls.append(path.stem)
        return fake_claims(path)

    monkeypatch.setattr(campaign, "build_generator_claims", build)
    claims_dir = tmp_path / "claims"
    report = campaign.prepare_claims(
        DECLARATION, materialized, claims_dir
    )

    expected_order = [row["map"] for row in declaration["maps"]]
    assert calls == expected_order
    assert report["passed"] is True
    assert report["pass_count"] == 28
    assert [row["map"] for row in report["maps"]] == expected_order
    assert verify_stage_membership(
        declaration, claims_dir, "claims"
    )["passed"] is True
    for row in declaration["maps"]:
        for suffix in STAGE_SUFFIXES["materialized"]:
            assert (
                claims_dir / f"{row['map']}{suffix}"
            ).read_bytes() == (
                materialized / f"{row['map']}{suffix}"
            ).read_bytes()
        claims_path = claims_dir / f"{row['map']}.generator-claims.json"
        assert json.loads(claims_path.read_text())["map"] == row["map"]
        assert report["maps"][row["ordinal"]][
            "generator_claims_sha256"
        ] == file_sha256(claims_path)


def test_prepare_one_build_failure_publishes_no_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    declaration = write_stage(materialized, "materialized")
    bad = declaration["maps"][17]["map"]
    calls: list[str] = []

    def build(path: Path) -> dict:
        calls.append(path.stem)
        if path.stem == bad:
            raise ValueError("route node is not compiled-reachable")
        return fake_claims(path)

    monkeypatch.setattr(campaign, "build_generator_claims", build)
    claims_dir = tmp_path / "claims"
    report = campaign.prepare_claims(
        DECLARATION, materialized, claims_dir
    )

    assert calls == [row["map"] for row in declaration["maps"]]
    assert report["passed"] is False
    assert report["pass_count"] == 27
    assert not claims_dir.exists()
    assert report["maps"][17]["passed"] is False
    assert report["maps"][17]["error"] == "route node is not compiled-reachable"
    assert report["failures"] == [
        f"{bad}: route node is not compiled-reachable"
    ]


def test_prepare_refuses_existing_or_nested_claims_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    write_stage(materialized, "materialized")
    monkeypatch.setattr(campaign, "build_generator_claims", fake_claims)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(campaign.ClaimCampaignError, match="already exists"):
        campaign.prepare_claims(DECLARATION, materialized, existing)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(campaign.ClaimCampaignError, match="already exists"):
        campaign.prepare_claims(DECLARATION, materialized, dangling)
    with pytest.raises(campaign.ClaimCampaignError, match="separate non-nested"):
        campaign.prepare_claims(
            DECLARATION, materialized, materialized / "claims"
        )


def make_claims_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path]:
    materialized = tmp_path / "materialized"
    declaration = write_stage(materialized, "materialized")
    monkeypatch.setattr(campaign, "build_generator_claims", fake_claims)
    claims = tmp_path / "claims"
    report = campaign.prepare_claims(DECLARATION, materialized, claims)
    assert report["passed"] is True
    return declaration, claims


def test_validate_uses_separate_explicit_analysis_root_in_declared_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration, claims = make_claims_stage(tmp_path, monkeypatch)
    analysis = tmp_path / "analysis"
    write_stage(analysis, "analysis")
    b1_gate = tmp_path / "B1-GATE.json"
    b1_gate.write_text("{}\n")
    calls: list[tuple[Path, Path, Path]] = []

    def validate(map_path: Path, analysis_path: Path, *, b1_gate_path: Path):
        calls.append((map_path, analysis_path, b1_gate_path))
        return {
            "passed": True,
            "identities": {
                "bsp_sha256": file_sha256(map_path.with_suffix(".bsp")),
                "generator_claims_sha256": file_sha256(
                    map_path.with_suffix(".generator-claims.json")
                ),
            },
            "failures": [],
        }

    monkeypatch.setattr(campaign, "validate_generated_map", validate)
    report = campaign.validate_campaign(
        DECLARATION, claims, analysis, b1_gate
    )

    expected_names = [row["map"] for row in declaration["maps"]]
    assert report["passed"] is True
    assert report["pass_count"] == 28
    assert [row["map"] for row in report["maps"]] == expected_names
    assert [call[0] for call in calls] == [
        claims / f"{name}.map" for name in expected_names
    ]
    assert [call[1] for call in calls] == [
        analysis / f"{name}.analysis.manifest.json" for name in expected_names
    ]
    assert all(call[2] == b1_gate for call in calls)
    assert all(call[0].parent != call[1].parent for call in calls)
    assert (claims / f"{expected_names[0]}.routes.json").is_file()
    assert (analysis / f"{expected_names[0]}.objectives.json").is_file()
    assert not (analysis / f"{expected_names[0]}.routes.json").exists()


def test_validate_rejects_same_count_analysis_substitution_before_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration, claims = make_claims_stage(tmp_path, monkeypatch)
    analysis = tmp_path / "analysis"
    write_stage(analysis, "analysis")
    missing = f"{declaration['maps'][4]['map']}.atlas.bin"
    (analysis / missing).unlink()
    replacement = "b2g26_replacement_99999999.atlas.bin"
    (analysis / replacement).write_bytes(b"replacement")
    b1_gate = tmp_path / "B1-GATE.json"
    b1_gate.write_text("{}\n")
    calls = []
    monkeypatch.setattr(
        campaign, "validate_generated_map", lambda *args, **kwargs: calls.append(args)
    )

    report = campaign.validate_campaign(
        DECLARATION, claims, analysis, b1_gate
    )

    assert report["passed"] is False
    assert report["pass_count"] == 0
    assert calls == []
    assert f"analysis-stage: missing file {missing}" in report["failures"]
    assert f"analysis-stage: unexpected file {replacement}" in report["failures"]


def test_validate_rejects_same_count_claims_substitution_before_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration, claims = make_claims_stage(tmp_path, monkeypatch)
    analysis = tmp_path / "analysis"
    write_stage(analysis, "analysis")
    missing = f"{declaration['maps'][9]['map']}.generator-claims.json"
    (claims / missing).unlink()
    replacement = "b2g26_replacement_99999999.generator-claims.json"
    (claims / replacement).write_bytes(b"replacement")
    b1_gate = tmp_path / "B1-GATE.json"
    b1_gate.write_text("{}\n")
    calls = []
    monkeypatch.setattr(
        campaign, "validate_generated_map", lambda *args, **kwargs: calls.append(args)
    )

    report = campaign.validate_campaign(
        DECLARATION, claims, analysis, b1_gate
    )

    assert report["passed"] is False
    assert report["pass_count"] == 0
    assert calls == []
    assert f"claims-stage: missing file {missing}" in report["failures"]
    assert f"claims-stage: unexpected file {replacement}" in report["failures"]


def test_validate_refuses_adjacent_or_nested_analysis_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declaration, claims = make_claims_stage(tmp_path, monkeypatch)
    with pytest.raises(campaign.ClaimCampaignError, match="separate non-nested"):
        campaign.validate_campaign(
            DECLARATION, claims, claims, tmp_path / "B1-GATE.json"
        )
    with pytest.raises(campaign.ClaimCampaignError, match="separate non-nested"):
        campaign.validate_campaign(
            DECLARATION,
            claims,
            claims / "analysis",
            tmp_path / "B1-GATE.json",
        )


def test_cli_requires_declaration_and_exclusive_external_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    write_stage(materialized, "materialized")
    monkeypatch.setattr(campaign, "build_generator_claims", fake_claims)
    report_path = tmp_path / "reports/prepare.json"
    arguments = [
        "prepare",
        "--declaration", str(DECLARATION),
        "--materialized-dir", str(materialized),
        "--claims-dir", str(tmp_path / "claims"),
        "--output", str(report_path),
    ]

    assert campaign.main(arguments) == 0
    first = report_path.read_bytes()
    assert first == canonical_bytes(json.loads(first))
    assert campaign.main(arguments) == 1
    assert report_path.read_bytes() == first

    with pytest.raises(SystemExit):
        campaign.main([
            "prepare",
            "--materialized-dir", str(materialized),
            "--claims-dir", str(tmp_path / "other-claims"),
            "--output", str(tmp_path / "other-report.json"),
        ])
