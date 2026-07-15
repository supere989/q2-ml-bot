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
DECLARATION = ROOT / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
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
    assert (claims / f"{expected_names[0]}.routes.json").read_bytes() != (
        analysis / f"{expected_names[0]}.routes.json"
    ).read_bytes()


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
