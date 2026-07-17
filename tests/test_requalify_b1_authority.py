from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from harness import atlas_b1_authority as authority
from tools import requalify_b1_authority as requalify


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_GATE_PATH = ROOT / authority.HISTORICAL_GATE_RELATIVE_PATH


def verified_inputs() -> tuple[
    authority.B1AuthorityGate,
    authority.B1RuntimeAuthoritySeal,
    dict,
]:
    gate = authority.load_historical_b1_authority_gate(
        HISTORICAL_GATE_PATH, repo_root=ROOT
    )
    probe = "a" * 64
    collision_physics = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=cm;tool_identity="
            f"{gate.oracle_tool_identity};map={probe}"
        ).encode()
    ).hexdigest()
    constants = "fixture-pmove-constants"
    pmove_physics = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=pmove;tool_identity="
            f"{gate.oracle_tool_identity};map={probe};gravity=800;"
            f"airaccelerate=0;constants={constants}"
        ).encode()
    ).hexdigest()
    seal = authority.B1RuntimeAuthoritySeal(
        schema=authority.SEAL_SCHEMA,
        design_sha256=authority.HISTORICAL_DESIGN_SHA256,
        plan_sha256=authority.HISTORICAL_PLAN_SHA256,
        hook_parity_attestation_sha256=gate.hook_attestation_sha256,
        fixture_bsp_sha256=gate.fixture_bsp_sha256,
        analysis_bsp_sha256=probe,
        cm_executable_sha256=gate.cm_executable_sha256,
        pmove_executable_sha256=gate.pmove_executable_sha256,
        hook_executable_sha256=authority.HISTORICAL_HOOK_EXECUTABLE_SHA256,
        fall_executable_sha256=gate.fall_executable_sha256,
        collision_tool_identity=gate.oracle_tool_identity,
        collision_physics_identity=collision_physics,
        pmove_tool_identity=gate.oracle_tool_identity,
        pmove_physics_identity=pmove_physics,
        hook_tool_identity=gate.hook_tool_identity,
        hook_physics_identity=gate.hook_physics_identity,
        fall_tool_identity=gate.fall_tool_identity,
        fall_physics_identity=gate.fall_default_physics_identity,
    )
    identities = {
        "collision": {
            "tool_identity": seal.collision_tool_identity,
            "physics_identity": seal.collision_physics_identity,
        },
        "pmove": {
            "tool_identity": seal.pmove_tool_identity,
            "physics_identity": seal.pmove_physics_identity,
            "parameters": {
                "gravity": 800,
                "airaccelerate": 0,
                "constants": constants,
            },
        },
        "hook": {
            "tool_identity": seal.hook_tool_identity,
            "physics_identity": seal.hook_physics_identity,
        },
        "fall": {
            "tool_identity": seal.fall_tool_identity,
            "physics_identity": seal.fall_physics_identity,
        },
    }
    return gate, seal, identities


def test_build_requalified_gate_binds_amended_docs_and_live_evidence() -> None:
    gate, seal, identities = verified_inputs()
    historical = requalify._load_historical_document(HISTORICAL_GATE_PATH)

    candidate = requalify.build_requalified_gate(
        historical_document=historical,
        historical_gate=gate,
        historical_seal=seal,
        live_identities=identities,
        repo_root=ROOT,
        repository={"commit": "1" * 40, "tree": "2" * 40, "clean": True},
        recorded_at="2026-07-16T12:00:00Z",
    )

    admitted = authority._validate_gate_document(candidate, ROOT)
    record = candidate["authority_requalification"]
    assert admitted.design_sha256 == authority.ACCEPTED_DESIGN_SHA256
    assert admitted.plan_sha256 == authority.ACCEPTED_PLAN_SHA256
    assert record["historical_gate_sha256"] == authority.HISTORICAL_GATE_SHA256
    assert record["probe_bsp_sha256"] == seal.analysis_bsp_sha256
    assert candidate["tests"]["authority_requalification"] == {
        "historical_gate_sha256": authority.HISTORICAL_GATE_SHA256,
        "historical_evidence_reexecuted": False,
        "byte_identical_oracles_reverified": 4,
        "live_identity_probes_passed": 4,
        "normative_documents_rehashed": 2,
    }


def test_build_requalified_gate_rejects_unrooted_live_verification() -> None:
    gate, seal, identities = verified_inputs()
    historical = requalify._load_historical_document(HISTORICAL_GATE_PATH)
    forged = replace(seal, design_sha256="0" * 64)

    with pytest.raises(
        requalify.B1RequalificationError,
        match="not rooted in historical B1",
    ):
        requalify.build_requalified_gate(
            historical_document=historical,
            historical_gate=gate,
            historical_seal=forged,
            live_identities=identities,
            repo_root=ROOT,
            repository={"commit": "1" * 40, "tree": "2" * 40, "clean": True},
            recorded_at="2026-07-16T12:00:00Z",
        )


def test_historical_gate_loader_rejects_edited_evidence(tmp_path: Path) -> None:
    document = json.loads(HISTORICAL_GATE_PATH.read_text())
    document["gate"]["green"] = True
    candidate = tmp_path / "historical.json"
    candidate.write_text(json.dumps(document, sort_keys=True))

    with pytest.raises(
        requalify.B1RequalificationError,
        match="bytes differ",
    ):
        requalify._load_historical_document(candidate)
