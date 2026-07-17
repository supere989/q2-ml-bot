from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from harness import atlas_b1_authority as authority


ROOT = Path(__file__).resolve().parents[1]
ATTESTATION = (
    ROOT / "tests/fixtures/multires/hook-parity-pullspeed-1700.json"
)


def historical_gate() -> authority.B1AuthorityGate:
    return authority.load_historical_b1_authority_gate(
        ROOT / authority.GATE_RELATIVE_PATH, repo_root=ROOT
    )


def requalification_record(
    gate: authority.B1AuthorityGate, probe_bsp_sha256: str = "a" * 64
) -> dict:
    pmove = pmove_identity(gate, probe_bsp_sha256)
    collision = cm_identity(gate, probe_bsp_sha256)
    identities = {
        "collision": {
            "tool_identity": collision["tool_identity"],
            "physics_identity": collision["physics_identity"],
        },
        "pmove": {
            "tool_identity": pmove["tool_identity"],
            "physics_identity": pmove["physics_identity"],
            "parameters": pmove["parameters"],
        },
        "hook": {
            "tool_identity": gate.hook_tool_identity,
            "physics_identity": gate.hook_physics_identity,
        },
        "fall": {
            "tool_identity": gate.fall_tool_identity,
            "physics_identity": gate.fall_default_physics_identity,
        },
    }
    executables = {
        "cm_sha256": gate.cm_executable_sha256,
        "pmove_sha256": gate.pmove_executable_sha256,
        "hook_sha256": authority.HISTORICAL_HOOK_EXECUTABLE_SHA256,
        "fall_sha256": gate.fall_executable_sha256,
    }
    return {
        "schema": authority.REQUALIFICATION_SCHEMA,
        "status": "green",
        "recorded_at": "2026-07-16T12:00:00Z",
        "historical_gate_sha256": authority.HISTORICAL_GATE_SHA256,
        "historical_normative_documents": {
            "design_sha256": authority.HISTORICAL_DESIGN_SHA256,
            "plan_sha256": authority.HISTORICAL_PLAN_SHA256,
        },
        "current_normative_documents": {
            "design_sha256": authority.ACCEPTED_DESIGN_SHA256,
            "plan_sha256": authority.ACCEPTED_PLAN_SHA256,
        },
        "probe_bsp_sha256": probe_bsp_sha256,
        "repository": {"commit": "1" * 40, "tree": "2" * 40, "clean": True},
        "inputs": {
            "hook_parity_attestation_sha256": gate.hook_attestation_sha256,
            "executables": executables,
        },
        "live_identities": identities,
        "probe_runtime_authority_seal": {
            "schema": authority.SEAL_SCHEMA,
            "normative_documents": {
                "design_sha256": authority.ACCEPTED_DESIGN_SHA256,
                "plan_sha256": authority.ACCEPTED_PLAN_SHA256,
            },
            "hook_parity_attestation_sha256": gate.hook_attestation_sha256,
            "fixture_bsp_sha256": gate.fixture_bsp_sha256,
            "analysis_bsp_sha256": probe_bsp_sha256,
            "executables": executables,
            "identities": {
                name: {
                    "tool_identity": identity["tool_identity"],
                    "physics_identity": identity["physics_identity"],
                }
                for name, identity in identities.items()
            },
        },
        "checks": {
            "historical_gate_exact_bytes": True,
            "normative_documents_rehashed": True,
            "repository_clean": True,
            "executable_bytes_match_historical_gate": True,
            "hook_attestation_revalidated": True,
            "live_identities_recomputed": True,
            "live_identity_preimages_validated": True,
        },
        "failures": [],
    }


def requalified_gate_document() -> dict:
    document = json.loads((ROOT / authority.GATE_RELATIVE_PATH).read_text())
    document["normative_documents"] = {
        "design_sha256": authority.ACCEPTED_DESIGN_SHA256,
        "plan_sha256": authority.ACCEPTED_PLAN_SHA256,
    }
    document["amended_at"] = "2026-07-16T12:00:00Z"
    document["authority_requalification"] = requalification_record(
        historical_gate()
    )
    return document


def copied_authority_root(tmp_path: Path) -> Path:
    for relative in (
        authority.DESIGN_RELATIVE_PATH,
        authority.PLAN_RELATIVE_PATH,
        authority.GATE_RELATIVE_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (tmp_path / authority.GATE_RELATIVE_PATH).write_text(
        json.dumps(requalified_gate_document())
    )
    return tmp_path


def provenance(gate: authority.B1AuthorityGate) -> dict:
    return {
        "schema": "q2-oracle-tool-identity-v1",
        "tool_identity": gate.oracle_tool_identity,
        "source_closure_sha256": gate.oracle_source_closure_sha256,
        "source_closure_count": 36,
        "build_identity_sha256": "1" * 64,
        "compiler": {
            "command": "cc", "version": "cc fixture", "target": "x86_64",
            "executable_sha256": "2" * 64,
        },
        "archiver": {
            "command": "ar", "version": "ar fixture",
            "executable_sha256": "3" * 64,
        },
        "build": {"cflags": "-O2", "ldflags": "-lm"},
    }


def cm_identity(gate: authority.B1AuthorityGate, map_sha: str) -> dict:
    physics = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=cm;tool_identity="
            f"{gate.oracle_tool_identity};map={map_sha}"
        ).encode()
    ).hexdigest()
    return {
        "ok": True,
        "id": "b1-authority",
        "op": "identity",
        "schema": "q2-cm-oracle-v1",
        "tool_identity": gate.oracle_tool_identity,
        "physics_identity": physics,
        "map_sha256": map_sha,
        "map_checksum": 123,
        "provenance": provenance(gate),
        "source": {
            "collision_sha256": gate.collision_source_sha256,
            "shared_header_sha256": gate.shared_header_sha256,
            "shared_source_sha256": gate.shared_source_sha256,
        },
        "map": "/a/relocated/map.bsp",
        "model0": {"mins": [-1, -2, -3], "maxs": [4, 5, 6], "headnode": 0},
        "clusters": 1,
        "inline_models": 2,
    }


def pmove_identity(gate: authority.B1AuthorityGate, map_sha: str) -> dict:
    constants = "fixture-pmove-constants"
    physics = hashlib.sha256(
        (
            "schema=q2-physics-oracle-v1;kind=pmove;tool_identity="
            f"{gate.oracle_tool_identity};map={map_sha};gravity=800;"
            f"airaccelerate=0;constants={constants}"
        ).encode()
    ).hexdigest()
    return {
        "ok": True,
        "id": "b1-authority",
        "op": "identity",
        "schema": "q2-pmove-oracle-v1",
        "tool_identity": gate.oracle_tool_identity,
        "physics_identity": physics,
        "map_sha256": map_sha,
        "map_checksum": 123,
        "parameters": {"gravity": 800, "airaccelerate": 0, "constants": constants},
        "provenance": provenance(gate),
        "source": {
            "collision_sha256": gate.collision_source_sha256,
            "pmove_sha256": gate.pmove_source_sha256,
            "shared_header_sha256": gate.shared_header_sha256,
            "shared_source_sha256": gate.shared_source_sha256,
        },
    }


def fall_identity(gate: authority.B1AuthorityGate) -> dict:
    constants = (
        "player_model=255,noclip=1,grapple_fly=0,release_grace=0.2,"
        "delta_scale=0.0001,water1=0.5,water2=0.25,water3=suppress,"
        "footstep=1,short=15,damage=30,far=55,fall_value_scale=0.5,"
        "fall_value_max=40,fall_time=0.3,damage_divisor=2,df_no_falling=8"
    )
    source = {
        "shared_c_sha256": gate.fall_shared_source_sha256,
        "shared_h_sha256": gate.fall_shared_header_sha256,
        "integration_sha256": gate.fall_integration_sha256,
        "game_header_sha256": "da27f13498fb7120b037b2a6b6ce0a36f4e90a90d1caf0c09c7aaeb1c8310877",
        "constants_sha256": gate.fall_constants_sha256,
        "build_contract": "lithium-linux-c99-o1-f32-shared-fall-v1",
        "tool_closure_sha256": gate.fall_tool_identity,
    }
    parameters = {"fall_damagemod": 1, "deathmatch": True, "dmflags": 0}
    return {
        "ok": True,
        "id": "b1-authority",
        "op": "identity",
        "schema": gate.fall_schema,
        "physics_identity": authority._fall_physics_identity(
            parameters, source, constants
        ),
        "tool_identity": gate.fall_tool_identity,
        "parameters": parameters,
        "constants": constants,
        "source": source,
    }


def test_historical_gate_is_not_current_but_fresh_gate_is_admitted(
    tmp_path: Path,
) -> None:
    with pytest.raises(authority.B1AuthorityError):
        authority.load_b1_authority_gate(ROOT)

    root = copied_authority_root(tmp_path)
    gate = authority.load_b1_authority_gate(root)
    admitted = authority.admit_hook_parity_attestation(ATTESTATION, repo_root=root)

    assert gate.design_sha256 == authority.ACCEPTED_DESIGN_SHA256
    assert gate.plan_sha256 == authority.ACCEPTED_PLAN_SHA256
    assert admitted.attestation_sha256 == gate.hook_attestation_sha256
    assert admitted.fixture_bsp_sha256 == gate.fixture_bsp_sha256
    assert admitted.hook_pullspeed == 1700.0


def test_real_cm_finite_json_bounds_are_admitted_but_nonfinite_are_not() -> None:
    gate = historical_gate()
    map_sha = "a" * 64
    record = cm_identity(gate, map_sha)
    record["model0"]["mins"] = [-4096.0, -2048.0, -128.0]
    record["model0"]["maxs"] = [4096.0, 2048.0, 1024.0]
    authority._validate_cm_identity(record, gate, map_sha)

    record["model0"]["mins"][0] = -4095.5
    authority._validate_cm_identity(record, gate, map_sha)

    record["model0"]["mins"][0] = float("nan")
    with pytest.raises(authority.B1AuthorityError, match="finite"):
        authority._validate_cm_identity(record, gate, map_sha)


@pytest.mark.parametrize("change", ["schema", "batch", "status", "green", "extra"])
def test_gate_rejects_non_green_or_malformed_claims(tmp_path: Path, change: str) -> None:
    root = copied_authority_root(tmp_path)
    gate_path = root / authority.GATE_RELATIVE_PATH
    document = json.loads(gate_path.read_text())
    if change == "schema":
        document["schema"] = "self-declared-passed-v1"
    elif change == "batch":
        document["batch"] = "B2"
    elif change == "status":
        document["status"] = "passed"
    elif change == "green":
        document["gate"]["green"] = "true"
    else:
        document["self_declared_passed"] = True
    gate_path.write_text(json.dumps(document))

    with pytest.raises(authority.B1AuthorityError):
        authority.load_b1_authority_gate(root)


def test_gate_rejects_changed_normative_document(tmp_path: Path) -> None:
    root = copied_authority_root(tmp_path)
    with (root / authority.DESIGN_RELATIVE_PATH).open("a") as handle:
        handle.write("\nreplacement text\n")

    with pytest.raises(authority.B1AuthorityError, match="design bytes changed"):
        authority.load_b1_authority_gate(root)


@pytest.mark.parametrize("change", ["historical", "binary", "pmove", "check"])
def test_gate_rejects_forged_requalification(
    tmp_path: Path, change: str
) -> None:
    root = copied_authority_root(tmp_path)
    gate_path = root / authority.GATE_RELATIVE_PATH
    document = json.loads(gate_path.read_text())
    record = document["authority_requalification"]
    if change == "historical":
        record["historical_gate_sha256"] = "0" * 64
    elif change == "binary":
        record["inputs"]["executables"]["cm_sha256"] = "0" * 64
    elif change == "pmove":
        record["live_identities"]["pmove"]["parameters"]["gravity"] = 799
    else:
        record["checks"]["live_identity_preimages_validated"] = 1
    gate_path.write_text(json.dumps(document))

    with pytest.raises(authority.B1AuthorityError):
        authority.load_b1_authority_gate(root)


def test_recorded_absolute_artifact_path_is_not_authority(tmp_path: Path) -> None:
    root = copied_authority_root(tmp_path)
    gate_path = root / authority.GATE_RELATIVE_PATH
    document = json.loads(gate_path.read_text())
    document["artifacts"]["hook_parity_attestation"]["path"] = (
        "/does/not/exist/and/must/not/be-opened.json"
    )
    gate_path.write_text(json.dumps(document))

    admitted = authority.admit_hook_parity_attestation(
        ATTESTATION, repo_root=root
    )
    assert admitted.attestation_sha256 == document["artifacts"][
        "hook_parity_attestation"
    ]["sha256"]


def test_self_declared_passed_attestation_does_not_override_byte_digest(
    tmp_path: Path,
) -> None:
    document = json.loads(ATTESTATION.read_text())
    document["evidence"]["vector_results_sha256"] = "0" * 64
    document["passed"] = True
    candidate = tmp_path / "self-declared.json"
    candidate.write_bytes(authority._canonical_json_bytes(document))
    root = copied_authority_root(tmp_path / "authority-root")

    with pytest.raises(authority.B1AuthorityError, match="not the B1 artifact"):
        authority.admit_hook_parity_attestation(candidate, repo_root=root)


def test_attestation_semantics_reject_unknown_fields_even_with_rebound_digest() -> None:
    gate = historical_gate()
    document = json.loads(ATTESTATION.read_text())
    document["unexpected"] = True
    data = authority._canonical_json_bytes(document)
    rebound = replace(gate, hook_attestation_sha256=hashlib.sha256(data).hexdigest())

    with pytest.raises(authority.B1AuthorityError, match="fields differ"):
        authority._validate_hook_attestation_record(
            document, rebound, hashlib.sha256(data).hexdigest()
        )


def test_attestation_semantics_reject_case_and_check_tampering() -> None:
    gate = historical_gate()
    for mutation in ("case", "check"):
        document = json.loads(ATTESTATION.read_text())
        if mutation == "case":
            document["evidence"]["case_ids"][0] = "forged-case"
        else:
            document["checks"]["collision_parity"] = 1
        data = authority._canonical_json_bytes(document)
        rebound = replace(
            gate, hook_attestation_sha256=hashlib.sha256(data).hexdigest()
        )
        with pytest.raises(authority.B1AuthorityError):
            authority._validate_hook_attestation_record(
                document, rebound, hashlib.sha256(data).hexdigest()
            )


def test_cm_and_pmove_bind_actual_map_tool_source_and_canonical_physics() -> None:
    gate = historical_gate()
    map_sha = hashlib.sha256(b"analysis-bsp").hexdigest()
    authority._validate_cm_identity(cm_identity(gate, map_sha), gate, map_sha)
    authority._validate_pmove_identity(pmove_identity(gate, map_sha), gate, map_sha)

    stale = cm_identity(gate, map_sha)
    stale["source"]["collision_sha256"] = "0" * 64
    with pytest.raises(authority.B1AuthorityError, match="source is stale"):
        authority._validate_cm_identity(stale, gate, map_sha)

    forged = pmove_identity(gate, map_sha)
    forged["physics_identity"] = "0" * 64
    with pytest.raises(authority.B1AuthorityError, match="not canonical"):
        authority._validate_pmove_identity(forged, gate, map_sha)


def test_hook_runtime_identity_must_equal_attestation() -> None:
    gate = historical_gate()
    parity, document = authority._load_hook_parity_attestation(ATTESTATION, gate)
    hook = deepcopy(document["identities"]["hook"])
    record = {
        "ok": True,
        "id": "b1-authority",
        "op": "identity",
        "schema": hook["schema"],
        "physics_identity": hook["physics_identity"],
        "tool_identity": hook["tool_identity"],
        "parameters": deepcopy(document["parameters"]),
        "source": deepcopy(hook["source"]),
    }
    authority._validate_hook_identity(record, gate, parity, document)

    record["parameters"]["hook_pullspeed"] = 700
    with pytest.raises(authority.B1AuthorityError):
        authority._validate_hook_identity(record, gate, parity, document)


def test_fall_runtime_binds_executable_tool_source_and_default_physics() -> None:
    gate = historical_gate()
    record = fall_identity(gate)
    assert record["physics_identity"] == gate.fall_default_physics_identity
    authority._validate_fall_identity(record, gate)

    stale = deepcopy(record)
    stale["source"]["integration_sha256"] = "0" * 64
    with pytest.raises(authority.B1AuthorityError, match="adapter is stale"):
        authority._validate_fall_identity(stale, gate)


def test_stale_executable_bytes_are_rejected_before_identity_probe(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "oracle"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    with pytest.raises(authority.B1AuthorityError, match="bytes differ"):
        authority._admit_executable(executable, "0" * 64, "test oracle")
