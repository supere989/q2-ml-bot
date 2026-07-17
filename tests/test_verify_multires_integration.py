"""Unit tests for the fail-closed multires integration gate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from harness.multires_contract import (
    ACTION_DIM,
    DYN_DIM,
    FACTUAL_DIM,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    GUIDE_DIM,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
    RECOVERY_DIM,
)
from harness.multires_lineage import CHECKPOINT_FORMAT
from harness.multires_runtime import (
    B4_ACTION_MAGIC,
    B4_CAUSAL_VERSION,
    B4_CLIENT_WIRE_VERSION,
    B4_OBSERVATION_MAGIC,
    B4_PROTOCOL_GENERATION,
    B4_ROLLOUT_SCHEMA,
    B4_TEACHER_VERSION,
    adapt_b4_observation_descriptor,
    validate_runtime_evidence,
)
from tools.verify_multires_integration import (
    GATE_ORDER,
    REQUIRED_PRIVATE_CAUSAL_FACTS,
    canonical_report_bytes,
    run_gates,
)

ATLAS_SHA = hashlib.sha256(b"synthetic-bundle-v3-atlas").hexdigest()
OBJECTIVE_SHA = hashlib.sha256(b"synthetic-objective-identity").hexdigest()
TRAJECTORY_SHA = hashlib.sha256(b"synthetic-500-transition-trajectory").hexdigest()
RETIREMENT_SHA = hashlib.sha256(b"synthetic-retirement-manifest").hexdigest()

ACTION_CARDINALITIES = {
    "vertical_intent": POSTURE_CLASSES,
    "fire": 2,
    "hook": 4,
    "weapon": 10,
}


def _b4_descriptor() -> dict:
    return {
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "factual_dim": FACTUAL_DIM,
        "dyn_dim": DYN_DIM,
        "recovery_dim": RECOVERY_DIM,
        "objective_count": 4,
        "objective_dim": 15,
        "total_dim": OBS_DIM,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_telemetry_schema": B4_ROLLOUT_SCHEMA,
        "logical_action_dim": ACTION_DIM,
        "action_cardinalities": ACTION_CARDINALITIES,
        "teacher_privileged_packing": "physically-separate-qm3c-v2",
        "causal_magic": 0x514D3343,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": 80,
    }


def _runtime_manifest_sha256() -> str:
    evidence = adapt_b4_observation_descriptor(
        _b4_descriptor(), atlas_sha256=ATLAS_SHA, teacher_field_violations=0
    )
    return validate_runtime_evidence(
        evidence, expected_atlas_sha256=ATLAS_SHA
    ).runtime_manifest_sha256


def _guide_vector() -> list[float]:
    # Candidate 0 carries a live health objective; the rest are empty slots.
    candidate = [0.4, -0.1, 0.2, 1.75, 0.25, 0.9, 1.0, 0, 0, 1, 0, 0, 0, 0, 0]
    return [float(v) for v in candidate] + [0.0] * (GUIDE_DIM - 15)


def _synthetic_evidence() -> dict[str, dict]:
    fencing_client = lambda index: {
        "client_id": f"ml-client-{index}",
        "map_epoch": 3,
        "map_name": "mlstage_0001",
        "atlas_sha256": ATLAS_SHA,
    }
    return {
        "feature_action_contract": {
            "policy_generation": POLICY_GENERATION,
            "feature_schema": FEATURE_SCHEMA,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "observation_dim": OBS_DIM,
            "factual_dim": FACTUAL_DIM,
            "dyn_dim": DYN_DIM,
            "recovery_dim": RECOVERY_DIM,
            "guide_dim": GUIDE_DIM,
            "logical_action_dim": ACTION_DIM,
            "posture_classes": POSTURE_CLASSES,
            "action_cardinalities": ACTION_CARDINALITIES,
        },
        "bundle_v3_atlas": {
            "bundle_version": 3,
            "artifact_state": "admitted",
            "atlas_sha256": ATLAS_SHA,
            "objective_identity_sha256": OBJECTIVE_SHA,
            "objective_fixture": {
                "map_name": "mlstage_0001",
                "objective_classes_present": ["health", "weapon"],
                "guide_vector": _guide_vector(),
            },
        },
        "runtime_epoch_fencing": {
            "clients": [fencing_client(index) for index in range(4)],
            "actions_dispatched_during_epoch_barrier": 0,
            "stale_epoch_transitions_admitted": 0,
        },
        "b4_wire_generation": {
            "descriptor": _b4_descriptor(),
            "atlas_sha256": ATLAS_SHA,
            "public_teacher_field_violations": 0,
            "private_causal_facts": list(REQUIRED_PRIVATE_CAUSAL_FACTS),
        },
        "causal_reward_admission": {
            "frames": [
                {
                    "tick": 10,
                    "client_life_epoch": 1,
                    "authoritative_echo_valid": True,
                    "trainable_transition": True,
                    "action_generation": 1,
                    "target_id": 5,
                    "target_epoch": 1,
                    "actionable_exposure": True,
                    "post_command_aligned": True,
                },
                {
                    "tick": 11,
                    "client_life_epoch": 1,
                    "authoritative_echo_valid": True,
                    "trainable_transition": True,
                    "action_generation": 1,
                    "hazard_component_id": 7,
                    "hazard_component_epoch": 1,
                    "environmental_hazard_evidence": True,
                    "cost_to_safety": 2.5,
                },
            ],
        },
        "lineage_attestation": {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "policy_generation": POLICY_GENERATION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "atlas_sha256": ATLAS_SHA,
            "runtime_manifest_sha256": _runtime_manifest_sha256(),
            "initialization": "random",
            "legacy_resume_paths": [],
        },
        "legacy_selector_deactivation": {
            "retirement_manifest_sha256": RETIREMENT_SHA,
            "selectors": [
                {
                    "name": "public_network_thermal_bc_live_v2",
                    "legacy": True,
                    "active": False,
                },
                {
                    "name": "public_network_multires_atlas_fresh_v1",
                    "legacy": False,
                    "active": True,
                },
            ],
            "active_wire_versions": [B4_CLIENT_WIRE_VERSION],
            "active_observation_magics": [B4_OBSERVATION_MAGIC],
            "active_rollout_schemas": [B4_ROLLOUT_SCHEMA],
        },
        "deterministic_transitions": {
            "transition_count": 500,
            "runs": [
                {
                    "launch_id": "same_seed_run_a",
                    "stack": "multires-stack",
                    "fresh_subprocess": True,
                    "transition_count": 500,
                    "trajectory_sha256": TRAJECTORY_SHA,
                    "seed": 17,
                    "game_seed": 19,
                    "collector": "MultiresSynchronousCollector",
                    "spatial_provider": "RustAtlasSpatialProvider",
                    "lattice_crate": "q2_lattice",
                    "partial_admissions": 0,
                    "stale_admissions": 0,
                    "resync_admissions": 0,
                },
                {
                    "launch_id": "same_seed_run_b",
                    "stack": "multires-stack",
                    "fresh_subprocess": True,
                    "transition_count": 500,
                    "trajectory_sha256": TRAJECTORY_SHA,
                    "seed": 17,
                    "game_seed": 19,
                    "collector": "MultiresSynchronousCollector",
                    "spatial_provider": "RustAtlasSpatialProvider",
                    "lattice_crate": "q2_lattice",
                    "partial_admissions": 0,
                    "stale_admissions": 0,
                    "resync_admissions": 0,
                },
            ],
            "divergence_run": {
                "launch_id": "divergence_game_seed",
                "stack": "multires-stack",
                "fresh_subprocess": True,
                "transition_count": 500,
                "trajectory_sha256": hashlib.sha256(
                    b"synthetic-divergence-trajectory"
                ).hexdigest(),
                "seed": 17,
                "game_seed": 20,
                "collector": "MultiresSynchronousCollector",
                "spatial_provider": "RustAtlasSpatialProvider",
                "lattice_crate": "q2_lattice",
                "partial_admissions": 0,
                "stale_admissions": 0,
                "resync_admissions": 0,
            },
        },
        "wsl_b6_campaign": {
            "host_platform": "wsl-ubuntu-22.04",
            "g0_identity_pass": True,
            "g1_transport_pass": True,
            "atlas_sha256": ATLAS_SHA,
            "accepted_transitions": 16384,
            "failed_rounds": 0,
            "echo_timeouts": 0,
            "feature_assembly_p99_ms": 0.31,
            "resident_atlas_bytes": 20 * 1024 * 1024,
            "dyn_payload_bytes_combined": 1024 * 1024,
            "public_services_modified": False,
            "orphan_processes_after_teardown": 0,
        },
    }


def _write_envelope(root: Path, evidence: dict[str, dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for gate, payload in evidence.items():
        path = root / f"{gate}.json"
        path.write_text(json.dumps(payload, sort_keys=True))
        paths[gate] = path.name
    envelope = root / "envelope.json"
    envelope.write_text(json.dumps({
        "schema": "multires-integration-evidence-v1",
        "evidence": paths,
    }, sort_keys=True))
    return envelope


def _statuses(report: dict) -> dict[str, str]:
    return {entry["gate"]: entry["status"] for entry in report["gates"]}


def _reasons(report: dict, gate: str) -> str:
    return " ".join(
        entry["reasons"][0] for entry in report["gates"]
        if entry["gate"] == gate and entry["reasons"]
    )


def test_fully_synthetic_passing_envelope(tmp_path):
    envelope = _write_envelope(tmp_path / "a", _synthetic_evidence())
    report = run_gates(envelope)
    assert report["overall"] == "pass", report["failed_gates"]
    assert report["failed_gates"] == []
    assert [entry["gate"] for entry in report["gates"]] == list(GATE_ORDER)
    assert [entry["rank"] for entry in report["gates"]] == list(
        range(1, len(GATE_ORDER) + 1)
    )
    assert all(entry["evidence_sha256"] for entry in report["gates"])


def test_canonical_determinism_across_runs_and_locations(tmp_path):
    envelope_a = _write_envelope(tmp_path / "a", _synthetic_evidence())
    first = run_gates(envelope_a)
    second = run_gates(envelope_a)
    assert canonical_report_bytes(first) == canonical_report_bytes(second)
    assert first["report_sha256"] == second["report_sha256"]
    # Location independence: byte-identical evidence in another directory.
    moved = tmp_path / "elsewhere" / "nested"
    shutil.copytree(envelope_a.parent, moved)
    third = run_gates(moved / "envelope.json")
    assert third["report_sha256"] == first["report_sha256"]
    # The digest is over the canonical body without the digest field itself.
    body = dict(first)
    del body["report_sha256"]
    recomputed = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    assert recomputed == first["report_sha256"]


def test_missing_evidence_fails_but_all_gates_still_report(tmp_path):
    evidence = _synthetic_evidence()
    envelope = _write_envelope(tmp_path, evidence)
    (tmp_path / "bundle_v3_atlas.json").unlink()
    mapping = json.loads(envelope.read_text())
    del mapping["evidence"]["wsl_b6_campaign"]
    envelope.write_text(json.dumps(mapping, sort_keys=True))
    report = run_gates(envelope)
    statuses = _statuses(report)
    assert report["overall"] == "fail"
    assert statuses["bundle_v3_atlas"] == "fail"
    assert statuses["wsl_b6_campaign"] == "fail"
    # Every other gate is still evaluated in the same run.
    assert statuses["feature_action_contract"] == "pass"
    assert statuses["deterministic_transitions"] == "pass"
    assert len(report["gates"]) == len(GATE_ORDER)


def test_mixed_epochs_and_mixed_digests_fail(tmp_path):
    evidence = _synthetic_evidence()
    evidence["runtime_epoch_fencing"]["clients"][2]["map_epoch"] = 4
    evidence["lineage_attestation"]["atlas_sha256"] = hashlib.sha256(
        b"some-other-atlas"
    ).hexdigest()
    report = run_gates(_write_envelope(tmp_path, evidence))
    statuses = _statuses(report)
    assert statuses["runtime_epoch_fencing"] == "fail"
    assert "mixed client map epochs" in _reasons(report, "runtime_epoch_fencing")
    assert statuses["lineage_attestation"] == "fail"
    assert "disagrees" in _reasons(report, "lineage_attestation")


def test_universal_zero_guide_and_placeholder_evidence_fail(tmp_path):
    evidence = _synthetic_evidence()
    fixture = evidence["bundle_v3_atlas"]["objective_fixture"]
    fixture["guide_vector"] = [0.0] * GUIDE_DIM
    evidence["deterministic_transitions"]["runs"][0]["trajectory_sha256"] = "0" * 64
    evidence["wsl_b6_campaign"]["host_platform"] = "TBD"
    report = run_gates(_write_envelope(tmp_path, evidence))
    statuses = _statuses(report)
    assert statuses["bundle_v3_atlas"] == "fail"
    assert "universal-zero" in _reasons(report, "bundle_v3_atlas")
    assert statuses["deterministic_transitions"] == "fail"
    assert statuses["wsl_b6_campaign"] == "fail"
    assert "placeholder" in _reasons(report, "wsl_b6_campaign")


def test_zero_p99_latency_is_placeholder_not_pass(tmp_path):
    evidence = _synthetic_evidence()
    evidence["wsl_b6_campaign"]["feature_assembly_p99_ms"] = 0.0
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["wsl_b6_campaign"] == "fail"
    assert "positive measurement" in _reasons(report, "wsl_b6_campaign")


def test_missing_private_causal_facts_fail(tmp_path):
    evidence = _synthetic_evidence()
    facts = evidence["b4_wire_generation"]["private_causal_facts"]
    facts.remove("causal_flags")
    facts.remove("action_generation")
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["b4_wire_generation"] == "fail"
    reason = _reasons(report, "b4_wire_generation")
    assert "causal_flags" in reason and "action_generation" in reason


def test_legacy_selection_fails(tmp_path):
    evidence = _synthetic_evidence()
    evidence["legacy_selector_deactivation"]["selectors"][0]["active"] = True
    evidence["legacy_selector_deactivation"]["active_wire_versions"].append(4)
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["legacy_selector_deactivation"] == "fail"
    assert "still active" in _reasons(report, "legacy_selector_deactivation")

    evidence = _synthetic_evidence()
    evidence["legacy_selector_deactivation"]["active_wire_versions"] = [4]
    report = run_gates(_write_envelope(tmp_path / "wire", evidence))
    assert _statuses(report)["legacy_selector_deactivation"] == "fail"
    assert "non-B4 generation" in _reasons(report, "legacy_selector_deactivation")


def test_legacy_wire_descriptor_rejected(tmp_path):
    evidence = _synthetic_evidence()
    descriptor = evidence["b4_wire_generation"]["descriptor"]
    descriptor["client_wire_version"] = 4
    descriptor["observation_magic"] = 0x514D4C50
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["b4_wire_generation"] == "fail"


def test_deterministic_evidence_requires_two_fresh_full_stack_runs(tmp_path):
    evidence = _synthetic_evidence()
    evidence["deterministic_transitions"]["transition_count"] = 499
    report = run_gates(_write_envelope(tmp_path / "count", evidence))
    assert _statuses(report)["deterministic_transitions"] == "fail"

    evidence = _synthetic_evidence()
    evidence["deterministic_transitions"]["runs"][1]["launch_id"] = (
        "same_seed_run_a"
    )
    report = run_gates(_write_envelope(tmp_path / "launch", evidence))
    assert _statuses(report)["deterministic_transitions"] == "fail"
    assert "launch identity" in _reasons(report, "deterministic_transitions")

    evidence = _synthetic_evidence()
    evidence["deterministic_transitions"]["runs"][1]["trajectory_sha256"] = (
        hashlib.sha256(b"diverged").hexdigest()
    )
    report = run_gates(_write_envelope(tmp_path / "hash", evidence))
    assert _statuses(report)["deterministic_transitions"] == "fail"
    assert "diverge" in _reasons(report, "deterministic_transitions")

    evidence = _synthetic_evidence()
    evidence["deterministic_transitions"]["divergence_run"][
        "trajectory_sha256"
    ] = TRAJECTORY_SHA
    report = run_gates(_write_envelope(tmp_path / "no-divergence", evidence))
    assert _statuses(report)["deterministic_transitions"] == "fail"
    assert "did not change" in _reasons(report, "deterministic_transitions")


def test_inadmissible_reward_frames_fail(tmp_path):
    evidence = _synthetic_evidence()
    evidence["causal_reward_admission"]["frames"] = []
    report = run_gates(_write_envelope(tmp_path / "empty", evidence))
    assert _statuses(report)["causal_reward_admission"] == "fail"

    evidence = _synthetic_evidence()
    evidence["causal_reward_admission"]["frames"][0]["trainable_transition"] = False
    report = run_gates(_write_envelope(tmp_path / "resync", evidence))
    assert _statuses(report)["causal_reward_admission"] == "fail"

    evidence = _synthetic_evidence()
    evidence["causal_reward_admission"]["frames"][0]["teacher_only_health"] = 1.0
    report = run_gates(_write_envelope(tmp_path / "unknown", evidence))
    assert _statuses(report)["causal_reward_admission"] == "fail"


def test_nonadmitted_bundle_state_fails(tmp_path):
    evidence = _synthetic_evidence()
    evidence["bundle_v3_atlas"]["artifact_state"] = "published"
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["bundle_v3_atlas"] == "fail"
    assert "non-admissible" in _reasons(report, "bundle_v3_atlas")


def test_unreadable_envelope_fails_every_gate(tmp_path):
    envelope = tmp_path / "envelope.json"
    envelope.write_text("{not json")
    report = run_gates(envelope)
    assert report["overall"] == "fail"
    assert all(entry["status"] == "fail" for entry in report["gates"])
    assert len(report["gates"]) == len(GATE_ORDER)
