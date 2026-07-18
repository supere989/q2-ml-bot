"""Unit tests for the fail-closed multires integration gate."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest
import tools.verify_multires_integration as verifier

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
from harness import client_protocol as public_wire
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


def _public_privilege_separation() -> dict:
    packet = bytearray(public_wire.TEACHER_SAMPLE_SIZE)
    struct.pack_into(
        "<III", packet, 0, public_wire.ML_TEACHER_MAGIC,
        public_wire.ML_TEACHER_VERSION, public_wire.TEACHER_SAMPLE_SIZE,
    )
    clients = [
        {
            "client_id": f"qual-{slot:02d}",
            "datagrams_seen": 33,
            "public_packets_decoded": 33,
            "routed_packets_accepted": 33,
            "malformed_packets_rejected": 0,
            "foreign_client_packets_rejected": 0,
            "stale_packets_rejected": 0,
            "teacher_packets_detected": 0,
        }
        for slot in range(4)
    ]
    return {
        "source_abi": {
            "public": {
                "magic": public_wire.ML_CLIENT_TELEM_MAGIC,
                "packet_bytes": public_wire.CLIENT_TELEMETRY_SIZE,
                "pod": "ml_client_telemetry_t",
                "packing_path": "ml_client_telemetry.c:ML_ClientTelemetryFrame",
                "fields": ["public_header", "ml_obs_t", "ml_causal_telemetry_t"],
                "teacher_action_tail_present": False,
            },
            "teacher": {
                "magic": public_wire.ML_TEACHER_MAGIC,
                "version": public_wire.ML_TEACHER_VERSION,
                "packet_bytes": public_wire.TEACHER_SAMPLE_SIZE,
                "pod": "ml_teacher_sample_t",
                "packing_path": "ml_bridge.c:ML_TeacherSend",
                "fields": [
                    "teacher_header", "ml_obs_t", "ml_causal_telemetry_t",
                    "ml_action_t",
                ],
            },
            "yamagi_teacher_identity_present": False,
            "qualification_teacher_enabled": False,
            "public_and_teacher_magic_distinct": True,
            "public_and_teacher_size_distinct": True,
        },
        "normal_run_audit": {
            "derivation": "sealed-baseline-cold-1-public-datagram-accounting-v1",
            "scenario_proof": {
                "raw_evidence_sha256": hashlib.sha256(b"baseline-raw").hexdigest(),
            },
            "clients": clients,
            "totals": {
                "datagrams_seen": 132,
                "public_packets_decoded": 132,
                "routed_packets_accepted": 132,
                "malformed_packets_rejected": 0,
                "foreign_client_packets_rejected": 0,
                "stale_packets_rejected": 0,
                "teacher_packets_detected": 0,
            },
        },
        "negative_probe": {
            "injection": "exact-teacher-magic-version-size-zero-body",
            "packet_magic": public_wire.ML_TEACHER_MAGIC,
            "packet_version": public_wire.ML_TEACHER_VERSION,
            "packet_bytes": len(packet),
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
            "public_parser_result": "fatal-public-privilege-violation",
        },
    }


def _guide_vector() -> list[float]:
    # Candidate 0 carries a live health objective; the rest are empty slots.
    candidate = [0.4, -0.1, 0.2, 1.75, 0.25, 0.9, 1.0, 0, 0, 1, 0, 0, 0, 0, 0]
    return [float(v) for v in candidate] + [0.0] * (GUIDE_DIM - 15)


def _b6_evidence() -> dict:
    digest = lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest()
    record = lambda name: {"bytes": len(name) + 1, "sha256": digest(name)}
    checkpoint = record("checkpoint")
    state_sha = digest("public-state")
    authority_path = (
        Path(__file__).resolve().parents[1]
        / "docs/multires/B6-PUBLIC-HOST-AUTHORITY.json"
    )
    authority_bytes = authority_path.read_bytes()
    authority = json.loads(authority_bytes)
    value = {
        "schema": "q2-multires-b6-wsl-g1-v1",
        "tool": "assemble_b6_wsl_g1_campaign",
        "status": "green",
        "host": {
            "hostname": "DESKTOP-RTX2080",
            "kernel_release": "6.6.0-microsoft-standard-WSL2",
            "architecture": "x86_64",
            "machine_identity_sha256": digest("wsl-machine"),
        },
        "source_repositories": {},
        "bindings": {
            **{
                name: record(name) for name in (
                    "runtime_manifest", "training_manifest", "objectives",
                    "bundle_manifest", "atlas", "atlas_manifest", "b2_gate",
                    "b3_gate", "b4_evidence", "b5_gate", "lineage_evidence",
                    "retirement_evidence",
                )
            },
            "checkpoint": checkpoint,
            "retirement_manifest": {"bytes": 10, "sha256": RETIREMENT_SHA},
            "atlas_sha256": ATLAS_SHA,
            "runtime_manifest_identity_sha256": _runtime_manifest_sha256(),
            "training_config_sha256": digest("training-config"),
            "objective_identity_sha256": OBJECTIVE_SHA,
            "network_barrier_execution_evidence_sha256": digest("barrier-execution"),
            "b3_gate_sha256": digest("b3-gate-seal"),
            "b4_evidence_sha256": digest("b4-seal"),
            "b5_gate_sha256": digest("b5-seal"),
            "hook_necessity_runtime": {
                "hook_walk_budget_ticks": 15,
                "game_tick_hz": 10,
                "walk_speed_q8_per_second": 76800,
            },
            "rust_extension": {
                "path": "/isolated/q2_lattice_rs.so", "bytes": 4096,
                "sha256": digest("rust-extension"),
                "repository_tree": "a" * 40,
                "source_closure_sha256": digest("rust-source"),
                "qualification_commands_sha256": digest("rust-tests"),
            },
        },
        "raw_evidence": {
            **{
                name: record(name) for name in (
                    "one_run", "fault_probe", "fault_execution",
                    "public_pre_probe", "public_post_probe",
                    "controller_ledger", "controller_plan"
                )
            },
            "one_run_evidence_sha256": digest("one-run-seal"),
            "fault_probe_evidence_sha256": digest("fault-seal"),
            "controller_ledger_sha256": digest("controller-ledger-seal"),
        },
        "g1": {
            "accepted_transitions": 16384,
            "echo_attempts": 16384,
            "authoritative_echo_accept_rate": 1.0,
            "vertical_samples": 16384,
            "vertical_matches": 16384,
            "vertical_match_rate": 1.0,
            "water_samples": 8192,
            "land_samples": 8192,
            "water_projection_mismatches": 0,
            "land_projection_mismatches": 0,
            "water_land_projection_skew": 0,
            "failed_rounds": 0,
            "echo_timeouts": 0,
            "declared_resyncs": 2,
            "declared_resync_limit": 4,
            "accepted_frame_discontinuities": [],
            "trajectory_sha256": digest("b6-trajectory"),
            "map_epoch_recovery_exercised": True,
            "telemetry_gap_recovery_exercised": True,
            "partial_client_timeout_fatal": True,
        },
        "no_update": {
            "mode": "collect-only-no-update-v1",
            "checkpoint_sha256_before": checkpoint["sha256"],
            "checkpoint_sha256_after": checkpoint["sha256"],
            "policy_state_sha256_before": digest("policy-state"),
            "policy_state_sha256_after": digest("policy-state"),
            "optimizer_state_sha256_before": digest("optimizer-state"),
            "optimizer_state_sha256_after": digest("optimizer-state"),
            "policy_updates": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "performance": {
            "four_client_feature_assembly_p99_ns": 310000,
            "atlas_resident_bytes": 20 * 1024 * 1024,
            "four_dyn_resident_bytes": 1024 * 1024,
            "atlas_build_peak_rss_bytes": 256 * 1024 * 1024,
            "host": "DESKTOP-RTX2080",
            "kernel_release": "6.6.0-microsoft-standard-WSL2",
            "machine_identity_sha256": digest("wsl-machine"),
            "lan_payload": {
                "basis": "wire-abi-struct-calcsize-v1",
                "accepted_packet_samples": 16384,
                "client_telemetry_bytes": 1248,
                "action_packet_bytes": 28,
                "telemetry_components": {
                    "header_bytes": 112,
                    "engine_observation_bytes": 1056,
                    "causal_telemetry_bytes": 80,
                },
                "atlas_wire_fields": 0,
                "atlas_wire_bytes_per_frame": 0,
                "dyn_wire_fields": 0,
                "dyn_wire_bytes_per_frame": 0,
            },
        },
        "public_state": {
            "host": {
                "hostname": authority["hostname"],
                "kernel_release": "6.12.0",
                "architecture": authority["architecture"],
                "machine_identity_sha256": authority[
                    "machine_identity_sha256"
                ],
            },
            "authority": {
                "bytes": len(authority_bytes),
                "sha256": hashlib.sha256(authority_bytes).hexdigest(),
                "authority_sha256": authority["authority_sha256"],
            },
            "pre_state_sha256": state_sha,
            "post_state_sha256": state_sha,
            "run_nonce": digest("run-nonce"),
            "launch_id": "b6-" + digest("launch")[:60],
            "attestation_key_id": authority["probe_attestation"]["key_id"],
            "pre_evidence_sha256": digest("public-pre-evidence"),
            "post_evidence_sha256": digest("public-post-evidence"),
            "controller_ledger_sha256": digest("controller-ledger-seal"),
            "ordering_basis": "signed-hash-chain-plus-controller-monotonic-ns-v1",
            "modified": False,
        },
        "teardown": {
            "process_records": [
                {"role": "q2ded" if i == 0 else f"network-client-{i-1:02d}",
                 "pid": 100 + i, "start_ticks": 1000 + i}
                for i in range(5)
            ],
            "orphan_processes_after_teardown": 0,
            "owned_ports": [
                {"role": f"socket-{i}", "address": "127.0.0.1", "port": 27000 + i,
                 "transport": "udp", "available_after_teardown": True}
                for i in range(6)
            ],
            "qport_identities": [
                {"client_index": i, "qport": 28000 + i} for i in range(4)
            ],
        },
        "gate_sha256": "",
    }
    value["teardown"]["terminated_process_records"] = value["teardown"]["process_records"]
    body = dict(value)
    body.pop("gate_sha256")
    value["gate_sha256"] = hashlib.sha256(json.dumps(
        {"domain": value["schema"], "gate": body},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return value


def _reseal_b6(value: dict) -> None:
    body = dict(value)
    body.pop("gate_sha256", None)
    value["gate_sha256"] = hashlib.sha256(json.dumps(
        {"domain": value["schema"], "gate": body},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


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
            "public_privilege_separation": _public_privilege_separation(),
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
            "mode": "production",
            "admissible": True,
            "production_pass": True,
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
        "wsl_b6_campaign": _b6_evidence(),
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
    evidence["wsl_b6_campaign"]["host"]["hostname"] = "TBD"
    report = run_gates(_write_envelope(tmp_path, evidence))
    statuses = _statuses(report)
    assert statuses["bundle_v3_atlas"] == "fail"
    assert "universal-zero" in _reasons(report, "bundle_v3_atlas")
    assert statuses["deterministic_transitions"] == "fail"
    assert statuses["wsl_b6_campaign"] == "fail"
    assert "placeholder" in _reasons(report, "wsl_b6_campaign")


def test_zero_p99_latency_is_placeholder_not_pass(tmp_path):
    evidence = _synthetic_evidence()
    evidence["wsl_b6_campaign"]["performance"][
        "four_client_feature_assembly_p99_ns"
    ] = 0
    _reseal_b6(evidence["wsl_b6_campaign"])
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["wsl_b6_campaign"] == "fail"
    assert "zero or out of budget" in _reasons(report, "wsl_b6_campaign")


def test_legacy_b6_summary_booleans_are_forbidden(tmp_path):
    evidence = _synthetic_evidence()
    evidence["wsl_b6_campaign"]["g1_transport_pass"] = True
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["wsl_b6_campaign"] == "fail"
    assert "legacy summary field" in _reasons(report, "wsl_b6_campaign")


@pytest.mark.parametrize("mutation,reason", [
    (lambda b: b["g1"].update(authoritative_echo_accept_rate=0.99), "acceptance derivation"),
    (lambda b: b["no_update"].update(optimizer_state_sha256_after=hashlib.sha256(b"changed").hexdigest()), "optimizer_state changed"),
    (lambda b: b["performance"].update(machine_identity_sha256=hashlib.sha256(b"other-host").hexdigest()), "machine identities differ"),
    (lambda b: b["public_state"].update(post_state_sha256=hashlib.sha256(b"changed-public").hexdigest()), "pre/post state differs"),
    (lambda b: b["bindings"].update(hook_necessity_runtime={"hook_walk_budget_ticks": 14, "game_tick_hz": 10, "walk_speed_q8_per_second": 76800}), "budget/cadence"),
])
def test_b6_recomputed_seal_does_not_hide_semantic_forgery(
    tmp_path, mutation, reason
):
    evidence = _synthetic_evidence()
    mutation(evidence["wsl_b6_campaign"])
    _reseal_b6(evidence["wsl_b6_campaign"])
    report = run_gates(_write_envelope(tmp_path, evidence))
    assert _statuses(report)["wsl_b6_campaign"] == "fail"
    assert reason in _reasons(report, "wsl_b6_campaign")


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


def test_retirement_gate_checks_live_entrypoint_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source-root"
    for relative in (
        "tools/live_round_train.py",
        "tools/live_match_onnx.py",
        "tools/ml_vs_ml.py",
        "tools/behavior_clone_aim.py",
        "train/ppo.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# retired:\nreturn 2\n", encoding="utf-8")
    client = root / "harness/client_env.py"
    client.parent.mkdir(parents=True, exist_ok=True)
    client.write_text(
        "# vector policy input requires an attested multires provider\n"
        "# spatial_seed belongs to the retired legacy reward path\n",
        encoding="utf-8",
    )
    (root / "tools/live_match_onnx.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8"
    )
    monkeypatch.setattr(verifier, "ROOT", root)
    report = run_gates(_write_envelope(tmp_path / "evidence", _synthetic_evidence()))
    assert _statuses(report)["legacy_selector_deactivation"] == "fail"
    assert "still operational" in _reasons(report, "legacy_selector_deactivation")


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
