from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.multires_contract import ACTION_DIM, DYN, OBS_DIM
from tools.run_multires_500_transition_proof import (
    COLLECTOR_CLASS_NAME,
    LATTICE_CRATE_NAME,
    ONE_RUN_PROTOCOL_VERSION,
    ONE_RUN_SCHEMA,
    PYTHON_COLLECTOR_SCHEMA,
    RUST_PROVIDER_SCHEMA,
    SPATIAL_PROVIDER_CLASS_NAME,
    make_transition_record,
    trajectory_sha256,
)
import tools.assemble_b6_wsl_g1_campaign as b6
import tools.assemble_b6_g1_fault_probe as fault_producer
from tools.capture_b6_public_state import _machine_identity_sha256, _start_ticks
from tools.qualify_network_client_frame_barrier import (
    QualificationError,
    _machine_identity_sha256 as _execution_machine_identity_sha256,
)
from train.multires_one_run import (
    OneRunError, _machine_identity_sha256 as _one_run_machine_identity_sha256,
    _owned_udp_ports, _publish_exclusive, _qport_identities,
)


LAUNCH_ID = "b6-" + "a" * 60


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _seal(value: dict, field: str = "evidence_sha256") -> dict:
    value.pop(field, None)
    value[field] = hashlib.sha256(b6.canonical_bytes(value)).hexdigest()
    return value


def _bindings() -> dict:
    checkpoint = {"bytes": 10, "sha256": _digest("checkpoint")}
    return {
        "checkpoint": checkpoint,
        "training_config_sha256": _digest("training"),
        "objective_identity_sha256": _digest("objectives"),
        "bundle_manifest": {"bytes": 10, "sha256": _digest("bundle")},
        "network_barrier_execution_evidence_sha256": _digest("barrier"),
    }


def _one_run(count: int = 8) -> dict:
    atlas = _digest("atlas")
    runtime = _digest("runtime")
    bindings = _bindings()
    records = []
    built = []
    for index in range(count):
        round_id, client_index = divmod(index, 4)
        observation = [0.0] * OBS_DIM
        action = [0.0] * ACTION_DIM
        action[4] = 1.0
        rust = observation[DYN.slice]
        record = make_transition_record(
            index=index, observation=observation, action=action, reward=0.0,
            client_id=f"client-{client_index:02d}", server_frame=100 + round_id,
            batch_round_id=round_id, policy_version=7, map_name="q2dm1",
            map_epoch=3, atlas_sha256=atlas,
            runtime_manifest_sha256=runtime, rust_features=rust,
        )
        item = record.to_mapping()
        item["rust_features"] = list(rust)
        item["transport_audit"] = {
            "authoritative_echo_valid": True,
            "trainable_transition": True,
            "requested_vertical": 1,
            "echoed_vertical": 1,
            "applied_upmove": 0,
            "water_vertical_mode": index % 2 == 0,
        }
        records.append(item)
        built.append(record)
    process_records = [
        {"role": "q2ded" if i == 0 else f"network-client-{i-1:02d}",
         "pid": 100 + i, "start_ticks": 1000 + i}
        for i in range(5)
    ]
    ports = [
        {"role": role, "address": "127.0.0.1", "port": 27000 + i,
         "transport": "udp", "available_after_teardown": True}
        for i, role in enumerate((
            "q2ded", "telemetry", "harness-00", "harness-01",
            "harness-02", "harness-03",
        ))
    ]
    state = _digest("policy")
    optimizer = _digest("optimizer")
    raw = {
        "schema": ONE_RUN_SCHEMA,
        "protocol_version": ONE_RUN_PROTOCOL_VERSION,
        "synthetic": False,
        "legacy": False,
        "started_at_unix_ns": 200,
        "completed_at_unix_ns": 300,
        "collector": COLLECTOR_CLASS_NAME,
        "python_collector_schema": PYTHON_COLLECTOR_SCHEMA,
        "spatial_provider": SPATIAL_PROVIDER_CLASS_NAME,
        "rust_provider_schema": RUST_PROVIDER_SCHEMA,
        "lattice_crate": LATTICE_CRATE_NAME,
        "campaign_mode": b6.B6_CAMPAIGN_MODE,
        "atlas_sha256": atlas,
        "runtime_manifest_sha256": runtime,
        "bundle_manifest_sha256": bindings["bundle_manifest"]["sha256"],
        "checkpoint_sha256": bindings["checkpoint"]["sha256"],
        "training_manifest_sha256": bindings["training_config_sha256"],
        "objective_identity_sha256": bindings["objective_identity_sha256"],
        "retirement_manifest_sha256": _digest("retirement"),
        "network_barrier_execution_evidence_sha256": _digest("barrier"),
        "source_repositories": {"bot": {}},
        "transition_count": count,
        "received_inputs": {
            "campaign_mode": b6.B6_CAMPAIGN_MODE, "transition_count": count,
            "launch_id": LAUNCH_ID,
        },
        "host": {
            "hostname": "DESKTOP-RTX2080",
            "kernel_release": "6.6.0-microsoft-standard-WSL2",
            "architecture": "x86_64",
            "machine_identity_sha256": _digest("machine"),
        },
        "map_name": "q2dm1",
        "map_epoch": 3,
        "policy_version": 7,
        "partial_admissions": 0,
        "stale_admissions": 0,
        "resync_admissions": 2,
        "records": records,
        "trajectory_sha256": trajectory_sha256(tuple(built)),
        "no_update": {
            "mode": "collect-only-no-update-v1",
            "checkpoint_sha256_before": bindings["checkpoint"]["sha256"],
            "checkpoint_sha256_after": bindings["checkpoint"]["sha256"],
            "policy_state_sha256_before": state,
            "policy_state_sha256_after": state,
            "optimizer_state_sha256_before": optimizer,
            "optimizer_state_sha256_after": optimizer,
            "policy_updates": 0, "optimizer_steps": 0, "backward_calls": 0,
        },
        "transport_metrics": {
            "network_client/transitions_accepted": count,
            "network_client/stale_echoes_rejected": 0,
            "network_client/mismatched_echoes_rejected": 0,
            "network_client/failed_rounds": 0,
            "network_client/echo_timeouts": 0,
            "boundary_rounds": 2,
            "declared_resync_limit": 4,
            "protocol_payload_accounting": {
                "basis": "wire-abi-struct-calcsize-v1",
                "accepted_packet_samples": count,
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
        "process_records": process_records,
        "terminated_process_records": process_records,
        "orphan_processes_after_teardown": 0,
        "owned_ports": ports,
        "qport_identities": [
            {"client_index": i, "qport": 28000 + i} for i in range(4)
        ],
    }
    return _seal(raw)


def _validate(raw: dict, monkeypatch):
    monkeypatch.setattr(b6, "REQUIRED_TRANSITIONS", 8)
    return b6._validate_one_run(
        raw, bindings=_bindings(), atlas_sha256=_digest("atlas"),
        runtime_manifest_sha256=_digest("runtime"),
            retirement_sha256=_digest("retirement"),
            expected_sources={"bot": {}},
            expected_launch_id=LAUNCH_ID,
        )


def test_one_run_derives_g1_from_raw_records(monkeypatch):
    result = _validate(_one_run(), monkeypatch)
    assert result["accepted_transitions"] == 8
    assert result["vertical_match_rate"] == 1.0
    assert result["water_samples"] == result["land_samples"] == 4
    assert len(result["owned_ports"]) == 6
    assert len(result["qport_identities"]) == 4


def test_qports_are_protocol_identities_not_owned_udp_sockets():
    config = {
        "client_count": 4, "server_port": 27910, "telemetry_port": 27911,
        "harness_port_base": 27920, "qport_base": 27920,
    }
    sockets = _owned_udp_ports(config)
    assert [item["port"] for item in sockets] == [27910, 27911, 27920, 27921, 27922, 27923]
    assert all(not item["role"].startswith("qport") for item in sockets)
    assert [item["qport"] for item in _qport_identities(config)] == [27920, 27921, 27922, 27923]


def test_one_run_exclusive_publication_refuses_overwrite(tmp_path):
    destination = tmp_path / "one-run.json"
    _publish_exclusive(destination, {"first": True})
    original = destination.read_bytes()
    with pytest.raises(OneRunError, match="already exists"):
        _publish_exclusive(destination, {"second": True})
    assert destination.read_bytes() == original


@pytest.mark.parametrize("mutation,match", [
    (lambda raw: raw["records"].pop(), "exactly 16,384 raw records"),
    (lambda raw: raw["no_update"].update(optimizer_steps=1), "update"),
    (lambda raw: raw["no_update"].update(optimizer_state_sha256_after=_digest("changed")), "changed"),
    (lambda raw: raw["records"][0]["transport_audit"].update(echoed_vertical=2), "vertical match rate"),
    (lambda raw: [item["transport_audit"].update(water_vertical_mode=False) for item in raw["records"]], "both water and land"),
    (lambda raw: raw["transport_metrics"].update(declared_resync_limit=65), "campaign cap"),
])
def test_one_run_rejects_forged_or_incomplete_evidence(monkeypatch, mutation, match):
    raw = _one_run()
    mutation(raw)
    _seal(raw)
    with pytest.raises(b6.B6CampaignError, match=match):
        _validate(raw, monkeypatch)


def _fault(host: dict) -> dict:
    source = {"bytes": 100, "sha256": _digest("raw"),
              "evidence_sha256": _digest("raw-seal")}
    bindings = {"runtime_manifest": {"bytes": 1, "sha256": _digest("runtime-file")}}
    raw = {
        "schema": b6.FAULT_SCHEMA,
        "tool": "assemble_b6_g1_fault_probe",
        "synthetic": False,
        "host": host,
        "source_repositories": {"bot": {}},
        "bindings": bindings,
        "full_network_execution": {
            "bytes": 10, "sha256": _digest("execution"),
            "execution_evidence_sha256": _digest("execution-seal"),
        },
        "scenarios": [
            {
                "name": "map-epoch-recovery", "source_scenario": "epoch-drain",
                "source": source, "observed_outcome": "completed", "epoch_drains": 1,
                "new_epoch_bootstrap_frames": 1,
                "actions_dispatched_during_epoch_drain": 0, "boundary_rounds": 1,
                "accepted_after_recovery": 4, "recovered_map_epoch": 2,
            },
            {
                "name": "whole-batch-telemetry-gap-recovery",
                "source_scenario": "whole-batch-telemetry-gap-recovery",
                "source": source,
                "observed_outcome": "completed", "boundary_rounds": 1,
                "accepted_synchronized_frames": 32, "fault_event_count": 0,
                "accepted_after_recovery": 4,
                "frame_discontinuities": [{
                    "prior_frame": 4, "current_frame": 6, "delta": 2,
                    "trajectory_ordinal": 5,
                }],
                "timeout_client_ids": [f"qual-{slot:02d}" for slot in range(4)],
                "relay_event_count": 4,
                "relay_events_sha256": _digest("gap-relay-events"),
                "transport_metrics": {
                    "telemetry_gap_resyncs": 1, "failed_rounds": 0,
                    "echo_timeouts": 0,
                },
            },
            {
                "name": "partial-client-timeout-fatal",
                "source_scenario": "partial-client-telemetry-timeout",
                "source": source, "observed_outcome": "fatal",
                "fault_event_count": 0, "exception": "AuthoritativeEchoError",
                "accepted_synchronized_frames": 4,
                "timeout_client_ids": ["qual-00"],
                "relay_event_count": 1,
                "relay_events_sha256": _digest("partial-relay-events"),
                "transport_metrics": {
                    "telemetry_gap_resyncs": 0, "failed_rounds": 1,
                    "echo_timeouts": 1,
                },
            },
        ],
    }
    return _seal(raw)


def _bind_fault_execution(tmp_path: Path, raw: dict) -> tuple[dict, Path, dict]:
    source_by_name = {
        item["source_scenario"]: item["source"] for item in raw["scenarios"]
    }
    execution = {
        "test_mode": False,
        "full_network_executed": True,
        "execution_evidence_sha256": _digest("execution-seal"),
        "scenario_evidence": [
            {
                "scenario": name,
                "size": source_by_name[name]["bytes"],
                "sha256": source_by_name[name]["sha256"],
                "raw_evidence_sha256": source_by_name[name]["evidence_sha256"],
            }
            for name in (
                "epoch-drain", "whole-batch-telemetry-gap-recovery",
                "partial-client-telemetry-timeout",
            )
        ],
    }
    path = tmp_path / "frame-barrier-execution.json"
    path.write_bytes(b6.canonical_bytes(execution) + b"\n")
    raw["full_network_execution"] = {
        **b6.file_record(path),
        "execution_evidence_sha256": execution["execution_evidence_sha256"],
    }
    return _seal(raw), path, execution


def test_fault_probe_derives_recovery_from_raw_scenario_bindings(
    tmp_path, monkeypatch,
):
    host = _one_run()["host"]
    raw, execution_path, execution = _bind_fault_execution(
        tmp_path, _fault(host)
    )
    monkeypatch.setattr(
        b6, "validate_network_execution_evidence",
        lambda value, path: execution,
    )
    result = b6._validate_fault_probe(
        raw, expected_bindings={
            "runtime_manifest": {"bytes": 1, "sha256": _digest("runtime-file")}
        }, expected_sources={"bot": {}}, expected_host=host,
        execution_path=execution_path,
    )
    assert all(result.values())


def test_fault_probe_rejects_cross_host_relabel(tmp_path):
    host = _one_run()["host"]
    raw = _fault(host)
    raw["host"] = {**host, "machine_identity_sha256": _digest("other-machine")}
    _seal(raw)
    with pytest.raises(b6.B6CampaignError, match="machine identities"):
        b6._validate_fault_probe(
            raw, expected_bindings=raw["bindings"],
            expected_sources=raw["source_repositories"], expected_host=host,
            execution_path=tmp_path / "not-consulted.json",
        )


def test_fault_probe_rejects_self_sealed_summary_without_exact_execution(
    tmp_path, monkeypatch,
):
    host = _one_run()["host"]
    raw, execution_path, execution = _bind_fault_execution(
        tmp_path, _fault(host)
    )
    raw["full_network_execution"]["sha256"] = _digest("forged-file")
    _seal(raw)
    monkeypatch.setattr(
        b6, "validate_network_execution_evidence",
        lambda value, path: execution,
    )
    with pytest.raises(b6.B6CampaignError, match="does not bind the supplied"):
        b6._validate_fault_probe(
            raw, expected_bindings=raw["bindings"],
            expected_sources=raw["source_repositories"], expected_host=host,
            execution_path=execution_path,
        )




def test_proc_start_ticks_parses_comm_with_spaces(monkeypatch):
    tail = ["S", *("0" for _ in range(18)), "777"]
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: "123 (name with spaces) " + " ".join(tail))
    assert _start_ticks(123) == 777


def test_machine_identity_rejects_empty_or_malformed(tmp_path):
    path = tmp_path / "machine-id"
    path.write_bytes(b"\n")
    with pytest.raises(Exception, match="malformed"):
        _machine_identity_sha256(path)
    path.write_bytes(b"g" * 32)
    with pytest.raises(Exception, match="malformed"):
        _machine_identity_sha256(path)
    path.write_bytes(b"a" * 32 + b"\n")
    assert _machine_identity_sha256(path) == hashlib.sha256(b"a" * 32).hexdigest()
    assert _execution_machine_identity_sha256(path) == hashlib.sha256(b"a" * 32).hexdigest()
    assert _one_run_machine_identity_sha256(path) == hashlib.sha256(b"a" * 32).hexdigest()
    path.write_bytes(b"")
    with pytest.raises(QualificationError, match="malformed"):
        _execution_machine_identity_sha256(path)
    for malformed in (b"a" * 32 + b"\n\n", b"a" * 32 + b" ", b" a" * 16):
        path.write_bytes(malformed)
        with pytest.raises(Exception, match="malformed"):
            _machine_identity_sha256(path)
        with pytest.raises(Exception, match="malformed"):
            _execution_machine_identity_sha256(path)
        with pytest.raises(Exception, match="malformed"):
            _one_run_machine_identity_sha256(path)


def test_fault_producer_uses_launch_host_and_hashed_raw_scenarios(
    tmp_path, monkeypatch
):
    paths = {}
    for name in (
        "execution", "runtime", "training", "objectives", "checkpoint",
        "bundle", "atlas", "atlas_manifest", "b3_gate",
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_bytes(name.encode())
    execution_digest = _digest("execution-seal")
    runtime_identity = _digest("runtime-identity")
    training_identity = _digest("training-identity")
    source_identity = {"commit": "a" * 40, "tree": "b" * 40, "clean": True}
    sources = {name: source_identity for name in ("bot", "client", "game")}
    launch_host = {
        "hostname": "DESKTOP-RTX2080",
        "kernel_release": "6.6.0-microsoft-standard-WSL2",
        "architecture": "x86_64",
        "machine_identity_sha256": _digest("wsl-machine"),
    }
    execution = {
        "test_mode": False, "full_network_executed": True,
        "execution_evidence_sha256": execution_digest,
        "execution_host": launch_host, "source_repositories": sources,
        "scenario_evidence": [],
    }
    rust_extension = {"bytes": 100, "sha256": _digest("rust-extension")}
    manifest = {"semantic": {
        "runtime_config": {
            "network_barrier_execution_evidence_sha256": execution_digest,
            "training_config_sha256": training_identity,
        },
        "artifacts": {"rust_lattice": {
            "enabled": True, "size": rust_extension["bytes"],
            "sha256": rust_extension["sha256"],
        }},
    }}
    objectives_sha = hashlib.sha256(paths["objectives"].read_bytes()).hexdigest()
    atlas_sha = hashlib.sha256(paths["atlas"].read_bytes()).hexdigest()

    def fake_load(path, _label):
        if Path(path) == paths["execution"]:
            return execution
        if Path(path) == paths["objectives"]:
            return {"objective_identity_sha256": _digest("objective-id")}
        if Path(path) == paths["bundle"]:
            return {"analysis_files": {paths["objectives"].name: objectives_sha}}
        if Path(path) == paths["atlas_manifest"]:
            return {
                "recovery_physics": {
                    "hook_walk_budget_ticks": 15, "game_tick_hz": 10,
                    "walk_speed_q8_per_second": 76800,
                },
                "artifacts": {paths["atlas"].name: {
                    "sha256_uncompressed": atlas_sha,
                    "uncompressed_size": paths["atlas"].stat().st_size,
                }},
            }
        if Path(path) == paths["b3_gate"]:
            return {
                "schema": "q2-multires-b3-gate-v1", "batch": "B3",
                "status": "green", "gate_sha256": _digest("b3-gate"),
                "recovery_guide": {
                    "hook_walk_budget_ticks": 15, "game_tick_hz": 10,
                    "walk_speed_q8_per_second": 76800,
                    "rust_extension": rust_extension,
                },
            }
        raise AssertionError(path)

    source_record = {"bytes": 10, "sha256": _digest("raw-file"),
                     "evidence_sha256": _digest("raw-seal")}
    rows = [{"clients": [{"map_epoch": 2} for _ in range(4)]}]
    gap_rows = [
        {"clients": [
            {"map_epoch": 1, "server_frame": frame} for _ in range(4)
        ]}
        for frame in (4, 6)
    ]
    gap_metrics = {
        "rounds_dispatched": 33, "actions_dispatched": 132,
        "transitions_accepted": 128, "failed_rounds": 0,
        "echo_timeouts": 0, "telemetry_gap_resyncs": 1,
        "realtime_catchup_resyncs": 1,
    }
    partial_metrics = {
        "rounds_dispatched": 5, "actions_dispatched": 20,
        "transitions_accepted": 16, "failed_rounds": 1,
        "echo_timeouts": 1, "telemetry_gap_resyncs": 0,
        "realtime_catchup_resyncs": 0,
    }
    raw_scenarios = {
        "epoch-drain": {"record": source_record, "raw": {
            "observed_outcome": "completed", "epoch_drains": 1,
            "new_epoch_bootstrap_frames": 1,
            "actions_dispatched_during_epoch_drain": 0, "boundary_rounds": 1,
            "trajectory_rows": rows,
        }},
        "whole-batch-telemetry-gap-recovery": {"record": source_record, "raw": {
            "observed_outcome": "completed", "boundary_rounds": 1,
            "accepted_synchronized_frames": 32, "fault_event_count": 0,
            "trajectory_rows": gap_rows,
            "transport_fault_events": [{
                "event": "udp_telemetry_held_and_released",
                "client_id": f"qual-{slot:02d}", "server_frame": 5,
            } for slot in range(4)],
            "transport_metrics": gap_metrics,
        }},
        "partial-client-telemetry-timeout": {"record": source_record, "raw": {
            "observed_outcome": "fatal", "fault_event_count": 0,
            "exception": "AuthoritativeEchoError",
            "accepted_synchronized_frames": 4,
            "transport_fault_events": [{
                "event": "udp_telemetry_held_and_released",
                "client_id": "qual-00", "server_frame": 5,
            }],
            "transport_metrics": partial_metrics,
        }},
    }
    monkeypatch.setattr(fault_producer, "_load_json_file", fake_load)
    monkeypatch.setattr(fault_producer, "validate_b3_gate", lambda value: value)
    monkeypatch.setattr(fault_producer, "_validate_execution_evidence", lambda value, path: value)
    monkeypatch.setattr(fault_producer, "load_runtime_manifest", lambda path: manifest)
    monkeypatch.setattr(fault_producer, "verify_runtime_manifest", lambda value: SimpleNamespace(valid=True, digest=runtime_identity))
    monkeypatch.setattr(fault_producer, "_git_identity", lambda path, label: source_identity)
    monkeypatch.setattr(fault_producer, "_training_identity", lambda path: training_identity)
    monkeypatch.setattr(fault_producer, "_raw_scenarios", lambda value, path: raw_scenarios)

    result = fault_producer.assemble(
        execution_path=paths["execution"], runtime_manifest_path=paths["runtime"],
        training_manifest_path=paths["training"], objectives_path=paths["objectives"],
        checkpoint_path=paths["checkpoint"], bundle_manifest_path=paths["bundle"],
        atlas_path=paths["atlas"], atlas_manifest_path=paths["atlas_manifest"],
        b3_gate_path=paths["b3_gate"], repo_root=tmp_path,
    )
    assert result["host"] == launch_host
    assert result["scenarios"][0]["source"] == source_record
    assert result["evidence_sha256"] == hashlib.sha256(
        b6.canonical_bytes({k: v for k, v in result.items() if k != "evidence_sha256"})
    ).hexdigest()
