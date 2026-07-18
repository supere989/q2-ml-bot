from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

from harness.multires_contract import (
    ACTION_DIM,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
)
from harness.multires_lineage import CHECKPOINT_FORMAT
from harness.multires_runtime import (
    B4_ACTION_MAGIC,
    B4_CAUSAL_MAGIC,
    B4_CAUSAL_PACKET_BYTES,
    B4_CAUSAL_VERSION,
    B4_CLIENT_WIRE_VERSION,
    B4_OBSERVATION_MAGIC,
    B4_PROTOCOL_GENERATION,
    B4_ROLLOUT_SCHEMA,
    B4_TEACHER_VERSION,
)
from tools.run_multires_pretraining_validation import (
    CAMPAIGN_SCHEMA,
    campaign_result_sha256,
    canonical_bytes,
    canonical_sha256,
    file_record,
)
from tools.assemble_b4_evidence import (
    DOCUMENT_NAMES as B4_DOCUMENT_NAMES,
    EPOCH_SCHEMA as B4_EPOCH_SCHEMA,
    EXPECTED_CONTRACT as B4_EXPECTED_CONTRACT,
    EXPECTED_DESCRIPTOR as B4_EXPECTED_DESCRIPTOR,
    FEATURE_SCHEMA as B4_FEATURE_SCHEMA,
    PRIVATE_CAUSAL_FACTS as B4_PRIVATE_CAUSAL_FACTS,
    REWARD_SCHEMA as B4_REWARD_SCHEMA,
    WIRE_SCHEMA as B4_WIRE_SCHEMA,
    _canonical_bytes as b4_canonical_bytes,
    _seal as b4_seal,
)
from tools.assemble_b2_gate import ActiveFinalAuthority


SOURCE = {
    "clean": True,
    "commit": "12" * 20,
    "tree": "34" * 20,
}
RUNTIME_IDENTITY = "78" * 32
AUTHORITY = ActiveFinalAuthority(
    cohort_id="b2g99_final_99999",
    declaration_sha256="94" * 32,
    immutable_declaration_path="docs/multires/B2-GENERATED-COHORT-99999-DECLARATION.json",
    qualification_successor_paths=frozenset({
        "docs/multires/B2-GENERATED-COHORT-99999-DECLARATION.json"
    }),
)
B3_GATE_SHA256 = "95" * 32
B3_ATLAS_SET_SHA256 = "96" * 32


def _write_b3_gate(root: Path) -> Path:
    path = root / "B3-GATE.json"
    path.write_bytes(canonical_bytes({
        "schema": "q2-multires-b3-gate-v1",
        "batch": "B3",
        "status": "green",
        "repository": {
            "repository_commit": SOURCE["commit"],
            "repository_tree": SOURCE["tree"],
            "git_clean": True,
        },
        "predecessor": {
            "status": "green",
            "cohort_id": AUTHORITY.cohort_id,
            "declaration_sha256": AUTHORITY.declaration_sha256,
        },
        "recovery_guide": {"atlas_set_sha256": B3_ATLAS_SET_SHA256},
        "gate_sha256": B3_GATE_SHA256,
    }))
    return path


def _write_b4_evidence(
    root: Path, *, b3_gate: Path, atlas_sha256: str, runtime_identity: str,
) -> Path:
    directory = root / "b4"
    directory.mkdir()
    scenario_digest = "91" * 32
    packet_digest = "92" * 32
    totals = {
        "datagrams_seen": 32,
        "public_packets_decoded": 32,
        "routed_packets_accepted": 32,
        "malformed_packets_rejected": 0,
        "foreign_client_packets_rejected": 0,
        "stale_packets_rejected": 0,
        "teacher_packets_detected": 0,
    }
    normal = {
        "scenario_proof": {"raw_evidence_sha256": scenario_digest},
        "totals": totals,
    }
    negative = {
        "packet_sha256": packet_digest,
        "public_parser_result": "fatal-public-privilege-violation",
    }
    source_repositories = {
        "bot": dict(SOURCE),
        "client": {"clean": True, "commit": "56" * 20, "tree": "57" * 20},
        "game": {"clean": True, "commit": "58" * 20, "tree": "59" * 20},
    }
    source_closure = hashlib.sha256(
        b4_canonical_bytes(source_repositories)
    ).hexdigest()
    scenario_proofs = {
        "baseline_cold_1": {"raw_evidence_sha256": scenario_digest},
        "epoch_drain": {"raw_evidence_sha256": "97" * 32},
        "old_telemetry": {"raw_evidence_sha256": "98" * 32},
    }
    frames = [
        {
            "tick": 100 + index,
            "client_life_epoch": 1,
            "authoritative_echo_valid": True,
            "trainable_transition": True,
            "state_resync": False,
            "teacher_field_violations": 0,
            "action_generation": index + 1,
            "requested_vertical": 1,
        }
        for index in range(4)
    ]
    documents = {
        "feature_action_contract": b4_seal({
            "schema": B4_FEATURE_SCHEMA,
            **B4_EXPECTED_CONTRACT,
            "source_closure_sha256": source_closure,
        }),
        "runtime_epoch_fencing": b4_seal({
            "schema": B4_EPOCH_SCHEMA,
            "clients": [
                {
                    "client_id": f"qual-{index:02d}",
                    "map_epoch": 1,
                    "map_name": "fixture-map",
                    "atlas_sha256": atlas_sha256,
                }
                for index in range(4)
            ],
            "actions_dispatched_during_epoch_barrier": 0,
            "stale_epoch_transitions_admitted": 0,
            "scenario_proofs": scenario_proofs,
        }),
        "b4_wire_generation": b4_seal({
            "schema": B4_WIRE_SCHEMA,
            "descriptor": B4_EXPECTED_DESCRIPTOR,
            "atlas_sha256": atlas_sha256,
            "runtime_manifest_sha256": runtime_identity,
            "public_teacher_field_violations": 0,
            "public_privilege_separation": {
                "normal_run_audit": normal,
                "negative_probe": negative,
            },
            "private_causal_facts": list(B4_PRIVATE_CAUSAL_FACTS),
            "source_abi": {},
            "qualification_evidence_sha256": "83" * 32,
        }),
        "causal_reward_admission": b4_seal({
            "schema": B4_REWARD_SCHEMA,
            "frames": frames,
            "derivation": "baseline-cold-1-authoritative-echo-v1",
            "scenario_proof": scenario_proofs["baseline_cold_1"],
        }),
    }
    document_records = {}
    for name, filename in B4_DOCUMENT_NAMES.items():
        payload = b4_canonical_bytes(documents[name]) + b"\n"
        (directory / filename).write_bytes(payload)
        document_records[name] = {
            "name": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "evidence_sha256": documents[name]["evidence_sha256"],
        }
    aggregate = b4_seal({
        "schema": "q2-multires-b4-evidence-v1",
        "milestone": "B4",
        "status": "green",
        "atlas_sha256": atlas_sha256,
        "predecessor": {
            "b3_gate": {
                "name": "B3-gate",
                "sha256": file_record(b3_gate)["sha256"],
                "size": file_record(b3_gate)["bytes"],
            },
            "b3_gate_sha256": B3_GATE_SHA256,
            "status": "green",
            "cohort_id": AUTHORITY.cohort_id,
            "declaration_sha256": AUTHORITY.declaration_sha256,
            "atlas_set_sha256": B3_ATLAS_SET_SHA256,
            "repository_commit": SOURCE["commit"],
            "repository_tree": SOURCE["tree"],
        },
        "source_repositories": source_repositories,
        "source_closure_sha256": source_closure,
        "normative_documents": [],
        "runtime_binaries": {},
        "runtime_manifest": {
            "name": "runtime-manifest", "sha256": "81" * 32,
            "size": 1, "manifest_sha256": runtime_identity,
        },
        "network_qualification": {
            "name": "network-qualification", "sha256": "82" * 32,
            "size": 1, "evidence_sha256": "83" * 32,
            "execution_evidence_sha256": "84" * 32,
            "runtime_closure_sha256": "85" * 32,
        },
        "scenario_proofs": scenario_proofs,
        "public_privilege_proof": {
            "baseline_raw_evidence_sha256": scenario_digest,
            "datagrams_seen": totals["datagrams_seen"],
            "public_packets_decoded": totals["public_packets_decoded"],
            "teacher_packets_detected": 0,
            "negative_probe_packet_sha256": packet_digest,
            "negative_probe_result": "fatal-public-privilege-violation",
        },
        "documents": document_records,
        "component_evidence": documents,
        "gate": {
            "clean_exact_sources": True,
            "atomic_wire_generation": True,
            "real_network_barrier_replayed": True,
            "runtime_and_binary_seals_bound": True,
            "epoch_and_stale_fencing_proven": True,
            "public_teacher_violations_zero": True,
            "causal_reward_frames_admissible": True,
            "active_authority_b3_predecessor_bound": True,
        },
    })
    path = directory / "B4-EVIDENCE.json"
    path.write_bytes(b4_canonical_bytes(aggregate) + b"\n")
    return path


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths["b3_gate"] = _write_b3_gate(tmp_path)
    for index, name in enumerate((
        "checkpoint", "training_manifest", "atlas",
    )):
        path = tmp_path / name
        path.write_bytes(f"b5:{name}:{index}\n".encode("ascii"))
        paths[name] = path
    atlas_sha256 = file_record(paths["atlas"])["sha256"]
    objectives = tmp_path / "objectives"
    objectives.write_bytes(canonical_bytes({
        "objective_identity_sha256": "93" * 32,
        "objectives": [],
    }))
    paths["objectives"] = objectives
    bundle = tmp_path / "bundle_manifest"
    bundle.write_bytes(canonical_bytes({
        "bundle_version": 3, "artifact_state": "admitted",
        "name": "fixture-map", "atlas_sha256": atlas_sha256,
    }))
    paths["bundle_manifest"] = bundle
    runtime = {
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "posture_classes": POSTURE_CLASSES,
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_schema": B4_ROLLOUT_SCHEMA,
        "atlas_sha256": atlas_sha256,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
        "runtime_manifest_sha256": RUNTIME_IDENTITY,
    }
    runtime_path = tmp_path / "runtime_manifest"
    runtime_path.write_bytes(canonical_bytes(runtime))
    paths["runtime_manifest"] = runtime_path
    paths["b4_evidence"] = _write_b4_evidence(
        tmp_path, b3_gate=paths["b3_gate"], atlas_sha256=atlas_sha256,
        runtime_identity=RUNTIME_IDENTITY
    )
    return paths


def input_bindings(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: file_record(path) for name, path in paths.items()}


def results_for(mode: str) -> dict[str, Any]:
    if mode in {"guide_on", "guide_off"}:
        return {
            "scenario_identity_sha256": "56" * 32,
            "guide_enabled": mode == "guide_on",
            "task_attempts": 12,
            "task_successes": 3,
            "guide_nonzero_samples": 20 if mode == "guide_on" else 0,
            "guide_dropout_samples": 4 if mode == "guide_on" else 24,
        }
    if mode == "hazard_hook":
        return {
            "hazard_scenarios": 8,
            "hook_scenarios": 7,
            "safe_arrivals": 3,
            "environmental_deaths": 2,
            "valid_hook_attachments": 5,
            "invalid_hook_attempts": 1,
            "reward_replay_violations": 0,
            "rate_reward_violations": 0,
            "hook_necessity_label_violations": 0,
        }
    if mode == "posture_water_crouch":
        return {
            "posture_fixtures": 4,
            "water_fixtures": 3,
            "crouch_fixtures": 5,
            "fixtures_passed": 12,
            "vertical_echo_mismatches": 0,
            "standing_blocked_mismatches": 0,
        }
    assert mode == "aim_combat_holdout"
    return {
        "holdout_samples": 100,
        "visible_contacts": 20,
        "actionable_exposures": 18,
        "permitted_fire": 11,
        "executed_fire": 10,
        "hits": 4,
        "repeat_hits": 2,
        "kills": 1,
        "hidden_fire": 0,
        "yaw_mae_degrees": 17.5,
        "pitch_mae_degrees": 8.25,
    }


def campaign(
    mode: str, replicate: int, bindings: Mapping[str, Mapping[str, Any]],
    *, seed: int = 17, game_seed: int = 23, transitions: int = 64,
    runtime_identity: str = RUNTIME_IDENTITY,
) -> dict[str, Any]:
    results = results_for(mode)
    if mode == "guide_off":
        results["guide_dropout_samples"] = transitions * 4
    guide_drops = results.get("guide_dropout_samples", 0)
    season_report = {
        "schema": "multires-atlas-season-v1",
        "season_id": f"b5-{mode}",
        "atlas_sha256": bindings["atlas"]["sha256"],
        "policy_start_version": 1,
        "policy_end_version": 1,
        "accepted_transitions": transitions,
        "transport": {"command_echo_match_rate": 1.0, "state_resyncs": 0},
        "privilege": {
            "scope": "not-measured-offline-no-public-conduit",
            "upstream_evidence_required": "sealed-b4-real-public-datagram-audit-v1",
        },
        "guides": {
            "global_drop_rate": 1.0 if mode == "guide_off" else 0.0,
            "candidate_drop_rate": float(guide_drops) / float(transitions * 4),
        },
        "hazard": {
            "safe_arrivals": results.get("safe_arrivals", 0),
            "environmental_deaths": results.get("environmental_deaths", 0),
        },
        "hook": {"invalid_attempts": results.get("invalid_hook_attempts", 0)},
        "combat": {
            "actionable_exposure": results.get("actionable_exposures", 0),
            "fire_permission": results.get("permitted_fire", 0),
            "executed_fire": results.get("executed_fire", 0),
            "hits": results.get("hits", 0),
            "repeated_hits": results.get("repeat_hits", 0),
            "kills": results.get("kills", 0),
            "hidden_fire": results.get("hidden_fire", 0),
            "visible_contact_yaw_mae_deg": results.get("yaw_mae_degrees", 0.0),
            "visible_contact_pitch_mae_deg": results.get("pitch_mae_degrees", 0.0),
        },
    }
    row: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": mode,
        "replicate": replicate,
        "status": "passed",
        "no_update": True,
        "seed": seed,
        "game_seed": game_seed,
        "transition_count": transitions,
        "source": {"commit": SOURCE["commit"], "tree": SOURCE["tree"]},
        "bindings": {
            "runtime_manifest_sha256": runtime_identity,
            "checkpoint_sha256_before": bindings["checkpoint"]["sha256"],
            "checkpoint_sha256_after": bindings["checkpoint"]["sha256"],
            "policy_state_sha256_before": "8a" * 32,
            "policy_state_sha256_after": "8a" * 32,
            "optimizer_state_sha256_before": "8b" * 32,
            "optimizer_state_sha256_after": "8b" * 32,
            "training_manifest_sha256": bindings["training_manifest"]["sha256"],
            "bundle_manifest_sha256": bindings["bundle_manifest"]["sha256"],
            "atlas_sha256": bindings["atlas"]["sha256"],
            "objective_identity_sha256": bindings["objectives"]["sha256"],
        },
        "counters": {
            "optimizer_steps": 0,
            "backward_parameter_gradients": 0,
        },
        "trajectory_sha256": {
            "guide_on": "61" * 32,
            "guide_off": "62" * 32,
            "hazard_hook": "63" * 32,
            "posture_water_crouch": "64" * 32,
            "aim_combat_holdout": "65" * 32,
        }[mode],
        "season_report": season_report,
        "season_report_sha256": canonical_sha256(season_report),
        "results": results,
    }
    row["result_sha256"] = campaign_result_sha256(row)
    return row


def proof(
    bindings: Mapping[str, Mapping[str, Any]], *, seed: int = 17,
    game_seed: int = 23, runtime_identity: str = RUNTIME_IDENTITY,
) -> dict[str, Any]:
    return {
        "schema": "q2-multires-500-transition-proof-v1",
        "tool": "run_multires_500_transition_proof",
        "mode": "production",
        "admissible": True,
        "production_pass": True,
        "transition_count": 500,
        "required_client_count": 4,
        "transitions_per_client": 125,
        "seed": seed,
        "game_seed": game_seed,
        "divergence_game_seed": game_seed + 1,
        "same_seed_match": True,
        "different_seed_diverges": True,
        "partial_admissions": 0,
        "stale_admissions": 0,
        "resync_admissions": 0,
        "orphan_processes_after_teardown": 0,
        "records_required": True,
        "digest_only_admissible": False,
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "collector": "MultiresSynchronousCollector",
        "spatial_provider": "RustAtlasSpatialProvider",
        "lattice_crate": "q2_lattice",
        "runtime_manifest_sha256": runtime_identity,
        "atlas_sha256": bindings["atlas"]["sha256"],
        "production_admission": {
            "bundle_version": 3,
            "artifact_state": "admitted",
            "objective_identity_sha256": bindings["objectives"]["sha256"],
            "checkpoint_sha256": bindings["checkpoint"]["sha256"],
            "training_manifest_sha256": bindings["training_manifest"]["sha256"],
            "checkpoint_initialization": "random",
            "checkpoint_training_step": 0,
            "checkpoint_lineage_root_sha256": "79" * 32,
            "q2ded": "/runtime/q2ded",
            "client_binary": "/runtime/q2-client",
            "runtime_root": "/runtime",
            "trainer_argv_prefix": [
                str(Path(sys.executable).resolve()),
                str((Path(__file__).resolve().parents[1] / "train/multires_one_run.py").resolve()),
            ],
            "evidence_dir": "/evidence",
        },
        "verifier_evidence": {"gate": "cross-language-500-transition-golden"},
        "failures": [],
    }
