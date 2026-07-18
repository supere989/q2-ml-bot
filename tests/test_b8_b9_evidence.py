from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from harness.runtime_attestation import semantic_digest
from tools import assemble_b8_b9_evidence as gate


SHA = "1" * 64
STAGE_SHA = "2" * 64
LINEAGE_SHA = "3" * 64


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _seal(body: dict, field: str = "evidence_sha256") -> dict:
    return {**body, field: gate.canonical_sha256(body)}


def _record(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _runtime(retirement_sha: str) -> dict:
    semantic = {
        "runtime": "multires", "generation": 2,
        "runtime_config": {"retirement_manifest_sha256": retirement_sha},
    }
    return {
        "schema": "q2-runtime-attestation-v2",
        "semantic": semantic,
        "manifest_sha256": semantic_digest(semantic),
        "diagnostics": {},
    }


def _integration(envelope_path: Path) -> dict:
    envelope_bytes = envelope_path.read_bytes()
    envelope = json.loads(envelope_bytes)
    body = {
        "schema": "multires-integration-report-v1",
        "tool": "verify_multires_integration",
        "policy_generation": 2,
        "feature_schema_sha256": "4" * 64,
        "envelope_sha256": hashlib.sha256(envelope_bytes).hexdigest(),
        "gates": [
            {"rank": index, "gate": name, "status": "pass",
             "evidence_sha256": hashlib.sha256(
                 (envelope_path.parent / envelope["evidence"][name]).read_bytes()
             ).hexdigest(), "reasons": []}
            for index, name in enumerate(gate.INTEGRATION_GATE_ORDER, start=1)
        ],
        "failed_gates": [],
        "overall": "pass",
    }
    return {**body, "report_sha256": gate.canonical_sha256(body)}


def _source() -> dict:
    return _seal({
        "schema": "q2-multires-source-identity-v1",
        "repositories": {
            name: {"commit": character * 40, "tree": character * 40, "clean": True}
            for name, character in (("bot", "a"), ("client", "b"), ("game", "c"))
        },
        "clean": True,
    })


def _legacy() -> dict:
    return _seal({
        "schema": gate.LEGACY_ABSENCE_SCHEMA,
        "legacy_runtime_selectors": [],
        "legacy_policy_selectors": [],
        "legacy_optimizer_selectors": [],
        "legacy_dyn_selectors": [],
        "legacy_protocol_selectors": [],
        "operational_fallbacks": [],
        "passed": True,
    })


def _b7(
    runtime_sha: str, atlas_catalog_sha: str, completed: Path, evaluator: Path,
) -> dict:
    return _seal({
        "schema": gate.B7_GATE_SCHEMA,
        "decision": "passed",
        "stage_id": 7,
        "stage_name": "full-guide-off-ablation",
        "stage_configuration_sha256": STAGE_SHA,
        "runtime_manifest_sha256": runtime_sha,
        "atlas_catalog_sha256": atlas_catalog_sha,
        "lineage_root_sha256": LINEAGE_SHA,
        "accepted_transitions": 20_000,
        "policy_updates": 40,
        "optimizer_steps": 80,
        "artifacts": {
            "completed_stage": {"path": str(completed),
                                "sha256": _record(completed)["sha256"]},
            "evaluator": {"path": str(evaluator),
                          "sha256": _record(evaluator)["sha256"]},
        },
        "automatic_promotion": False,
    })


def _causal(kind: str, atlas_sha: str) -> dict:
    del kind
    events = {
        "actionable_exposure": 100, "post_command_alignment": 80,
        "fire_permission": 70, "executed_fire": 60, "hits": 55,
        "repeated_hits": 20, "kills": 5,
    }
    return {
        "schema": "multires-atlas-season-v1",
        "season_id": f"{atlas_sha[:8]}-season",
        "atlas_sha256": atlas_sha,
        "policy_start_version": 100,
        "policy_end_version": 200,
        "accepted_transitions": 20_000,
        "transport": {"command_echo_match_rate": 1.0, "state_resyncs": 0},
        "posture": {},
        "movement": {
            "forward_command_mean": 0.60, "backward_command_mean": 0.25,
            "speed_mean": 180.0, "true_view_pitch_mean_deg": 1.0,
            "true_view_pitch_abs_mean_deg": 4.0,
            "downlook_over_15deg_rate": 0.05,
        },
        "hazard": {}, "hook": {},
        "guides": {"global_drop_rate": 0.25, "candidate_drop_rate": 0.5,
                   "per_class_drop_rate": {}},
        "privilege": {"teacher_field_violations": 0},
        "combat": {**events, "hidden_fire": 0, "aligned_fire_precision": 0.90,
                   "visible_contact_yaw_mae_deg": 6.0,
                   "visible_contact_pitch_mae_deg": 4.0},
        "atlas": {}, "dyn": {},
        "observer_coverage": {
            "movement_command_samples": 20_000,
            "movement_speed_samples": 20_000,
            "true_view_pitch_samples": 20_000,
            "guide_dropout_samples": 20_000,
            "missing_movement_command_samples": 0,
            "missing_movement_speed_samples": 0,
            "missing_true_view_pitch_samples": 0,
            "missing_guide_dropout_samples": 0,
        },
        "private_causal_payload_serialized": False,
    }


def _network(kind: str) -> dict:
    return {
        "network_client/transitions_accepted": 20_000,
        "network_client/failed_rounds": 0,
        "network_client/echo_timeouts": 0,
        "network_client/authoritative_echo_accept_rate": 0.99,
        "network_client/map_epoch_resyncs": 2 if kind == "generated" else 3,
        "network_client/telemetry_gap_resyncs": 1,
    }


def _current(
    kind: str, runtime_sha: str, atlas_sha: str, atlas_catalog_sha: str,
    policy_sha: str,
) -> dict:
    causal = _causal(kind, atlas_sha)
    network = _network(kind)
    return _seal({
        "schema": gate.CURRENT_SEASON_SCHEMA,
        "training_core_schema": "q2-multires-continuous-training-v1",
        "health": "training-active", "promotion_claim": False,
        "stage_id": 7, "stage_name": "full-guide-off-ablation",
        "stage_configuration_sha256": STAGE_SHA,
        "runtime_manifest_sha256": runtime_sha,
        "atlas_sha256": atlas_sha,
        "atlas_catalog_sha256": atlas_catalog_sha,
        "lineage_root_sha256": LINEAGE_SHA,
        "counters": {"accepted_transitions": 20_000, "policy_updates": 40,
                     "optimizer_steps": 80, "next_policy_version": 41},
        "last_policy_version": 40, "last_rollout_evidence_sha256": "8" * 64,
        "last_update_evidence_sha256": "9" * 64,
        "last_checkpoint": "checkpoints/policy.pt", "checkpoint_sha256": policy_sha,
        "checkpoint_manifest": {"training_step": 20_000},
        "last_reward_mean": 1.0, "last_ppo_metrics": {},
        "causal_metrics_window_sha256": gate.canonical_sha256(causal),
        "causal_metrics_window": causal,
        "network_metrics_window_sha256": gate.canonical_sha256(network),
        "network_metrics_window": network,
    })


def _evaluation(kind: str, current: dict, files: dict[str, Path], b7: dict) -> dict:
    causal = current["causal_metrics_window"]
    combat = causal["combat"]
    events = {name: combat[name] for name in (
        "actionable_exposure", "post_command_alignment", "fire_permission",
        "executed_fire", "hits", "repeated_hits", "kills",
    )}
    return _seal({
        "schema": gate.SEASON_EVALUATION_SCHEMA,
        "season_id": causal["season_id"], "season_kind": kind,
        "current_season_evidence_sha256": current["evidence_sha256"],
        "runtime_manifest_sha256": current["runtime_manifest_sha256"],
        "atlas_sha256": current["atlas_sha256"],
        "atlas_catalog_sha256": current["atlas_catalog_sha256"],
        "lineage_root_sha256": current["lineage_root_sha256"],
        "b7_gate_evidence_sha256": b7["evidence_sha256"],
        "artifact_sha256": {role: hashlib.sha256(path.read_bytes()).hexdigest()
                            for role, path in files.items() if role != "evaluation"},
        "g1": {"vertical_intent_echo_match_rate": 1.0,
               "water_land_projection_skew": 0,
               "partial_client_timeout_fatal_exercised": True},
        "g2": {
            "stage_configuration_sha256": STAGE_SHA,
            "no_visible_target_samples": 5000, "moving_at_least_96_samples": 4000,
            "downlook_rate": 0.05, "moving_mean_pitch_deg": 2.0,
            "forward_command_rate": 0.60, "backward_command_rate": 0.25,
            "unnecessary_actual_crouch_rate": 0.05,
            "unnecessary_actual_crouch_max": 0.10,
            "action_collapse_audits": {
                name: {"observed_rate": 0.10, "minimum_rate": 0.0,
                       "maximum_rate": 0.50, "collapsed": False}
                for name in ("jump", "crouch", "strafe", "hook")
            },
        },
        "g3": {
            "section17_fixtures_passed": True, "stock_determinism_passed": True,
            "matched_seed_sha256": "a" * 64,
            "baseline_episodes": 100, "treatment_episodes": 100,
            "baseline_safe_arrivals": 40, "treatment_safe_arrivals": 70,
            "baseline_environmental_deaths": 30,
            "treatment_environmental_deaths": 10,
            "progress_credit_cap_violations": 0,
            "boundary_oscillation_credits": 0,
            "invalid_hook_net_positive_events": 0,
            "engine_invalid_anchor_recoveries": 0,
        },
        "g4": {
            "matched_seed_sha256": gate.canonical_sha256({
                "domain": "q2-multires-g4-matched-seed-v1",
                "matched_seed": 77,
            }),
            "matched_task_set_sha256": "c" * 64,
            "b7_stage_evaluation_evidence_sha256": json.loads(
                files["b7_stage_evaluation"].read_text()
            )["evidence_sha256"],
            "matched_attempts": 100, "guide_on_successes": 80,
            "guide_off_successes": 70, "neutral_successes": 20,
            "global_dropout_after_stage4_rate": 0.25,
            "guide_off_downlook_rate": 0.05, "guide_off_backward_rate": 0.25,
            "guide_off_combat_events": events,
        },
        "g5": {"visible_contact_yaw_samples": 100,
               "visible_contact_pitch_samples": 100,
               "aligned_fire_precision_min": 0.85,
               "visible_contact_yaw_mae_max_deg": 12.0,
               "visible_contact_pitch_mae_max_deg": 8.0},
        "automatic_promotion": False,
    })


def _freeze_archive(root: Path, kind: str, season_kind, files: dict[str, Path]) -> Path:
    manifest = root / "archive-manifest.json"
    body = {
        "schema": gate.ARCHIVE_SCHEMA,
        "archive_id": f"{kind}-{season_kind or 'shared'}-archive",
        "archive_kind": kind, "season_kind": season_kind, "read_only": True,
        "files": {role: {"path": path.relative_to(root).as_posix(), **_record(path)}
                  for role, path in sorted(files.items())},
    }
    _write(manifest, _seal(body, "manifest_sha256"))
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            (Path(directory) / filename).chmod(0o444)
    for directory, _, _ in list(os.walk(root, topdown=False)):
        Path(directory).chmod(0o555)
    return manifest


def _catalog() -> dict:
    maps = [
        {"map_name": name, "atlas_sha256": hashlib.sha256(data).hexdigest()}
        for name, data in (
            ("generated-map", b"atlas-generated"),
            ("stock-map", b"atlas-stock"),
            ("shadow-inventory", b"atlas-inventory"),
        )
    ]
    body = {"domain": gate.ATLAS_CATALOG_DOMAIN, "maps": maps}
    return {
        "schema": gate.ATLAS_CATALOG_SCHEMA,
        **body,
        "map_count": len(maps),
        "atlas_catalog_sha256": gate.canonical_sha256(body),
    }


def _common_files(root: Path, *, atlas_bytes: bytes = b"atlas-generated") -> dict[str, Path]:
    retirement = {
        "schema": "q2-multires-runtime-retirement-v1",
        "status": "legacy-runtime-retired", "fallback_allowed": False,
    }
    retirement_path = root / "retirement_manifest.json"
    _write(retirement_path, retirement)
    values = {
        "runtime_manifest": _runtime(_record(retirement_path)["sha256"]),
        "policy": b"same-policy", "atlas": atlas_bytes,
        "atlas_catalog": _catalog(),
        "dyn_schema": b"Q2LAT002-schema", "reward_configuration": {"reward": "v2"},
        "source_identity": _source(), "legacy_selector_absence": _legacy(),
        "retirement_manifest": retirement,
    }
    result = {}
    for role, value in values.items():
        path = root / f"{role}.json"
        _write(path, value)
        result[role] = path
    integration_root = root / "integration"
    integration_root.mkdir(parents=True, exist_ok=True)
    evidence_paths = {}
    for name in gate.INTEGRATION_GATE_ORDER:
        path = integration_root / f"{name}.json"
        payload = {
            "source_repositories": _source()["repositories"],
            "schema": "q2-multires-b6-wsl-g1-v1" if name == "wsl_b6_campaign"
            else f"test-{name}",
        }
        if name == "wsl_b6_campaign":
            payload["bindings"] = {
                "runtime_manifest_identity_sha256": values[
                    "runtime_manifest"
                ]["manifest_sha256"],
                "retirement_manifest": _record(retirement_path),
            }
        _write(path, payload)
        result[gate.INTEGRATION_EVIDENCE_ROLES[name]] = path
        evidence_paths[name] = path.name
    envelope_path = integration_root / "envelope.json"
    _write(envelope_path, {
        "schema": "multires-integration-evidence-v1", "evidence": evidence_paths,
    })
    result["integration_envelope"] = envelope_path
    report_path = root / "integration_report.json"
    report_path.write_bytes(gate.canonical_bytes(_integration(envelope_path)) + b"\n")
    result["integration_report"] = report_path
    return result


def _reconstruction_manifest(b8: dict, files: dict[str, Path]) -> dict:
    identities = b8["identities"]
    return _seal({
        "schema": gate.RECONSTRUCTION_MANIFEST_SCHEMA,
        "reconstruction_id": "cold-reconstruction-1",
        "b8_gate_sha256": b8["gate_sha256"],
        "runtime_manifest_sha256": identities["runtime_manifest_sha256"],
        "atlas_catalog_sha256": identities["atlas_catalog_sha256"],
        "lineage_root_sha256": identities["lineage_root_sha256"],
        "artifacts": {
            role: _record(files[role]) for role in gate.IDENTITY_ROLES
        },
        "complete": True,
    }, "manifest_sha256")


@pytest.fixture(autouse=True)
def _integration_boundary(monkeypatch):
    monkeypatch.setattr(gate, "rerun_integration_gates", _integration)
    monkeypatch.setattr(gate, "validate_b6_campaign", lambda value: None)


def _season_archive(root: Path, kind: str, *, mutate=None) -> Path:
    files = _common_files(root, atlas_bytes=f"atlas-{kind}".encode())
    runtime_sha = json.loads(files["runtime_manifest"].read_text())["manifest_sha256"]
    policy_sha = _record(files["policy"])["sha256"]
    atlas_sha = _record(files["atlas"])["sha256"]
    atlas_catalog_sha = json.loads(
        files["atlas_catalog"].read_text()
    )["atlas_catalog_sha256"]
    current = _current(
        kind, runtime_sha, atlas_sha, atlas_catalog_sha, policy_sha,
    )
    files["current_season"] = root / "current_season.json"
    _write(files["current_season"], current)
    files["b7_stage_evaluation"] = root / "b7_stage_evaluation.json"
    stage_evaluator = _seal({
        "schema": "q2-multires-b7-stage-evaluation-v1",
        "decision": "passed", "stage_id": 7,
        "stage_name": "full-guide-off-ablation",
        "stage_configuration_sha256": STAGE_SHA,
        "runtime_manifest_sha256": runtime_sha,
        "atlas_sha256": atlas_sha,
        "atlas_catalog_sha256": atlas_catalog_sha,
        "lineage_root_sha256": LINEAGE_SHA,
        "completed_season": {"path": str(files["current_season"]),
                             "sha256": _record(files["current_season"])["sha256"]},
        "minimum_accepted_transitions": 1,
        "predicates": [{
            "name": "guide_degradation", "path": "g4.degradation",
            "operator": "<=", "threshold": 0.15, "observed": 0.125,
            "passed": True,
        }],
        "stage_specific": {
            "guide_on_reference": {"path": "/sealed/guide-on-reference.json",
                                   "sha256": "4" * 64},
            "guide_on_completed_season": {"path": "/sealed/guide-on-season.json",
                                          "sha256": "5" * 64},
            "guide_on_checkpoint": {"path": "/sealed/guide-on-checkpoint.pt",
                                    "sha256": "6" * 64},
            "guide_on_policy_version": 80,
            "matched_seed": 77,
            "guide_on_task_success": 0.80,
            "guide_off_task_success": 0.70,
            "degradation_fraction": 0.125,
            "maximum_degradation_fraction": 0.15,
            "global_dropout_rate": 0.25,
            "passed": True,
        },
        "automatic_promotion": False,
    })
    files["b7_stage_evaluation"].write_bytes(
        gate.canonical_bytes(stage_evaluator) + b"\n"
    )
    b7 = _b7(
        runtime_sha, atlas_catalog_sha,
        files["current_season"], files["b7_stage_evaluation"],
    )
    files["b7_gate"] = root / "b7_gate.json"
    _write(files["b7_gate"], b7)
    files["evaluation"] = root / "evaluation.json"
    evaluation = _evaluation(kind, current, files, b7)
    if mutate:
        mutate(evaluation, current, files)
        current["causal_metrics_window_sha256"] = gate.canonical_sha256(
            current["causal_metrics_window"]
        )
        current["network_metrics_window_sha256"] = gate.canonical_sha256(
            current["network_metrics_window"]
        )
        current.pop("evidence_sha256", None)
        current = _seal(current)
        _write(files["current_season"], current)
        stage_evaluator = json.loads(files["b7_stage_evaluation"].read_text())
        stage_evaluator["completed_season"]["sha256"] = _record(
            files["current_season"]
        )["sha256"]
        stage_evaluator.pop("evidence_sha256", None)
        stage_evaluator = _seal(stage_evaluator)
        files["b7_stage_evaluation"].write_bytes(
            gate.canonical_bytes(stage_evaluator) + b"\n"
        )
        b7 = json.loads(files["b7_gate"].read_text())
        b7["artifacts"]["completed_stage"]["sha256"] = _record(
            files["current_season"]
        )["sha256"]
        b7["artifacts"]["evaluator"]["sha256"] = _record(
            files["b7_stage_evaluation"]
        )["sha256"]
        b7.pop("evidence_sha256", None)
        b7 = _seal(b7)
        _write(files["b7_gate"], b7)
        evaluation["current_season_evidence_sha256"] = current["evidence_sha256"]
        evaluation["b7_gate_evidence_sha256"] = b7["evidence_sha256"]
        evaluation["artifact_sha256"]["current_season"] = _record(
            files["current_season"]
        )["sha256"]
        evaluation["artifact_sha256"]["b7_stage_evaluation"] = _record(
            files["b7_stage_evaluation"]
        )["sha256"]
        evaluation["artifact_sha256"]["b7_gate"] = _record(
            files["b7_gate"]
        )["sha256"]
        evaluation.pop("evidence_sha256", None)
        evaluation = _seal(evaluation)
    _write(files["evaluation"], evaluation)
    return _freeze_archive(root, "season", kind, files)


def _publish(path: Path, value: dict) -> None:
    _write(path, value)


def _schema(name: str) -> dict:
    return json.loads((gate.ROOT / "schemas" / name).read_text())


def _green_b8(root: Path) -> tuple[dict, Path]:
    generated = gate.assemble_season(
        _season_archive(root / "generated", "generated")
    )
    stock = gate.assemble_season(_season_archive(root / "stock", "stock"))
    generated_path = root / "generated-gate.json"
    stock_path = root / "stock-gate.json"
    _publish(generated_path, generated)
    _publish(stock_path, stock)
    shadow_root = root / "shadow"
    files = _common_files(shadow_root, atlas_bytes=b"atlas-inventory")
    files["shadow_evidence"] = shadow_root / "shadow_evidence.json"
    body = {
        "schema": gate.SHADOW_EVIDENCE_SCHEMA, "shadow_id": "shadow-adversarial",
        "season_gate_sha256": {
            "generated": generated["gate_sha256"],
            "stock": stock["gate_sha256"],
        },
        "artifact_sha256": {},
        "topology": {"maxclients": 6, "ml_clients": 4, "human_slots": 2},
        "vps_work": {"compile_processes": 0, "analyze_processes": 0,
                     "compile_launches": 0, "analyze_launches": 0},
        "rotation": {"stock_maps": 1, "generated_maps": 1,
                     "interlaced": True, "queue_prefix_isolation": True},
        "events": {"human_joins": 1, "human_leaves": 1, "map_changes": 2},
        "transport": {
            "accepted_transitions": 20_000, "failed_rounds": 0,
            "echo_timeouts": 0, "authoritative_echo_accept_rate": 0.99,
            "vertical_intent_echo_match_rate": 1.0,
            "water_land_projection_skew": 0, "map_epoch_recoveries": 1,
            "telemetry_gap_recoveries": 1,
            "partial_client_timeout_remained_fatal": True,
        },
        "legacy_absence_evidence_sha256": _legacy()["evidence_sha256"],
        "shadow_only": True, "automatic_promotion": False,
        "public_mutations": [],
    }
    body["artifact_sha256"] = {
        role: _record(path)["sha256"] for role, path in files.items()
        if role != "shadow_evidence"
    }
    _write(files["shadow_evidence"], _seal(body))
    b8 = gate.assemble_b8(
        generated_path, stock_path,
        _freeze_archive(shadow_root, "shadow", None, files),
    )
    b8_path = root / "b8.json"
    _publish(b8_path, b8)
    return b8, b8_path


def _restart_archive(
    root: Path, b8: dict, *, omit_reconstruction: bool = False,
    wrong_reconstruction_digest: bool = False,
    rebound_reconstruction_artifact: bool = False,
) -> tuple[Path, dict[str, Path]]:
    files = _common_files(root, atlas_bytes=b"atlas-inventory")
    files["b8_gate"] = root / "b8_gate.json"
    _publish(files["b8_gate"], b8)
    if not omit_reconstruction:
        files["reconstruction_manifest"] = root / "reconstruction_manifest.json"
        reconstruction_manifest = _reconstruction_manifest(b8, files)
        if rebound_reconstruction_artifact:
            reconstruction_manifest["artifacts"]["policy"]["sha256"] = "d" * 64
            reconstruction_manifest.pop("manifest_sha256")
            reconstruction_manifest = _seal(
                reconstruction_manifest, "manifest_sha256",
            )
        _write(
            files["reconstruction_manifest"],
            reconstruction_manifest,
        )
    reconstruction_sha256 = (
        "d" * 64 if omit_reconstruction
        else _record(files["reconstruction_manifest"])["sha256"]
    )
    if wrong_reconstruction_digest:
        reconstruction_sha256 = "e" * 64
    files["cold_restart_evidence"] = root / "cold.json"
    body = {
        "schema": gate.COLD_RESTART_SCHEMA, "restart_id": root.name,
        "b8_gate_sha256": b8["gate_sha256"], "artifact_sha256": {},
        "shadow_stopped_cleanly": True,
        "reconstructed_from_attested_archive": True,
        "reconstruction_manifest_sha256": reconstruction_sha256,
        "server_reconnected": True, "ml_clients_reconnected": 4,
        "policy_loaded": True, "stock_maps_loaded": 1,
        "generated_maps_loaded": 1, "map_changes": 2,
        "accepted_transitions": 4, "failed_rounds": 0, "echo_timeouts": 0,
        "legacy_absence_evidence_sha256": _legacy()["evidence_sha256"],
        "operational_legacy_selector_matches": [],
        "automatic_promotion": False, "public_mutations": [],
    }
    body["artifact_sha256"] = {
        role: _record(path)["sha256"] for role, path in files.items()
        if role != "cold_restart_evidence"
    }
    _write(files["cold_restart_evidence"], _seal(body))
    return _freeze_archive(root, "restart", None, files), files


_SEASON_IDENTITY_PATHS = {
    "season_id": ("season_id",),
    "archive_id": ("archive", "archive_id"),
    "archive_manifest_sha256": ("archive", "manifest_sha256"),
    "archive_manifest_file_sha256": ("archive", "sha256"),
    "current_season_file_sha256": ("inputs", "current_season", "sha256"),
    "evaluation_file_sha256": ("inputs", "evaluation", "sha256"),
    "current_season_evidence_sha256": (
        "inputs", "current_season_evidence_sha256",
    ),
    "evaluation_evidence_sha256": (
        "inputs", "evaluation_evidence_sha256",
    ),
    "causal_metrics_window_sha256": (
        "window", "causal_metrics_window_sha256",
    ),
    "network_metrics_window_sha256": (
        "window", "network_metrics_window_sha256",
    ),
    "atlas_sha256": ("identities", "atlas_sha256"),
    "atlas_artifact_sha256": (
        "identities", "artifacts", "atlas", "sha256",
    ),
}


def _copy_path(target: dict, source: dict, path: tuple[str, ...]) -> None:
    target_owner = target
    source_owner = source
    for name in path[:-1]:
        target_owner = target_owner[name]
        source_owner = source_owner[name]
    target_owner[path[-1]] = source_owner[path[-1]]


def _reseal_gate(value: dict) -> None:
    value.pop("gate_sha256", None)
    value["gate_sha256"] = gate.canonical_sha256({
        "domain": gate.SEASON_GATE_SCHEMA, "gate": value,
    })


def test_generated_and_stock_seasons_b8_and_b9_pass_without_authorizing_mutation(tmp_path):
    generated_manifest = _season_archive(tmp_path / "generated", "generated")
    stock_manifest = _season_archive(tmp_path / "stock", "stock")
    generated = gate.assemble_season(generated_manifest)
    stock = gate.assemble_season(stock_manifest)
    assert generated["passed"] and stock["passed"]
    g0 = generated["predicates"]["G0"]
    assert g0["passed"] and g0["failures"] == []
    summary = g0["evidence"]
    assert summary["schema"] == gate.G0_SUMMARY_SCHEMA
    assert all(summary["assertions"].values())
    assert summary["evidence_sha256"] == gate.canonical_sha256({
        key: value for key, value in summary.items() if key != "evidence_sha256"
    })
    assert (generated["identities"]["atlas_sha256"] !=
            stock["identities"]["atlas_sha256"])
    assert (generated["identities"]["atlas_catalog_sha256"] ==
            stock["identities"]["atlas_catalog_sha256"])
    assert _schema("q2-multires-b8-season-gate-v1.schema.json")["$id"].endswith(
        gate.SEASON_GATE_SCHEMA
    )
    assert _schema("q2-multires-quality-archive-v1.schema.json")["$id"].endswith(
        gate.ARCHIVE_SCHEMA
    )
    generated_path, stock_path = tmp_path / "generated-gate.json", tmp_path / "stock-gate.json"
    _publish(generated_path, generated)
    _publish(stock_path, stock)

    shadow_root = tmp_path / "shadow"
    shadow_files = _common_files(shadow_root, atlas_bytes=b"atlas-inventory")
    shadow_files["shadow_evidence"] = shadow_root / "shadow_evidence.json"
    shadow_body = {
        "schema": gate.SHADOW_EVIDENCE_SCHEMA, "shadow_id": "shadow-1",
        "season_gate_sha256": {"generated": generated["gate_sha256"],
                               "stock": stock["gate_sha256"]},
        "artifact_sha256": {},
        "topology": {"maxclients": 6, "ml_clients": 4, "human_slots": 2},
        "vps_work": {"compile_processes": 0, "analyze_processes": 0,
                     "compile_launches": 0, "analyze_launches": 0},
        "rotation": {"stock_maps": 4, "generated_maps": 4, "interlaced": True,
                     "queue_prefix_isolation": True},
        "events": {"human_joins": 2, "human_leaves": 2, "map_changes": 8},
        "transport": {"accepted_transitions": 20_000, "failed_rounds": 0,
                      "echo_timeouts": 0, "authoritative_echo_accept_rate": 0.99,
                      "vertical_intent_echo_match_rate": 1.0,
                      "water_land_projection_skew": 0, "map_epoch_recoveries": 4,
                      "telemetry_gap_recoveries": 1,
                      "partial_client_timeout_remained_fatal": True},
        "legacy_absence_evidence_sha256": _legacy()["evidence_sha256"],
        "shadow_only": True, "automatic_promotion": False, "public_mutations": [],
    }
    shadow_body["artifact_sha256"] = {
        role: _record(path)["sha256"] for role, path in shadow_files.items()
        if role != "shadow_evidence"
    }
    _write(shadow_files["shadow_evidence"], _seal(shadow_body))
    shadow_manifest = _freeze_archive(shadow_root, "shadow", None, shadow_files)
    b8 = gate.assemble_b8(generated_path, stock_path, shadow_manifest)
    assert b8["passed"] and b8["cold_restart_pending"]
    assert all(
        b8["season_independence"]["generated"][field] !=
        b8["season_independence"]["stock"][field]
        for field in gate.SEASON_DISTINCT_FIELDS
    )
    assert (generated["identities"]["artifacts"]["policy"] ==
            stock["identities"]["artifacts"]["policy"])
    assert (generated["identities"]["runtime_manifest_sha256"] ==
            stock["identities"]["runtime_manifest_sha256"])
    assert (b8["identities"]["atlas_catalog_sha256"] ==
            generated["identities"]["atlas_catalog_sha256"])
    assert _schema("q2-multires-b8-gate-v1.schema.json")["$id"].endswith(
        gate.B8_GATE_SCHEMA
    )
    assert not b8["public_mutation_authorized"]
    b8_path = tmp_path / "b8.json"
    _publish(b8_path, b8)

    restart_root = tmp_path / "restart"
    restart_files = _common_files(restart_root, atlas_bytes=b"atlas-inventory")
    restart_files["b8_gate"] = restart_root / "b8_gate.json"
    _publish(restart_files["b8_gate"], b8)
    restart_files["reconstruction_manifest"] = (
        restart_root / "reconstruction_manifest.json"
    )
    _write(
        restart_files["reconstruction_manifest"],
        _reconstruction_manifest(b8, restart_files),
    )
    restart_files["cold_restart_evidence"] = restart_root / "cold.json"
    cold_body = {
        "schema": gate.COLD_RESTART_SCHEMA, "restart_id": "restart-1",
        "b8_gate_sha256": b8["gate_sha256"], "artifact_sha256": {},
        "shadow_stopped_cleanly": True, "reconstructed_from_attested_archive": True,
        "reconstruction_manifest_sha256": _record(
            restart_files["reconstruction_manifest"]
        )["sha256"],
        "server_reconnected": True,
        "ml_clients_reconnected": 4, "policy_loaded": True,
        "stock_maps_loaded": 1, "generated_maps_loaded": 1, "map_changes": 2,
        "accepted_transitions": 4, "failed_rounds": 0, "echo_timeouts": 0,
        "legacy_absence_evidence_sha256": _legacy()["evidence_sha256"],
        "operational_legacy_selector_matches": [], "automatic_promotion": False,
        "public_mutations": [],
    }
    cold_body["artifact_sha256"] = {
        role: _record(path)["sha256"] for role, path in restart_files.items()
        if role != "cold_restart_evidence"
    }
    _write(restart_files["cold_restart_evidence"], _seal(cold_body))
    restart_manifest = _freeze_archive(restart_root, "restart", None, restart_files)
    decision = gate.assemble_b9(b8_path, restart_manifest)
    assert decision["eligible"]
    decision_path = tmp_path / "b9-decision.json"
    _publish(decision_path, decision)
    assert gate.validate_document(decision_path) == decision
    assert _schema("q2-multires-b9-promotion-decision-v1.schema.json")[
        "$id"
    ].endswith(gate.B9_DECISION_SCHEMA)
    reconstruction_schema = _schema(
        "q2-multires-b9-reconstruction-manifest-v1.schema.json"
    )
    assert reconstruction_schema["$id"].endswith(
        gate.RECONSTRUCTION_MANIFEST_SCHEMA
    )
    assert reconstruction_schema["properties"]["artifacts"]["minProperties"] == 20
    decision_schema = _schema("q2-multires-b9-promotion-decision-v1.schema.json")
    assert "reconstruction_manifest" in decision_schema["properties"]["inputs"][
        "required"
    ]
    assert decision["decision"] == "eligible-for-root-manual-promotion"
    assert decision["root_manual_promotion_required"]
    assert decision["inputs"]["reconstruction_manifest"] == {
        **_record(restart_files["reconstruction_manifest"]),
        "manifest_sha256": json.loads(
            restart_files["reconstruction_manifest"].read_text()
        )["manifest_sha256"],
    }
    assert not decision["automatic_promotion"]
    assert not decision["public_mutation_performed"]
    assert not decision["public_mutation_authorized"]
    bad_cold = json.loads(json.dumps(_seal(cold_body)))
    bad_cold["server_reconnected"] = False
    bad_cold.pop("evidence_sha256")
    bad_cold = _seal(bad_cold)
    _, cold_predicate, _ = gate._validate_cold_restart(
        bad_cold, restart_files, b8, b8["gate_sha256"]
    )
    assert not cold_predicate["passed"]
    assert "server_reconnected" in " ".join(cold_predicate["failures"])
    mutating = json.loads(json.dumps(bad_cold))
    mutating["automatic_promotion"] = True
    mutating.pop("evidence_sha256")
    mutating = _seal(mutating)
    _, mutating_predicate, _ = gate._validate_cold_restart(
        mutating, restart_files, b8, b8["gate_sha256"]
    )
    assert not mutating_predicate["passed"]
    assert "mutation selector" in " ".join(mutating_predicate["failures"])


def test_b9_reconstruction_manifest_failures_are_rejected(tmp_path):
    b8, b8_path = _green_b8(tmp_path / "authority")

    absent, _ = _restart_archive(
        tmp_path / "restart-absent", b8, omit_reconstruction=True,
    )
    with pytest.raises(gate.GateError, match="missing=.*reconstruction_manifest"):
        gate.assemble_b9(b8_path, absent)

    symlinked, symlinked_files = _restart_archive(
        tmp_path / "restart-symlink", b8,
    )
    symlinked_root = symlinked.parent
    symlinked_root.chmod(0o755)
    reconstruction = symlinked_files["reconstruction_manifest"]
    reconstruction.unlink()
    reconstruction.symlink_to("atlas.json")
    symlinked_root.chmod(0o555)
    with pytest.raises(gate.GateError, match="non-symlink"):
        gate.assemble_b9(b8_path, symlinked)

    tampered, tampered_files = _restart_archive(
        tmp_path / "restart-tampered", b8,
    )
    tampered_root = tampered.parent
    tampered_root.chmod(0o755)
    reconstruction = tampered_files["reconstruction_manifest"]
    reconstruction.chmod(0o644)
    tampered_bytes = bytearray(reconstruction.read_bytes())
    tampered_bytes[0] ^= 1
    reconstruction.write_bytes(tampered_bytes)
    reconstruction.chmod(0o444)
    tampered_root.chmod(0o555)
    with pytest.raises(gate.GateError, match="reconstruction_manifest SHA-256 differs"):
        gate.assemble_b9(b8_path, tampered)

    wrong_digest, _ = _restart_archive(
        tmp_path / "restart-wrong-digest", b8,
        wrong_reconstruction_digest=True,
    )
    with pytest.raises(
        gate.GateError, match="reconstruction manifest digest differs",
    ):
        gate.assemble_b9(b8_path, wrong_digest)

    rebound, _ = _restart_archive(
        tmp_path / "restart-rebound-artifact", b8,
        rebound_reconstruction_artifact=True,
    )
    with pytest.raises(gate.GateError, match="reconstruction policy binding differs"):
        gate.assemble_b9(b8_path, rebound)

    hardlinked, hardlinked_files = _restart_archive(
        tmp_path / "restart-hardlinked", b8,
    )
    os.link(
        hardlinked_files["reconstruction_manifest"],
        tmp_path / "external-reconstruction-hardlink.json",
    )
    with pytest.raises(gate.GateError, match="hard-linked member rejected"):
        gate.assemble_b9(b8_path, hardlinked)


@pytest.mark.parametrize("field", gate.SEASON_DISTINCT_FIELDS)
def test_b8_rejects_duplicate_or_relabelled_season_evidence(tmp_path, field):
    generated = gate.assemble_season(
        _season_archive(tmp_path / f"generated-{field}", "generated")
    )
    stock = gate.assemble_season(
        _season_archive(tmp_path / f"stock-{field}", "stock")
    )
    _copy_path(stock, generated, _SEASON_IDENTITY_PATHS[field])
    _reseal_gate(stock)
    generated_path = tmp_path / f"generated-{field}.json"
    stock_path = tmp_path / f"stock-{field}.json"
    _publish(generated_path, generated)
    _publish(stock_path, stock)

    with pytest.raises(gate.GateError, match=f"reuse {field}"):
        gate.assemble_b8(
            generated_path, stock_path, tmp_path / "unreachable-shadow.json",
        )


def test_b8_validator_rejects_resealed_duplicate_season_record(tmp_path):
    generated = gate.assemble_season(
        _season_archive(tmp_path / "generated-record", "generated")
    )
    stock = gate.assemble_season(
        _season_archive(tmp_path / "stock-record", "stock")
    )
    records = {
        "generated": gate._season_identity(generated, "generated"),
        "stock": gate._season_identity(stock, "stock"),
    }
    records["stock"]["season_id"] = records["generated"]["season_id"]
    forged = {
        "schema": gate.B8_GATE_SCHEMA,
        "predicates": {
            name: {"passed": True, "failures": []} for name in (
                "G0", "G1_generated", "G1_stock", "G2_generated", "G2_stock",
                "G3_generated", "G3_stock", "G4_generated", "G4_stock",
                "G5_generated", "G5_stock", "G6_shadow",
            )
        },
        "failures": [], "passed": True, "cold_restart_pending": True,
        "automatic_promotion": False, "public_mutation_authorized": False,
        "season_independence": records,
    }
    forged["gate_sha256"] = gate.canonical_sha256({
        "domain": gate.B8_GATE_SCHEMA, "gate": forged,
    })
    with pytest.raises(gate.GateError, match="reuse season_id"):
        gate._validate_b8_gate(forged)


@pytest.mark.parametrize("mutation,expected", [
    (lambda e, c, f: (
        c["counters"].update(accepted_transitions=16_383),
        c["causal_metrics_window"].update(accepted_transitions=16_383),
        c["network_metrics_window"].update({
            "network_client/transitions_accepted": 16_383,
        }),
        c.update(
            causal_metrics_window_sha256=gate.canonical_sha256(
                c["causal_metrics_window"]
            ),
            network_metrics_window_sha256=gate.canonical_sha256(
                c["network_metrics_window"]
            ),
        ),
    ), "accepted transitions"),
    (lambda e, c, f: e["g1"].update(vertical_intent_echo_match_rate=0.98),
     "vertical echo match"),
    (lambda e, c, f: e["g2"].update(downlook_rate=0.11), "down-look rate"),
    (lambda e, c, f: e["g2"].update(backward_command_rate=0.41),
     "backward command rate"),
    (lambda e, c, f: e["g2"]["action_collapse_audits"]["hook"].update(
        collapsed=True), "hook action collapsed"),
    (lambda e, c, f: e["g3"].update(treatment_safe_arrivals=30),
     "safe-arrival rate did not improve"),
    (lambda e, c, f: e["g3"].update(boundary_oscillation_credits=1),
     "boundary oscillation"),
    (lambda e, c, f: e["g4"].update(guide_off_successes=60),
     "guide-off degradation"),
    (lambda e, c, f: e["g4"].update(global_dropout_after_stage4_rate=0),
     "global guide dropout"),
    (lambda e, c, f: e["g5"].update(aligned_fire_precision_min=0.80),
     "weaker than 85%"),
    (lambda e, c, f: c["causal_metrics_window"]["combat"].update(hidden_fire=1),
     "hidden fire"),
    (lambda e, c, f: c["causal_metrics_window"]["combat"].update(
        visible_contact_yaw_mae_deg=13.0), "yaw MAE"),
])
def test_season_critical_bars_fail_closed(tmp_path, mutation, expected):
    # Current-report mutations require resealing and rebinding; these cases use
    # evaluator fields except the first malformed identity test.
    manifest = _season_archive(tmp_path / "season", "generated", mutate=mutation)
    result = gate.assemble_season(manifest)
    assert not result["passed"]
    assert expected in " ".join(result["failures"])


@pytest.mark.parametrize("mutation,expected", [
    (lambda e, c, f: e["g4"].update(
        b7_stage_evaluation_evidence_sha256="d" * 64,
    ), "different B7 stage evaluator"),
    (lambda e, c, f: c["causal_metrics_window"]["movement"].update(
        backward_command_mean=0.20,
    ), "backward rate differs from the causal window"),
    (lambda e, c, f: c["causal_metrics_window"]["combat"].update(
        kills=6,
    ), "combat ladder differs from the causal window"),
])
def test_g4_measurements_bind_archived_raw_evidence(
    tmp_path, mutation, expected,
):
    result = gate.assemble_season(
        _season_archive(tmp_path / "measurement", "generated", mutate=mutation)
    )
    assert not result["passed"]
    assert expected in " ".join(result["failures"])


def test_checked_schema_rejects_resealed_unknown_fields_and_type_drift(tmp_path):
    assembled = gate.assemble_season(
        _season_archive(tmp_path / "schema-source", "generated")
    )
    unknown = json.loads(json.dumps(assembled))
    unknown["inputs"]["undeclared"] = {"bytes": 1, "sha256": "d" * 64}
    _reseal_gate(unknown)
    unknown_path = tmp_path / "unknown-season-gate.json"
    _publish(unknown_path, unknown)
    with pytest.raises(gate.GateError, match="schema fields differ; extra"):
        gate.validate_document(unknown_path)

    drift = json.loads(json.dumps(assembled))
    drift["window"]["policy_start_version"] = "100"
    _reseal_gate(drift)
    drift_path = tmp_path / "type-drift-season-gate.json"
    _publish(drift_path, drift)
    with pytest.raises(gate.GateError, match="schema type"):
        gate.validate_document(drift_path)

    with pytest.raises(gate.GateError, match="unsupported schema keywords"):
        gate._audit_checked_schema(
            {"type": "object", "oneOf": []}, "mutated checked-in schema",
        )


def test_b8_and_b9_checked_schemas_reject_resealed_unknown_fields(tmp_path):
    b8, b8_path = _green_b8(tmp_path / "checked-chain")
    restart, _ = _restart_archive(tmp_path / "checked-restart", b8)
    decision = gate.assemble_b9(b8_path, restart)

    forged_b8 = json.loads(json.dumps(b8))
    forged_b8["undeclared"] = True
    forged_b8.pop("gate_sha256")
    forged_b8["gate_sha256"] = gate.canonical_sha256({
        "domain": gate.B8_GATE_SCHEMA, "gate": forged_b8,
    })
    b8_unknown = tmp_path / "b8-unknown.json"
    _publish(b8_unknown, forged_b8)
    with pytest.raises(gate.GateError, match="schema fields differ; extra"):
        gate.validate_document(b8_unknown)

    forged_b9 = json.loads(json.dumps(decision))
    forged_b9["undeclared"] = True
    forged_b9.pop("decision_sha256")
    forged_b9["decision_sha256"] = gate.canonical_sha256({
        "domain": gate.B9_DECISION_SCHEMA, "decision": forged_b9,
    })
    b9_unknown = tmp_path / "b9-unknown.json"
    _publish(b9_unknown, forged_b9)
    with pytest.raises(gate.GateError, match="schema fields differ; extra"):
        gate.validate_document(b9_unknown)


def test_g0_summary_cannot_be_resealed_only_at_the_outer_gate(tmp_path):
    assembled = gate.assemble_season(
        _season_archive(tmp_path / "g0-source", "generated")
    )
    forged = json.loads(json.dumps(assembled))
    forged["predicates"]["G0"]["evidence"]["assertions"][
        "legacy_selectors_absent"
    ] = False
    _reseal_gate(forged)
    path = tmp_path / "forged-g0.json"
    _publish(path, forged)
    with pytest.raises(gate.GateError, match="schema const|seal differs"):
        gate.validate_document(path)


def test_archive_tamper_symlink_and_legacy_fallback_are_rejected(tmp_path):
    manifest = _season_archive(tmp_path / "tamper", "generated")
    root = manifest.parent
    root.chmod(0o755)
    policy = root / "policy.json"
    policy.chmod(0o644)
    policy.write_bytes(b"same-polici")
    policy.chmod(0o444)
    root.chmod(0o555)
    with pytest.raises(gate.GateError, match="differs"):
        gate.assemble_season(manifest)

    manifest = _season_archive(tmp_path / "symlink", "generated")
    root = manifest.parent
    root.chmod(0o755)
    policy = root / "policy.json"
    policy.chmod(0o644)
    policy.unlink()
    policy.symlink_to("atlas.json")
    root.chmod(0o555)
    with pytest.raises(gate.GateError, match="non-symlink"):
        gate.assemble_season(manifest)

    manifest = _season_archive(tmp_path / "legacy", "generated")
    root = manifest.parent
    root.chmod(0o755)
    legacy_path = root / "legacy_selector_absence.json"
    legacy_path.chmod(0o644)
    legacy = json.loads(legacy_path.read_text())
    legacy["legacy_policy_selectors"] = ["old.pt"]
    legacy.pop("evidence_sha256")
    _write(legacy_path, _seal(legacy))
    legacy_path.chmod(0o444)
    root.chmod(0o555)
    # A changed archive member is rejected before its self-assertion can help.
    with pytest.raises(gate.GateError, match="differs"):
        gate.assemble_season(manifest)


def test_source_identity_must_equal_archived_b6_source(tmp_path):
    files = _common_files(tmp_path / "source-rebound")
    b6_path = files[gate.INTEGRATION_EVIDENCE_ROLES["wsl_b6_campaign"]]
    b6 = json.loads(b6_path.read_text())
    b6["source_repositories"]["bot"]["tree"] = "d" * 40
    _write(b6_path, b6)
    report = _integration(files["integration_envelope"])
    files["integration_report"].write_bytes(gate.canonical_bytes(report) + b"\n")
    with pytest.raises(gate.GateError, match="source identity differs from archived B6"):
        gate._validate_identity_artifacts(files)


def test_identity_validation_rejects_missing_atlas_catalog_as_gate_error(tmp_path):
    files = _common_files(tmp_path / "catalog-missing")
    files.pop("atlas_catalog")
    with pytest.raises(gate.GateError, match="lack the admitted Atlas catalog"):
        gate._validate_identity_artifacts(files)


def test_b7_gate_records_must_bind_archived_completed_stage_and_evaluator(tmp_path):
    manifest = _season_archive(tmp_path / "b7-binding", "generated")
    _, files = gate.validate_archive(manifest, kind="season")
    b7 = json.loads(files["b7_gate"].read_text())
    rebound = tmp_path / "rebound-current.json"
    _write(rebound, {"different": True})
    substituted = dict(files)
    substituted["current_season"] = rebound
    with pytest.raises(gate.GateError, match="completed-stage binding differs"):
        gate._validate_b7_gate(b7, substituted)
    substituted = dict(files)
    substituted["b7_stage_evaluation"] = rebound
    with pytest.raises(gate.GateError, match="evaluator binding differs"):
        gate._validate_b7_gate(b7, substituted)


@pytest.mark.parametrize("field,value,expected", [
    ("maxclients", 8, "maxclients is not 6"),
    ("compile_processes", 1, "compile_processes is nonzero"),
    ("human_joins", 0, "human_joins was not exercised"),
    ("accepted_transitions", 16_383, "fewer than 16384"),
])
def test_g6_shadow_critical_rejections(field, value, expected):
    evidence = {
        "topology": {"maxclients": 6, "ml_clients": 4, "human_slots": 2},
        "vps_work": {"compile_processes": 0, "analyze_processes": 0,
                     "compile_launches": 0, "analyze_launches": 0},
        "rotation": {"stock_maps": 1, "generated_maps": 1, "interlaced": True,
                     "queue_prefix_isolation": True},
        "events": {"human_joins": 1, "human_leaves": 1, "map_changes": 2},
        "transport": {"accepted_transitions": 16_384, "failed_rounds": 0,
                      "echo_timeouts": 0, "authoritative_echo_accept_rate": 0.99,
                      "vertical_intent_echo_match_rate": 1.0,
                      "water_land_projection_skew": 0, "map_epoch_recoveries": 1,
                      "telemetry_gap_recoveries": 1,
                      "partial_client_timeout_remained_fatal": True},
    }
    owner = next(
        section for section in (evidence["topology"], evidence["vps_work"],
                                evidence["events"], evidence["transport"])
        if field in section
    )
    owner[field] = value
    result = gate._g6(evidence)
    assert not result["passed"]
    assert expected in " ".join(result["failures"])

def test_cli_exclusive_output_and_validation(tmp_path):
    source_manifest = _season_archive(tmp_path / "source-season", "generated")
    _, source_files = gate.validate_archive(source_manifest, kind="season")
    archive = tmp_path / "authored-season"
    arguments = [
        "archive", "--kind", "season", "--season-kind", "generated",
        "--archive-id", "generated-quality-1", "--out-dir", str(archive),
    ]
    for role, path in source_files.items():
        if role not in gate.INTEGRATION_EVIDENCE_ROLES.values():
            arguments.extend(["--artifact", f"{role}={path}"])
    assert gate.main(arguments) == 0
    manifest = archive / "archive-manifest.json"
    assert not (manifest.stat().st_mode & 0o222)
    assert not (archive.stat().st_mode & 0o222)
    output = tmp_path / "gate.json"
    assert gate.main(["season", "--archive-manifest", str(manifest),
                      "--out", str(output)]) == 0
    assert gate.main(["validate", "--document", str(output)]) == 0
    assert gate.main(["season", "--archive-manifest", str(manifest),
                      "--out", str(output)]) == 2
