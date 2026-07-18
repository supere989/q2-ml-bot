import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

try:
    import torch
except ImportError:  # qualification hosts must provide it; unit host may not
    torch = None

from harness.multires_contract import (
    ACTION_DIM,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
)
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
from harness.multires_training_config import MultiresTrainingConfiguration
from harness.multires_lineage import save_attested_checkpoint
from harness.atlas_catalog import (
    ATLAS_MAP_SPEC_SCHEMA,
    author_catalog,
    canonical_bytes as catalog_bytes,
    load_atlas_catalog,
)
from harness.runtime_attestation import MANIFEST_SCHEMA, semantic_digest
from harness.multires_reward import CausalRewardConfig
from harness.multires_contract import GuideDropoutConfig
from train.multires_one_run import (
    OneRunError,
    _process_record as _one_run_process_record,
    _process_record_alive as _one_run_process_record_alive,
    _records,
    preflight,
)
import train.multires_one_run as one_run_module
from tools.qualify_network_client_frame_barrier import (
    _validate_execution_evidence as _real_validate_execution_evidence,
)
from train.multires_service import (
    MultiresServiceError,
    _owned_process_tree,
    _acquire_lease,
    _release_lease,
    _transfer_lease,
    _process_record,
    _record_alive,
    _reconcile_zero_update_attempt,
    _state_path,
    _write_state,
    parse_args as service_parse_args,
    service_status,
    service_preflight,
    start_service,
    stop_service,
)
import train.multires_service as service_module
from train.multires_train import CurriculumStage
from train.multires_primary import (
    PRIMARY_ATTEMPT_FIELDS,
    PRIMARY_TERMINAL_FIELDS,
    PrimaryTrainingError,
    _write_attempt_evidence,
    _write_terminal_evidence,
    admit_primary_training,
)

if torch is not None:
    from models.multires_policy import MultiresQ2BotPolicy


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, name: str) -> dict:
    return {"name": name, "sha256": sha(path), "size": path.stat().st_size}


def materialize(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    root = tmp_path.resolve()
    runtime = root / "runtime"
    bundle = root / "bundle"
    q2 = root / "q2"
    for directory in (runtime, bundle, q2 / "lithium" / "maps", root / "clients"):
        directory.mkdir(parents=True)
    q2ded = q2 / "q2ded"
    client = root / "yquake2"
    game = q2 / "lithium" / "game.so"
    extension = root / "q2_lattice_rs.so"
    for path, payload in (
        (q2ded, b"q2ded"), (client, b"client"), (game, b"game"),
        (extension, b"rust-extension"),
    ):
        path.write_bytes(payload)
    q2ded.chmod(0o755)
    client.chmod(0o755)

    name = "qualmap"
    bsp = bundle / f"{name}.bsp"
    bsp.write_bytes(b"bsp")
    installed = q2 / "lithium" / "maps" / bsp.name
    installed.write_bytes(bsp.read_bytes())
    objectives = bundle / f"{name}.objectives.json"
    objectives.write_text(json.dumps({
        "schema": "q2-atlas-objectives-v1",
        "atlas_sha256": "f" * 64,
        "canonical_map_id": name,
        "origin": [0, 0, 0],
        "objectives": [],
    }, sort_keys=True), encoding="utf-8")
    atlas = bundle / f"{name}.atlas.bin"
    atlas.write_bytes(b"atlas")
    atlas_sha = sha(atlas)
    objectives.write_text(json.dumps({
        "schema": "q2-atlas-objectives-v1",
        "atlas_sha256": atlas_sha,
        "canonical_map_id": name,
        "origin": [0, 0, 0],
        "objectives": [],
    }, sort_keys=True), encoding="utf-8")
    atlas_manifest = bundle / f"{name}.atlas.manifest.json"
    atlas_manifest.write_text(json.dumps({
        "grid": {"origin": [0, 0, 0]},
        "bsp": {"canonical_map_id": name, "sha256": sha(bsp),
                "size_bytes": bsp.stat().st_size},
        "artifacts": {f"{name}.atlas.bin": {
            "sha256_uncompressed": atlas_sha,
            "uncompressed_size": atlas.stat().st_size,
        }},
    }, sort_keys=True), encoding="utf-8")
    bundle_manifest = bundle / f"{name}.bundle.json"
    bundle_manifest.write_text(json.dumps({
        "bundle_version": 3,
        "artifact_state": "admitted",
        "name": name,
        "files": {bsp.name: sha(bsp)},
        "analysis_files": {
            objectives.name: sha(objectives),
            atlas_manifest.name: sha(atlas_manifest),
        },
    }, sort_keys=True), encoding="utf-8")

    class FakeDynRuntime:
        @staticmethod
        def empty(atlas_id, map_id, origin, epoch, client_id, count, steps):
            return SimpleNamespace(
                atlas_sha256=atlas_id, map_sha256=map_id,
                origin=tuple(origin), map_epoch=epoch, client_id=client_id,
                client_count=count, environment_steps=steps,
                client_life_epoch=0, server_frame=0, last_event_id=0,
                accepted_event_count=0, cell_count=0,
            )

    fake_extension = SimpleNamespace(
        __file__=str(extension.resolve()), DynRuntime=FakeDynRuntime,
    )
    monkeypatch.setattr(one_run_module, "_load_extension", lambda _path: fake_extension)
    atlas_catalog = root / "atlas-catalog.json"
    catalog_document = author_catalog(
        [{
            "schema": ATLAS_MAP_SPEC_SCHEMA, "map_name": name,
            "bsp": str(bsp.resolve()), "atlas": str(atlas.resolve()),
            "atlas_manifest": str(atlas_manifest.resolve()),
            "bundle_manifest": str(bundle_manifest.resolve()),
            "objectives": str(objectives.resolve()),
        }],
        catalog_path=atlas_catalog,
        rust_extension_path=extension,
        extension_module=fake_extension,
    )
    atlas_catalog.write_bytes(catalog_bytes(catalog_document) + b"\n")
    atlas_catalog_sha = catalog_document["atlas_catalog_sha256"]

    training = MultiresTrainingConfiguration.create(
        reward=CausalRewardConfig(),
        guide_dropout=GuideDropoutConfig(),
        ppo={
            "clip_coef": 0.2, "value_coef": 0.5, "entropy_coef": 0.01,
            "max_grad_norm": 0.5, "epochs": 4,
            "normalize_advantage": True,
        },
    )
    training_manifest = root / "training.json"
    training_manifest.write_text(training.to_json(), encoding="utf-8")
    run = root / "runs" / "public_network_multires_atlas_fresh_v1"
    run_roots = {
        "current_run_root": run,
        "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard",
        "rollout_root": run / "evidence" / "rollouts",
        "update_root": run / "evidence" / "updates",
        "season_report_root": run / "season",
    }
    for path in run_roots.values():
        path.mkdir(parents=True, exist_ok=True)
    checkpoint = run_roots["checkpoint_root"] / "fresh-step-zero.pt"
    checkpoint_attestation = runtime / "checkpoint-attestation.json"
    retirement = ROOT / "docs" / "multires" / "M4-RUNTIME-RETIREMENT.json"
    barrier = root / "network-barrier.json"
    source_repositories = {
        name: {"commit": character * 40, "tree": character * 40, "clean": True}
        for name, character in (("bot", "a"), ("client", "b"), ("game", "c"))
    }
    source_closure = hashlib.sha256(_canonical(source_repositories)).hexdigest()
    execution_payload = {
        "schema": "q2-network-client-frame-barrier-execution-v1",
        "test_mode": False,
        "full_network_executed": True,
        "design_sha256": sha(ROOT / "docs" / "NETWORK-CLIENT-FRAME-BARRIER.md"),
        "source_repositories": source_repositories,
        "source_closure_sha256": source_closure,
        "runtime_binaries": {
            "q2ded": record(q2ded, "q2ded"),
            "game_module": record(game, "game.so"),
            "client_binary": record(client, "yquake2"),
        },
        "mode": "client-telemetry-frame-ack-v1",
        "protocol_version": 1,
        "enabled_cvar": "ml_client_frame_barrier",
        "fault_injection_passed": True,
        "ack_timeout_rejection_passed": True,
        "unsupported_mode_rejected": True,
        "deterministic_client_id_slot_admission": True,
        "all_clients_registered_before_bootstrap": True,
        "action_free_bootstrap_frames": 1,
        "fresh_usercmd_per_client_frame": True,
        "modulo_generation_enforced": True,
        "usercmd_application_order": "client-id-then-slot",
        "reliable_hook_weapon_deferred_ordered": True,
        "automatic_promotion": False,
    }
    execution = {
        **execution_payload,
        "execution_evidence_sha256": hashlib.sha256(
            _canonical(execution_payload)
        ).hexdigest(),
    }
    monkeypatch.setattr(
        one_run_module, "_validate_execution_evidence",
        lambda document, source: dict(document),
    )
    monkeypatch.setattr(
        one_run_module, "_current_source_repositories",
        lambda: source_repositories,
    )
    optimizer = {
        "class": "torch.optim.Adam", "learning_rate": 1e-5, "kwargs": {}
    }
    dyn = []
    dyn_records = []
    for index in range(4):
        path = root / f"client{index}.q2lat002"
        path.write_bytes(b"Q2LAT002" + bytes([index]))
        dyn.append(str(path))
        dyn_records.append({
            "client_id": index, "sha256": sha(path),
            "size": path.stat().st_size,
        })
    semantic = {
        "artifacts": {
            "q2ded": record(q2ded, "q2ded"),
            "game_module": record(game, "lithium/game.so"),
            "rust_lattice": {"enabled": True, **record(extension, "q2_lattice_rs")},
        },
        "maps": [{"name": name, "files": [record(
            installed, f"lithium/maps/{name}.bsp"
        )]}],
        "runtime_config": {
            "proof_module": "train.multires_one_run",
            "trainer_module": "train.multires_primary",
            "legacy_fallback_enabled": False,
            "deterministic_collection": True,
            "client_count": 4,
            "use_startobserver": 0,
            "use_startchasecam": 0,
            "training_config_sha256": training.sha256,
            "retirement_manifest_sha256": sha(retirement),
            "network_barrier_execution_evidence_sha256": execution[
                "execution_evidence_sha256"
            ],
            "optimizer": optimizer,
            "client_binary_sha256": sha(client),
            "client_binary_size": client.stat().st_size,
            "dyn_snapshots": dyn_records,
        },
    }
    runtime_digest = semantic_digest(semantic)
    (runtime / "runtime-manifest.json").write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "semantic": semantic,
        "manifest_sha256": runtime_digest,
    }, sort_keys=True), encoding="utf-8")
    barrier_payload = {
        "schema": "q2-network-client-frame-barrier-qualification-v1",
        "passed": True,
        "mode": "client-telemetry-frame-ack-v1",
        "protocol_version": 1,
        "test_mode": False,
        "non_admissible_for_training": True,
        "runtime_manifest_sha256": runtime_digest,
        "execution_evidence_sha256": execution["execution_evidence_sha256"],
        "runtime_closure_sha256": hashlib.sha256(_canonical({
            "runtime_manifest_sha256": runtime_digest,
            "execution_evidence_sha256": execution[
                "execution_evidence_sha256"
            ],
        })).hexdigest(),
        "execution_evidence": execution,
    }
    barrier.write_text(json.dumps({
        **barrier_payload,
        "evidence_sha256": hashlib.sha256(_canonical(barrier_payload)).hexdigest(),
    }, sort_keys=True), encoding="utf-8")
    evidence = {
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
        "atlas_sha256": atlas_sha,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
        "runtime_manifest_sha256": runtime_digest,
    }
    runtime_evidence = root / "runtime-evidence.json"
    runtime_evidence.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    if torch is not None:
        torch.manual_seed(13)
        checkpoint_policy = MultiresQ2BotPolicy()
        checkpoint_optimizer = torch.optim.Adam(
            checkpoint_policy.parameters(), lr=float(optimizer["learning_rate"])
        )
        checkpoint_manifest = save_attested_checkpoint(
            checkpoint,
            checkpoint_policy,
            atlas_catalog_sha256=atlas_catalog_sha,
            runtime_manifest_sha256=runtime_digest,
            training_config=training,
            initialization="random",
            training_step=0,
            optimizer=checkpoint_optimizer,
        )
    else:
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("unavailable/data.pkl", b"torch unavailable fixture")
        checkpoint_manifest = SimpleNamespace(
            checkpoint_format="q2-multires-attested-checkpoint-v1",
            policy_generation="multires-atlas-policy-v1",
            architecture="models.multires_policy.MultiresQ2BotPolicy",
            initialization="random",
            training_step=0,
            observation_dim=298,
            action_dim=8,
            posture_classes=3,
            lineage_root_sha256="a" * 64,
        )
    checkpoint_attestation.write_text(json.dumps({
        "schema": "q2-multires-cold-checkpoint-v1",
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_format": checkpoint_manifest.checkpoint_format,
        "policy_generation": checkpoint_manifest.policy_generation,
        "architecture": checkpoint_manifest.architecture,
        "initialization": checkpoint_manifest.initialization,
        "training_step": checkpoint_manifest.training_step,
        "observation_dim": checkpoint_manifest.observation_dim,
        "action_dim": checkpoint_manifest.action_dim,
        "posture_classes": checkpoint_manifest.posture_classes,
        "atlas_catalog_sha256": atlas_catalog_sha,
        "optimizer_state": "fresh-empty",
        "normalization_state": "fresh-empty",
    }, sort_keys=True), encoding="utf-8")
    stage = CurriculumStage.create(1, {
        "objective": "transport-posture-water-death-screen-echo",
        "map_name": name,
    })
    primary_training = runtime / "multires-training-runtime.json"
    primary_training.write_text(json.dumps({
        "schema": "q2-multires-primary-training-runtime-v1",
        "runtime_manifest_sha256": runtime_digest,
        "atlas_catalog": {"path": str(atlas_catalog), "sha256": sha(atlas_catalog)},
        "atlas_catalog_sha256": atlas_catalog_sha,
        "network_barrier_execution_evidence_sha256": execution[
            "execution_evidence_sha256"
        ],
        "seed": 7,
        "game_seed": 8,
        "map_name": name,
        "map_epoch": 0,
        "collector": {
            "transitions_per_client": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "maximum_boundary_rounds": 64,
        },
        "optimizer": optimizer,
        "curriculum": {
            "stage_id": stage.stage_id,
            "configuration": json.loads(stage.configuration_json)["configuration"],
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": None,
            "predecessor_gate": None,
        },
        "execution": {
            "maximum_updates": 1,
            "continuous": False,
            "device": "cpu",
            "deterministic_collection": True,
            "deterministic_algorithms": True,
            "cuda_cublas_workspace_config": None,
            "shutdown_grace_seconds": 600.0,
        },
        "checkpoint": {
            "mode": "fresh-step-zero",
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "lineage_root_sha256": checkpoint_manifest.lineage_root_sha256,
            "current_season_report": None,
        },
    }, sort_keys=True), encoding="utf-8")

    # The service consumes only an exact precomputed report that byte-matches
    # a fresh invocation of the final B2-B6 integration verifier.  Most tests
    # in this module exercise service ownership rather than the verifier's
    # nine gate algorithms, so use a sealed synthetic result while preserving
    # the real service-side identity and file-binding checks.
    integration_root = root / "integration"
    integration_root.mkdir()
    integration_bot_source = {"commit": "a" * 40, "tree": "b" * 40}
    monkeypatch.setattr(
        service_module,
        "_current_bot_source_identity",
        lambda: {**integration_bot_source, "clean": True},
    )
    integration_paths = {}
    for gate in service_module.INTEGRATION_GATE_ORDER:
        path = integration_root / f"{gate}.json"
        payload = {}
        if gate == "bundle_v3_atlas":
            payload = {"atlas_sha256": atlas_sha}
        elif gate == "lineage_attestation":
            payload = {
                "atlas_sha256": atlas_sha,
                "runtime_manifest_sha256": runtime_digest,
            }
        elif gate == "legacy_selector_deactivation":
            payload = {"retirement_manifest_sha256": sha(retirement)}
        elif gate == "wsl_b6_campaign":
            bound_paths = {
                "runtime_evidence": runtime_evidence,
                "runtime_manifest": runtime / "runtime-manifest.json",
                "checkpoint": checkpoint,
                "training_manifest": training_manifest,
                "objectives": objectives,
                "bundle_manifest": bundle_manifest,
                "atlas": atlas,
                "retirement_manifest": retirement,
            }
            payload = {
                "source_repositories": {
                    "bot": {**integration_bot_source, "clean": True}
                },
                "bindings": {
                    **{
                        key: {
                            "bytes": value.stat().st_size,
                            "sha256": sha(value),
                        }
                        for key, value in bound_paths.items()
                    },
                    "atlas_sha256": atlas_sha,
                    "runtime_manifest_identity_sha256": runtime_digest,
                },
            }
        path.write_bytes(_canonical(payload) + b"\n")
        integration_paths[gate] = path.name
    integration_envelope = integration_root / "envelope.json"
    integration_envelope.write_bytes(_canonical({
        "schema": service_module.INTEGRATION_ENVELOPE_SCHEMA,
        "evidence": integration_paths,
    }) + b"\n")
    integration_body = {
        "schema": service_module.INTEGRATION_REPORT_SCHEMA,
        "tool": "verify_multires_integration",
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "envelope_sha256": sha(integration_envelope),
        "gates": [
            {
                "rank": rank,
                "gate": gate,
                "status": "pass",
                "evidence_sha256": sha(integration_root / integration_paths[gate]),
                "reasons": [],
            }
            for rank, gate in enumerate(
                service_module.INTEGRATION_GATE_ORDER, start=1
            )
        ],
        "failed_gates": [],
        "overall": "pass",
    }
    integration_report_value = {
        **integration_body,
        "report_sha256": hashlib.sha256(_canonical(integration_body)).hexdigest(),
    }
    integration_report = integration_root / "report.json"
    integration_report.write_bytes(_canonical(integration_report_value) + b"\n")
    frozen_integration_report = json.loads(json.dumps(integration_report_value))
    monkeypatch.setattr(
        service_module,
        "run_integration_gates",
        lambda _path: json.loads(json.dumps(frozen_integration_report)),
    )
    cold_start = runtime / "cold-start.json"
    cold_start.write_text(json.dumps({
        "schema": "q2-multires-cold-start-v2",
        "retirement_manifest_sha256": sha(retirement),
        "runtime_manifest_sha256": runtime_digest,
        "atlas_catalog_sha256": atlas_catalog_sha,
        "optimizer": optimizer,
        "selectors": {
            "service_module": "train.multires_service",
            "proof_module": "train.multires_one_run",
            "trainer_module": "train.multires_primary",
            "client_builder": "harness.client_batch.build_network_client_batch",
            "collector": "harness.multires_collector.MultiresSynchronousCollector",
            "policy_class": "models.multires_policy.MultiresQ2BotPolicy",
            "provider_class": (
                "harness.rust_multires_provider.RustAtlasSpatialProvider"
            ),
            "rust_dyn_call": "q2_lattice_rs.DynRuntime.commit_frame",
        },
        "lineage": {
            "run_tag": "public_network_multires_atlas_fresh_v1",
            "initialization": "random",
            "training_step": 0,
            "optimizer_state": "fresh-empty",
            "normalization_state": "fresh-empty",
            **{key: str(value) for key, value in run_roots.items()},
        },
        "inputs": {
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            "checkpoint_attestation": {
                "path": str(checkpoint_attestation),
                "sha256": sha(checkpoint_attestation),
            },
            "runtime_evidence": {
                "path": str(runtime_evidence), "sha256": sha(runtime_evidence),
            },
            "training_manifest": {
                "path": str(training_manifest), "sha256": sha(training_manifest),
            },
            "bundle_manifest": {
                "path": str(bundle_manifest), "sha256": sha(bundle_manifest),
            },
            "atlas_catalog": {
                "path": str(atlas_catalog), "sha256": sha(atlas_catalog),
            },
            "dyn_snapshots": [
                {"path": value, "sha256": sha(Path(value))} for value in dyn
            ],
            "training_runtime": {
                "path": str(primary_training), "sha256": sha(primary_training),
            },
            "trainer_checkpoint": {
                "path": str(checkpoint), "sha256": sha(checkpoint),
            },
            "trainer_current_season": None,
            "integration_envelope": {
                "path": str(integration_envelope),
                "sha256": sha(integration_envelope),
            },
            "integration_report": {
                "path": str(integration_report),
                "sha256": sha(integration_report),
            },
            "integration_bot_source": integration_bot_source,
        },
    }, sort_keys=True), encoding="utf-8")
    config = {
        "schema": "q2-multires-one-run-runtime-v1",
        "q2_root": str(q2), "rust_extension": str(extension),
        "dyn_snapshots": dyn, "retirement_manifest": str(retirement),
        "network_barrier_qualification": str(barrier),
        "client_count": 4, "server_host": "127.0.0.1", "server_port": 28010,
        "telemetry_host": "127.0.0.1", "telemetry_port": 28011,
        "telemetry_token_env": "Q2_ML_CLIENT_TELEMETRY_TOKEN",
        "harness_host": "127.0.0.1", "harness_port_base": 39000,
        "qport_base": 49000, "client_data_root": str(root / "clients"),
        "client_id_prefix": "qual", "name_prefix": "qual", "game": "lithium",
        "client_timeout": 8.0, "round_timeout": 2.0,
        "max_rejected_echoes": 16, "movement_tolerance": 0.05,
        "look_tolerance": 0.25,
        "maximum_boundary_rounds": 0, "device": "cpu", "optimizer": optimizer,
        "debug_clients": False, "server_warmup_seconds": 0.0,
    }
    (runtime / "multires-one-run.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setenv("Q2_ML_CLIENT_TELEMETRY_TOKEN", "x" * 32)
    return SimpleNamespace(
        seed=7, game_seed=8, q2ded=q2ded, client_binary=client,
        runtime_root=runtime, bundle_manifest=bundle_manifest,
        objectives=objectives, atlas_bin=atlas, checkpoint=checkpoint,
        training_manifest=training_manifest, runtime_evidence=runtime_evidence,
        atlas_catalog=atlas_catalog,
        transition_count=500, policy_version=2, map_epoch=0, map_name=name,
        out=root / "out.json", launch_id="same-a",
        expected_atlas_sha256=atlas_sha,
        expected_atlas_catalog_sha256=atlas_catalog_sha,
        expected_runtime_manifest_sha256=runtime_digest,
        retirement_manifest=retirement, retirement_cold_start=cold_start,
        primary_training=primary_training,
        integration_envelope=integration_envelope,
        integration_report=integration_report,
        integration_bot_source=integration_bot_source,
    )


def _integration_guard_config(args: SimpleNamespace) -> dict:
    return {
        "retirement_manifest_sha256": sha(args.retirement_manifest),
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "retirement_manifest": str(args.retirement_manifest),
        "proof": {
            "runtime_evidence": str(args.runtime_evidence),
            "checkpoint": str(args.checkpoint),
            "training_manifest": str(args.training_manifest),
            "objectives": str(args.objectives),
            "bundle_manifest": str(args.bundle_manifest),
            "atlas_bin": str(args.atlas_bin),
            "atlas_catalog": str(args.atlas_catalog),
            "expected_atlas_sha256": args.expected_atlas_sha256,
            "expected_atlas_catalog_sha256": args.expected_atlas_catalog_sha256,
        },
        "integration_admission": {
            "envelope": {
                "path": str(args.integration_envelope),
                "sha256": sha(args.integration_envelope),
            },
            "report": {
                "path": str(args.integration_report),
                "sha256": sha(args.integration_report),
            },
            "bot_source": dict(args.integration_bot_source),
        },
    }


def test_preflight_checks_exact_total_then_admits_executable_barrier(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    args.transition_count = 501
    with pytest.raises(OneRunError, match="divide exactly"):
        preflight(args)
    args.transition_count = 500
    if torch is None:
        with pytest.raises(OneRunError, match="PyTorch"):
            preflight(args)
    else:
        admission = preflight(args)
        assert admission.runtime_manifest_sha256 == (
            args.expected_runtime_manifest_sha256
        )


def test_one_run_server_config_seals_playable_client_role(tmp_path, monkeypatch):
    q2_root = tmp_path / "q2"
    (q2_root / "lithium").mkdir(parents=True)
    monkeypatch.setenv("Q2_ML_CLIENT_TELEMETRY_TOKEN", "x" * 32)
    admission = SimpleNamespace(
        args=SimpleNamespace(launch_id="role-seal", game_seed=8, map_name="q2dm1"),
        q2_root=q2_root,
        runtime_config={
            "telemetry_token_env": "Q2_ML_CLIENT_TELEMETRY_TOKEN",
            "game": "lithium",
            "client_count": 4,
            "telemetry_port": 27949,
        },
    )
    destination = one_run_module._write_server_config(admission)
    text = destination.read_text(encoding="utf-8")
    assert "set use_startobserver 0\n" in text
    assert "set use_startchasecam 0\n" in text
    assert destination.stat().st_mode & 0o077 == 0


def test_random_advertised_runtime_digest_is_rejected(tmp_path, monkeypatch):
    args = materialize(tmp_path, monkeypatch)
    args.expected_runtime_manifest_sha256 = "e" * 64
    with pytest.raises(OneRunError, match="sealed runtime manifest digest"):
        preflight(args)


def test_missing_network_frame_barrier_qualification_fails_closed(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    config_path = args.runtime_root / "multires-one-run.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["network_barrier_qualification"]).unlink()
    with pytest.raises(OneRunError, match="frame-barrier qualification"):
        preflight(args)


@pytest.mark.skipif(torch is None, reason="real weights-only loader requires PyTorch")
def test_marker_only_checkpoint_is_rejected_before_barrier_admission(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    with zipfile.ZipFile(args.checkpoint, "w") as archive:
        archive.writestr(
            "marker/data.pkl",
            b"checkpoint_format manifest policy_state optimizer_state "
            b"q2-multires-attested-checkpoint-v1 multires-atlas-policy-v1 "
            b"models.multires_policy.MultiresQ2BotPolicy",
        )
    with pytest.raises(OneRunError, match="step-zero checkpoint admission failed"):
        preflight(args)


def test_records_are_exact_500_time_major_client_minor(tmp_path, monkeypatch):
    args = materialize(tmp_path, monkeypatch)
    admission = SimpleNamespace(
        args=args,
        runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
    )
    observations = np.zeros((4, 125, OBS_DIM), dtype=np.float32)
    actions = np.zeros((4, 125, ACTION_DIM), dtype=np.float32)
    rewards = np.zeros((4, 125), dtype=np.float32)
    infos = []
    for tick in range(125):
        infos.append(tuple({
            "client_id": f"qual-{client:02d}",
            "server_frame": 100 + tick,
            "batch_round_id": tick,
            "policy_version": 2,
            "map": "qualmap",
            "_multires_spatial_attestation": {
                "map_epoch": 0,
                "atlas_sha256": args.expected_atlas_sha256,
                "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
            },
        } for client in range(4)))
    rollout = SimpleNamespace(
        observations=observations, actions=actions, rewards=rewards,
        infos=tuple(infos),
    )
    records, digest = _records(rollout, admission)
    assert len(records) == 500 and len(digest) == 64
    assert [record["client_id"] for record in records[:5]] == [
        "qual-00", "qual-01", "qual-02", "qual-03", "qual-00"
    ]
    assert [record["index"] for record in records] == list(range(500))
    assert all(len(record["rust_features"]) == 24 for record in records)


def test_service_is_fail_closed_until_executable_barrier_and_wrapper_is_exact(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    monkeypatch.setattr(
        one_run_module,
        "_validate_execution_evidence",
        _real_validate_execution_evidence,
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    one_run_config = json.loads(
        (args.runtime_root / "multires-one-run.json").read_text(encoding="utf-8")
    )
    retirement = Path(one_run_config["retirement_manifest"])
    service = {
        "schema": "q2-multires-service-v2",
        "retirement_manifest_sha256": sha(retirement),
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "retirement_manifest": str(args.retirement_manifest),
        "retirement_cold_start": str(args.retirement_cold_start),
        "operational_roots": [str(tmp_path.resolve())],
        "service_selectors": [str(ROOT / "tools" / "train_service.sh")],
        "modules": {
            "proof_module": "train.multires_one_run",
            "trainer_module": "train.multires_primary",
        },
        "proof": {
            "seed": args.seed,
            "game_seed": args.game_seed,
            "divergence_game_seed": args.game_seed + 1,
            "transition_count": 500,
            "policy_version": args.policy_version,
            "map_name": args.map_name,
            "map_epoch": args.map_epoch,
            "timeout_seconds": 60.0,
            "q2ded": str(args.q2ded),
            "client_binary": str(args.client_binary),
            "bundle_manifest": str(args.bundle_manifest),
            "objectives": str(args.objectives),
            "atlas_bin": str(args.atlas_bin),
            "atlas_catalog": str(args.atlas_catalog),
            "checkpoint": str(args.checkpoint),
            "training_manifest": str(args.training_manifest),
            "runtime_evidence": str(args.runtime_evidence),
            "expected_atlas_sha256": args.expected_atlas_sha256,
            "expected_atlas_catalog_sha256": args.expected_atlas_catalog_sha256,
            "expected_runtime_manifest_sha256": (
                args.expected_runtime_manifest_sha256
            ),
        },
        "training_runtime": {
            "path": str(args.primary_training),
            "sha256": sha(args.primary_training),
        },
        "integration_admission": {
            "envelope": {
                "path": str(args.integration_envelope),
                "sha256": sha(args.integration_envelope),
            },
            "report": {
                "path": str(args.integration_report),
                "sha256": sha(args.integration_report),
            },
            "bot_source": dict(args.integration_bot_source),
        },
        "evidence_dir": str(evidence),
        "log_path": str(args.runtime_root / "multires-service.log"),
        "tensorboard": {
            "enabled": False, "executable": "", "port": 6006,
            "bind_all": False,
        },
    }
    (args.runtime_root / "multires-service.json").write_text(
        json.dumps(service, sort_keys=True), encoding="utf-8"
    )
    if torch is None:
        with pytest.raises(
            MultiresServiceError, match="cold-start retirement validation failed"
        ):
            service_preflight(args.runtime_root)
    else:
        with pytest.raises(
            OneRunError,
            match="network frame-barrier execution evidence differs",
        ):
            service_preflight(args.runtime_root)

    wrapper = (Path(__file__).parents[1] / "tools" / "train_service.sh").read_text()
    assert "pkill" not in wrapper
    assert "train.ppo" not in wrapper and "models.policy" not in wrapper
    assert "train.multires_service" in wrapper
    service_source = (ROOT / "train" / "multires_service.py").read_text()
    assert 'PROOF_ROLE = "qualification-proof"' in service_source
    assert 'PRIMARY_ROLE = "primary-trainer"' in service_source
    assert 'choices=("preflight", "prove", "start", "stop", "status")' in service_source
    assert "run_training" not in service_source
    with pytest.raises(SystemExit):
        service_parse_args(["--runtime_root", str(tmp_path), "train"])


def test_service_preflight_exposes_primary_and_tb_watches_only_cold_root(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    one_run_config = json.loads(
        (args.runtime_root / "multires-one-run.json").read_text(encoding="utf-8")
    )
    retirement = Path(one_run_config["retirement_manifest"])
    service = {
        "schema": "q2-multires-service-v2",
        "retirement_manifest_sha256": sha(retirement),
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "retirement_manifest": str(args.retirement_manifest),
        "retirement_cold_start": str(args.retirement_cold_start),
        "operational_roots": [str(tmp_path.resolve())],
        "service_selectors": [str(ROOT / "tools" / "train_service.sh")],
        "modules": {
            "proof_module": "train.multires_one_run",
            "trainer_module": "train.multires_primary",
        },
        "proof": {
            "seed": args.seed, "game_seed": args.game_seed,
            "divergence_game_seed": args.game_seed + 1,
            "transition_count": 500, "policy_version": args.policy_version,
            "map_name": args.map_name, "map_epoch": args.map_epoch,
            "timeout_seconds": 60.0, "q2ded": str(args.q2ded),
            "client_binary": str(args.client_binary),
            "bundle_manifest": str(args.bundle_manifest),
            "objectives": str(args.objectives), "atlas_bin": str(args.atlas_bin),
            "atlas_catalog": str(args.atlas_catalog),
            "checkpoint": str(args.checkpoint),
            "training_manifest": str(args.training_manifest),
            "runtime_evidence": str(args.runtime_evidence),
            "expected_atlas_sha256": args.expected_atlas_sha256,
            "expected_atlas_catalog_sha256": args.expected_atlas_catalog_sha256,
            "expected_runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        },
        "training_runtime": {
            "path": str(args.primary_training), "sha256": sha(args.primary_training),
        },
        "integration_admission": {
            "envelope": {
                "path": str(args.integration_envelope),
                "sha256": sha(args.integration_envelope),
            },
            "report": {
                "path": str(args.integration_report),
                "sha256": sha(args.integration_report),
            },
            "bot_source": dict(args.integration_bot_source),
        },
        "evidence_dir": str(evidence),
        "log_path": str(args.runtime_root / "multires-service.log"),
        "tensorboard": {
            "enabled": True, "executable": str(Path(sys.executable).resolve()),
            "port": 6006, "bind_all": False,
        },
    }
    (args.runtime_root / "multires-service.json").write_text(
        json.dumps(service, sort_keys=True), encoding="utf-8"
    )
    retirement_report = {
        "schema": "q2-multires-retirement-validation-v1",
        "status": "pass", "read_only": True,
        "manifest_sha256": sha(retirement),
    }
    monkeypatch.setattr(
        service_module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps(retirement_report), stderr=""
        ),
    )

    def admitted_one_run(namespace):
        catalog = load_atlas_catalog(
            args.atlas_catalog,
            expected_sha256=args.expected_atlas_catalog_sha256,
            extension_module=one_run_module._load_extension(
                Path(one_run_config["rust_extension"])
            ),
        )
        return SimpleNamespace(
            args=namespace,
            runtime_root=args.runtime_root,
            runtime_config=one_run_config,
            runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
            network_barrier_execution_evidence_sha256=json.loads(
                args.primary_training.read_text(encoding="utf-8")
            )["network_barrier_execution_evidence_sha256"],
            runtime_evidence=json.loads(args.runtime_evidence.read_text()),
            training_configuration=MultiresTrainingConfiguration.from_json(
                args.training_manifest.read_text()
            ),
            retirement_manifest_sha256=sha(retirement),
            q2_root=Path(one_run_config["q2_root"]),
            q2ded=args.q2ded,
            client_binary=args.client_binary,
            bundle_manifest=args.bundle_manifest,
            objectives=args.objectives,
            atlas_bin=args.atlas_bin,
            dyn_snapshots=tuple(Path(value) for value in one_run_config["dyn_snapshots"]),
            rust_extension=Path(one_run_config["rust_extension"]),
            atlas_catalog=catalog,
        )

    monkeypatch.setattr(service_module, "one_run_preflight", admitted_one_run)
    admission = service_preflight(args.runtime_root)
    assert admission.primary.current_run_root == args.checkpoint.parents[1]
    assert admission.primary.tensorboard_root == args.checkpoint.parents[1] / "tensorboard"
    assert admission.tensorboard_command[1:3] == (
        "--logdir", str(admission.primary.tensorboard_root)
    )
    assert admission.trainer_command[2:4] == ("-m", "train.multires_primary")
    assert any("train.multires_one_run" in value for value in admission.proof_command)
    for field, bad_value in (
        ("enabled", 1), ("bind_all", "false"), ("port", "6006")
    ):
        broken = json.loads(json.dumps(service))
        broken["tensorboard"][field] = bad_value
        (args.runtime_root / "multires-service.json").write_text(
            json.dumps(broken, sort_keys=True), encoding="utf-8"
        )
        with pytest.raises(MultiresServiceError, match="tensorboard.*types"):
            service_preflight(args.runtime_root)

    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    event = admission.primary.tensorboard_root / "events.out.tfevents.stale"
    event.write_bytes(b"stale-run")
    with pytest.raises(PrimaryTrainingError, match="tensorboard_root must be empty"):
        admit_primary_training(
            SimpleNamespace(one_run=admission.one_run),
            config_path=args.primary_training,
            config_sha256=sha(args.primary_training),
            cold_document=cold,
        )
    event.unlink()

    broken_training = json.loads(args.primary_training.read_text(encoding="utf-8"))
    broken_training["curriculum"]["configuration_sha256"] = "d" * 64
    broken_path = args.runtime_root / "broken-primary-training.json"
    broken_path.write_text(json.dumps(broken_training), encoding="utf-8")
    with pytest.raises(PrimaryTrainingError, match="curriculum configuration digest"):
        admit_primary_training(
            SimpleNamespace(one_run=admission.one_run),
            config_path=broken_path,
            config_sha256=sha(broken_path),
            cold_document=cold,
        )


def test_service_v1_b4_era_config_is_rejected_before_runtime_state(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    legacy = {
        "schema": "q2-multires-service-v1",
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
    }
    (args.runtime_root / "multires-service.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    with pytest.raises(
        MultiresServiceError, match="service configuration fields/schema differ"
    ):
        service_preflight(args.runtime_root)
    assert not (args.runtime_root / "multires-service.lock").exists()
    assert not _state_path(args.runtime_root).exists()


def test_integration_report_rehash_cannot_rebind_stale_verification(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    config = _integration_guard_config(args)
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    args.integration_report.write_bytes(args.integration_report.read_bytes() + b"\n")
    rebound_sha = sha(args.integration_report)
    config["integration_admission"]["report"]["sha256"] = rebound_sha
    cold["inputs"]["integration_report"]["sha256"] = rebound_sha
    with pytest.raises(
        MultiresServiceError,
        match="report bytes differ from fresh verification",
    ):
        service_module._verify_integration_admission(
            runtime_root=args.runtime_root,
            config=config,
            cold_inputs=cold["inputs"],
            retirement_manifest=args.retirement_manifest,
        )


def test_integration_missing_report_is_rejected(tmp_path, monkeypatch):
    args = materialize(tmp_path, monkeypatch)
    config = _integration_guard_config(args)
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    args.integration_report.unlink()
    with pytest.raises(
        MultiresServiceError,
        match="precomputed integration report must be an absolute non-symlink file",
    ):
        service_module._verify_integration_admission(
            runtime_root=args.runtime_root,
            config=config,
            cold_inputs=cold["inputs"],
            retirement_manifest=args.retirement_manifest,
        )


def test_integration_stale_report_digest_is_rejected(tmp_path, monkeypatch):
    args = materialize(tmp_path, monkeypatch)
    config = _integration_guard_config(args)
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    args.integration_report.write_bytes(args.integration_report.read_bytes() + b"\n")
    with pytest.raises(
        MultiresServiceError,
        match="precomputed integration report byte digest differs",
    ):
        service_module._verify_integration_admission(
            runtime_root=args.runtime_root,
            config=config,
            cold_inputs=cold["inputs"],
            retirement_manifest=args.retirement_manifest,
        )


def test_integration_b6_exact_file_binding_rejects_changed_atlas(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    config = _integration_guard_config(args)
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    args.atlas_bin.write_bytes(b"changed-after-b6")
    with pytest.raises(
        MultiresServiceError,
        match="B6 atlas binding differs from service input",
    ):
        service_module._verify_integration_admission(
            runtime_root=args.runtime_root,
            config=config,
            cold_inputs=cold["inputs"],
            retirement_manifest=args.retirement_manifest,
        )


def test_integration_rejects_dirty_or_drifted_live_bot_source(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    config = _integration_guard_config(args)
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        service_module,
        "_current_bot_source_identity",
        lambda: {
            **args.integration_bot_source,
            "tree": "c" * 40,
            "clean": True,
        },
    )
    with pytest.raises(
        MultiresServiceError,
        match="runtime/Atlas/retirement/source identities differ",
    ):
        service_module._verify_integration_admission(
            runtime_root=args.runtime_root,
            config=config,
            cold_inputs=cold["inputs"],
            retirement_manifest=args.retirement_manifest,
        )


def test_current_bot_source_requires_exact_clean_git_worktree(
    tmp_path, monkeypatch
):
    repo = (tmp_path / "bot-source").resolve()
    repo.mkdir()
    monkeypatch.setattr(service_module, "ROOT", repo)
    with pytest.raises(
        MultiresServiceError, match="exact bot Git worktree root"
    ):
        service_module._current_bot_source_identity()

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    source = repo / "source.py"
    source.write_text("frozen = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "freeze"],
        check=True,
        capture_output=True,
    )
    identity = service_module._current_bot_source_identity()
    assert identity["clean"] is True
    assert len(identity["commit"]) == 40
    assert len(identity["tree"]) == 40

    source.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(MultiresServiceError, match="dirty or unreadable"):
        service_module._current_bot_source_identity()


def test_start_and_tensorboard_cannot_spawn_before_integration_revalidation(
    tmp_path, monkeypatch
):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    popen_called = False

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("Popen reached before integration admission")

    def reject(_admission):
        raise MultiresServiceError("fresh integration verification rejected")

    monkeypatch.setattr(service_module, "_revalidate_service_integration", reject)
    monkeypatch.setattr(service_module.subprocess, "Popen", forbidden_popen)
    with pytest.raises(
        MultiresServiceError, match="fresh integration verification rejected"
    ):
        start_service(SimpleNamespace(runtime_root=runtime))
    assert popen_called is False
    assert not (runtime / "multires-service.lock").exists()
    assert not _state_path(runtime).exists()
    assert not (runtime / "multires-service.log").exists()


def test_tensorboard_cannot_spawn_before_child_admission(
    tmp_path, monkeypatch
):
    runtime = (tmp_path / "runtime").resolve()
    run = (tmp_path / "run").resolve()
    runtime.mkdir()
    (run / "tensorboard").mkdir(parents=True)
    (run / "season").mkdir(parents=True)
    admission = SimpleNamespace(
        runtime_root=runtime,
        trainer_command=(sys.executable, "-c", "import time; time.sleep(60)"),
        tensorboard_command=(sys.executable, "-c", "import time; time.sleep(60)"),
        log_path=runtime / "multires-service.log",
        one_run=SimpleNamespace(runtime_manifest_sha256="e" * 64),
        current_run_root=run,
        tensorboard_root=run / "tensorboard",
        current_season_report=run / "season/current.json",
        child_inventory=runtime / "multires-primary-children.json",
        terminal_evidence=runtime / "multires-primary-terminal.json",
        launch_config=tmp_path / "multires_one_run_primary-trainer.cfg",
        shutdown_grace_seconds=5.0,
    )
    monkeypatch.setattr(
        service_module, "_revalidate_service_integration", lambda _value: None
    )
    monkeypatch.setattr(
        service_module,
        "_wait_for_primary_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MultiresServiceError("child admission rejected")
        ),
    )
    real_popen = subprocess.Popen
    commands = []

    def tracked_popen(*args, **kwargs):
        commands.append(tuple(args[0]))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(service_module.subprocess, "Popen", tracked_popen)
    with pytest.raises(MultiresServiceError, match="child admission rejected"):
        start_service(admission)
    assert commands == [admission.trainer_command]
    assert not _state_path(runtime).exists()
    assert not (runtime / "multires-service.lock").exists()


def test_child_admission_attempt_is_exactly_bound_before_observability(tmp_path):
    runtime = (tmp_path / "runtime").resolve()
    current = (tmp_path / "run").resolve()
    checkpoint_root = current / "checkpoints"
    tensorboard_root = current / "tensorboard"
    rollout_root = current / "evidence/rollouts"
    update_root = current / "evidence/updates"
    season_root = current / "season"
    for directory in (
        runtime, checkpoint_root, tensorboard_root, rollout_root,
        update_root, season_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / "step-zero.pt"
    checkpoint.write_bytes(b"step-zero")
    catalog_sha256 = "d" * 64
    admission = SimpleNamespace(
        runtime_root=runtime,
        one_run=SimpleNamespace(
            runtime_manifest_sha256="a" * 64,
            atlas_catalog=SimpleNamespace(
                atlas_catalog_sha256=catalog_sha256,
            ),
        ),
        primary=SimpleNamespace(
            config_sha256="b" * 64,
            checkpoint_mode="fresh-step-zero",
            checkpoint=checkpoint,
            checkpoint_root=checkpoint_root,
            rollout_root=rollout_root,
            update_root=update_root,
        ),
        current_run_root=current,
        tensorboard_root=tensorboard_root,
        current_season_report=season_root / "current.json",
    )
    selector_token = "c" * 64
    token_sha256 = hashlib.sha256(selector_token.encode("ascii")).hexdigest()
    trainer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        record = _process_record(service_module.PRIMARY_ROLE, trainer)
        attempt_path = runtime / service_module.PRIMARY_ATTEMPT_NAME
        producer = SimpleNamespace(
            service=SimpleNamespace(one_run=admission.one_run),
            config_sha256=admission.primary.config_sha256,
            checkpoint_mode=admission.primary.checkpoint_mode,
            checkpoint=checkpoint,
            current_run_root=current,
            checkpoint_root=checkpoint_root,
            tensorboard_root=tensorboard_root,
            rollout_root=rollout_root,
            update_root=update_root,
            season_report_root=season_root,
        )
        owner = {**record, "role": service_module.PRIMARY_TRAINER_MODULE}
        attempt = _write_attempt_evidence(
            attempt_path,
            producer,
            owner=owner,
            selector_token_sha256=token_sha256,
        )
        assert set(attempt) == PRIMARY_ATTEMPT_FIELDS
        admitted = service_module._wait_for_primary_admission(
            admission, trainer, record, selector_token, timeout_seconds=0.5
        )
        assert admitted["evidence_sha256"] == attempt["evidence_sha256"]

        attempt_path.unlink()
        mismatched_catalog = "e" * 64
        mismatched_producer = SimpleNamespace(
            **{
                **producer.__dict__,
                "service": SimpleNamespace(one_run=SimpleNamespace(
                    runtime_manifest_sha256=admission.one_run.runtime_manifest_sha256,
                    atlas_catalog=SimpleNamespace(
                        atlas_catalog_sha256=mismatched_catalog,
                    ),
                )),
            }
        )
        _write_attempt_evidence(
            attempt_path,
            mismatched_producer,
            owner=owner,
            selector_token_sha256=token_sha256,
        )
        with pytest.raises(
            MultiresServiceError,
            match="primary child admission attempt fields/bindings differ",
        ):
            service_module._wait_for_primary_admission(
                admission, trainer, record, selector_token, timeout_seconds=0.5
            )
    finally:
        trainer.terminate()
        trainer.wait(timeout=5)


def test_service_cli_rejects_removed_run_alias(tmp_path):
    with pytest.raises(SystemExit):
        service_parse_args(["--runtime_root", str(tmp_path), "run"])


def test_service_stop_terminates_nested_owned_process_sessions(tmp_path):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    child_code = "import time; time.sleep(60)"
    root_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", root_code], start_new_session=True
    )
    record = _process_record("primary-trainer", process)
    run = (tmp_path / "run").resolve()
    (run / "tensorboard").mkdir(parents=True)
    (run / "season").mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    child_inventory = runtime / "multires-primary-children.json"
    terminal_evidence = runtime / "multires-primary-terminal.json"
    launch_config = tmp_path / "multires_one_run_primary-trainer.cfg"
    _write_state(
        _state_path(runtime),
        [record],
        runtime_manifest_sha256="a" * 64,
        atlas_catalog_sha256="b" * 64,
        current_run_root=run,
        tensorboard_root=run / "tensorboard",
        current_season_report=run / "season" / "current.json",
        child_inventory=child_inventory,
        terminal_evidence=terminal_evidence,
        launch_config=launch_config,
        shutdown_grace_seconds=30.0,
    )
    assert (_state_path(runtime).stat().st_mode & 0o777) == 0o600
    tree = []
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            tree = _owned_process_tree(record)
            if len(tree) >= 2:
                break
            time.sleep(0.02)
        assert len(tree) >= 2
        result = stop_service(runtime)
        assert set(int(item["pid"]) for item in tree).issubset(result["terminated"])
        assert all(not _record_alive(item) for item in tree)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5.0)


def test_service_health_requires_live_primary_and_optimizer_update_evidence(tmp_path):
    runtime = (tmp_path / "runtime").resolve()
    run = (tmp_path / "run").resolve()
    runtime.mkdir()
    (run / "tensorboard").mkdir(parents=True)
    (run / "season").mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        for _index in range(5)
    ]
    try:
        record = _process_record("primary-trainer", process)
        child_inventory = runtime / "multires-primary-children.json"
        terminal_evidence = runtime / "multires-primary-terminal.json"
        launch_config = tmp_path / "multires_one_run_primary-trainer.cfg"
        _write_state(
            _state_path(runtime), [record],
            runtime_manifest_sha256="b" * 64,
            atlas_catalog_sha256="d" * 64,
            current_run_root=run,
            tensorboard_root=run / "tensorboard",
            current_season_report=run / "season/current.json",
            child_inventory=child_inventory,
            terminal_evidence=terminal_evidence,
            launch_config=launch_config,
            shutdown_grace_seconds=30.0,
        )
        (runtime / "multires-service.lock").write_text(json.dumps({
            "schema": "q2-multires-service-lease-v1",
            "mode": "start",
            "owner": record,
            "selector_token_sha256": "f" * 64,
        }), encoding="utf-8")
        starting = service_status(runtime)
        assert starting["health"] == "starting"
        assert starting["healthy"] is False
        assert starting["current_season_evidence"] is False

        checkpoint = run / "checkpoints/checkpoint-000000000512.pt"
        checkpoint.write_bytes(b"attested-fixture")
        season = {
            "schema": "q2-multires-current-season-v1",
            "health": "training-active",
            "runtime_manifest_sha256": "b" * 64,
            "atlas_catalog_sha256": "d" * 64,
            "lineage_root_sha256": "c" * 64,
            "counters": {
                "accepted_transitions": 512, "policy_updates": 1,
                "optimizer_steps": 4, "next_policy_version": 1,
            },
            "last_checkpoint": "checkpoints/checkpoint-000000000512.pt",
            "checkpoint_sha256": sha(checkpoint),
            "checkpoint_manifest": {
                "training_step": 512,
                "runtime_manifest_sha256": "b" * 64,
                "atlas_catalog_sha256": "d" * 64,
            },
        }
        season["evidence_sha256"] = hashlib.sha256(json.dumps(
            season, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        (run / "season/current.json").write_text(
            json.dumps(season), encoding="utf-8"
        )
        child_records = [
            _process_record(
                "q2ded" if index == 0 else f"network-client-{index - 1:02d}",
                child,
            )
            for index, child in enumerate(children)
        ]
        child_inventory.write_text(json.dumps({
            "schema": "q2-multires-primary-child-inventory-v1",
            "runtime_manifest_sha256": "b" * 64,
            "owner": {**record, "role": "train.multires_primary"},
            "launch_config": str(launch_config),
            "expected_client_count": 4,
            "complete": True,
            "processes": child_records,
        }), encoding="utf-8")
        active = service_status(runtime)
        assert active["health"] == "training-active"
        assert active["healthy"] is True
        assert active["current_season_evidence"] is True

        process.terminate()
        process.wait(timeout=5)
        crashed = service_status(runtime)
        assert crashed["health"] == "crashed"
        assert crashed["terminal_completion_evidence"] is False

        summary = SimpleNamespace(
            current_season_report=run / "season/current.json",
            lineage_root_sha256="c" * 64,
            accepted_transitions=512,
            policy_updates=1,
            optimizer_steps=4,
            next_policy_version=1,
        )
        terminal_producer = SimpleNamespace(
            service=SimpleNamespace(one_run=SimpleNamespace(
                runtime_manifest_sha256="b" * 64,
                atlas_catalog=SimpleNamespace(atlas_catalog_sha256="d" * 64),
            )),
            current_run_root=run,
            checkpoint_root=run / "checkpoints",
        )
        terminal = _write_terminal_evidence(
            terminal_evidence, terminal_producer, summary
        )
        assert set(terminal) == PRIMARY_TERMINAL_FIELDS
        completed = service_status(runtime)
        assert completed["health"] == "completed"
        assert completed["terminal_completion_evidence"] is True

        mismatched_terminal_producer = SimpleNamespace(
            service=SimpleNamespace(one_run=SimpleNamespace(
                runtime_manifest_sha256="b" * 64,
                atlas_catalog=SimpleNamespace(atlas_catalog_sha256="e" * 64),
            )),
            current_run_root=run,
            checkpoint_root=run / "checkpoints",
        )
        _write_terminal_evidence(
            terminal_evidence, mismatched_terminal_producer, summary
        )
        mismatched = service_status(runtime)
        assert mismatched["health"] == "crashed"
        assert mismatched["terminal_completion_evidence"] is False
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        for child in children:
            child.terminate()
            child.wait(timeout=5)
        _state_path(runtime).unlink(missing_ok=True)
        child_inventory.unlink(missing_ok=True)
        terminal_evidence.unlink(missing_ok=True)
        (runtime / "multires-service.lock").unlink(missing_ok=True)


def test_runtime_lease_is_exclusive_until_explicit_release(tmp_path):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    token = _acquire_lease(runtime, "prove")
    assert len(token) == 64
    with pytest.raises(MultiresServiceError, match="active exclusive runtime lease"):
        _acquire_lease(runtime, "start")
    with pytest.raises(MultiresServiceError, match="refusing to unlink live prove"):
        stop_service(runtime)
    assert (runtime / "multires-service.lock").is_file()
    _release_lease(runtime)
    with pytest.raises(MultiresServiceError, match="not operational"):
        _acquire_lease(runtime, "train")
    second = _acquire_lease(runtime, "start")
    assert len(second) == 64
    _release_lease(runtime)

    proof_token = _acquire_lease(runtime, "prove")
    proof = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    proof_record = _process_record("qualification-proof", proof)
    _transfer_lease(runtime, "prove", proof_record, proof_token)
    with pytest.raises(MultiresServiceError, match="live prove selector"):
        stop_service(runtime)
    proof.terminate()
    proof.wait(timeout=5)
    reconciled = stop_service(runtime)
    assert reconciled["already_stopped"] is True
    assert not (runtime / "multires-service.lock").exists()


def test_supervised_start_owns_primary_session_lease_and_stop(
    tmp_path, monkeypatch
):
    runtime = (tmp_path / "runtime").resolve()
    run = (tmp_path / "run").resolve()
    runtime.mkdir()
    (run / "tensorboard").mkdir(parents=True)
    (run / "season").mkdir(parents=True)
    admission = SimpleNamespace(
        runtime_root=runtime,
        trainer_command=(
            sys.executable, "-c", "import time; time.sleep(60)"
        ),
        tensorboard_command=None,
        log_path=runtime / "multires-service.log",
        one_run=SimpleNamespace(
            runtime_manifest_sha256="e" * 64,
            atlas_catalog=SimpleNamespace(atlas_catalog_sha256="d" * 64),
        ),
        current_run_root=run,
        tensorboard_root=run / "tensorboard",
        current_season_report=run / "season/current.json",
        child_inventory=runtime / "multires-primary-children.json",
        terminal_evidence=runtime / "multires-primary-terminal.json",
        launch_config=tmp_path / "multires_one_run_primary-trainer.cfg",
        shutdown_grace_seconds=30.0,
    )
    monkeypatch.setattr(
        service_module, "_revalidate_service_integration", lambda _value: None
    )
    monkeypatch.setattr(
        service_module,
        "_wait_for_primary_admission",
        lambda *_args, **_kwargs: {"evidence_sha256": "f" * 64},
    )
    started = start_service(admission)
    assert started["started"] is True
    assert started["processes"][0]["process_group"] == started["processes"][0]["pid"]
    status = service_status(runtime)
    assert status["health"] == "starting"
    assert status["exclusive_runtime_lease"] is True
    stopped = stop_service(runtime)
    assert stopped["already_stopped"] is False
    assert stopped["terminated"]
    assert not (runtime / "multires-service.lock").exists()
    assert not _state_path(runtime).exists()


def test_zero_update_fresh_attempt_reconciliation_is_exact_and_fail_closed(
    tmp_path, monkeypatch
):
    args = materialize(tmp_path, monkeypatch)
    runtime = args.runtime_root
    cold = json.loads(args.retirement_cold_start.read_text(encoding="utf-8"))
    lineage = cold["lineage"]
    run = Path(lineage["current_run_root"])
    checkpoint_root = Path(lineage["checkpoint_root"])
    tensorboard_root = Path(lineage["tensorboard_root"])
    rollout_root = Path(lineage["rollout_root"])
    update_root = Path(lineage["update_root"])
    season = Path(lineage["season_report_root"]) / "current.json"
    service_config = {
        "schema": "q2-multires-service-v2",
        "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        "retirement_cold_start": str(args.retirement_cold_start),
        "training_runtime": {
            "path": str(args.primary_training),
            "sha256": sha(args.primary_training),
        },
    }
    (runtime / "multires-service.json").write_text(
        json.dumps(service_config), encoding="utf-8"
    )

    dead = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    owner = _process_record("train.multires_primary", dead)
    dead.terminate()
    dead.wait(timeout=5)
    selector_digest = "f" * 64
    attempt_path = runtime / "multires-primary-attempt.json"
    producer = SimpleNamespace(
        service=SimpleNamespace(one_run=SimpleNamespace(
            runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
            atlas_catalog=SimpleNamespace(
                atlas_catalog_sha256=args.expected_atlas_catalog_sha256,
            ),
        )),
        config_sha256=sha(args.primary_training),
        checkpoint_mode="fresh-step-zero",
        checkpoint=args.checkpoint,
        current_run_root=run,
        checkpoint_root=checkpoint_root,
        tensorboard_root=tensorboard_root,
        rollout_root=rollout_root,
        update_root=update_root,
        season_report_root=season.parent,
    )
    wrong_producer = SimpleNamespace(
        **{
            **producer.__dict__,
            "service": SimpleNamespace(one_run=SimpleNamespace(
                runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
                atlas_catalog=SimpleNamespace(atlas_catalog_sha256="e" * 64),
            )),
        }
    )
    _write_attempt_evidence(
        attempt_path,
        wrong_producer,
        owner=owner,
        selector_token_sha256=selector_digest,
    )
    with pytest.raises(
        MultiresServiceError, match="Atlas catalog differs from training runtime"
    ):
        _reconcile_zero_update_attempt(runtime)
    attempt_path.unlink()
    attempt = _write_attempt_evidence(
        attempt_path,
        producer,
        owner=owner,
        selector_token_sha256=selector_digest,
    )
    assert set(attempt) == PRIMARY_ATTEMPT_FIELDS
    suffix = str(attempt["tensorboard_filename_suffix"])
    pending = checkpoint_root / (
        f".checkpoint-000000000512-{owner['pid']}.pending"
    )
    pending.write_bytes(b"interrupted-unpublished-checkpoint")
    pending_temporary = checkpoint_root / (
        f"..checkpoint-000000000512-{owner['pid']}.pending.atomic123.tmp"
    )
    pending_temporary.write_bytes(b"interrupted-atomic-checkpoint-write")
    event = tensorboard_root / (
        f"events.out.tfevents.1.testhost.{owner['pid']}.0{suffix}"
    )
    event.write_bytes(b"unsealed-observer-output")
    unexpected = checkpoint_root / "checkpoint-000000000512-deadbeefdeadbeef.pt"
    unexpected.write_bytes(b"not-safe-to-reconcile")

    with pytest.raises(
        MultiresServiceError, match="non-attempt content"
    ):
        _reconcile_zero_update_attempt(runtime)
    assert args.checkpoint.is_file()
    assert (
        pending.is_file() and pending_temporary.is_file()
        and event.is_file() and attempt_path.is_file()
    )

    unexpected.unlink()
    result = _reconcile_zero_update_attempt(runtime)
    assert result["status"] == "zero-update-reconciled"
    assert args.checkpoint.is_file()
    assert (
        not pending.exists() and not pending_temporary.exists()
        and not event.exists() and not attempt_path.exists()
    )
    assert list(checkpoint_root.iterdir()) == [args.checkpoint]


def test_stop_after_trainer_crash_kills_inherited_group_and_removes_token_cfg(
    tmp_path,
):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    child_pid_path = tmp_path / "child.pid"
    root_code = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(0.2)"
    )
    root_process = subprocess.Popen(
        [sys.executable, "-c", root_code], start_new_session=True
    )
    root_record = _process_record("primary-trainer", root_process)
    run = (tmp_path / "run").resolve()
    (run / "tensorboard").mkdir(parents=True)
    (run / "season").mkdir(parents=True)
    launch_config = tmp_path / "multires_one_run_primary-trainer.cfg"
    launch_config.write_text('set ml_client_telemetry_token "secret"')
    _write_state(
        _state_path(runtime), [root_record],
        runtime_manifest_sha256="d" * 64,
        atlas_catalog_sha256="e" * 64,
        current_run_root=run,
        tensorboard_root=run / "tensorboard",
        current_season_report=run / "season/current.json",
        child_inventory=runtime / "multires-primary-children.json",
        terminal_evidence=runtime / "multires-primary-terminal.json",
        launch_config=launch_config,
        shutdown_grace_seconds=30.0,
    )
    root_process.wait(timeout=5)
    child_pid = int(child_pid_path.read_text())
    tree = _owned_process_tree(root_record)
    assert child_pid in {record["pid"] for record in tree}
    result = stop_service(runtime)
    assert child_pid in result["terminated"]
    assert not Path(f"/proc/{child_pid}").exists()
    assert not launch_config.exists()
    assert not _state_path(runtime).exists()


def test_one_run_process_evidence_binds_pid_to_start_ticks():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = _one_run_process_record("network-client-00", process.pid)
        assert set(record) == {"role", "pid", "start_ticks"}
        assert _one_run_process_record_alive(record) is True
    finally:
        process.terminate()
        process.wait(timeout=5.0)
    assert _one_run_process_record_alive(record) is False
