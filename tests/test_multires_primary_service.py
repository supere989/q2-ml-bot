import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from train.multires_primary import (
    PrimaryOwnedResources,
    PrimaryTrainingError,
    admit_primary_training,
    execute_primary,
    run_admitted_core,
)
from train.multires_train import CurriculumStage, MultiresContinuousTrainingCore


ATLAS = "a" * 64
RUNTIME = "b" * 64
LINEAGE = "c" * 64


class FakeMetrics:
    def __init__(self):
        self.transitions = 0

    def snapshot(self, *, policy_end_version):
        return {
            "schema": "multires-atlas-season-v1",
            "season_id": "service-fake",
            "atlas_sha256": ATLAS,
            "policy_start_version": 0,
            "policy_end_version": policy_end_version,
            "accepted_transitions": self.transitions,
            "privilege": {"teacher_field_violations": 0},
            "private_causal_payload_serialized": False,
        }


class FakeNetworkMetrics:
    def __init__(self):
        self.transitions = 0

    def as_dict(self):
        return {
            "network_client/rounds_dispatched": self.transitions,
            "network_client/rounds_accepted": self.transitions,
            "network_client/failed_rounds": 0,
            "network_client/actions_dispatched": self.transitions,
            "network_client/transitions_accepted": self.transitions,
            "network_client/stale_policy_rounds_rejected": 0,
            "network_client/stale_echoes_rejected": 0,
            "network_client/mismatched_echoes_rejected": 0,
            "network_client/echo_timeouts": 0,
            "network_client/map_epoch_resyncs": 0,
            "network_client/telemetry_gap_resyncs": 0,
            "network_client/realtime_catchup_resyncs": 0,
            "network_client/action_state_resyncs": 0,
            "network_client/preflight_packets_drained": 0,
            "network_client/max_observed_frame_span": 1,
            "network_client/fire_gate_suppressions": 0,
            "network_client/authoritative_echo_accept_rate": 1.0,
        }


class FakeCheckpointRuntime:
    def __init__(self):
        self.runtime = SimpleNamespace(
            atlas_sha256=ATLAS, runtime_manifest_sha256=RUNTIME
        )
        self.lineage_root_sha256 = None

    def checkpoint(self, path, *, training_step, optimizer):
        Path(path).write_bytes(f"checkpoint:{training_step}".encode())
        self.lineage_root_sha256 = LINEAGE
        return {
            "training_step": training_step,
            "lineage_root_sha256": LINEAGE,
            "atlas_sha256": ATLAS,
            "runtime_manifest_sha256": RUNTIME,
        }


class FakeRollout:
    policy_version = 0
    valid = np.ones((2, 2), dtype=np.bool_)
    rewards = np.ones((2, 2), dtype=np.float32)
    boundary_rounds = 0

    def validate(self):
        assert self.valid.all()

    def deterministic_sha256(self):
        return hashlib.sha256(b"primary-service-rollout").hexdigest()


class FakeTrainer:
    def __init__(self, causal_metrics):
        self.runtime = FakeCheckpointRuntime()
        self.optimizer = object()
        self.causal_metrics = causal_metrics
        self.network_metrics = FakeNetworkMetrics()
        self.collector = SimpleNamespace(
            batch=SimpleNamespace(metrics=self.network_metrics)
        )
        self.calls = 0

    def train_update(self, *, policy_version):
        assert policy_version == 0
        self.calls += 1
        self.causal_metrics.transitions += 4
        self.network_metrics.transitions += 4
        return SimpleNamespace(
            rollout=FakeRollout(),
            ppo_metrics={"policy_loss": 0.25, "optimizer_steps": 4},
        )


class FakeWriter:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.values.append((tag, scalar_value, global_step))

    def flush(self):
        pass


def test_finite_primary_path_reaches_one_real_core_update(tmp_path):
    metrics = FakeMetrics()
    trainer = FakeTrainer(metrics)
    writer = FakeWriter()
    core = MultiresContinuousTrainingCore(
        trainer,
        output_root=tmp_path.resolve(),
        stage=CurriculumStage.create(1, {"map": "service-fake"}),
        writer=writer,
        causal_metrics=metrics,
    )
    summary = run_admitted_core(
        core,
        {"maximum_updates": 1, "continuous": False},
        lambda: False,
    )
    assert trainer.calls == 1
    assert summary.policy_updates == 1
    assert summary.optimizer_steps == 4
    assert summary.accepted_transitions == 4
    assert summary.current_season_report.is_file()
    assert list((tmp_path / "checkpoints").glob("*.pt"))


def test_primary_resource_owner_closes_writer_batch_server_and_config(tmp_path):
    events = []

    class Writer:
        def flush(self): events.append("writer-flush")
        def close(self): events.append("writer-close")

    class Batch:
        def close(self): events.append("batch-close")

    launch = tmp_path / "primary.cfg"
    launch.write_text("fixture")
    server = object()
    resources = PrimaryOwnedResources(
        server=server,
        launch_config=launch,
        stop_server=lambda value: events.append(
            "server-stop" if value is server else "wrong-server"
        ),
        batch=Batch(),
        writer=Writer(),
    )
    resources.close()
    resources.close()
    assert events == ["writer-flush", "writer-close", "batch-close", "server-stop"]
    assert not launch.exists()


def test_primary_cleanup_aggregates_failures_without_skipping_server_or_cfg(tmp_path):
    events = []

    class Writer:
        def flush(self):
            events.append("writer-flush")
            raise RuntimeError("flush-failed")

        def close(self):
            events.append("writer-close")
            raise RuntimeError("close-failed")

    class Batch:
        def close(self):
            events.append("batch-close")
            raise RuntimeError("batch-failed")

    launch = tmp_path / "primary.cfg"
    launch.write_text("token-bearing-fixture")
    resources = PrimaryOwnedResources(
        server=object(), launch_config=launch,
        stop_server=lambda _server: events.append("server-stop"),
        batch=Batch(), writer=Writer(),
    )
    with pytest.raises(PrimaryTrainingError, match="writer_flush=.*batch="):
        resources.close()
    assert events == [
        "writer-flush", "writer-close", "batch-close", "server-stop"
    ]
    assert not launch.exists()


def test_primary_startup_fails_before_import_or_launch_without_private_token(
    monkeypatch,
):
    monkeypatch.delenv("Q2_ML_CLIENT_TELEMETRY_TOKEN", raising=False)
    admission = SimpleNamespace(
        config={
            "execution": {"cuda_cublas_workspace_config": None},
        },
        service=SimpleNamespace(one_run=SimpleNamespace(runtime_config={
            "telemetry_token_env": "Q2_ML_CLIENT_TELEMETRY_TOKEN"
        })),
    )
    with pytest.raises(PrimaryTrainingError, match="missing or malformed"):
        execute_primary(admission)


def test_direct_primary_module_requires_owned_selector_before_mutating(tmp_path, monkeypatch):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    inventory = runtime / "multires-primary-children.json"
    terminal = runtime / "multires-primary-terminal.json"
    inventory.write_text("preserve-inventory")
    terminal.write_text("preserve-terminal")
    monkeypatch.setenv("Q2_ML_CLIENT_TELEMETRY_TOKEN", "t" * 32)
    monkeypatch.delenv("Q2_MULTIRES_PRIMARY_SELECTOR_TOKEN", raising=False)
    admission = SimpleNamespace(
        config={"execution": {"cuda_cublas_workspace_config": None}},
        service=SimpleNamespace(one_run=SimpleNamespace(
            runtime_root=runtime,
            runtime_config={
                "telemetry_token_env": "Q2_ML_CLIENT_TELEMETRY_TOKEN"
            },
        )),
    )
    with pytest.raises(PrimaryTrainingError, match="selector token"):
        execute_primary(admission)
    assert inventory.read_text() == "preserve-inventory"
    assert terminal.read_text() == "preserve-terminal"


def test_same_lineage_resume_uses_only_explicit_checkpoint_and_sealed_season(tmp_path):
    run = (tmp_path / "run").resolve()
    roots = {
        "current_run_root": run,
        "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard",
        "rollout_root": run / "evidence/rollouts",
        "update_root": run / "evidence/updates",
        "season_report_root": run / "season",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    proof_checkpoint = roots["checkpoint_root"] / "fresh-step-zero.pt"
    proof_checkpoint.write_bytes(b"proof-step-zero")
    selected = roots["checkpoint_root"] / "checkpoint-000000000004.pt"
    selected.write_bytes(b"selected-resume")
    # A lexicographically later file proves admission does not scan/select latest.
    (roots["checkpoint_root"] / "checkpoint-999999999999.pt").write_bytes(
        b"not-selected"
    )
    barrier = tmp_path / "barrier.json"
    barrier.write_text("{}")
    stage = CurriculumStage.create(1, {"map": "resume-fixture"})
    report = {
        "schema": "q2-multires-current-season-v1",
        "health": "training-active",
        "promotion_claim": False,
        "stage_configuration_sha256": stage.configuration_sha256,
        "runtime_manifest_sha256": RUNTIME,
        "atlas_sha256": ATLAS,
        "lineage_root_sha256": LINEAGE,
        "counters": {
            "accepted_transitions": 4,
            "policy_updates": 1,
            "optimizer_steps": 4,
            "next_policy_version": 1,
        },
        "last_checkpoint": "checkpoints/checkpoint-000000000004.pt",
        "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "checkpoint_manifest": {
            "training_step": 4,
            "runtime_manifest_sha256": RUNTIME,
            "atlas_sha256": ATLAS,
            "lineage_root_sha256": LINEAGE,
        },
    }
    report["evidence_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    season = roots["season_report_root"] / "current.json"
    season.write_text(json.dumps(report), encoding="utf-8")
    optimizer = {
        "class": "torch.optim.Adam", "learning_rate": 1e-5, "kwargs": {}
    }
    config = {
        "schema": "q2-multires-primary-training-runtime-v1",
        "runtime_manifest_sha256": RUNTIME,
        "network_barrier_execution_evidence_sha256": "e" * 64,
        "seed": 7, "game_seed": 8, "map_name": "resume-fixture", "map_epoch": 0,
        "collector": {
            "transitions_per_client": 2, "gamma": 0.99,
            "gae_lambda": 0.95, "maximum_boundary_rounds": 4,
        },
        "optimizer": optimizer,
        "curriculum": {
            "stage_id": 1,
            "configuration": json.loads(stage.configuration_json)["configuration"],
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": None,
            "predecessor_gate": None,
        },
        "execution": {
            "maximum_updates": 1, "continuous": False, "device": "cpu",
            "deterministic_collection": True, "deterministic_algorithms": True,
            "cuda_cublas_workspace_config": None,
            "shutdown_grace_seconds": 120.0,
        },
        "checkpoint": {
            "mode": "same-lineage-resume", "path": str(selected),
            "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "lineage_root_sha256": LINEAGE,
            "current_season_report": {
                "path": str(season),
                "sha256": hashlib.sha256(season.read_bytes()).hexdigest(),
            },
        },
    }
    config_path = tmp_path / "primary.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    one_run = SimpleNamespace(
        runtime_manifest_sha256=RUNTIME,
        network_barrier_execution_evidence_sha256="e" * 64,
        runtime_config={
            "network_barrier_qualification": str(barrier),
            "optimizer": optimizer,
            "client_timeout": 2.0,
            "round_timeout": 2.0,
        },
        args=SimpleNamespace(
            map_name="resume-fixture",
            checkpoint=proof_checkpoint,
            expected_atlas_sha256=ATLAS,
        ),
    )
    admission = admit_primary_training(
        SimpleNamespace(one_run=one_run),
        config_path=config_path,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        cold_document={"optimizer": optimizer, "lineage": {
            key: str(value) for key, value in roots.items()
        }},
    )
    assert admission.checkpoint == selected
    assert admission.current_season_report == season

    report["lineage_root_sha256"] = "d" * 64
    report["evidence_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in report.items() if key != "evidence_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    season.write_text(json.dumps(report), encoding="utf-8")
    config["checkpoint"]["current_season_report"]["sha256"] = hashlib.sha256(
        season.read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(PrimaryTrainingError, match="lineage selection differs"):
        admit_primary_training(
            SimpleNamespace(one_run=one_run),
            config_path=config_path,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            cold_document={"optimizer": optimizer, "lineage": {
                key: str(value) for key, value in roots.items()
            }},
        )


def test_real_summarywriter_is_rooted_only_at_declared_tensorboard_dir(tmp_path):
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch is unavailable on this unit host")
    from torch.utils.tensorboard import SummaryWriter

    root = (tmp_path / "run/tensorboard").resolve()
    root.mkdir(parents=True)
    writer = SummaryWriter(log_dir=str(root))
    try:
        writer.add_scalar("train/policy_updates", 1.0, 4)
        writer.add_scalar("train/optimizer_steps", 4.0, 4)
        writer.flush()
        assert Path(writer.log_dir).resolve() == root
    finally:
        writer.close()
    events = list(root.glob("events.out.tfevents.*"))
    assert len(events) == 1
    assert list((tmp_path / "run").glob("events.out.tfevents.*")) == []


def test_real_primary_checkpoint_load_is_exact_lineage_when_torch_available(tmp_path):
    try:
        import torch
    except ImportError:
        pytest.skip("torch is unavailable on this unit host")

    from harness.multires_contract import (
        ACTION_DIM, FEATURE_SCHEMA_SHA256, OBS_DIM, POLICY_GENERATION,
        POSTURE_CLASSES, GuideDropoutConfig,
    )
    from harness.multires_lineage import save_attested_checkpoint
    from harness.multires_reward import CausalRewardConfig
    from harness.multires_runtime import (
        B4_ACTION_MAGIC, B4_CAUSAL_MAGIC, B4_CAUSAL_PACKET_BYTES,
        B4_CAUSAL_VERSION, B4_CLIENT_WIRE_VERSION, B4_OBSERVATION_MAGIC,
        B4_PROTOCOL_GENERATION, B4_ROLLOUT_SCHEMA, B4_TEACHER_VERSION,
    )
    from harness.multires_training_config import MultiresTrainingConfiguration
    from models.multires_policy import MultiresQ2BotPolicy
    from train.multires_ppo import MultiresPPOConfig
    from train.multires_runtime import MultiresTrainerRuntime

    reward = CausalRewardConfig()
    dropout = GuideDropoutConfig()
    ppo = MultiresPPOConfig()
    training = MultiresTrainingConfiguration.create(
        reward=reward, guide_dropout=dropout, ppo=ppo
    )
    runtime_evidence = {
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
        "atlas_sha256": ATLAS,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
        "runtime_manifest_sha256": RUNTIME,
    }
    policy = MultiresQ2BotPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    checkpoint = tmp_path / "selected.pt"
    saved = save_attested_checkpoint(
        checkpoint, policy,
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        training_config=training,
        initialization="random",
        training_step=0,
        optimizer=optimizer,
    )

    def optimizer_factory(parameters):
        return torch.optim.Adam(parameters, lr=1e-5)

    _runtime, loaded_optimizer, loaded = MultiresTrainerRuntime.resume(
        checkpoint,
        runtime_evidence,
        expected_atlas_sha256=ATLAS,
        optimizer_factory=optimizer_factory,
        reward_config=reward,
        guide_dropout=dropout,
        ppo_config=ppo,
        expected_lineage_root_sha256=saved.lineage_root_sha256,
    )
    assert loaded.lineage_root_sha256 == saved.lineage_root_sha256
    assert loaded.training_step == 0
    assert loaded_optimizer.state_dict()["state"] == {}
