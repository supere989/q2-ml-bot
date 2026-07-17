import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness.multires_metrics import MultiresSeasonMetrics
from harness.multires_reward import CausalRewardFrame, CausalRewardReducer
from train.multires_train import (
    CURRICULUM_GATE_SCHEMA,
    CurriculumStage,
    MultiresContinuousTrainingCore,
    MultiresTrainingCoreError,
    SanitizedCausalMetricsAccumulator,
    TrainingProgress,
    create_curriculum_gate_evidence,
    progress_from_season_report,
)


ATLAS = "a" * 64
RUNTIME = "b" * 64
LINEAGE = "c" * 64


class FakeCheckpointRuntime:
    def __init__(self, *, lineage_root_sha256=None):
        self.runtime = SimpleNamespace(
            atlas_sha256=ATLAS,
            runtime_manifest_sha256=RUNTIME,
        )
        self.lineage_root_sha256 = lineage_root_sha256
        self.training_steps = []

    def checkpoint(self, path, *, training_step, optimizer):
        assert optimizer is not None
        destination = Path(path)
        temporary = destination.with_suffix(".publishing")
        temporary.write_bytes(f"checkpoint:{training_step}".encode("ascii"))
        os.replace(temporary, destination)
        self.training_steps.append(training_step)
        if self.lineage_root_sha256 is None:
            self.lineage_root_sha256 = LINEAGE
        return {
            "training_step": training_step,
            "lineage_root_sha256": self.lineage_root_sha256,
            "atlas_sha256": self.runtime.atlas_sha256,
            "runtime_manifest_sha256": self.runtime.runtime_manifest_sha256,
        }


class FakeRollout:
    def __init__(self, policy_version, *, transitions=6):
        assert transitions == 6
        self.policy_version = policy_version
        self.valid = np.ones((2, 3), dtype=np.bool_)
        self.rewards = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
        self.boundary_rounds = policy_version

    def validate(self):
        assert self.valid.all()

    def deterministic_sha256(self):
        return hashlib.sha256(
            f"fake-rollout:{self.policy_version}".encode("ascii")
        ).hexdigest()


class FakeNetworkMetrics:
    def __init__(self):
        self.transitions = 0
        self.rounds = 0

    def as_dict(self):
        return {
            "network_client/rounds_dispatched": self.rounds,
            "network_client/rounds_accepted": self.rounds,
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


class FakeTrainer:
    def __init__(self, observer, *, runtime=None):
        self.runtime = runtime or FakeCheckpointRuntime()
        self.optimizer = object()
        self.observer = observer
        self.policy_versions = []
        self.tick = 0
        self.reducer = CausalRewardReducer()
        self.network_metrics = FakeNetworkMetrics()
        self.collector = SimpleNamespace(
            batch=SimpleNamespace(metrics=self.network_metrics)
        )

    def train_update(self, *, policy_version):
        self.policy_versions.append(policy_version)
        rollout = FakeRollout(policy_version)
        self.network_metrics.transitions += int(rollout.valid.sum())
        self.network_metrics.rounds += rollout.valid.shape[1]
        if self.observer is not None:
            for _index in range(int(rollout.valid.sum())):
                self.tick += 1
                frame = CausalRewardFrame(
                    tick=self.tick,
                    client_life_epoch=1,
                    authoritative_echo_valid=True,
                    trainable_transition=True,
                    action_generation=self.tick,
                )
                result = self.reducer.step(frame)
                self.observer("c0", frame, result, {
                    "client_id": "c0",
                    "action_debug_movement": [1.0, 0.0, 0.0, 0.0],
                    "movement_speed": 240.0,
                    "true_view_pitch_deg": 0.0,
                    "guide_dropped": [False, False, False, False],
                    "guide_classes": [0, 1, None, None],
                    "global_guide_drop": False,
                })
        return SimpleNamespace(
            rollout=rollout,
            ppo_metrics={
                "policy_loss": -0.25, "value_loss": 0.5,
                "optimizer_steps": 4,
            },
        )


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flushes = 0

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def flush(self):
        self.flushes += 1


def stage_one():
    return CurriculumStage.create(1, {
        "maps": ["transport-fixture"],
        "minimum_accepted_transitions": 6,
    })


def accumulator(*, start_policy_version=0, season_id="test-season"):
    return SanitizedCausalMetricsAccumulator(MultiresSeasonMetrics(
        season_id, ATLAS, start_policy_version
    ))


def make_core(tmp_path, *, runtime=None, stage=None, progress=TrainingProgress(),
              predecessor_gate=None, start_policy_version=None):
    if start_policy_version is None:
        start_policy_version = progress.next_policy_version
    metrics = accumulator(start_policy_version=start_policy_version)
    trainer = FakeTrainer(metrics, runtime=runtime)
    writer = FakeWriter()
    core = MultiresContinuousTrainingCore(
        trainer,
        output_root=tmp_path.resolve(),
        stage=stage or stage_one(),
        writer=writer,
        causal_metrics=metrics,
        predecessor_gate=predecessor_gate,
        progress=progress,
    )
    return core, trainer, writer, metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_zero_update_cannot_emit_training_health(tmp_path):
    core, trainer, _writer, _metrics = make_core(tmp_path)
    with pytest.raises(MultiresTrainingCoreError, match="at least one"):
        core.run(maximum_updates=0)
    with pytest.raises(MultiresTrainingCoreError, match="zero policy"):
        core.run(should_stop=lambda: True)
    assert trainer.policy_versions == []
    assert not core.current_season_report.exists()
    assert list((tmp_path / "checkpoints").iterdir()) == []


def test_updates_checkpoint_cumulative_transitions_and_emit_complete_evidence(
    tmp_path,
):
    core, trainer, writer, _metrics = make_core(tmp_path)
    summary = core.run(maximum_updates=2)

    assert trainer.policy_versions == [0, 1]
    assert trainer.runtime.training_steps == [6, 12]
    assert summary.accepted_transitions == 12
    assert summary.policy_updates == 2
    assert summary.optimizer_steps == 8
    assert summary.next_policy_version == 2
    assert summary.lineage_root_sha256 == LINEAGE
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2
    assert len(list((tmp_path / "evidence/rollouts").glob("*.json"))) == 2
    assert len(list((tmp_path / "evidence/updates").glob("*.json"))) == 2
    assert list(tmp_path.rglob("*.tmp")) == []
    assert list(tmp_path.rglob("*.publishing")) == []

    current = read_json(summary.current_season_report)
    assert current["health"] == "training-active"
    assert current["promotion_claim"] is False
    assert current["counters"] == {
        "accepted_transitions": 12,
        "next_policy_version": 2,
        "policy_updates": 2,
        "optimizer_steps": 8,
    }
    assert current["checkpoint_manifest"]["training_step"] == 12
    assert current["causal_metrics_window"]["accepted_transitions"] == 12
    assert current["causal_metrics_window"]["privilege"] == {
        "teacher_field_violations": 0
    }
    assert current["causal_metrics_window"][
        "private_causal_payload_serialized"
    ] is False
    assert current["causal_metrics_window"]["movement"][
        "forward_command_mean"
    ] == 1.0
    assert current["network_metrics_window"][
        "network_client/transitions_accepted"
    ] == 12
    tags = {tag for tag, _value, _step in writer.scalars}
    assert {
        "train/accepted_transitions",
        "train/policy_updates",
        "train/optimizer_steps",
        "train/policy_version",
        "train/policy_loss",
        "rollout/reward_mean",
        "curriculum/stage_id",
        "season/transport/command_echo_match_rate",
        "season/movement/speed_mean",
        "season/combat/kills",
        "season/privilege/teacher_field_violations",
        "network_client/transitions_accepted",
        "network_client/telemetry_gap_resyncs",
        "network_client/action_state_resyncs",
    } <= tags
    assert writer.flushes == 2


def test_tensorboard_observer_emits_only_after_sealed_season_publication(tmp_path):
    metrics = accumulator()
    trainer = FakeTrainer(metrics)
    season_path = tmp_path / "season/current.json"

    class OrderingWriter(FakeWriter):
        def add_scalar(self, tag, scalar_value, global_step):
            report = read_json(season_path)
            unsigned = dict(report)
            declared = unsigned.pop("evidence_sha256")
            assert declared == hashlib.sha256(json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()
            super().add_scalar(tag, scalar_value, global_step)

    writer = OrderingWriter()
    core = MultiresContinuousTrainingCore(
        trainer, output_root=tmp_path.resolve(), stage=stage_one(),
        writer=writer, causal_metrics=metrics,
    )
    core.run(maximum_updates=1)
    assert writer.scalars


def test_same_policy_retry_reuses_identical_content_artifacts_without_wedge(tmp_path):
    first, _trainer, _writer, _metrics = make_core(tmp_path)
    first.run(maximum_updates=1)
    first.current_season_report.unlink()

    retried, trainer, _writer, _metrics = make_core(tmp_path)
    summary = retried.run(maximum_updates=1)
    assert trainer.policy_versions == [0]
    assert summary.policy_updates == 1
    assert summary.optimizer_steps == 4
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 1
    assert len(list((tmp_path / "evidence/rollouts").glob("*.json"))) == 1
    assert len(list((tmp_path / "evidence/updates").glob("*.json"))) == 1
    assert summary.current_season_report.is_file()


def test_resume_preserves_lineage_counters_and_policy_versions(tmp_path):
    first, first_trainer, _writer, _metrics = make_core(tmp_path)
    first.run(maximum_updates=1)
    current = read_json(first.current_season_report)
    progress = progress_from_season_report(
        current,
        stage_one(),
        runtime_manifest_sha256=RUNTIME,
        atlas_sha256=ATLAS,
        lineage_root_sha256=LINEAGE,
    )
    assert progress == TrainingProgress(
        accepted_transitions=6, policy_updates=1, optimizer_steps=4,
        next_policy_version=1, lineage_root_sha256=LINEAGE,
    )

    resumed_runtime = FakeCheckpointRuntime(lineage_root_sha256=LINEAGE)
    resumed, resumed_trainer, _writer, _metrics = make_core(
        tmp_path, runtime=resumed_runtime, progress=progress,
        start_policy_version=1,
    )
    summary = resumed.run(maximum_updates=1)

    assert first_trainer.policy_versions == [0]
    assert resumed_trainer.policy_versions == [1]
    assert resumed_runtime.training_steps == [12]
    assert summary.accepted_transitions == 12
    assert summary.policy_updates == 2
    assert summary.optimizer_steps == 8
    assert summary.next_policy_version == 2
    assert summary.lineage_root_sha256 == LINEAGE
    report = read_json(resumed.current_season_report)
    # The causal aggregation window is explicit and restarts with this process;
    # cumulative checkpoint counters remain monotonic independently.
    assert report["causal_metrics_window"]["accepted_transitions"] == 6
    assert report["counters"]["accepted_transitions"] == 12


def test_causal_metric_delta_must_match_the_admitted_rollout(tmp_path):
    metrics = accumulator()
    trainer = FakeTrainer(None)
    core = MultiresContinuousTrainingCore(
        trainer,
        output_root=tmp_path.resolve(),
        stage=stage_one(),
        writer=FakeWriter(),
        causal_metrics=metrics,
    )
    with pytest.raises(MultiresTrainingCoreError, match="transition delta"):
        core.run(maximum_updates=1)
    assert not core.current_season_report.exists()
    assert list((tmp_path / "checkpoints").iterdir()) == []


def test_curriculum_predecessor_gate_is_explicit_and_carries_progress(tmp_path):
    first_stage = stage_one()
    second_stage = CurriculumStage.create(
        2,
        {"maps": ["standing-crouched-fixture"]},
        predecessor_stage_sha256=first_stage.configuration_sha256,
    )
    gate = create_curriculum_gate_evidence(
        first_stage,
        decision="passed",
        runtime_manifest_sha256=RUNTIME,
        lineage_root_sha256=LINEAGE,
        accepted_transitions=6,
        policy_updates=1,
        optimizer_steps=4,
    )
    assert gate["schema"] == CURRICULUM_GATE_SCHEMA
    runtime = FakeCheckpointRuntime(lineage_root_sha256=LINEAGE)
    progress = TrainingProgress(
        accepted_transitions=6, policy_updates=1, optimizer_steps=4,
        next_policy_version=1, lineage_root_sha256=LINEAGE,
    )

    with pytest.raises(MultiresTrainingCoreError, match="requires a passed"):
        make_core(tmp_path / "missing", runtime=runtime, stage=second_stage)
    failed = create_curriculum_gate_evidence(
        first_stage,
        decision="failed",
        runtime_manifest_sha256=RUNTIME,
        lineage_root_sha256=LINEAGE,
        accepted_transitions=6,
        policy_updates=1,
        optimizer_steps=4,
    )
    with pytest.raises(MultiresTrainingCoreError, match="gate differs"):
        make_core(
            tmp_path / "failed", runtime=runtime, stage=second_stage,
            predecessor_gate=failed, progress=progress,
        )
    with pytest.raises(MultiresTrainingCoreError, match="progress differs"):
        make_core(
            tmp_path / "regressed", runtime=runtime, stage=second_stage,
            predecessor_gate=gate,
        )

    core, trainer, _writer, _metrics = make_core(
        tmp_path / "accepted",
        runtime=runtime,
        stage=second_stage,
        predecessor_gate=gate,
        progress=progress,
    )
    summary = core.run(maximum_updates=1)
    assert trainer.policy_versions == [1]
    assert summary.accepted_transitions == 12
    assert summary.policy_updates == 2
    assert summary.optimizer_steps == 8


def test_progress_rejects_policy_or_transition_regression():
    with pytest.raises(MultiresTrainingCoreError, match="cannot trail"):
        TrainingProgress(accepted_transitions=1, policy_updates=2,
                         optimizer_steps=2, next_policy_version=2,
                         lineage_root_sha256=LINEAGE).validate()
    with pytest.raises(MultiresTrainingCoreError, match="one-update-per-version"):
        TrainingProgress(accepted_transitions=6, policy_updates=1,
                         optimizer_steps=4, next_policy_version=2,
                         lineage_root_sha256=LINEAGE).validate()


def test_real_torch_checkpoint_is_attested_when_torch_is_available(tmp_path):
    try:
        import torch
    except ImportError:
        pytest.skip("torch is unavailable on this unit host")

    from harness.multires_contract import GuideDropoutConfig
    from harness.multires_lineage import load_attested_checkpoint
    from harness.multires_reward import CausalRewardConfig
    from harness.multires_training_config import MultiresTrainingConfiguration
    from models.multires_policy import MultiresQ2BotPolicy
    from train.multires_ppo import MultiresPPOConfig
    from train.multires_runtime import MultiresTrainerRuntime

    policy = MultiresQ2BotPolicy()
    reward_config = CausalRewardConfig()
    dropout = GuideDropoutConfig()
    ppo = MultiresPPOConfig()
    training = MultiresTrainingConfiguration.create(
        reward=reward_config, guide_dropout=dropout, ppo=ppo
    )
    runtime = MultiresTrainerRuntime(
        policy=policy,
        runtime=SimpleNamespace(
            atlas_sha256=ATLAS, runtime_manifest_sha256=RUNTIME
        ),
        reward_config=reward_config,
        guide_dropout=dropout,
        ppo_config=ppo,
        training_config=training,
        initialization="random",
    )
    metrics = accumulator()
    trainer = FakeTrainer(metrics, runtime=runtime)
    trainer.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    core = MultiresContinuousTrainingCore(
        trainer,
        output_root=tmp_path.resolve(),
        stage=stage_one(),
        writer=FakeWriter(),
        causal_metrics=metrics,
    )
    summary = core.run(maximum_updates=1)
    checkpoints = list((tmp_path / "checkpoints").glob(
        "checkpoint-000000000006-*.pt"
    ))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]

    target = MultiresQ2BotPolicy()
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-5)
    manifest = load_attested_checkpoint(
        checkpoint,
        target,
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
        expected_training_config=training,
        optimizer=target_optimizer,
        expected_lineage_root_sha256=summary.lineage_root_sha256,
    )
    assert manifest.training_step == 6
    assert manifest.lineage_root_sha256 == summary.lineage_root_sha256
