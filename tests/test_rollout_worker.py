from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness.distributed_runtime import BackoffPolicy
from harness.rollout_protocol import (
    CoordinatorRequestError,
    CoordinatorTransportError,
    PPO_BEHAVIOR_METRIC_KEYS,
    PPO_EPISODE_SUMMARY_COLUMNS,
)
from tools.rollout_worker import (
    _attest_worker_runtime,
    _finalize_batch_telemetry,
    _new_batch_telemetry,
    _new_episode_accumulators,
    _record_q2_telemetry,
    _prepare_deterministic_environment,
    _retry_coordinator,
    _worker_runtime_config,
)


def _attestation_args(**overrides):
    values = {
        "n_bots": 4,
        "n_ml": 4,
        "timescale": 10.0,
        "max_ep_steps": 1000,
        "steps": 256,
        "deterministic": 1,
        "deterministic_actions": False,
        "device": "auto",
        "map_name": "mltrain_test",
        "runtime_manifest": Path("runtime-manifest.json"),
        "attestation_key_env": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_worker_telemetry_preserves_partial_episode_across_rollouts():
    episodes = _new_episode_accumulators(2)
    first_rollout = _new_batch_telemetry()
    _record_q2_telemetry(
        episodes,
        first_rollout,
        env_index=0,
        reward=1.0,
        done=False,
        info={
            "reward_base": 0.7,
            "spatial_bonus": 0.3,
            "ammo_depleted": 1.0,
        },
    )
    first = _finalize_batch_telemetry(first_rollout)
    assert first["episode_summaries"].shape == (
        0, len(PPO_EPISODE_SUMMARY_COLUMNS)
    )
    assert first["behavior_samples"].tolist() == [1]
    ammo_index = PPO_BEHAVIOR_METRIC_KEYS.index("ammo_depleted")
    assert first["behavior_sums"][ammo_index] == 1.0

    second_rollout = _new_batch_telemetry()
    _record_q2_telemetry(
        episodes,
        second_rollout,
        env_index=0,
        reward=2.0,
        done=True,
        info={
            "reward_base": 1.5,
            "spatial_bonus": 0.5,
            "kills": 1.0,
            "outcome_sample": 1.0,
            "outcome_win": 1.0,
            "outcome_bonus": 0.5,
        },
    )
    second = _finalize_batch_telemetry(second_rollout)
    columns = {
        name: index for index, name in enumerate(PPO_EPISODE_SUMMARY_COLUMNS)
    }
    summary = second["episode_summaries"][0]
    assert summary[columns["reward"]] == 3.0
    assert np.isclose(summary[columns["base_reward"]], 2.2)
    assert summary[columns["spatial_reward"]] == 0.8
    assert summary[columns["kills"]] == 1.0
    assert summary[columns["deaths"]] == 0.0
    assert summary[columns["length"]] == 2.0
    assert second["behavior_samples"].tolist() == [1]
    assert second["behavior_sums"][
        PPO_BEHAVIOR_METRIC_KEYS.index("outcome_sample")
    ] == 1.0
    assert all(values[0] == 0 for values in episodes.values())
    assert all(values[1] == 0 for values in episodes.values())


def test_worker_reconnect_retries_transient_transport_with_bounded_backoff():
    attempts = []
    waits = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise CoordinatorTransportError("offline")
        return "connected"

    policy = BackoffPolicy(
        initial_delay=0.25,
        maximum_delay=1.0,
        multiplier=2.0,
        jitter_fraction=0.0,
        max_attempts=3,
    )
    assert _retry_coordinator(
        operation, policy, "worker-a", wait=waits.append
    ) == "connected"
    assert waits == [0.25, 0.5]

    with pytest.raises(CoordinatorRequestError):
        _retry_coordinator(
            lambda: (_ for _ in ()).throw(
                CoordinatorRequestError(401, "unauthorized")
            ),
            policy,
            "worker-a",
            wait=waits.append,
        )


def test_deterministic_worker_requires_hash_seed_from_process_start(monkeypatch):
    args = SimpleNamespace(deterministic=1)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        _prepare_deterministic_environment(args)
    assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _prepare_deterministic_environment(args)


def test_worker_runtime_config_requires_core_and_rebinds_present_options():
    args = _attestation_args(n_ml=3, timescale=8.0, steps=64)
    core_only = {
        "n_bots": 99,
        "n_ml": 99,
        "timescale": 99.0,
        "extension": "preserved",
    }
    assert _worker_runtime_config(core_only, args, "cuda") == {
        "n_bots": 4,
        "n_ml": 3,
        "timescale": 8.0,
        "extension": "preserved",
    }

    fully_bound = {
        **core_only,
        "max_ep_steps": 1,
        "steps": 1,
        "deterministic": False,
        "deterministic_actions": True,
        "device": "cpu",
    }
    actual = _worker_runtime_config(fully_bound, args, "cuda")
    assert actual["max_ep_steps"] == 1000
    assert actual["steps"] == 64
    assert actual["deterministic"] is True
    assert actual["deterministic_actions"] is False
    assert actual["device"] == "cuda"

    with pytest.raises(RuntimeError, match="timescale"):
        _worker_runtime_config({"n_bots": 4, "n_ml": 4}, args, "cuda")


def test_worker_attestation_uses_actual_config_and_source_revision(monkeypatch):
    import harness.runtime_attestation as attestation

    digest = "a" * 64
    expected_config = {
        "n_bots": 4,
        "n_ml": 4,
        "timescale": 10.0,
        "max_ep_steps": 1000,
        "steps": 256,
        "deterministic": True,
        "deterministic_actions": False,
        "device": "cuda",
    }
    expected = {"semantic": {"runtime_config": expected_config}}
    captured = {}
    monkeypatch.setattr(attestation, "load_runtime_manifest", lambda _path: expected)

    def fake_build_runtime_manifest(**kwargs):
        captured.update(kwargs)
        return {"semantic": {"runtime_config": kwargs["runtime_config"]}}

    def fake_verify(manifest, *, expected=None, **_kwargs):
        valid = (
            expected is None
            or manifest["semantic"]["runtime_config"]
            == expected["semantic"]["runtime_config"]
        )
        return SimpleNamespace(
            valid=valid,
            digest=digest,
            errors=() if valid else ("semantic mismatch",),
        )

    monkeypatch.setattr(attestation, "build_runtime_manifest", fake_build_runtime_manifest)
    monkeypatch.setattr(attestation, "verify_runtime_manifest", fake_verify)
    monkeypatch.setenv("Q2_ROOT", "/isolated/q2")
    monkeypatch.setenv("Q2_SOURCE_REVISION", "actual-revision")
    artifact = SimpleNamespace(runtime_manifest_sha256=digest)

    with pytest.raises(RuntimeError, match="no runtime manifest digest"):
        _attest_worker_runtime(
            SimpleNamespace(runtime_manifest_sha256=""),
            _attestation_args(),
            "cuda",
        )

    assert _attest_worker_runtime(
        artifact, _attestation_args(), "cuda"
    ) == digest
    assert captured["runtime_config"] == expected_config
    assert captured["source_revision"] == "actual-revision"

    with pytest.raises(RuntimeError, match="runtime attestation failed"):
        _attest_worker_runtime(
            artifact, _attestation_args(n_ml=3), "cuda"
        )
