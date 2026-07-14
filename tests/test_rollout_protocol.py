import numpy as np
import pytest

from harness.rollout_protocol import (
    CoordinatorClient,
    CoordinatorServer,
    PPO_BEHAVIOR_METRIC_KEYS,
    PPO_EPISODE_SUMMARY_COLUMNS,
    PPO_TELEMETRY_SCHEMA,
    PolicyArtifact,
    RolloutBatch,
    RolloutCoordinator,
    deterministic_synthetic_batch,
    merge_ppo_batches,
)


def _policy(version=1, payload=b"policy-one", runtime_digest=""):
    return PolicyArtifact.create(
        version,
        payload,
        config_hash="cfg-test",
        runtime_manifest_sha256=runtime_digest,
    )


def test_behavior_schema_carries_engagement_and_posture_audit():
    required = {
        "level_aim_movement_reward",
        "entity_count",
        "enemy_count",
        "enemy_visible_any",
        "enemy_visible_count",
        "aim_yaw_error",
        "aim_pitch_error",
        "aim_tracking_quality",
        "damage_dealt",
        "damage_taken",
        "kills",
        "deaths",
    }
    assert required.issubset(PPO_BEHAVIOR_METRIC_KEYS)


def test_policy_and_batch_round_trip_are_deterministic_and_pickle_free():
    policy = _policy()
    assert PolicyArtifact.decode(policy.encode()) == policy
    batch = deterministic_synthetic_batch(policy, "worker-a", 1, 10, 20, 0)
    first = batch.encode()
    second = batch.encode()
    assert first == second
    restored = RolloutBatch.decode(first)
    assert restored.metadata == batch.metadata
    assert restored.rollout_hash() == batch.rollout_hash()
    for name in batch.arrays:
        assert np.array_equal(restored.arrays[name], batch.arrays[name])

    corrupted = bytearray(first)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="hash"):
        RolloutBatch.decode(bytes(corrupted))


def test_coordinator_quorum_duplicate_stale_and_determinism_rejection():
    coordinator = RolloutCoordinator(quorum=2)
    policy = _policy()
    coordinator.publish(policy)
    first = deterministic_synthetic_batch(policy, "worker-a", 1, 10, 20, 0)
    twin = deterministic_synthetic_batch(policy, "worker-b", 1, 10, 20, 0)

    accepted = coordinator.submit(first.encode())
    assert accepted.accepted and accepted.quorum_count == 1
    duplicate = coordinator.submit(first.encode())
    assert duplicate.status == "duplicate" and not duplicate.accepted
    mismatch = deterministic_synthetic_batch(policy, "worker-c", 1, 10, 20, 0)
    mismatch.arrays["rewards"][0] += 1.0
    assert coordinator.submit(mismatch.encode()).status == "determinism_mismatch"
    accepted_twin = coordinator.submit(twin.encode())
    assert accepted_twin.accepted and accepted_twin.quorum_count == 2
    assert len(coordinator.wait_for_quorum(1, 0.1)) == 2
    closed = deterministic_synthetic_batch(policy, "worker-d", 1, 12, 22, 1)
    assert coordinator.submit(closed.encode()).status == "generation_closed"

    coordinator.publish(_policy(2, b"policy-two"))
    stale = deterministic_synthetic_batch(policy, "worker-a", 2, 11, 21, 1)
    assert coordinator.submit(stale.encode()).status == "stale"


def test_loopback_http_fetch_submit_status_and_authentication():
    coordinator = RolloutCoordinator(quorum=1)
    policy = _policy()
    coordinator.publish(policy)
    server = CoordinatorServer(coordinator, "127.0.0.1", 0, token="secret").start()
    host, port = server.address
    try:
        client = CoordinatorClient(f"http://{host}:{port}", token="secret")
        assert client.fetch_policy() == policy
        assert client.status()["accepted_for_current"] == 0
        batch = deterministic_synthetic_batch(policy, "loopback", 1, 3, 4, 0)
        decision = client.submit(batch)
        assert decision.accepted
        assert client.status()["accepted_for_current"] == 1
        assert len(coordinator.wait_for_quorum(1, 0.5)) == 1

        unauthorized = CoordinatorClient(f"http://{host}:{port}")
        with pytest.raises(RuntimeError, match="401"):
            unauthorized.fetch_policy()
    finally:
        server.close()


def test_ppo_runtime_digest_is_required_and_bound_to_policy_and_status():
    runtime_digest = "a" * 64
    policy = _policy(runtime_digest=runtime_digest)
    assert PolicyArtifact.decode(policy.encode()) == policy

    with pytest.raises(ValueError, match="expected runtime"):
        RolloutCoordinator(quorum=1, schema="ppo")
    coordinator = RolloutCoordinator(
        quorum=1,
        schema="ppo",
        expected_runtime_manifest_sha256=runtime_digest,
    )
    with pytest.raises(ValueError, match="policy runtime manifest"):
        coordinator.publish(_policy())
    with pytest.raises(ValueError, match="policy runtime manifest"):
        coordinator.publish(_policy(runtime_digest="b" * 64))
    coordinator.publish(policy)
    status = coordinator.status()
    assert status["runtime_manifest_sha256"] == runtime_digest
    assert status["policy_sha256"] == policy.sha256
    assert status["config_hash"] == policy.config_hash


def test_ppo_schema_rejects_incomplete_batches_and_accepts_exact_shapes():
    runtime_digest = "a" * 64
    policy = _policy(runtime_digest=runtime_digest)
    coordinator = RolloutCoordinator(
        quorum=1,
        schema="ppo",
        expected_runtime_manifest_sha256=runtime_digest,
    )
    coordinator.publish(policy)
    incomplete = deterministic_synthetic_batch(policy, "worker-a", 1, 1, 2, 0)
    assert coordinator.submit(incomplete.encode()).status == "invalid_schema"

    steps, envs, obs_dim = 4, 2, 219
    metadata = dict(
        incomplete.metadata,
        producer="q2",
        map_name="map-a",
        n_envs=envs,
        lattice_mode="fresh_worker_session",
        deterministic_actions=False,
        runtime_manifest_sha256=runtime_digest,
        telemetry_schema=PPO_TELEMETRY_SCHEMA,
    )
    zeros = np.zeros
    batch = RolloutBatch(metadata, {
        "obs": zeros((steps, envs, obs_dim), np.float32),
        "actions": zeros((steps, envs, 8), np.float32),
        "rewards": zeros((steps, envs), np.float32),
        "dones": zeros((steps, envs), np.uint8),
        "values": zeros((steps, envs), np.float32),
        "log_probs": zeros((steps, envs), np.float32),
        "h_states": zeros((steps, envs, 256), np.float32),
        "c_states": zeros((steps, envs, 256), np.float32),
        "last_obs": zeros((envs, obs_dim), np.float32),
        "last_h": zeros((envs, 256), np.float32),
        "last_c": zeros((envs, 256), np.float32),
        "episode_summaries": np.array(
            [[10.0, 7.0, 3.0, 2.0, 1.0, 40.0]], dtype=np.float64
        ),
        "behavior_sums": np.arange(
            len(PPO_BEHAVIOR_METRIC_KEYS), dtype=np.float64
        ),
        "behavior_samples": np.array([steps * envs], dtype=np.int64),
    })
    batch.arrays["dones"][-1, 0] = 1
    assert coordinator.submit(batch.encode()).accepted
    runtime_guard = RolloutCoordinator(
        quorum=1,
        schema="ppo",
        expected_runtime_manifest_sha256=runtime_digest,
    )
    runtime_guard.publish(policy)
    wrong_runtime = RolloutBatch(dict(
        metadata,
        worker_id="worker-wrong-runtime",
        runtime_manifest_sha256="b" * 64,
    ), {name: array.copy() for name, array in batch.arrays.items()})
    decision = runtime_guard.submit(wrong_runtime.encode())
    assert decision.status == "wrong_runtime"
    assert decision.quorum_count == 0
    second = RolloutBatch(dict(
        metadata, worker_id="worker-b", map_name="map-b"
    ), {
        name: array.copy() for name, array in batch.arrays.items()
    })
    second.arrays["episode_summaries"][0] = np.array(
        [20.0, 12.0, 8.0, 1.0, 0.0, 24.0], dtype=np.float64
    )
    second.arrays["behavior_sums"] *= 2.0
    merged = merge_ppo_batches([batch, second])
    assert merged["obs"].shape == (steps, envs * 2, obs_dim)
    assert merged["last_h"].shape == (envs * 2, 256)
    assert merged["episode_summaries"].shape == (
        2, len(PPO_EPISODE_SUMMARY_COLUMNS)
    )
    assert merged["episode_summaries"][:, 0].tolist() == [10.0, 20.0]
    assert merged["episode_map_names"] == ("map-a", "map-b")
    assert np.array_equal(
        merged["behavior_sums"],
        np.arange(len(PPO_BEHAVIOR_METRIC_KEYS), dtype=np.float64) * 3.0,
    )
    assert merged["behavior_samples"].tolist() == [steps * envs * 2]

    mismatched_runtime = RolloutBatch(dict(
        second.metadata,
        worker_id="worker-runtime-b",
        runtime_manifest_sha256="b" * 64,
    ), {name: array.copy() for name, array in second.arrays.items()})
    with pytest.raises(ValueError, match="different policies"):
        merge_ppo_batches([batch, mismatched_runtime])

    malformed = RolloutBatch(dict(metadata, worker_id="worker-c"), {
        name: array.copy() for name, array in batch.arrays.items()
    })
    malformed.arrays["behavior_sums"] = np.zeros(1, dtype=np.float64)
    with pytest.raises(ValueError, match="behavior_sums"):
        malformed.validate_ppo_schema()

    missing_episode = RolloutBatch(dict(metadata, worker_id="worker-d"), {
        name: array.copy() for name, array in batch.arrays.items()
    })
    missing_episode.arrays["episode_summaries"] = np.empty(
        (0, len(PPO_EPISODE_SUMMARY_COLUMNS)), dtype=np.float64
    )
    with pytest.raises(ValueError, match="completed episodes"):
        missing_episode.validate_ppo_schema()

    restored = RolloutBatch.decode(batch.encode())
    assert np.array_equal(
        restored.arrays["episode_summaries"], batch.arrays["episode_summaries"]
    )
    assert np.array_equal(
        restored.arrays["behavior_sums"], batch.arrays["behavior_sums"]
    )
