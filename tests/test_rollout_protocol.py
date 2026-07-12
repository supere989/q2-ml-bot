import numpy as np
import pytest

from harness.rollout_protocol import (
    CoordinatorClient,
    CoordinatorServer,
    PolicyArtifact,
    RolloutBatch,
    RolloutCoordinator,
    deterministic_synthetic_batch,
    merge_ppo_batches,
)


def _policy(version=1, payload=b"policy-one"):
    return PolicyArtifact.create(version, payload, config_hash="cfg-test")


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


def test_ppo_schema_rejects_incomplete_batches_and_accepts_exact_shapes():
    policy = _policy()
    coordinator = RolloutCoordinator(quorum=1, schema="ppo")
    coordinator.publish(policy)
    incomplete = deterministic_synthetic_batch(policy, "worker-a", 1, 1, 2, 0)
    assert coordinator.submit(incomplete.encode()).status == "invalid_schema"

    steps, envs, obs_dim = 4, 2, 219
    metadata = dict(
        incomplete.metadata,
        producer="q2",
        n_envs=envs,
        lattice_mode="fresh_worker_session",
        deterministic_actions=False,
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
    })
    assert coordinator.submit(batch.encode()).accepted
    second = RolloutBatch(dict(metadata, worker_id="worker-b"), {
        name: array.copy() for name, array in batch.arrays.items()
    })
    merged = merge_ppo_batches([batch, second])
    assert merged["obs"].shape == (steps, envs * 2, obs_dim)
    assert merged["last_h"].shape == (envs * 2, 256)
