import gzip
import json
import time

import numpy as np
import pytest

from harness.distributed_runtime import LearnerLatticeStore
from harness.rollout_protocol import (
    CoordinatorClient,
    CoordinatorRecoveryConfig,
    CoordinatorRequestError,
    CoordinatorServer,
    PPO_BEHAVIOR_METRIC_KEYS,
    PPO_EPISODE_SUMMARY_COLUMNS,
    PPO_TELEMETRY_SCHEMA,
    PolicyArtifact,
    RolloutBatch,
    RolloutCoordinator,
)


RUNTIME_DIGEST = "d" * 64
CONFIG_HASH = "c" * 64


def _lattice_payload(env_steps, n_envs=2):
    document = {
        "version": 1,
        "env_steps": env_steps,
        "instances": [
            {"maps": {"map-a": [{"cell": [index, 2, 3], "visits": 1.0}]}}
            for index in range(n_envs)
        ],
    }
    return gzip.compress(json.dumps(document).encode(), mtime=0)


def _leased_batch(policy, lease, lattice_payload):
    assignment = lease.assignment
    steps, envs, obs_dim = assignment.steps, assignment.n_envs, 219
    metadata = {
        **assignment.batch_contract(),
        "worker_id": lease.worker_id,
        "sequence": lease.epoch,
        "lease_id": lease.lease_id,
        "lease_epoch": lease.epoch,
        "producer": "q2",
        "device": "cpu",
        "deterministic_actions": False,
        "runtime_manifest_sha256": policy.runtime_manifest_sha256,
        "telemetry_schema": PPO_TELEMETRY_SCHEMA,
    }
    zeros = np.zeros
    return RolloutBatch(metadata, {
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
        "episode_summaries": zeros(
            (0, len(PPO_EPISODE_SUMMARY_COLUMNS)), np.float64
        ),
        "behavior_sums": zeros(len(PPO_BEHAVIOR_METRIC_KEYS), np.float64),
        "behavior_samples": np.array([steps * envs], dtype=np.int64),
        "lattice_payload": np.frombuffer(lattice_payload, dtype=np.uint8).copy(),
    })


def _coordinator(tmp_path, lease_ttl=1.0):
    store = LearnerLatticeStore(tmp_path / "learner-lattices", "learner-a", CONFIG_HASH)
    recovery = CoordinatorRecoveryConfig(
        learner_id="learner-a",
        steps=4,
        n_envs=2,
        maps=("map-a",),
        base_seed=10,
        base_game_seed=20,
        lease_ttl=lease_ttl,
        max_attempts=3,
        lattice_store=store,
    )
    coordinator = RolloutCoordinator(
        1,
        schema="ppo",
        expected_runtime_manifest_sha256=RUNTIME_DIGEST,
        recovery=recovery,
    )
    return coordinator, store


def test_loopback_lease_fencing_duplicate_receipt_and_lattice_recovery(tmp_path):
    coordinator, store = _coordinator(tmp_path)
    first_policy = PolicyArtifact.create(
        1, b"policy-one", CONFIG_HASH, RUNTIME_DIGEST
    )
    coordinator.publish(first_policy)
    server = CoordinatorServer(coordinator, "127.0.0.1", 0, token="secret").start()
    host, port = server.address
    client = CoordinatorClient(f"http://{host}:{port}", token="secret")
    try:
        lease = client.claim_assignment("worker-a")
        assert lease.assignment.policy_sha256 == first_policy.sha256
        assert client.status()["assignments_leased"] == 1
        renewed = client.heartbeat(lease)
        assert renewed.lease_id == lease.lease_id
        batch = _leased_batch(first_policy, lease, _lattice_payload(9))

        decision = client.submit(batch)
        assert decision.accepted
        assert decision.assignment_id == lease.assignment.assignment_id
        assert decision.lattice_artifact_sha256
        assert store.latest(0).artifact_sha256 == decision.lattice_artifact_sha256

        # A lost HTTP response can safely retry the exact bytes after the
        # lease has completed and still receive the learner's adoption receipt.
        duplicate = client.submit(batch)
        assert duplicate.status == "duplicate"
        assert duplicate.lattice_artifact_sha256 == decision.lattice_artifact_sha256
        assert len(coordinator.wait_for_quorum(1, 0.2)) == 1

        second_policy = PolicyArtifact.create(
            2, b"policy-two", CONFIG_HASH, RUNTIME_DIGEST
        )
        coordinator.publish(second_policy)
        second = client.claim_assignment("worker-a", preferred_lane=0)
        assert (
            second.assignment.lattice_artifact_sha256
            == decision.lattice_artifact_sha256
        )
        recovered = client.fetch_lattice(
            second.assignment.lane_index,
            second.assignment.lattice_artifact_sha256,
        )
        recovered.validate_recovery_for(second.assignment)

        with pytest.raises(CoordinatorRequestError) as stale:
            client.heartbeat(lease)
        assert stale.value.status == 409
    finally:
        server.close()


def test_expired_assignment_is_reissued_with_same_identity_and_new_fence(tmp_path):
    coordinator, _store = _coordinator(tmp_path, lease_ttl=0.02)
    policy = PolicyArtifact.create(1, b"policy", CONFIG_HASH, RUNTIME_DIGEST)
    coordinator.publish(policy)
    server = CoordinatorServer(coordinator, "127.0.0.1", 0).start()
    host, port = server.address
    client = CoordinatorClient(f"http://{host}:{port}")
    try:
        first = client.claim_assignment("failed-worker")
        time.sleep(0.03)
        replacement = client.claim_assignment("replacement-worker")
        assert replacement.assignment == first.assignment
        assert replacement.epoch == first.epoch + 1
        assert replacement.lease_id != first.lease_id
        with pytest.raises(CoordinatorRequestError) as stale:
            client.heartbeat(first)
        assert stale.value.status == 409
    finally:
        server.close()
