import gzip
import json

import pytest

from harness.distributed_runtime import (
    AssignmentLeaseBook,
    AssignmentUnavailableError,
    BackoffPolicy,
    LatticeArtifact,
    LearnerLatticeStore,
    ReconnectBackoff,
    RetryExhaustedError,
    RolloutAssignment,
    StaleLeaseError,
    build_generation_assignments,
    is_retryable_http_status,
    validate_batch_lease,
    validate_lattice_payload,
)


LEARNER = "shadow-learner"
CONFIG_HASH = "b" * 64
POLICY_HASH = "a" * 64


def _assignments(
    *,
    policy_version=100,
    rollout_index=7,
    lanes=1,
    n_envs=2,
    lattice_artifacts=None,
):
    return build_generation_assignments(
        learner_id=LEARNER,
        config_hash=CONFIG_HASH,
        policy_version=policy_version,
        policy_sha256=POLICY_HASH,
        rollout_index=rollout_index,
        lanes=lanes,
        steps=256,
        n_envs=n_envs,
        maps=("mltrain_00005208", "mltrain_00005209"),
        base_seed=929,
        base_game_seed=1931,
        lattice_artifacts=lattice_artifacts,
    )


def _lattice_payload(env_steps, n_envs=2, marker=1):
    document = {
        "version": 1,
        "env_steps": env_steps,
        "instances": [
            {
                "maps": {
                    "mltrain_00005208": [
                        {
                            "cell": [index, marker, 3],
                            "visits": float(marker),
                        }
                    ]
                }
            }
            for index in range(n_envs)
        ],
    }
    return gzip.compress(
        (json.dumps(document, separators=(",", ":")) + "\n").encode(),
        mtime=0,
    )


def test_generation_assignments_are_stable_lane_owned_and_policy_specific():
    first = _assignments(lanes=2)
    repeated = _assignments(lanes=2)
    assert first == repeated
    assert [value.assignment_id for value in first] == [
        value.assignment_id for value in repeated
    ]
    assert len({value.assignment_id for value in first}) == 2
    assert len({value.seed for value in first}) == 2
    assert len({value.game_seed for value in first}) == 2
    assert first[0].lane_id == (
        "q2l1-d104f524319d7f4a0acde0b62f95d666960a9dbfd55766e7d894935e5dc9abf6"
    )
    assert first[0].assignment_id == (
        "q2a1-0e865b88c3c54cc1e9cc1d4347e55e02f0e4e49c4247360703b9c57a24b62b4d"
    )
    assert (first[0].seed, first[0].game_seed) == (1198012780, 1250963339)
    assert RolloutAssignment.from_dict(first[0].as_dict()) == first[0]
    assert [value.map_name for value in first] == [
        "mltrain_00005208",
        "mltrain_00005209",
    ]

    next_generation = _assignments(policy_version=200, rollout_index=8, lanes=2)
    assert [value.lane_id for value in next_generation] == [
        value.lane_id for value in first
    ]
    assert [value.assignment_id for value in next_generation] != [
        value.assignment_id for value in first
    ]
    assert [value.seed for value in next_generation] == [value.seed for value in first]

    with pytest.raises(ValueError, match="unknown lanes"):
        _assignments(lanes=1, lattice_artifacts={4: "c" * 64})


def test_lease_heartbeat_expiration_replacement_and_stale_fencing():
    assignment = _assignments()[0]
    leases = AssignmentLeaseBook(lease_ttl=10.0, max_attempts=3)
    leases.add([assignment])

    first = leases.claim("worker-a", now=100.0)
    assert first is not None
    assert type(first).from_dict(first.as_dict()) == first
    assert first.assignment == assignment
    assert first.epoch == 1
    renewed = leases.heartbeat(first.lease_id, "worker-a", now=105.0)
    assert renewed.issued_at == 100.0
    assert renewed.expires_at == 115.0
    assert leases.expire(now=114.999) == ()
    assert leases.expire(now=115.0) == (assignment.assignment_id,)

    replacement = leases.claim(
        "worker-b", now=115.0, assignment_id=assignment.assignment_id
    )
    assert replacement is not None
    assert replacement.assignment.assignment_id == first.assignment.assignment_id
    assert replacement.assignment.seed == first.assignment.seed
    assert replacement.assignment.game_seed == first.assignment.game_seed
    assert replacement.epoch == 2
    assert replacement.lease_id != first.lease_id

    with pytest.raises(StaleLeaseError):
        leases.heartbeat(first.lease_id, "worker-a", now=116.0)
    with pytest.raises(StaleLeaseError):
        leases.complete(first.lease_id, "worker-a", now=116.0)

    metadata = {
        **assignment.batch_contract(),
        "worker_id": "worker-b",
        "lease_id": replacement.lease_id,
        "lease_epoch": replacement.epoch,
    }
    validate_batch_lease(metadata, assignment, replacement, now=116.0)
    with pytest.raises(ValueError, match="seed"):
        validate_batch_lease({**metadata, "seed": assignment.seed + 1}, assignment, replacement)

    completed = leases.complete(replacement.lease_id, "worker-b", now=117.0)
    assert completed == assignment
    assert leases.snapshot(assignment.assignment_id)["state"] == "completed"
    with pytest.raises(AssignmentUnavailableError):
        leases.claim("worker-c", now=118.0, assignment_id=assignment.assignment_id)


def test_lease_retry_budget_fails_closed_after_repeated_expiration():
    assignment = _assignments()[0]
    leases = AssignmentLeaseBook(
        lease_ttl=5.0, max_attempts=2, incarnation_id="old-learner-process"
    )
    leases.add([assignment])

    first = leases.claim("worker-a", now=0.0)
    assert first is not None
    assert leases.expire(now=5.0) == (assignment.assignment_id,)
    second = leases.claim("worker-b", now=5.0)
    assert second is not None and second.epoch == 2
    assert leases.expire(now=10.0) == (assignment.assignment_id,)
    assert leases.claim("worker-c", now=10.0) is None
    snapshot = leases.snapshot(assignment.assignment_id)
    assert snapshot["state"] == "failed"
    assert snapshot["attempts"] == 2
    assert snapshot["last_failure"] == "lease_expired"

    # A learner restart gets a new fencing-token namespace even if its
    # in-memory attempt counter must be restored separately later.
    restarted = AssignmentLeaseBook(
        lease_ttl=5.0, max_attempts=2, incarnation_id="new-learner-process"
    )
    restarted.add([assignment])
    after_restart = restarted.claim("worker-a", now=0.0)
    assert after_restart is not None
    assert after_restart.lease_id != first.lease_id


def test_reconnect_backoff_is_bounded_stable_per_worker_and_resettable():
    policy = BackoffPolicy(
        initial_delay=1.0,
        maximum_delay=5.0,
        multiplier=2.0,
        jitter_fraction=0.25,
        max_attempts=4,
    )
    first = ReconnectBackoff(policy, jitter_key="worker-a")
    twin = ReconnectBackoff(policy, jitter_key="worker-a")
    other = ReconnectBackoff(policy, jitter_key="worker-b")
    first_delays = [first.next_delay() for _ in range(4)]
    assert first_delays == [twin.next_delay() for _ in range(4)]
    assert first_delays != [other.next_delay() for _ in range(4)]
    assert all(0.0 <= delay <= 5.0 for delay in first_delays)
    with pytest.raises(RetryExhaustedError):
        first.next_delay()
    first.reset()
    assert first.attempts == 0
    assert first.next_delay() == first_delays[0]
    assert is_retryable_http_status(429)
    assert is_retryable_http_status(503)
    assert not is_retryable_http_status(401)
    assert not is_retryable_http_status(409)


def test_lattice_artifact_round_trip_validates_assignment_and_tampering():
    assignment = _assignments()[0]
    payload = _lattice_payload(512)
    info = validate_lattice_payload(payload)
    assert (info.state_version, info.env_steps, info.instance_count) == (1, 512, 2)

    artifact = LatticeArtifact.from_assignment(assignment, payload)
    restored = LatticeArtifact.decode(artifact.encode())
    assert restored == artifact
    assert restored.artifact_sha256 == artifact.artifact_sha256
    restored.validate_source_assignment(assignment)

    corrupted = bytearray(artifact.encode())
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        LatticeArtifact.decode(bytes(corrupted))

    wrong_lane = _assignments(lanes=2)[1]
    with pytest.raises(ValueError, match="source assignment"):
        restored.validate_source_assignment(wrong_lane)
    with pytest.raises(ValueError, match="instance count"):
        LatticeArtifact.from_assignment(assignment, _lattice_payload(512, n_envs=1))


def test_learner_store_adopts_parent_chain_and_recovers_exact_reference(tmp_path):
    store = LearnerLatticeStore(tmp_path / "lattices", LEARNER, CONFIG_HASH)
    first_assignment = _assignments(policy_version=100, rollout_index=1)[0]
    first, first_path = store.adopt(first_assignment, _lattice_payload(512, marker=1))
    assert first_path.is_file()
    assert store.latest(0) == first
    assert store.publish(first, first_assignment) == first_path

    second_assignment = _assignments(
        policy_version=200,
        rollout_index=2,
        lattice_artifacts={0: first.artifact_sha256},
    )[0]
    recovered = store.recover_for(second_assignment)
    assert recovered == first
    recovered.validate_recovery_for(second_assignment)

    second, second_path = store.adopt(
        second_assignment, _lattice_payload(1024, marker=2)
    )
    assert second_path.is_file()
    assert second.parent_artifact_sha256 == first.artifact_sha256
    assert store.latest(0) == second
    assert store.get(0, first.artifact_sha256) == first

    conflicting = LatticeArtifact.from_assignment(
        second_assignment, _lattice_payload(1025, marker=3)
    )
    with pytest.raises(ValueError, match="conflicting"):
        store.publish(conflicting, second_assignment)

    wrong_store = LearnerLatticeStore(tmp_path / "wrong", "other-learner", CONFIG_HASH)
    with pytest.raises(ValueError, match="different learner"):
        wrong_store.publish(second, second_assignment)

    missing_reference = _assignments(
        policy_version=300,
        rollout_index=3,
        lattice_artifacts={0: "f" * 64},
    )[0]
    with pytest.raises(FileNotFoundError):
        store.recover_for(missing_reference)


def test_lattice_payload_validation_rejects_unsafe_or_wrong_shapes():
    with pytest.raises(ValueError, match="unsupported lattice state"):
        validate_lattice_payload(json.dumps({
            "version": 2,
            "env_steps": 0,
            "instances": [],
        }).encode())
    with pytest.raises(ValueError, match="coordinate"):
        validate_lattice_payload(json.dumps({
            "version": 1,
            "env_steps": 0,
            "instances": [{"maps": {"q2dm1": [{"cell": [1, 2]}]}}],
        }).encode())
    with pytest.raises(ValueError, match="invalid JSON constant"):
        validate_lattice_payload(
            b'{"version":1,"env_steps":NaN,"instances":[]}'
        )
