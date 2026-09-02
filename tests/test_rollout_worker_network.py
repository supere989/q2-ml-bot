"""Network-native rollout-worker backend tests (torch-free).

Drives tools/rollout_worker._drive_network_rollout over scripted envs with
stub inference and asserts the produced RolloutBatch satisfies the PPO wire
schema — including the fire_allowed mask channel — survives coordinator
submit with a quorum of 2, merges along the env axis, and never buffers
boundary rounds. The torch policy path of collect_network_batch itself
needs a GPU host and is verified on WSL.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from harness.client_batch import Q2NetworkClientMultiEnv
from harness.client_env import ClientActionDispatch, ClientTelemetryDrain
from harness.client_protocol import ClientTelemetry
from harness.protocol import Action
from harness.protocol import (
    ML_ACTION_GENERATION_SHIFT,
    ML_FIRE_GATE_SUPPRESSED,
)
from harness.rollout_protocol import (
    PPO_TELEMETRY_SCHEMA,
    PolicyArtifact,
    RolloutBatch,
    RolloutCoordinator,
    merge_ppo_batches,
)
from tools.rollout_worker import (
    _drive_network_rollout,
    _new_batch_telemetry,
    _new_episode_accumulators,
)

RUNTIME_DIGEST = "a" * 64
STEPS = 6
N_ENVS = 2


class _SpatialStub:
    def __init__(self):
        self.finalized = []

    def finalize_episode(self, *, terminal_reason, truncated):
        self.finalized.append((terminal_reason, truncated))
        return 0.25, {"outcome_bonus": 0.25}


def _telemetry(client_id, slot, sequence, frame, *, echo_tick=0, accepted=0,
               forward=0.0, right=0.0, look_yaw=0.0, look_pitch=0.0,
               jump=False, fire=False, hook=0, weapon=0, terminal=False,
               map_name="mltrain_x", gate_flags=0, action_generation=10):
    debug = np.zeros(12, dtype=np.float32)
    debug[0] = echo_tick
    debug[1] = accepted
    debug[4] = forward
    debug[5] = right
    debug[6] = look_yaw
    debug[7] = look_pitch
    debug[8] = int(jump)
    debug[9] = int(fire)
    debug[10] = int(hook)
    debug[3] = int(weapon)
    if accepted and not (int(gate_flags) & 0x00FF0000):
        gate_flags = int(gate_flags) | (
            ((int(action_generation) % 192) + 1)
            << ML_ACTION_GENERATION_SHIFT
        )
    debug[11] = int(gate_flags)
    obs = SimpleNamespace(
        action_debug=debug,
        is_terminal=terminal,
        terminal_reason=1 if terminal else 0,
    )
    return ClientTelemetry(
        sequence=sequence, client_slot=slot, server_frame=frame,
        client_id=client_id, map_name=map_name, observation=obs,
    )


class _ScriptedEnv:
    def __init__(self, client_id, slot, script, *, preflight=(),
                 preflight_at=0):
        self.client_id = client_id
        self.slot = slot
        self.script = list(script)
        self.preflight = list(preflight)
        self.preflight_at = int(preflight_at)
        self._drain_calls = 0
        self._last = None
        self._spatial = _SpatialStub()

    def start(self):
        self._last = _telemetry(self.client_id, self.slot, 1, 10)
        return self._last

    def initial_result(self, current, *, vector=False):
        value = np.array([current.server_frame, self.slot, -1.0],
                         dtype=np.float32)
        return value, {"map": current.map_name, "client_id": self.client_id}

    def dispatch_action(self, action):
        return ClientActionDispatch(
            client_id=self.client_id,
            client_slot=self.slot,
            after_sequence=self._last.sequence,
            action_tick=self._last.server_frame,
            map_name=self._last.map_name,
            action=action,
        )

    def drain_latest_telemetry(self):
        previous = self._last
        self._drain_calls += 1
        if not self.preflight or self._drain_calls <= self.preflight_at:
            return ClientTelemetryDrain(previous, previous, 0, ())
        drained = self.preflight
        self.preflight = []
        latest = drained[-1]
        self._last = latest
        return ClientTelemetryDrain(
            previous=previous, latest=latest,
            packet_count=len(drained),
            map_names=tuple(sample.map_name for sample in drained),
        )

    def receive_telemetry(self, *, after_sequence, timeout=None):
        if not self.script:
            raise TimeoutError("script exhausted")
        current = self.script.pop(0)
        assert current.sequence > after_sequence
        self._last = current
        return current

    def transition_result(self, current, *, vector=False):
        value = np.array([current.server_frame, self.slot, 1.0],
                         dtype=np.float32)
        info = {
            "map": current.map_name,
            "client_id": self.client_id,
            "spatial_bonus": 0.0,
            "terminal_reason": int(current.observation.terminal_reason),
        }
        return (
            value,
            1.5,
            current.observation.is_terminal,
            False,
            info,
        )

    def reset_episode_vector(self):
        return np.array([-7.0, self.slot, 99.0], dtype=np.float32)

    def close(self):
        pass


def _matched(client_id, slot, sequence, frame, action, dispatch_frame,
             *, terminal=False, suppressed=False):
    return _telemetry(
        client_id, slot, sequence, frame,
        echo_tick=frame, accepted=1,
        forward=action.move_forward, right=action.move_right,
        look_yaw=action.look_yaw, look_pitch=action.look_pitch,
        jump=action.jump,
        fire=False if suppressed else action.fire,
        hook=action.hook, weapon=action.weapon,
        gate_flags=ML_FIRE_GATE_SUPPRESSED if suppressed else 0,
        action_generation=dispatch_frame, terminal=terminal,
    )


# Round plan (start frame 10): see module docstring scenario.
#   k=0 dispatch@10 -> echo f10            [row 0]
#   k=1 dispatch@10 -> echo f11, env0 fire suppressed [row 1]
#   k=2 dispatch@11 -> echo f12            [row 2]
#   k=3 dispatch@12 -> echo f13, env0 dies [row 3]
#   k=4 catchup boundary (preflight f14)   [no row]
#   k=5 dispatch@14 -> echo f15            [row 4]
#   k=6 dispatch@15 -> echo f16            [row 5]
#   k=7 dispatch@16 -> echo f17            [discarded tail]
_ACTIONS = [
    [Action(move_forward=0.1 * (k + 1), look_yaw=0.02 * k, fire=(k == 1)),
     Action(move_forward=-0.1, jump=bool(k % 2))]
    for k in range(8)
]


def _make_envs():
    def script_for(client_id, slot):
        return [
            _matched(client_id, slot, 2, 10, _ACTIONS[0][slot], 10),
            _matched(client_id, slot, 3, 11, _ACTIONS[1][slot], 10,
                     suppressed=(slot == 0)),
            _matched(client_id, slot, 4, 12, _ACTIONS[2][slot], 11),
            _matched(client_id, slot, 5, 13, _ACTIONS[3][slot], 12,
                     terminal=(slot == 0)),
            _matched(client_id, slot, 7, 15, _ACTIONS[5][slot], 14),
            _matched(client_id, slot, 8, 16, _ACTIONS[6][slot], 15),
            _matched(client_id, slot, 9, 17, _ACTIONS[7][slot], 16),
        ]

    env_a = _ScriptedEnv(
        "client-a", 0, script_for("client-a", 0),
        preflight=[_telemetry("client-a", 0, 6, 14)], preflight_at=4,
    )
    env_b = _ScriptedEnv(
        "client-b", 1, script_for("client-b", 1),
        preflight=[_telemetry("client-b", 1, 6, 14)], preflight_at=4,
    )
    return [env_a, env_b]


def _stub_act_factory(counter):
    def act(obs_in):
        index = counter[0]
        counter[0] += 1
        return {
            "current_obs": np.asarray(obs_in, dtype=np.float32).copy(),
            "h_step": np.zeros((N_ENVS, 256), dtype=np.float32),
            "c_step": np.zeros((N_ENVS, 256), dtype=np.float32),
            "actions": np.array(
                [[a.move_forward, a.move_right, a.look_yaw, a.look_pitch,
                  float(a.jump), float(a.fire), float(a.hook),
                  float(a.weapon)] for a in _ACTIONS[index]],
                dtype=np.float32,
            ),
            "values": np.full(N_ENVS, 0.5, dtype=np.float32),
            "log_probs": np.full(N_ENVS, -2.0, dtype=np.float32),
            "fire_allowed": np.ones(N_ENVS, dtype=np.bool_),
            "fire_metadata": {
                "fire_allowed": np.ones(N_ENVS, dtype=np.bool_),
                "raw_fire_probability": np.full(N_ENVS, 0.8, np.float32),
                "raw_fire_log_probability": np.full(
                    N_ENVS, np.log(0.8), np.float32
                ),
            },
        }
    return act


POLICY = PolicyArtifact.create(1, b"network-policy", "cfg", RUNTIME_DIGEST)


def _collect_batch(worker_id, sequence, rollout_index):
    srv = Q2NetworkClientMultiEnv(_make_envs(), max_ep_steps=1000)
    try:
        vectors = srv.reset_all()
        obs = np.stack(vectors)
        hidden = [np.zeros((1, 1, 256), dtype=np.float32)
                  for _ in range(N_ENVS)]
        episode_accumulators = _new_episode_accumulators(N_ENVS)
        batch_telemetry = _new_batch_telemetry()
        arrays, final, accepted = _drive_network_rollout(
            srv,
            steps=STEPS,
            policy_version=1,
            obs=obs,
            hidden=hidden,
            act=_stub_act_factory([0]),
            init_hidden=lambda bi: np.zeros((1, 1, 256), dtype=np.float32),
            episode_accumulators=episode_accumulators,
            batch_telemetry=batch_telemetry,
            hidden_dim=256,
        )
        from tools.rollout_worker import _finalize_batch_telemetry
        arrays["last_obs"] = np.asarray(final, dtype=np.float32)
        arrays["last_h"] = np.zeros((N_ENVS, 256), dtype=np.float32)
        arrays["last_c"] = np.zeros((N_ENVS, 256), dtype=np.float32)
        arrays.update(_finalize_batch_telemetry(batch_telemetry))
        metadata = {
            "worker_id": worker_id,
            "sequence": sequence,
            "policy_version": 1,
            "policy_sha256": POLICY.sha256,
            "config_hash": POLICY.config_hash,
            "seed": 1,
            "game_seed": 1,
            "rollout_index": rollout_index,
            "determinism_key": (
                f"q2-network:v1:{POLICY.sha256}:cfg={POLICY.config_hash}:"
                f"seed=1:game=1:rollout={rollout_index}:"
                f"map=mltrain_x:steps={STEPS}:envs={N_ENVS}"
            ),
            "producer": "q2",
            "collection_mode": "network",
            "fire_mask": "explicit",
            "seed_semantics": "policy_sampling_and_spatial_only",
            "map_name": "mltrain_x",
            "n_envs": N_ENVS,
            "device": "cpu",
            "deterministic_actions": True,
            "telemetry_schema": PPO_TELEMETRY_SCHEMA,
            "runtime_manifest_sha256": RUNTIME_DIGEST,
            "lattice_mode": "fresh_worker_session",
        }
        return RolloutBatch(metadata, arrays)
    finally:
        srv.close()


def test_network_batch_schema_reconciliation_and_boundary_handling():
    batch = _collect_batch("worker-a", 1, 0)
    batch.validate_ppo_schema()

    # Fire-suppression reconciliation landed before row 1 was written:
    # sampled fire becomes no-fire, its raw log-probability is subtracted,
    # and the closed mask is recorded.
    assert batch.arrays["actions"][1, 0, 5] == 0.0
    assert batch.arrays["fire_allowed"][1, 0] == 0
    assert batch.arrays["log_probs"][1, 0] == pytest.approx(
        -2.0 - float(np.log(0.8))
    )
    assert batch.arrays["fire_allowed"][1, 1] == 1
    assert batch.arrays["fire_allowed"][0, 0] == 1

    # The boundary round left no rows: every row is a trainable accepted
    # round, and row 4 is the post-boundary round (frame-14 initial obs).
    assert batch.arrays["obs"][4, 0, 0] == 14.0
    assert batch.arrays["obs"][4, 0, 2] == -1.0
    assert np.isfinite(batch.arrays["rewards"]).all()

    # Death round: done recorded, episode summary counted exactly once.
    assert batch.arrays["dones"][3, 0] == 1
    assert batch.arrays["dones"][:, 1].sum() == 0
    assert batch.arrays["episode_summaries"].shape[0] == 1
    # Death-round reward carries the finalize_episode outcome bonus.
    assert batch.arrays["rewards"][3, 0] == pytest.approx(1.5 + 0.25)
    assert int(batch.arrays["behavior_samples"][0]) == STEPS * N_ENVS

    # Wire round-trip preserves everything.
    decoded = RolloutBatch.decode(batch.encode())
    assert decoded.rollout_hash() == batch.rollout_hash()
    np.testing.assert_array_equal(
        decoded.arrays["fire_allowed"], batch.arrays["fire_allowed"]
    )


def test_network_batches_submit_and_merge_with_quorum_of_two():
    coordinator = RolloutCoordinator(
        quorum=2,
        schema="ppo",
        expected_runtime_manifest_sha256=RUNTIME_DIGEST,
    )
    coordinator.publish(POLICY)
    first = _collect_batch("worker-a", 1, 0)
    second = _collect_batch("worker-b", 1, 1)
    decision_a = coordinator.submit(first.encode())
    assert decision_a.accepted and decision_a.quorum_count == 1
    decision_b = coordinator.submit(second.encode())
    assert decision_b.accepted and decision_b.quorum_count == 2

    quorum = coordinator.wait_for_quorum(1, 0.1)
    assert len(quorum) == 2
    merged = merge_ppo_batches(quorum)
    assert merged["obs"].shape == (STEPS, 2 * N_ENVS, 3)
    assert merged["fire_allowed"].shape == (STEPS, 2 * N_ENVS)
    assert merged["fire_allowed"][1, 0] == 0
    assert merged["fire_allowed"][1, 2] == 0  # both lanes suppressed at row 1
    assert merged["fire_allowed"][0, 1] == 1
    assert merged["dones"][3, 0] == 1
    assert int(merged["behavior_samples"][0]) == 2 * STEPS * N_ENVS


def test_merge_rejects_mixed_fire_mask_quorum():
    explicit = _collect_batch("worker-a", 1, 0)
    legacy_arrays = {
        name: array for name, array in explicit.arrays.items()
        if name != "fire_allowed"
    }
    legacy_metadata = dict(explicit.metadata)
    del legacy_metadata["fire_mask"]
    legacy = RolloutBatch(legacy_metadata, legacy_arrays)
    legacy.validate_ppo_schema()  # older producers stay valid
    with pytest.raises(ValueError, match="different policies"):
        merge_ppo_batches([explicit, legacy])


def test_explicit_fire_mask_requires_the_array():
    batch = _collect_batch("worker-a", 1, 0)
    arrays = {
        name: array for name, array in batch.arrays.items()
        if name != "fire_allowed"
    }
    broken = RolloutBatch(dict(batch.metadata), arrays)
    with pytest.raises(ValueError, match="fire_allowed"):
        broken.validate_ppo_schema()
    flipped = RolloutBatch(
        {**batch.metadata, "fire_mask": "ones"}, batch.arrays
    )
    with pytest.raises(ValueError, match="fire_mask"):
        flipped.validate_ppo_schema()
