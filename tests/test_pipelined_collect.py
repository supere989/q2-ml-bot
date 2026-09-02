"""Equivalence tests for the pipelined round driver.

The pipelined driver (Q2NetworkClientBatch.collect_rounds_pipelined) must
emit a BatchRound sequence identical to serial collect_round calls over the
same scripted session — accepted rounds, boundary rounds, rewards, feature
vectors, infos, tags, and final metrics — while overlapping the previous
round's assembly with the next round's echo wait.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from harness.client_batch import (
    Q2NetworkClientBatch,
    StalePolicyVersionError,
)
from harness.client_env import ClientActionDispatch, ClientTelemetryDrain
from harness.client_protocol import ClientTelemetry
from harness.protocol import Action
from harness.protocol import (
    ML_ACTION_GENERATION_SHIFT,
)


def _telemetry(
    client_id,
    slot,
    sequence,
    frame,
    *,
    echo_tick=0,
    accepted=0,
    forward=0.0,
    right=0.0,
    look_yaw=0.0,
    look_pitch=0.0,
    jump=False,
    fire=False,
    hook=0,
    weapon=0,
    terminal=False,
    damage_dealt=0.0,
    map_name="q2dm1",
    gate_flags=0,
    action_generation=10,
):
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
        reward_damage_dealt=damage_dealt,
    )
    return ClientTelemetry(
        sequence=sequence,
        client_slot=slot,
        server_frame=frame,
        client_id=client_id,
        map_name=map_name,
        observation=obs,
    )


class _ScriptedEnv:
    """Per-env packet script; drains are quiet unless preflight is queued.

    ``preflight_at`` is the drain-call index (0-based across collect calls)
    at which the queued preflight packets surface; earlier drains report no
    new packets, so a mid-stream catchup can be scripted.
    """

    def __init__(self, client_id, slot, script, *, preflight=(),
                 preflight_at=0):
        self.client_id = client_id
        self.slot = slot
        self.script = list(script)
        self.preflight = list(preflight)
        self.preflight_at = int(preflight_at)
        self._drain_calls = 0
        self._last = None

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
            previous=previous,
            latest=latest,
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
        value = np.array(
            [current.server_frame, self.slot,
             current.observation.reward_damage_dealt],
            dtype=np.float32,
        )
        info = {
            "map": current.map_name,
            "client_id": self.client_id,
            "spatial_bonus": 0.0,
        }
        return (
            value,
            float(current.observation.reward_damage_dealt),
            current.observation.is_terminal,
            False,
            info,
        )

    def reset_episode_vector(self):
        return np.array([self._last.server_frame, self.slot, -1.0],
                        dtype=np.float32)

    def close(self):
        pass


def _matched(client_id, slot, sequence, frame, action, dispatch_frame,
             **kwargs):
    """Echo packet for an action dispatched at dispatch_frame.

    The modulo-192 action generation must equal the dispatch's action_tick
    (the server frame of the last consumed packet), not the echo's frame.
    """
    return _telemetry(
        client_id, slot, sequence, frame,
        echo_tick=frame, accepted=1,
        forward=action.move_forward, right=action.move_right,
        look_yaw=action.look_yaw, look_pitch=action.look_pitch,
        jump=action.jump, fire=action.fire,
        hook=action.hook, weapon=action.weapon,
        action_generation=dispatch_frame,
        **kwargs,
    )


def _assert_rounds_equal(serial_rounds, pipelined_rounds):
    assert len(serial_rounds) == len(pipelined_rounds)
    for index, (expected, actual) in enumerate(
        zip(serial_rounds, pipelined_rounds)
    ):
        assert actual.round_id == expected.round_id, f"round {index}"
        assert actual.policy_version == expected.policy_version
        assert actual.tags == expected.tags
        np.testing.assert_array_equal(
            actual.observations, expected.observations,
            err_msg=f"round {index} observations",
        )
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.terminated, expected.terminated)
        np.testing.assert_array_equal(actual.truncated, expected.truncated)
        assert actual.infos == expected.infos, f"round {index} infos"


def _run_serial(envs, actions_per_round, policy_version=7):
    batch = Q2NetworkClientBatch(envs, round_timeout=1.0)
    rounds = []
    try:
        batch.reset()
        for actions in actions_per_round:
            rounds.append(
                batch.collect_round(actions, policy_version=policy_version)
            )
    finally:
        metrics = batch.metrics
        batch.close()
    return rounds, metrics


def _run_pipelined(envs, actions_per_round, policy_version=7):
    batch = Q2NetworkClientBatch(envs, round_timeout=1.0)
    rounds = []
    script = iter(actions_per_round)
    try:
        observations, _infos = batch.reset()
        batch.collect_rounds_pipelined(
            observations,
            rounds=len(actions_per_round),
            infer=lambda _vectors, _round_id: next(script),
            policy_version=policy_version,
            on_round=rounds.append,
        )
    finally:
        metrics = batch.metrics
        batch.close()
    return rounds, metrics


def test_happy_path_rounds_are_identical_to_serial():
    actions_per_round = [
        [Action(move_forward=0.1 * (k + 1), look_yaw=0.05 * k),
         Action(move_forward=-0.2, jump=bool(k % 2))]
        for k in range(5)
    ]

    def make_envs():
        envs = []
        for client_id, slot in (("client-a", 0), ("client-b", 1)):
            script = []
            for k in range(5):
                # Round k's dispatch happens at the previous echo's frame
                # (round 0 dispatches at the start packet's frame 10).
                dispatch_frame = 10 if k == 0 else 10 + (k - 1)
                script.append(_matched(
                    client_id, slot, 2 + k, 10 + k,
                    actions_per_round[k][slot], dispatch_frame,
                    damage_dealt=float(k + slot),
                ))
            envs.append(_ScriptedEnv(client_id, slot, script))
        return envs

    serial_rounds, serial_metrics = _run_serial(make_envs(), actions_per_round)
    pipelined_rounds, pipelined_metrics = _run_pipelined(
        make_envs(), actions_per_round
    )
    _assert_rounds_equal(serial_rounds, pipelined_rounds)
    assert [r.round_id for r in pipelined_rounds] == [0, 1, 2, 3, 4]
    assert serial_metrics == pipelined_metrics
    assert pipelined_metrics.rounds_accepted == 5
    assert pipelined_metrics.transitions_accepted == 10


def test_preflight_catchup_flushes_pending_before_boundary_round():
    actions_per_round = [[Action(), Action()] for _ in range(4)]

    def make_envs():
        # Round 2 (index 2) begins with an already-advanced drain: the
        # collector must emit a nontrainable catchup boundary, and the
        # pipelined driver must flush round 1 first.
        envs = []
        for client_id, slot in (("client-a", 0), ("client-b", 1)):
            script = [
                _matched(client_id, slot, 2, 10, actions_per_round[0][slot],
                         10, damage_dealt=1.0),
                _matched(client_id, slot, 3, 11, actions_per_round[1][slot],
                         10, damage_dealt=2.0),
                _matched(client_id, slot, 5, 13, actions_per_round[3][slot],
                         12, damage_dealt=3.0),
            ]
            preflight = [_telemetry(client_id, slot, 4, 12)]
            envs.append(_ScriptedEnv(client_id, slot, script,
                                     preflight=preflight, preflight_at=2))
        return envs

    serial_rounds, serial_metrics = _run_serial(make_envs(), actions_per_round)
    pipelined_rounds, pipelined_metrics = _run_pipelined(
        make_envs(), actions_per_round
    )
    _assert_rounds_equal(serial_rounds, pipelined_rounds)
    assert serial_metrics == pipelined_metrics
    trainable = [
        bool(r.infos[0].get("trainable_transition")) for r in pipelined_rounds
    ]
    assert trainable == [True, True, False, True]
    assert pipelined_rounds[2].infos[0]["realtime_catchup_resync"]


def test_action_state_resync_round_matches_serial():
    actions = [Action(move_forward=0.5, look_yaw=1.0), Action()]
    resync_a = _telemetry(
        "client-a", 0, 2, 10, echo_tick=10, accepted=1,
        forward=0.5, look_yaw=3.0,  # engine-owned look diverged
        action_generation=10,
    )
    matched_b = _matched("client-b", 1, 2, 10, actions[1], 10)

    def make_envs():
        return [
            _ScriptedEnv("client-a", 0, [resync_a]),
            _ScriptedEnv("client-b", 1, [matched_b]),
        ]

    serial_rounds, serial_metrics = _run_serial(make_envs(), [actions])
    pipelined_rounds, pipelined_metrics = _run_pipelined(make_envs(), [actions])
    _assert_rounds_equal(serial_rounds, pipelined_rounds)
    assert serial_metrics == pipelined_metrics
    assert pipelined_metrics.action_state_resyncs == 1
    assert not any(
        info["trainable_transition"] for info in pipelined_rounds[0].infos
    )


def test_pipelined_driver_enforces_policy_monotonicity():
    def make_envs():
        return [
            _ScriptedEnv("client-a", 0, [
                _matched("client-a", 0, 2, 10, Action(), 10),
            ]),
        ]

    serial_batch = Q2NetworkClientBatch(make_envs(), round_timeout=1.0)
    try:
        serial_batch.reset()
        serial_batch.collect_round([Action()], policy_version=5)
        with pytest.raises(StalePolicyVersionError):
            serial_batch.collect_round([Action()], policy_version=4)
        assert serial_batch.metrics.stale_policy_rounds_rejected == 1
    finally:
        serial_batch.close()

    pipelined_batch = Q2NetworkClientBatch(make_envs(), round_timeout=1.0)
    try:
        pipelined_batch.reset()
        pipelined_batch.collect_round([Action()], policy_version=5)
        with pytest.raises(StalePolicyVersionError):
            pipelined_batch.collect_rounds_pipelined(
                np.zeros((1, 3), dtype=np.float32),
                rounds=1,
                infer=lambda _vectors, _round_id: [Action()],
                policy_version=4,
            )
        assert pipelined_batch.metrics.stale_policy_rounds_rejected == 1
    finally:
        pipelined_batch.close()


def test_pipelined_driver_validates_action_count():
    envs = [_ScriptedEnv("client-a", 0, [])]
    batch = Q2NetworkClientBatch(envs, round_timeout=1.0)
    try:
        batch.reset()
        with pytest.raises(ValueError, match="expected 1 actions"):
            batch.collect_rounds_pipelined(
                np.zeros((1, 3), dtype=np.float32),
                rounds=1,
                infer=lambda _vectors, _round_id: [Action(), Action()],
                policy_version=0,
            )
    finally:
        batch.close()
