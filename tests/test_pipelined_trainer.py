"""Integration test: PipelinedNetworkRollout vs the serial adapter path.

Drives the same scripted session through (a) the serial
Q2NetworkClientMultiEnv.step_all loop with ppo-style per-round processing
and (b) PipelinedNetworkRollout with stub inference, and asserts identical
inference inputs, buffered transitions, done/boundary handling, and
admission counters — without torch.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from harness.client_batch import Q2NetworkClientMultiEnv
from harness.client_env import ClientActionDispatch, ClientTelemetryDrain
from harness.client_protocol import ClientTelemetry
from harness.pipelined_trainer import PipelinedNetworkRollout
from harness.protocol import Action
from harness.protocol import (
    ML_ACTION_GENERATION_SHIFT,
)


class _SpatialStub:
    def __init__(self):
        self.finalized = []

    def finalize_episode(self, *, terminal_reason, truncated):
        self.finalized.append((terminal_reason, truncated))
        return 0.25, {"outcome_bonus": 0.25}


def _telemetry(client_id, slot, sequence, frame, *, echo_tick=0, accepted=0,
               forward=0.0, right=0.0, look_yaw=0.0, look_pitch=0.0,
               jump=False, fire=False, hook=0, weapon=0, terminal=False,
               map_name="q2dm1", gate_flags=0, action_generation=10):
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
    """Scripted env with reset sentinels distinguishable from step vectors."""

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
        # Distinct from any step vector so tests can tell which one the
        # policy saw after a done.
        return np.array([-7.0, self.slot, 99.0], dtype=np.float32)

    def close(self):
        pass


def _matched(client_id, slot, sequence, frame, action, dispatch_frame,
             *, terminal=False):
    return _telemetry(
        client_id, slot, sequence, frame,
        echo_tick=frame, accepted=1,
        forward=action.move_forward, right=action.move_right,
        look_yaw=action.look_yaw, look_pitch=action.look_pitch,
        jump=action.jump, fire=action.fire,
        hook=action.hook, weapon=action.weapon,
        action_generation=dispatch_frame, terminal=terminal,
    )


# Round plan (start frame 10):
#   k=0: dispatch@10  -> echo f10   (accepted)
#   k=1: dispatch@10  -> echo f11   (accepted)
#   k=2: dispatch@11  -> echo f12   (accepted, env0 DEATH terminal)
#   k=3: drain advanced -> realtime-catchup boundary (no dispatch)
#   k=4: dispatch@13  -> echo f14   (accepted)
#   k=5: dispatch@14  -> echo f15   (accepted; discarded in-flight tail)
_ACTIONS = [
    [Action(move_forward=0.1 * (k + 1), look_yaw=0.02 * k),
     Action(move_forward=-0.1, jump=bool(k % 2))]
    for k in range(6)
]


def _make_envs():
    env_a = _ScriptedEnv("client-a", 0, [
        _matched("client-a", 0, 2, 10, _ACTIONS[0][0], 10),
        _matched("client-a", 0, 3, 11, _ACTIONS[1][0], 10),
        _matched("client-a", 0, 4, 12, _ACTIONS[2][0], 11, terminal=True),
        _matched("client-a", 0, 6, 14, _ACTIONS[4][0], 13),
        _matched("client-a", 0, 7, 15, _ACTIONS[5][0], 14),
    ], preflight=[_telemetry("client-a", 0, 5, 13)], preflight_at=3)
    env_b = _ScriptedEnv("client-b", 1, [
        _matched("client-b", 1, 2, 10, _ACTIONS[0][1], 10),
        _matched("client-b", 1, 3, 11, _ACTIONS[1][1], 10),
        _matched("client-b", 1, 4, 12, _ACTIONS[2][1], 11),
        _matched("client-b", 1, 6, 14, _ACTIONS[4][1], 13),
        _matched("client-b", 1, 7, 15, _ACTIONS[5][1], 14),
    ], preflight=[_telemetry("client-b", 1, 5, 13)], preflight_at=3)
    return [env_a, env_b]


def _make_stub_ctx(actions_index):
    return {
        "current_obs": None,  # filled by the act log comparison instead
        "h_step": np.zeros((2, 4), dtype=np.float32),
        "c_step": np.zeros((2, 4), dtype=np.float32),
        "actions": np.array(
            [[a.move_forward, a.move_right, a.look_yaw, a.look_pitch,
              float(a.jump), float(a.fire), float(a.hook), float(a.weapon)]
             for a in _ACTIONS[actions_index]],
            dtype=np.float32,
        ),
        "values": np.zeros(2, dtype=np.float32),
        "log_probs": np.zeros(2, dtype=np.float32),
        "fire_allowed": np.ones(2, dtype=np.bool_),
        "fire_metadata": None,
    }


def _run_serial_reference():
    srv = Q2NetworkClientMultiEnv(_make_envs(), max_ep_steps=1000)
    infer_log, buffer, resets, boundaries = [], [], [], []
    try:
        vectors = srv.reset_all()
        obs = np.stack(vectors)
        while len(buffer) < 4:
            infer_log.append(obs.copy())
            ctx = _make_stub_ctx(len(infer_log) - 1)
            results = srv.step_all(
                [ctx["actions"][k] for k in range(2)], policy_version=0
            )
            if not all(bool(info.get("trainable_transition", False))
                       for *_x, info in results):
                boundaries.append([dict(info) for *_x, info in results])
                for bi in range(2):
                    resets.append(("hidden", bi))
                # ppo.py sets obs from the boundary results before
                # recollecting the slot.
                obs = np.stack([o for o, _r, _t, _tr, _i in results])
                continue
            rewards = np.array([r for _o, r, _t, _tr, _i in results],
                               dtype=np.float32)
            dones = np.array([float(t or tr)
                              for _o, _r, t, tr, _i in results],
                             dtype=np.float32)
            buffer.append((ctx, rewards, dones,
                           [dict(info) for _o, _r, _t, _tr, info in results]))
            obs = np.stack([o for o, _r, _t, _tr, _i in results])
            for bi, (_o, _r, t, tr, _i) in enumerate(results):
                if t or tr:
                    resets.append(("hidden", bi))
                    obs[bi] = srv.reset_slot(bi)
        return {
            "infer": infer_log, "buffer": buffer, "resets": resets,
            "boundaries": boundaries, "ep_steps": list(srv._ep_steps),
            "metrics": srv.metrics,
            "finalized": [sp.finalized for sp in srv._spatial_rewards],
        }
    finally:
        srv.close()


def _run_pipelined_glue():
    srv = Q2NetworkClientMultiEnv(_make_envs(), max_ep_steps=1000)
    infer_log, buffer, resets, boundaries = [], [], [], []
    try:
        vectors = srv.reset_all()
        obs = np.stack(vectors)
        glue = PipelinedNetworkRollout(srv, n_steps=4, policy_version=0)

        def act(obs_in):
            ctx = _make_stub_ctx(len(infer_log))
            ctx["current_obs"] = np.asarray(obs_in, dtype=np.float32).copy()
            infer_log.append(ctx["current_obs"].copy())
            return ctx

        final = glue.collect(
            obs,
            act=act,
            buf_add=lambda ctx, rewards, dones: buffer.append(
                (ctx, rewards, dones)
            ),
            reset_hidden=lambda bi: resets.append(("hidden", bi)),
            on_boundary=lambda infos: boundaries.append(
                [dict(info) for info in infos]
            ),
        )
        return {
            "infer": infer_log, "buffer": buffer, "resets": resets,
            "boundaries": boundaries, "ep_steps": list(srv._ep_steps),
            "metrics": srv.metrics,
            "final_vectors": final,
            "finalized": [sp.finalized for sp in srv._spatial_rewards],
        }
    finally:
        srv.close()


def test_glue_matches_serial_reference():
    serial = _run_serial_reference()
    pipelined = _run_pipelined_glue()

    # The pipelined driver performs exactly one extra (discarded) inference
    # per rollout for the in-flight tail round; the shared prefix must be
    # identical, including the post-death reset vector.
    assert len(pipelined["infer"]) == len(serial["infer"]) + 1
    for index, (expected, actual) in enumerate(
        zip(serial["infer"], pipelined["infer"])
    ):
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"infer input {index}"
        )
    # The reset sentinel must appear right after the death round.
    assert any(
        np.array_equal(row[0], np.array([-7.0, 0.0, 99.0], dtype=np.float32))
        for row in pipelined["infer"]
    )

    # Same buffered transitions.
    assert len(serial["buffer"]) == len(pipelined["buffer"]) == 4
    for index, ((s_ctx, s_rewards, s_dones, s_infos),
                (p_ctx, p_rewards, p_dones)) in enumerate(
        zip(serial["buffer"], pipelined["buffer"])
    ):
        np.testing.assert_array_equal(p_rewards, s_rewards)
        np.testing.assert_array_equal(p_dones, s_dones)
        np.testing.assert_array_equal(p_ctx["actions"], s_ctx["actions"])
        assert p_dones.tolist() == ([0.0, 0.0] if index != 2 else [1.0, 0.0])
        # Rewards include the adapter's outcome bonus on the death round.
        if index == 2:
            assert p_rewards[0] == pytest.approx(1.5 + 0.25)
        else:
            assert p_rewards[0] == pytest.approx(1.5)

    # Same resets, boundary, and finalize calls. Per-slot step counters
    # differ by exactly one: the discarded in-flight tail round increments
    # them (and its episode bookkeeping) before should_stop fires. Those
    # side effects are absorbed by the synchronization boundary that opens
    # the next rollout, which resets episode state via initial_result.
    assert pipelined["resets"] == serial["resets"]
    assert len(pipelined["boundaries"]) == len(serial["boundaries"]) == 1
    assert pipelined["boundaries"][0][0]["realtime_catchup_resync"]
    assert (
        pipelined["ep_steps"][0] == serial["ep_steps"][0] + 1
        and pipelined["ep_steps"][1] == serial["ep_steps"][1] + 1
    )
    assert pipelined["finalized"] == serial["finalized"] == [[(1, False)], []]

    # Admission accounting counters intact.
    assert pipelined["metrics"].failed_rounds == 0
    assert pipelined["metrics"].echo_timeouts == 0
    assert pipelined["metrics"].telemetry_gap_resyncs == 0
    assert pipelined["metrics"].realtime_catchup_resyncs == 1
    assert serial["metrics"].failed_rounds == 0
    assert serial["metrics"].echo_timeouts == 0

    # Death round buffered with the adapter-adjusted done; final inference
    # vectors match serial's obs bookkeeping is out of scope here (the
    # driver returns them; the trainer assigns obs_np from the return).
    assert pipelined["final_vectors"] is not None


def test_glue_stops_exactly_at_n_steps():
    srv = Q2NetworkClientMultiEnv(_make_envs(), max_ep_steps=1000)
    buffer = []
    try:
        vectors = srv.reset_all()
        obs = np.stack(vectors)
        glue = PipelinedNetworkRollout(srv, n_steps=2, policy_version=0)
        call_index = [0]

        def act(obs_in):
            ctx = _make_stub_ctx(call_index[0])
            call_index[0] += 1
            return ctx

        glue.collect(
            obs,
            act=act,
            buf_add=lambda ctx, rewards, dones: buffer.append(
                (ctx, rewards, dones)
            ),
            reset_hidden=lambda bi: None,
            on_boundary=lambda infos: None,
        )
        assert glue.accepted_steps == 2
        assert len(buffer) == 2
    finally:
        srv.close()


def test_glue_handles_catchup_boundary_without_buffering():
    """A realtime-catchup boundary emits no buffered transitions and resets
    per-slot episode state."""
    env_a = _ScriptedEnv("client-a", 0, [
        _matched("client-a", 0, 2, 10, _ACTIONS[0][0], 10),
    ], preflight=[_telemetry("client-a", 0, 3, 11)], preflight_at=1)
    env_b = _ScriptedEnv("client-b", 1, [
        _matched("client-b", 1, 2, 10, _ACTIONS[0][1], 10),
    ])
    srv = Q2NetworkClientMultiEnv([env_a, env_b], max_ep_steps=1000)
    buffer, boundaries, resets = [], [], []
    try:
        vectors = srv.reset_all()
        obs = np.stack(vectors)
        glue = PipelinedNetworkRollout(srv, n_steps=1, policy_version=0)

        def act(obs_in):
            return _make_stub_ctx(0)

        glue.collect(
            obs,
            act=act,
            buf_add=lambda ctx, rewards, dones: buffer.append(
                (ctx, rewards, dones)
            ),
            reset_hidden=lambda bi: resets.append(bi),
            on_boundary=lambda infos: boundaries.append(infos),
        )
        assert glue.accepted_steps == 1
        assert len(buffer) == 1
        assert len(boundaries) == 1
        assert all(
            not bool(info.get("trainable_transition", False))
            for info in boundaries[0]
        )
        # Boundary resets hidden state for every client.
        assert sorted(resets) == [0, 1]
        assert srv._ep_steps == [0, 0]
        assert srv.metrics.realtime_catchup_resyncs == 1
        assert srv.metrics.failed_rounds == 0
        assert srv.metrics.echo_timeouts == 0
    finally:
        srv.close()
