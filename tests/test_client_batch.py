from types import SimpleNamespace

import numpy as np
import pytest

from harness.client_batch import (
    AuthoritativeEchoError,
    Q2NetworkClientBatch,
    Q2NetworkClientMultiEnv,
    StalePolicyVersionError,
)
from harness.client_env import ClientActionDispatch, ClientTelemetryDrain
from harness.client_env import Q2NetworkClientEnv
from harness.client_protocol import ClientTelemetry
from harness.protocol import Action
from harness.protocol import ML_FIRE_GATE_SUPPRESSED
from harness.protocol import (
    ML_ACTION_GENERATION_SHIFT,
)


class _Spatial:
    def __init__(self):
        self.finalized = 0

    def finalize_episode(self, *, terminal_reason, truncated):
        self.finalized += 1
        return 0.25, {"outcome_bonus": 0.25}


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
    terminal_reason=None,
    damage_dealt=0.0,
    map_name="q2dm1",
    gate_flags=0,
    pitch=0.0,
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
    debug[11] = int(gate_flags)
    obs = SimpleNamespace(
        action_debug=debug,
        is_terminal=terminal,
        terminal_reason=(
            int(terminal_reason)
            if terminal_reason is not None
            else (1 if terminal else 0)
        ),
        reward_damage_dealt=damage_dealt,
        pitch=pitch,
    )
    return ClientTelemetry(
        sequence=sequence,
        client_slot=slot,
        server_frame=frame,
        client_id=client_id,
        map_name=map_name,
        observation=obs,
    )


class _FakeEnv:
    def __init__(self, client_id, slot, script, events, *, preflight=()):
        self.client_id = client_id
        self.slot = slot
        self.script = list(script)
        self.events = events
        self._spatial = _Spatial()
        self._last = None
        self.last_action = None
        self.transition_sample = None
        self.initial_maps = []
        self.preflight = list(preflight)

    def start(self):
        self._last = _telemetry(self.client_id, self.slot, 1, 10)
        return self._last

    def initial_result(self, current, *, vector=False):
        self.initial_maps.append(current.map_name)
        value = np.array([current.server_frame, self.slot], dtype=np.float32)
        return value, {"map": current.map_name, "client_id": self.client_id}

    def dispatch_action(self, action):
        self.events.append(("dispatch", self.client_id))
        self.last_action = action
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
        if not self.preflight:
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
        self.events.append(("receive", self.client_id))
        if not self.script:
            raise TimeoutError("script exhausted")
        current = self.script.pop(0)
        assert current.sequence > after_sequence
        self._last = current
        return current

    def transition_result(self, current, *, vector=False):
        self.transition_sample = current
        value = np.array([current.server_frame, self.slot], dtype=np.float32)
        info = {
            "map": current.map_name,
            "client_id": self.client_id,
            "spatial_bonus": 0.0,
        }
        return value, 1.5, current.observation.is_terminal, False, info

    def reset_episode_vector(self):
        return np.array([self._last.server_frame, self.slot], dtype=np.float32)

    def close(self):
        self.events.append(("close", self.client_id))


def test_dispatches_whole_round_and_rejects_stale_and_mismatched_echoes():
    events = []
    env_a = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 10, echo_tick=10, accepted=1,
                terminal=True, damage_dealt=1.0,
            ),
            _telemetry(
                "client-a", 0, 3, 11, echo_tick=11, accepted=1,
                damage_dealt=2.0,
            ),
            _telemetry(
                "client-a", 0, 4, 12, echo_tick=12, accepted=1,
                forward=0.5, right=0.2, fire=True, damage_dealt=3.0,
            ),
        ],
        events,
    )
    env_b = _FakeEnv(
        "client-b",
        1,
        [
            _telemetry(
                "client-b", 1, 2, 11, echo_tick=11, accepted=1,
                forward=-0.4, right=-0.1, jump=True,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env_a, env_b], round_timeout=1.0)
    try:
        initial, _infos = batch.reset()
        assert initial.shape == (2, 2)
        result = batch.collect_round(
            [
                Action(move_forward=0.5, move_right=0.2, fire=True),
                Action(move_forward=-0.4, move_right=-0.1, jump=True),
            ],
            policy_version=42,
        )
        first_receive = next(i for i, event in enumerate(events) if event[0] == "receive")
        assert [event[0] for event in events[:first_receive]] == ["dispatch", "dispatch"]
        assert result.policy_version == 42
        assert [tag.action_tick for tag in result.tags] == [10, 10]
        assert [info["authoritative_echo_tick"] for info in result.infos] == [12, 11]
        assert all(info["trainable_transition"] for info in result.infos)
        assert result.infos[0]["stale_echoes_rejected"] == 1
        assert result.infos[0]["mismatched_echoes_rejected"] == 1
        assert bool(result.terminated[0])
        assert env_a.transition_sample.observation.reward_damage_dealt == 6.0
        assert env_a.transition_sample.observation.terminal_reason == 1
        metrics = batch.metrics
        assert metrics.transitions_accepted == 2
        assert metrics.stale_echoes_rejected == 1
        assert metrics.mismatched_echoes_rejected == 1
        assert metrics.as_dict()["network_client/authoritative_echo_accept_rate"] == 0.5

        with pytest.raises(StalePolicyVersionError):
            batch.collect_round(
                [Action(), Action()], policy_version=41
            )
        assert batch.metrics.stale_policy_rounds_rejected == 1
        assert batch.metrics.actions_dispatched == 2
    finally:
        batch.close()


def test_failed_echo_round_never_returns_a_trainable_transition():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [_telemetry("client-a", 0, 2, 11, echo_tick=11, accepted=1)],
        events,
    )
    batch = Q2NetworkClientBatch(
        [env], round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(move_forward=0.9)], policy_version=7
            )
        metrics = batch.metrics
        assert metrics.failed_rounds == 1
        assert metrics.mismatched_echoes_rejected == 1
        assert metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_look_echo_must_match_the_dispatched_policy_action():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=11, accepted=1,
                look_yaw=0.0, look_pitch=0.0,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch(
        [env], round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(look_yaw=8.0, look_pitch=-4.0)],
                policy_version=8,
            )
        assert batch.metrics.mismatched_echoes_rejected == 1
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_same_tick_echo_requires_matching_action_generation():
    events = []
    generation_flags = ((10 % 192) + 1) << ML_ACTION_GENERATION_SHIFT
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=10, accepted=1,
                look_yaw=5.5, look_pitch=-3.25,
                gate_flags=generation_flags,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        result = batch.collect_round(
            [Action(look_yaw=5.5, look_pitch=-3.25)], policy_version=8
        )
        assert result.infos[0]["authoritative_echo_tick"] == 10
        assert result.infos[0]["authoritative_echo_valid"] is True
    finally:
        batch.close()


def test_same_tick_echo_rejects_stale_action_generation():
    events = []
    stale_generation_flags = ((9 % 192) + 1) << ML_ACTION_GENERATION_SHIFT
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=10, accepted=1,
                look_yaw=5.5, look_pitch=-3.25,
                gate_flags=stale_generation_flags,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch(
        [env], round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(look_yaw=5.5, look_pitch=-3.25)], policy_version=8
            )
    finally:
        batch.close()


def test_same_tick_engine_pitch_clamp_returns_nontrainable_resync():
    events = []
    generation_flags = ((10 % 192) + 1) << ML_ACTION_GENERATION_SHIFT
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=10, accepted=1,
                look_yaw=-16.48, look_pitch=-15.59, pitch=-89.0,
                forward=-0.03, right=0.71, jump=True, hook=1, weapon=7,
                gate_flags=generation_flags,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        result = batch.collect_round(
            [Action(
                move_forward=-0.03, move_right=0.71,
                look_yaw=-16.48, look_pitch=-26.85,
                jump=True, hook=1, weapon=7,
            )],
            policy_version=8,
        )
        assert result.infos[0]["look_clamp_resync"] is True
        assert result.infos[0]["trainable_transition"] is False
        assert result.infos[0]["action_dispatched"] is True
        assert batch.metrics.look_clamp_resyncs == 1
        assert batch.metrics.failed_rounds == 0
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_wrapped_yaw_and_reliable_command_echoes_are_admitted():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=11, accepted=1,
                look_yaw=2.0, hook=2, weapon=7,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        result = batch.collect_round(
            [Action(look_yaw=2.0, hook=2, weapon=7)],
            policy_version=9,
        )
        assert result.infos[0]["authoritative_echo_valid"]
    finally:
        batch.close()


def test_mismatched_reliable_command_echo_is_rejected():
    events = []
    env = _FakeEnv(
        "client-a", 0,
        [_telemetry(
            "client-a", 0, 2, 11, echo_tick=11, accepted=1,
            hook=0, weapon=0,
        )],
        events,
    )
    batch = Q2NetworkClientBatch(
        [env], round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(hook=1, weapon=7)], policy_version=10
            )
    finally:
        batch.close()


def test_server_suppressed_fire_is_an_admitted_effective_action():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a",
                0,
                2,
                11,
                echo_tick=11,
                accepted=1,
                fire=False,
                gate_flags=ML_FIRE_GATE_SUPPRESSED,
            )
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        result = batch.collect_round(
            [Action(fire=True)], policy_version=9
        )
        assert result.infos[0]["fire_gate_suppressed"] is True
        assert result.infos[0]["effective_action_fire"] is False
        assert result.infos[0]["trainable_transition"] is True
        assert batch.metrics.fire_gate_suppressions == 1
        assert (
            batch.metrics.as_dict()["network_client/fire_gate_suppressions"]
            == 1
        )
    finally:
        batch.close()


def test_preflight_drains_large_backlog_without_dispatching_stale_action():
    events = []
    backlog_a = [
        _telemetry("client-a", 0, sequence, frame)
        for sequence, frame in zip(range(2, 22), range(11, 31))
    ]
    backlog_b = [
        _telemetry("client-b", 1, sequence, frame)
        for sequence, frame in zip(range(2, 19), range(11, 28))
    ]
    env_a = _FakeEnv(
        "client-a",
        0,
        [_telemetry(
            "client-a", 0, 22, 31, echo_tick=31, accepted=1, forward=0.3
        )],
        events,
        preflight=backlog_a,
    )
    env_b = _FakeEnv(
        "client-b",
        1,
        [_telemetry(
            "client-b", 1, 19, 28, echo_tick=28, accepted=1, forward=0.3
        )],
        events,
        preflight=backlog_b,
    )
    batch = Q2NetworkClientBatch([env_a, env_b], round_timeout=1.0)
    try:
        batch.reset()
        boundary = batch.collect_round(
            [Action(move_forward=0.3), Action(move_forward=0.3)],
            policy_version=60,
        )
        assert not any(event[0] == "dispatch" for event in events)
        assert boundary.rewards.tolist() == [0.0, 0.0]
        assert boundary.terminated.tolist() == [True, True]
        assert all(not info["trainable_transition"] for info in boundary.infos)
        assert all(info["realtime_catchup_resync"] for info in boundary.infos)
        assert all(not info["map_epoch_resync"] for info in boundary.infos)
        assert boundary.observations[:, 0].tolist() == [30.0, 27.0]
        metrics = batch.metrics
        assert metrics.realtime_catchup_resyncs == 1
        assert metrics.preflight_packets_drained == 37
        assert metrics.actions_dispatched == 0

        accepted = batch.collect_round(
            [Action(move_forward=0.3), Action(move_forward=0.3)],
            policy_version=60,
        )
        assert [tag.action_tick for tag in accepted.tags] == [30, 27]
        assert all(info["trainable_transition"] for info in accepted.infos)
        assert batch.metrics.actions_dispatched == 2
    finally:
        batch.close()


def test_preflight_map_change_resyncs_before_any_action_dispatch():
    events = []
    env_a = _FakeEnv(
        "client-a",
        0,
        [],
        events,
        preflight=[_telemetry("client-a", 0, 2, 1, map_name="q2dm2")],
    )
    env_b = _FakeEnv(
        "client-b",
        1,
        [_telemetry("client-b", 1, 2, 1, map_name="q2dm2")],
        events,
    )
    batch = Q2NetworkClientBatch([env_a, env_b], round_timeout=1.0)
    try:
        batch.reset()
        boundary = batch.collect_round(
            [Action(), Action()], policy_version=61
        )
        assert not any(event[0] == "dispatch" for event in events)
        assert all(info["map_epoch_resync"] for info in boundary.infos)
        assert all(info["map_epoch_target"] == "q2dm2" for info in boundary.infos)
        assert all(not info["trainable_transition"] for info in boundary.infos)
        assert batch.metrics.map_epoch_resyncs == 1
        assert batch.metrics.preflight_packets_drained == 1
    finally:
        batch.close()


def test_map_epoch_resyncs_every_client_and_returns_nontrainable_boundary():
    events = []
    env_a = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 1, map_name="q2dm2",
                damage_dealt=99.0, terminal=True,
            )
        ],
        events,
    )
    env_b = _FakeEnv(
        "client-b",
        1,
        [
            _telemetry(
                "client-b", 1, 2, 11, echo_tick=11, accepted=1,
                forward=0.2,
            ),
            _telemetry(
                "client-b", 1, 3, 1, map_name="q2dm2",
                damage_dealt=88.0,
            ),
        ],
        events,
    )
    batch = Q2NetworkClientBatch([env_a, env_b], round_timeout=1.0)
    try:
        batch.reset()
        waiting = batch.collect_round(
            [Action(move_forward=0.2), Action(move_forward=0.2)],
            policy_version=50,
        )
        assert all(info["map_epoch_pending"] for info in waiting.infos)
        assert waiting.observations[:, 0].tolist() == [1.0, 11.0]

        result = batch.collect_round(
            [Action(move_forward=0.2), Action(move_forward=0.2)],
            policy_version=50,
        )
        assert np.array_equal(result.rewards, np.zeros(2, dtype=np.float32))
        assert result.terminated.tolist() == [True, True]
        assert result.truncated.tolist() == [False, False]
        assert result.observations[:, 0].tolist() == [1.0, 1.0]
        assert all(info["map_epoch_resync"] for info in result.infos)
        assert all(not info["map_epoch_pending"] for info in result.infos)
        assert all(not info["trainable_transition"] for info in result.infos)
        assert all(not info["authoritative_echo_valid"] for info in result.infos)
        assert {info["map_epoch_target"] for info in result.infos} == {"q2dm2"}
        assert env_a.initial_maps == ["q2dm1", "q2dm2", "q2dm2"]
        assert env_b.initial_maps == ["q2dm1", "q2dm1", "q2dm2"]
        metrics = batch.metrics
        assert metrics.map_epoch_resyncs == 1
        assert metrics.rounds_accepted == 0
        assert metrics.transitions_accepted == 0
        assert metrics.as_dict()["network_client/map_epoch_resyncs"] == 1
    finally:
        batch.close()


def test_intermission_before_echo_is_nontrainable_until_new_map_is_active():
    events = []
    envs = []
    for slot in range(4):
        client_id = f"client-{slot}"
        envs.append(_FakeEnv(
            client_id,
            slot,
            [
                # q2ded freezes usercmd application during intermission. The
                # echo therefore cannot match while telemetry keeps flowing.
                _telemetry(
                    client_id,
                    slot,
                    2,
                    11,
                    echo_tick=10,
                    accepted=1,
                    terminal=True,
                    terminal_reason=2,
                ),
                _telemetry(
                    client_id, slot, 3, 1, map_name="mllive_test"
                ),
                _telemetry(
                    client_id,
                    slot,
                    4,
                    2,
                    echo_tick=2,
                    accepted=1,
                    forward=0.3,
                    map_name="mllive_test",
                ),
            ],
            events,
        ))
    batch = Q2NetworkClientBatch(
        envs, round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()

        waiting = batch.collect_round(
            [Action(move_forward=0.3)] * 4, policy_version=70
        )
        assert waiting.terminated.tolist() == [True] * 4
        assert waiting.rewards.tolist() == [0.0] * 4
        assert all(info["map_epoch_resync"] for info in waiting.infos)
        assert all(info["map_epoch_pending"] for info in waiting.infos)
        assert all(
            info["map_epoch_target"] == "q2dm1" for info in waiting.infos
        )
        assert all(not info["trainable_transition"] for info in waiting.infos)

        changed = batch.collect_round(
            [Action(move_forward=0.3)] * 4, policy_version=71
        )
        assert all(info["map_epoch_resync"] for info in changed.infos)
        assert all(not info["map_epoch_pending"] for info in changed.infos)
        assert all(
            info["map_epoch_target"] == "mllive_test"
            for info in changed.infos
        )
        assert all(not info["trainable_transition"] for info in changed.infos)

        accepted = batch.collect_round(
            [Action(move_forward=0.3)] * 4, policy_version=72
        )
        assert all(info["trainable_transition"] for info in accepted.infos)
        assert accepted.observations[:, 0].tolist() == [2.0] * 4
        assert batch.metrics.failed_rounds == 0
        assert batch.metrics.transitions_accepted == 4
        assert batch.metrics.map_epoch_resyncs == 1

        with pytest.raises(StalePolicyVersionError):
            batch.collect_round([Action()] * 4, policy_version=71)
    finally:
        batch.close()


def test_map_download_pause_and_staggered_clients_never_dispatch_actions():
    events = []
    envs = []
    for slot in range(4):
        client_id = f"client-{slot}"
        envs.append(_FakeEnv(
            client_id,
            slot,
            [
                _telemetry(
                    client_id,
                    slot,
                    2,
                    11,
                    echo_tick=10,
                    accepted=1,
                    terminal=True,
                    terminal_reason=2,
                ),
            ],
            events,
        ))
    batch = Q2NetworkClientBatch(
        envs, round_timeout=1.0, max_rejected_echoes=0
    )
    try:
        batch.reset()

        initial_boundary = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=80
        )
        assert all(info["map_epoch_pending"] for info in initial_boundary.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        download_pause = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=81
        )
        assert all(info["map_epoch_pending"] for info in download_pause.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        for env in envs[:2]:
            env.script.append(_telemetry(
                env.client_id, env.slot, 3, 1, map_name="mllive_staggered"
            ))
        partial = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=82
        )
        assert all(info["map_epoch_pending"] for info in partial.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        for env in envs[2:]:
            env.script.append(_telemetry(
                env.client_id, env.slot, 3, 1, map_name="mllive_staggered"
            ))
        ready = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=83
        )
        assert all(not info["map_epoch_pending"] for info in ready.infos)
        assert all(not info["trainable_transition"] for info in ready.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        for env in envs:
            env.script.append(_telemetry(
                env.client_id,
                env.slot,
                4,
                2,
                echo_tick=2,
                accepted=1,
                forward=0.4,
                map_name="mllive_staggered",
            ))
        accepted = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=84
        )
        assert all(info["trainable_transition"] for info in accepted.infos)
        assert sum(event[0] == "dispatch" for event in events) == 8
        assert batch.metrics.failed_rounds == 0
        assert batch.metrics.echo_timeouts == 0
        assert batch.metrics.map_epoch_resyncs == 1
    finally:
        batch.close()


def test_multi_env_adapter_matches_trainer_surface_and_tags_policy():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=11, accepted=1,
                forward=1.0, right=-1.0, jump=True, fire=True,
            )
        ],
        events,
    )
    adapter = Q2NetworkClientMultiEnv(
        [env], max_ep_steps=1, initial_policy_version=100
    )
    try:
        assert adapter.n_ml == 1
        assert len(adapter._spatial_rewards) == 1
        assert len(adapter.reset_all()) == 1
        results = adapter.step_all(
            [np.array([2, -2, 0, 0, 1, 1, 0, 0], dtype=np.float32)],
            policy_version=101,
        )
        observation, reward, terminated, truncated, info = results[0]
        assert observation.shape == (2,)
        assert reward == pytest.approx(1.75)
        assert not terminated
        assert truncated
        assert info["policy_version"] == 101
        assert info["action_tick"] == 10
        assert info["authoritative_echo_valid"] is True
        assert adapter.active_map == "q2dm1"
        assert adapter._spatial_rewards[0].finalized == 1
        assert adapter.reset_slot(0).shape == (2,)
    finally:
        adapter.close()


def test_multi_env_does_not_finalize_spatial_outcome_for_map_resync():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [_telemetry("client-a", 0, 2, 1, map_name="q2dm2", terminal=True)],
        events,
    )
    adapter = Q2NetworkClientMultiEnv([env], initial_policy_version=5)
    try:
        adapter.reset_all()
        results = adapter.step_all(
            [np.zeros(8, dtype=np.float32)], policy_version=5
        )
        _observation, reward, terminated, truncated, info = results[0]
        assert reward == 0.0
        assert terminated
        assert not truncated
        assert info["map_epoch_resync"] is True
        assert info["trainable_transition"] is False
        assert adapter.active_map == "q2dm2"
        assert adapter._spatial_rewards[0].finalized == 0
    finally:
        adapter.close()


def test_multi_env_does_not_finalize_same_map_realtime_catchup():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [],
        events,
        preflight=[_telemetry("client-a", 0, 2, 11)],
    )
    adapter = Q2NetworkClientMultiEnv([env], initial_policy_version=5)
    try:
        adapter.reset_all()
        results = adapter.step_all(
            [np.zeros(8, dtype=np.float32)], policy_version=5
        )
        _observation, reward, terminated, truncated, info = results[0]
        assert reward == 0.0
        assert terminated
        assert not truncated
        assert info["realtime_catchup_resync"] is True
        assert info["map_epoch_resync"] is False
        assert info["trainable_transition"] is False
        assert adapter._spatial_rewards[0].finalized == 0
        assert not any(event[0] == "dispatch" for event in events)
    finally:
        adapter.close()


def test_client_launch_isolates_home_downloads_and_avoids_stdout_pipe(
    tmp_path, monkeypatch
):
    launches = []

    class _Process:
        returncode = None
        stdout = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def _popen(args, **kwargs):
        launches.append((args, kwargs))
        return _Process()

    monkeypatch.setattr("harness.client_env.subprocess.Popen", _popen)
    env = Q2NetworkClientEnv(
        server="127.0.0.1:28000",
        telemetry_server="127.0.0.1:28049",
        telemetry_token="secret",
        client_binary=str(tmp_path / "quake2"),
        client_root=str(tmp_path / "runtime"),
        client_data_root=str(tmp_path / "sandboxes"),
        harness_port=0,
        client_id="client-a",
    )
    monkeypatch.setattr(env, "_receive", lambda after_sequence: "started")
    try:
        assert env.start() == "started"
        args, kwargs = launches[0]
        assert args[1:3] == ["-datadir", str((tmp_path / "runtime").resolve())]
        assert kwargs["stdout"] is not __import__("subprocess").PIPE
        assert kwargs["env"]["HOME"] == str(
            (tmp_path / "sandboxes/client-a/home").resolve()
        )
        assert kwargs["env"]["XDG_DATA_HOME"] == str(
            (tmp_path / "sandboxes/client-a/data").resolve()
        )
        assert (tmp_path / "sandboxes/client-a/home").is_dir()
        assert (tmp_path / "sandboxes/client-a/data").is_dir()
    finally:
        env.close()


def test_client_id_cannot_escape_its_sandbox(tmp_path):
    with pytest.raises(ValueError, match="letters, digits"):
        Q2NetworkClientEnv(
            server="127.0.0.1:28000",
            telemetry_server="127.0.0.1:28049",
            telemetry_token="secret",
            client_binary=str(tmp_path / "quake2"),
            client_root=str(tmp_path),
            client_id="../escape",
        )
