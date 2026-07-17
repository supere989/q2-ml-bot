from types import SimpleNamespace

import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_batch import (
    AuthoritativeEchoError,
    Q2NetworkClientBatch,
    Q2NetworkClientMultiEnv,
    StalePolicyVersionError,
    build_network_client_batch,
)
from harness.client_env import ClientActionDispatch, ClientTelemetryDrain
from harness.client_env import Q2NetworkClientEnv
from harness.client_protocol import ClientTelemetry
from harness.protocol import Action, ActionDebugIndex, VerticalIntent
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
    vertical_intent=VerticalIntent.NEUTRAL,
    fire=False,
    hook=0,
    weapon=0,
    terminal=False,
    terminal_reason=None,
    damage_dealt=0.0,
    map_name="q2dm1",
    gate_flags=0,
    pitch=0.0,
    action_generation=10,
    causal_facts_complete=True,
    causal_transition_trainable=True,
    causal_role_playing=True,
    causal_role_public_pm_normal=True,
    applied_action_tick=None,
    life_epoch=1,
):
    debug = np.zeros(15, dtype=np.float32)
    debug[0] = echo_tick
    debug[1] = accepted
    debug[4] = forward
    debug[5] = right
    debug[6] = look_yaw
    debug[7] = look_pitch
    debug[ActionDebugIndex.VERTICAL_INTENT] = int(vertical_intent)
    debug[ActionDebugIndex.APPLIED_UPMOVE] = {
        VerticalIntent.CROUCH_OR_DOWN: -320,
        VerticalIntent.NEUTRAL: 0,
        VerticalIntent.JUMP_OR_UP: 320,
    }[VerticalIntent(vertical_intent)]
    debug[ActionDebugIndex.FIRE] = int(fire)
    debug[ActionDebugIndex.HOOK] = int(hook)
    debug[3] = int(weapon)
    if accepted and not (int(gate_flags) & 0x00FF0000):
        gate_flags = int(gate_flags) | (
            ((int(action_generation) % 192) + 1)
            << ML_ACTION_GENERATION_SHIFT
        )
    debug[ActionDebugIndex.FLAGS] = int(gate_flags)
    echoed_generation = (
        int(gate_flags) >> ML_ACTION_GENERATION_SHIFT
    ) & 0xFF
    causal_echo = bool(accepted and echo_tick > 0 and echoed_generation > 0)
    causal_flags = (
        CausalFlags.FACTS_COMPLETE if causal_facts_complete
        else CausalFlags(0)
    )
    if causal_echo:
        causal_flags |= CausalFlags.ECHO_VALID
        if causal_facts_complete and causal_transition_trainable:
            causal_flags |= CausalFlags.TRANSITION_TRAINABLE
    if causal_role_playing:
        causal_flags |= CausalFlags.ROLE_PLAYING
    if causal_role_public_pm_normal:
        causal_flags |= CausalFlags.ROLE_PUBLIC_PM_NORMAL
    causal = CausalTelemetry(
        tick=frame,
        client_life_epoch=life_epoch,
        target_id=0,
        target_epoch=0,
        environmental_source_id=0,
        environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0,
        crouch_edge_id=0,
        crouch_edge_epoch=0,
        echo_tick=echo_tick if causal_echo else 0,
        action_generation=echoed_generation if causal_echo else 0,
        hook_zone_id=0,
        hook_attempt_tick=0,
        hook_action_generation=0,
        flags=causal_flags,
    )
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
        causal=causal,
        applied_action_tick=(
            max(0, frame - 1)
            if applied_action_tick is None else int(applied_action_tick)
        ),
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


@pytest.mark.parametrize(
    ("frame", "echo_tick", "applied_tick", "expected"),
    [
        (11, 10, 10, "matched"),
        (10, 10, 10, "mismatch"),
        (9, 9, 9, "mismatch"),
        (12, 11, 11, "mismatch"),
    ],
)
def test_deterministic_barrier_requires_exact_plus_one_echo(
    frame, echo_tick, applied_tick, expected
):
    env = _FakeEnv("trainer-00", 0, [], [])
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True
    )
    dispatch = ClientActionDispatch(
        client_id="trainer-00",
        client_slot=0,
        after_sequence=1,
        action_tick=10,
        map_name="q2dm1",
        action=Action(),
    )
    telemetry = _telemetry(
        "trainer-00", 0, 2, frame,
        echo_tick=echo_tick,
        accepted=1,
        action_generation=10,
        applied_action_tick=applied_tick,
    )
    try:
        assert batch._echo_state(telemetry, dispatch)[0] == expected
    finally:
        batch.close()


def test_reset_rejects_bootstrap_without_public_normal_role():
    env = _FakeEnv("trainer-00", 0, [], [])
    env.start = lambda: _telemetry(
        "trainer-00", 0, 1, 10,
        causal_role_public_pm_normal=False,
    )
    batch = Q2NetworkClientBatch([env])
    try:
        with pytest.raises(
            AuthoritativeEchoError, match="bootstrap lacks.*PM_NORMAL"
        ):
            batch.reset()
        assert batch._started is False
    finally:
        batch.close()


def test_deterministic_death_lifecycle_discards_until_new_life_is_primed():
    events = []
    env_a = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                terminal=True, terminal_reason=1,
            ),
            # Death camera owns view angles, but the same tick/generation and
            # all movement/reliable commands still prove command causality.
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                look_yaw=0.0, action_generation=11,
            ),
            # PutClientInServer advances the explicit transport life epoch.
            # This first new-life packet re-primes state and is not trained.
            _telemetry(
                "trainer-00", 0, 4, 13, echo_tick=12, accepted=1,
                look_yaw=0.0, life_epoch=2, action_generation=12,
                causal_transition_trainable=False,
            ),
            # Stock PMF_TIME_TELEPORT can span a variable number of full
            # commands.  Each exact causal generation remains explicitly
            # nontrainable while engine-owned look is clamped.
            _telemetry(
                "trainer-00", 0, 5, 14, echo_tick=13, accepted=1,
                look_yaw=0.0, life_epoch=2, action_generation=13,
                causal_transition_trainable=False,
            ),
            _telemetry(
                "trainer-00", 0, 6, 15, echo_tick=14, accepted=1,
                look_yaw=0.0, life_epoch=2, action_generation=14,
                causal_transition_trainable=False,
            ),
            # First exact, actionable command only primes the new life.
            _telemetry(
                "trainer-00", 0, 7, 16, echo_tick=15, accepted=1,
                look_yaw=5.0, life_epoch=2, action_generation=15,
            ),
            _telemetry(
                "trainer-00", 0, 8, 17, echo_tick=16, accepted=1,
                look_yaw=5.0, life_epoch=2, action_generation=16,
            ),
        ],
        events,
    )
    env_b = _FakeEnv(
        "trainer-01",
        1,
        [
            _telemetry("trainer-01", 1, 2, 11, echo_tick=10, accepted=1),
            _telemetry(
                "trainer-01", 1, 3, 12, echo_tick=11, accepted=1,
                look_yaw=5.0, action_generation=11,
            ),
            _telemetry(
                "trainer-01", 1, 4, 13, echo_tick=12, accepted=1,
                look_yaw=5.0, action_generation=12,
            ),
            _telemetry(
                "trainer-01", 1, 5, 14, echo_tick=13, accepted=1,
                look_yaw=5.0, action_generation=13,
            ),
            _telemetry(
                "trainer-01", 1, 6, 15, echo_tick=14, accepted=1,
                look_yaw=5.0, action_generation=14,
            ),
            _telemetry(
                "trainer-01", 1, 7, 16, echo_tick=15, accepted=1,
                look_yaw=5.0, action_generation=15,
            ),
            _telemetry(
                "trainer-01", 1, 8, 17, echo_tick=16, accepted=1,
                look_yaw=5.0, action_generation=16,
            ),
        ],
        events,
    )
    env_a.deterministic_frame_barrier = True
    env_b.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env_a, env_b],
        deterministic_frame_barrier=True,
        round_timeout=1.0,
    )
    try:
        batch.reset()

        death = batch.collect_round([Action(), Action()], policy_version=1)
        assert death.terminated.tolist() == [True, True]
        assert all(not info["trainable_transition"] for info in death.infos)
        assert death.infos[0]["death_lifecycle_phase"] == "death_terminal"
        assert death.infos[1]["death_lifecycle_phase"] == "peer"
        assert death.rewards.tolist() == [0.0, 0.0]

        dead = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=2,
        )
        assert dead.terminated.tolist() == [True, True]
        assert all(not info["trainable_transition"] for info in dead.infos)
        assert all(info["death_lifecycle_resync"] for info in dead.infos)
        assert [
            info["authoritative_client_life_epoch"] for info in dead.infos
        ] == [1, 1]

        reborn = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=3,
        )
        assert all(not info["trainable_transition"] for info in reborn.infos)
        assert all(info["death_lifecycle_resync"] for info in reborn.infos)
        assert [
            info["authoritative_client_life_epoch"] for info in reborn.infos
        ] == [2, 1]
        assert reborn.infos[0]["death_lifecycle_phase"] == "new_life_rebase"

        settling_one = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=4,
        )
        settling_two = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=5,
        )
        for settling in (settling_one, settling_two):
            assert all(not info["trainable_transition"] for info in settling.infos)
            assert settling.infos[0]["death_lifecycle_phase"] == (
                "new_life_teleport_settling"
            )
            assert settling.infos[0]["causal_echo_valid"] is True
            assert settling.infos[0]["causal_facts_complete"] is True
            assert settling.infos[0]["causal_transition_trainable"] is False

        prime = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=6,
        )
        assert all(not info["trainable_transition"] for info in prime.infos)
        assert prime.infos[0]["death_lifecycle_phase"] == (
            "new_life_actionable_prime"
        )
        assert prime.infos[0]["causal_transition_trainable"] is True

        resumed = batch.collect_round(
            [Action(look_yaw=5.0), Action(look_yaw=5.0)],
            policy_version=7,
        )
        assert all(info["trainable_transition"] for info in resumed.infos)
        assert [
            info["authoritative_client_life_epoch"] for info in resumed.infos
        ] == [2, 1]
        assert batch.metrics.transitions_accepted == 2
        assert batch.metrics.death_lifecycle_resyncs == 6
        assert batch.metrics.failed_rounds == 0
    finally:
        batch.close()


@pytest.mark.parametrize("life_epoch", [0, 3])
def test_deterministic_death_lifecycle_rejects_invalid_epoch(life_epoch):
    events = []
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                terminal=True, terminal_reason=1,
            ),
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                life_epoch=life_epoch, action_generation=11,
            ),
        ],
        events,
    )
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        batch.collect_round([Action()], policy_version=1)
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round([Action()], policy_version=2)
        assert batch.metrics.failed_rounds == 1
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_deterministic_death_lifecycle_never_relaxes_movement_echo():
    events = []
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                terminal=True, terminal_reason=1,
            ),
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                forward=0.0, action_generation=11,
            ),
        ],
        events,
    )
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        batch.collect_round([Action()], policy_version=1)
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(move_forward=0.5)], policy_version=2
            )
        assert batch.metrics.failed_rounds == 1
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_deterministic_death_terminal_with_camera_reset_is_boundary():
    events = []
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                terminal=True, terminal_reason=1, look_yaw=0.0,
            ),
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                action_generation=11,
            ),
        ],
        events,
    )
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        death = batch.collect_round(
            [Action(look_yaw=5.0)], policy_version=1
        )
        assert death.terminated.tolist() == [True]
        assert death.infos[0]["death_lifecycle_resync"] is True
        assert death.infos[0]["trainable_transition"] is False
        assert batch.metrics.transitions_accepted == 0

        corpse = batch.collect_round([Action()], policy_version=2)
        assert corpse.infos[0]["death_lifecycle_resync"] is True
        assert batch.metrics.death_lifecycle_resyncs == 2
        assert batch.metrics.failed_rounds == 0
    finally:
        batch.close()


def test_overlapping_deaths_are_both_registered_before_batch_boundary():
    events = []
    envs = []
    for slot in range(2):
        client_id = f"trainer-0{slot}"
        env = _FakeEnv(
            client_id,
            slot,
            [
                _telemetry(
                    client_id, slot, 2, 11, echo_tick=10, accepted=1,
                    terminal=True, terminal_reason=1,
                    look_yaw=0.0,
                ),
                _telemetry(
                    client_id, slot, 3, 12, echo_tick=11, accepted=1,
                    action_generation=11,
                ),
                _telemetry(
                    client_id, slot, 4, 13, echo_tick=12, accepted=1,
                    action_generation=12, life_epoch=2,
                ),
                    _telemetry(
                        client_id, slot, 5, 14, echo_tick=13, accepted=1,
                        action_generation=13, life_epoch=2,
                    ),
                    _telemetry(
                        client_id, slot, 6, 15, echo_tick=14, accepted=1,
                        action_generation=14, life_epoch=2,
                    ),
            ],
            events,
        )
        env.deterministic_frame_barrier = True
        envs.append(env)
    batch = Q2NetworkClientBatch(
        envs, deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        # Slot 0 camera-resyncs while slot 1 has an independently exact death
        # echo. The whole batch is discarded, but both deaths must be tracked.
        first = batch.collect_round(
            [Action(look_yaw=5.0), Action()], policy_version=1
        )
        assert all(info["death_lifecycle_resync"] for info in first.infos)

        corpse = batch.collect_round([Action(), Action()], policy_version=2)
        assert all(info["death_lifecycle_resync"] for info in corpse.infos)

        reborn = batch.collect_round([Action(), Action()], policy_version=3)
        assert [
            info["authoritative_client_life_epoch"] for info in reborn.infos
        ] == [2, 2]
        assert all(info["death_lifecycle_resync"] for info in reborn.infos)

        prime = batch.collect_round([Action(), Action()], policy_version=4)
        assert all(not info["trainable_transition"] for info in prime.infos)
        assert all(
            info["death_lifecycle_phase"] == "new_life_actionable_prime"
            for info in prime.infos
        )

        resumed = batch.collect_round([Action(), Action()], policy_version=5)
        assert all(info["trainable_transition"] for info in resumed.infos)
        assert batch.metrics.transitions_accepted == 2
        assert batch.metrics.death_lifecycle_resyncs == 4
        assert batch.metrics.failed_rounds == 0
    finally:
        batch.close()


def test_realtime_stale_death_terminal_is_boundary_without_reward_migration():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=9, accepted=1,
                action_generation=9, terminal=True, terminal_reason=1,
                damage_dealt=7.0,
            ),
            _telemetry(
                "client-a", 0, 3, 12, echo_tick=11, accepted=1,
                action_generation=11,
            ),
            _telemetry(
                "client-a", 0, 4, 13, echo_tick=12, accepted=1,
                action_generation=12, life_epoch=2,
            ),
            _telemetry(
                "client-a", 0, 5, 14, echo_tick=13, accepted=1,
                action_generation=13, life_epoch=2,
            ),
            _telemetry(
                "client-a", 0, 6, 15, echo_tick=14, accepted=1,
                action_generation=14, life_epoch=2,
            ),
        ],
        events,
    )
    batch = Q2NetworkClientBatch(
        [env], round_timeout=1.0, max_rejected_echoes=4
    )
    try:
        batch.reset()
        death = batch.collect_round([Action()], policy_version=1)
        assert death.infos[0]["death_lifecycle_resync"] is True
        assert death.rewards.tolist() == [0.0]

        corpse = batch.collect_round([Action()], policy_version=2)
        assert corpse.rewards.tolist() == [0.0]
        reborn = batch.collect_round([Action()], policy_version=3)
        assert reborn.rewards.tolist() == [0.0]

        prime = batch.collect_round([Action()], policy_version=4)
        assert prime.rewards.tolist() == [0.0]
        assert prime.infos[0]["death_lifecycle_phase"] == (
            "new_life_actionable_prime"
        )

        resumed = batch.collect_round([Action()], policy_version=5)
        assert resumed.infos[0]["trainable_transition"] is True
        assert env.transition_sample.observation.reward_damage_dealt == 0.0
        assert batch.metrics.stale_echoes_rejected == 1
        assert batch.metrics.transitions_accepted == 1
        assert batch.metrics.failed_rounds == 0
    finally:
        batch.close()


def test_deterministic_same_life_teleport_hold_settles_then_primes_once():
    events = []
    held = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                look_yaw=0.0, look_pitch=0.0, action_generation=10,
                causal_transition_trainable=False, damage_dealt=7.0,
            ),
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                look_yaw=0.0, look_pitch=0.0, action_generation=11,
                causal_transition_trainable=False, damage_dealt=11.0,
            ),
            _telemetry(
                "trainer-00", 0, 4, 13, echo_tick=12, accepted=1,
                look_yaw=5.0, look_pitch=2.0, action_generation=12,
                damage_dealt=13.0,
            ),
            _telemetry(
                "trainer-00", 0, 5, 14, echo_tick=13, accepted=1,
                look_yaw=5.0, look_pitch=2.0, action_generation=13,
            ),
        ],
        events,
    )
    peer = _FakeEnv(
        "trainer-01",
        1,
        [
            _telemetry(
                "trainer-01", 1, sequence, frame,
                echo_tick=frame - 1, accepted=1,
                look_yaw=5.0, look_pitch=2.0,
                action_generation=frame - 1,
            )
            for sequence, frame in ((2, 11), (3, 12), (4, 13), (5, 14))
        ],
        events,
    )
    held.deterministic_frame_barrier = True
    peer.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [held, peer], deterministic_frame_barrier=True, round_timeout=1.0
    )
    action = Action(look_yaw=5.0, look_pitch=2.0)
    try:
        batch.reset()
        for version in (1, 2):
            settling = batch.collect_round(
                [action, action], policy_version=version
            )
            assert settling.rewards.tolist() == [0.0, 0.0]
            assert all(
                not info["trainable_transition"] for info in settling.infos
            )
            assert settling.infos[0]["action_state_phase"] == (
                "same_life_hold_settling"
            )
            assert settling.infos[1]["action_state_phase"] == "peer"
            assert settling.infos[0]["causal_echo_valid"] is True
            assert settling.infos[0]["causal_facts_complete"] is True
            assert settling.infos[0]["causal_transition_trainable"] is False
            debug = held._last.observation.action_debug
            assert int(debug[ActionDebugIndex.TICK]) == 9 + version
            assert float(debug[ActionDebugIndex.LOOK_YAW]) == 0.0
            assert float(debug[ActionDebugIndex.LOOK_PITCH]) == 0.0

        prime = batch.collect_round([action, action], policy_version=3)
        assert prime.rewards.tolist() == [0.0, 0.0]
        assert prime.infos[0]["action_state_phase"] == (
            "same_life_actionable_prime"
        )
        prime_debug = held._last.observation.action_debug
        assert float(prime_debug[ActionDebugIndex.LOOK_YAW]) == 5.0
        assert float(prime_debug[ActionDebugIndex.LOOK_PITCH]) == 2.0

        admitted = batch.collect_round([action, action], policy_version=4)
        assert all(info["trainable_transition"] for info in admitted.infos)
        assert held.transition_sample.observation.reward_damage_dealt == 0.0
        assert batch.metrics.transitions_accepted == 2
        assert batch.metrics.action_state_resyncs == 3
        assert batch.metrics.failed_rounds == 0
    finally:
        batch.close()


def test_non_public_normal_nontrainable_echo_cannot_arm_same_life_hold():
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11,
                echo_tick=10, accepted=1, action_generation=10,
                causal_transition_trainable=False,
                causal_role_playing=True,
                causal_role_public_pm_normal=False,
            ),
        ],
        [],
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        boundary = batch.collect_round([Action()], policy_version=1)
        assert boundary.rewards.tolist() == [0.0]
        assert boundary.infos[0]["action_state_resync"] is True
        assert boundary.infos[0]["action_state_phase"] == "action_state_resync"
        assert "trainer-00" not in batch._action_hold_epochs
        assert batch.metrics.action_state_resyncs == 1
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_death_settling_rejects_trainable_flag_on_engine_altered_action():
    events = []
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
                terminal=True, terminal_reason=1,
            ),
            _telemetry(
                "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
                action_generation=11,
            ),
            _telemetry(
                "trainer-00", 0, 4, 13, echo_tick=12, accepted=1,
                action_generation=12, life_epoch=2, look_pitch=0.0,
                causal_transition_trainable=False,
            ),
            # Tamper: the engine-altered pitch claims it is trainable.
            _telemetry(
                "trainer-00", 0, 5, 14, echo_tick=13, accepted=1,
                action_generation=13, life_epoch=2, look_pitch=0.0,
                causal_transition_trainable=True,
            ),
        ],
        events,
    )
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        batch.collect_round([Action()], policy_version=1)
        batch.collect_round([Action()], policy_version=2)
        batch.collect_round([Action(look_pitch=2.0)], policy_version=3)
        with pytest.raises(AuthoritativeEchoError, match="batch round") as caught:
            batch.collect_round([Action(look_pitch=2.0)], policy_version=4)
        assert caught.value.__cause__ is not None
        assert "engine-owned nontrainable transition" in str(
            caught.value.__cause__
        )
        assert "trainer-00" in batch._death_life_epochs
        assert batch.metrics.transitions_accepted == 0
        assert batch.metrics.failed_rounds == 1
    finally:
        batch.close()


def test_death_during_same_life_hold_takes_terminal_precedence():
    env = _FakeEnv("trainer-00", 0, [
        _telemetry(
            "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
            look_pitch=0.0, causal_transition_trainable=False,
        ),
        _telemetry(
            "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
            look_pitch=0.0, action_generation=11,
            causal_transition_trainable=False,
            terminal=True, terminal_reason=1,
        ),
    ], [])
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        hold = batch.collect_round([Action(look_pitch=2.0)], policy_version=1)
        assert hold.infos[0]["action_state_phase"] == "same_life_hold_settling"
        assert "trainer-00" in batch._action_hold_epochs
        death = batch.collect_round([Action(look_pitch=2.0)], policy_version=2)
        assert death.infos[0]["death_lifecycle_phase"] == "death_terminal"
        assert "trainer-00" not in batch._action_hold_epochs
        assert batch._death_life_epochs["trainer-00"].previous_life_epoch == 1
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_peer_terminal_is_registered_while_other_client_enters_hold():
    held = _FakeEnv("trainer-00", 0, [
        _telemetry(
            "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
            look_pitch=0.0, causal_transition_trainable=False,
        ),
    ], [])
    dying = _FakeEnv("trainer-01", 1, [
        _telemetry(
            "trainer-01", 1, 2, 11, echo_tick=10, accepted=1,
            terminal=True, terminal_reason=1,
        ),
    ], [])
    held.deterministic_frame_barrier = True
    dying.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [held, dying], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        boundary = batch.collect_round(
            [Action(look_pitch=2.0), Action()], policy_version=1
        )
        assert all(not info["trainable_transition"] for info in boundary.infos)
        assert "trainer-00" in batch._action_hold_epochs
        assert batch._death_life_epochs["trainer-01"].previous_life_epoch == 1
    finally:
        batch.close()


def test_rapid_redeath_before_prime_advances_pending_life_identity():
    env = _FakeEnv("trainer-00", 0, [
        _telemetry(
            "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
            terminal=True, terminal_reason=1,
        ),
        _telemetry(
            "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
            action_generation=11, life_epoch=2,
            terminal=True, terminal_reason=1,
        ),
        _telemetry(
            "trainer-00", 0, 4, 13, echo_tick=12, accepted=1,
            action_generation=12, life_epoch=3, look_pitch=0.0,
            causal_transition_trainable=False,
        ),
    ], [])
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        batch.collect_round([Action()], policy_version=1)
        redeath = batch.collect_round([Action()], policy_version=2)
        assert redeath.infos[0]["death_lifecycle_phase"] == (
            "rapid_redeath_terminal"
        )
        assert batch._death_life_epochs["trainer-00"].previous_life_epoch == 2
        rebased = batch.collect_round(
            [Action(look_pitch=2.0)], policy_version=3
        )
        assert rebased.infos[0]["death_lifecycle_phase"] == "new_life_rebase"
        assert batch._death_life_epochs["trainer-00"].new_life_seen is True
    finally:
        batch.close()


def test_new_life_settling_rejects_rollback_to_old_life():
    env = _FakeEnv("trainer-00", 0, [
        _telemetry(
            "trainer-00", 0, 2, 11, echo_tick=10, accepted=1,
            terminal=True, terminal_reason=1,
        ),
        _telemetry(
            "trainer-00", 0, 3, 12, echo_tick=11, accepted=1,
            action_generation=11, life_epoch=2, look_pitch=0.0,
            causal_transition_trainable=False,
        ),
        _telemetry(
            "trainer-00", 0, 4, 13, echo_tick=12, accepted=1,
            action_generation=12, life_epoch=1, look_pitch=0.0,
            causal_transition_trainable=False,
        ),
    ], [])
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        batch.collect_round([Action()], policy_version=1)
        batch.collect_round([Action(look_pitch=2.0)], policy_version=2)
        with pytest.raises(AuthoritativeEchoError, match="batch round") as caught:
            batch.collect_round([Action(look_pitch=2.0)], policy_version=3)
        assert "violated the synchronized death boundary" in str(
            caught.value.__cause__
        )
        assert batch._death_life_epochs["trainer-00"].new_life_seen is True
    finally:
        batch.close()


def test_deterministic_barrier_has_zero_stale_skip_budget():
    events = []
    env = _FakeEnv(
        "trainer-00",
        0,
        [
            _telemetry(
                "trainer-00", 0, 2, 10, echo_tick=9, accepted=1,
                applied_action_tick=9, action_generation=9,
            ),
            _telemetry(
                "trainer-00", 0, 3, 11, echo_tick=10, accepted=1,
                applied_action_tick=10, action_generation=10,
            ),
        ],
        events,
    )
    env.deterministic_frame_barrier = True
    batch = Q2NetworkClientBatch(
        [env], deterministic_frame_barrier=True, round_timeout=1.0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round([Action()], policy_version=1)
    finally:
        batch.close()


def test_dispatches_whole_round_and_rejects_stale_and_mismatched_echoes():
    events = []
    env_a = _FakeEnv(
        "client-a",
        0,
        [
                _telemetry(
                    "client-a", 0, 2, 10, echo_tick=10, accepted=1,
                    damage_dealt=1.0, action_generation=9,
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
                forward=-0.4, right=-0.1,
                vertical_intent=VerticalIntent.JUMP_OR_UP,
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
                Action(
                    move_forward=-0.4,
                    move_right=-0.1,
                    vertical_intent=VerticalIntent.JUMP_OR_UP,
                ),
            ],
            policy_version=42,
        )
        first_receive = next(i for i, event in enumerate(events) if event[0] == "receive")
        assert [event[0] for event in events[:first_receive]] == ["dispatch", "dispatch"]
        assert result.policy_version == 42
        assert [tag.action_tick for tag in result.tags] == [10, 10]
        assert [info["authoritative_echo_tick"] for info in result.infos] == [12, 11]
        assert [info["requested_action_fire"] for info in result.infos] == [
            True, False,
        ]
        assert result.infos[0]["effective_action_fire"] is True
        assert result.infos[1]["effective_action_fire"] is False
        assert all(info["trainable_transition"] for info in result.infos)
        assert result.infos[0]["stale_echoes_rejected"] == 1
        assert result.infos[0]["mismatched_echoes_rejected"] == 1
        assert not bool(result.terminated[0])
        assert env_a.transition_sample.observation.reward_damage_dealt == 6.0
        assert env_a.transition_sample.observation.terminal_reason == 0
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


def test_midrun_routed_role_loss_is_fatal_and_never_admitted():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [_telemetry(
            "client-a", 0, 2, 11, echo_tick=10, accepted=1,
            causal_role_playing=False,
            causal_role_public_pm_normal=False,
        )],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError, match="batch round") as raised:
            batch.collect_round([Action()], policy_version=7)
        assert "not an authoritative Lithium player" in str(
            raised.value.__cause__
        )
        assert batch.metrics.transitions_accepted == 0
        assert batch.metrics.failed_rounds == 1
    finally:
        batch.close()


def test_private_causal_incompleteness_forces_whole_round_nontrainable():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [_telemetry(
            "client-a", 0, 2, 11, echo_tick=10, accepted=1,
            forward=0.4, causal_facts_complete=False,
        )],
        events,
    )
    batch = Q2NetworkClientBatch([env], round_timeout=1.0)
    try:
        batch.reset()
        result = batch.collect_round(
            [Action(move_forward=0.4)], policy_version=8
        )
        assert result.infos[0]["trainable_transition"] is False
        assert result.infos[0]["action_state_resync"] is True
        assert batch.metrics.transitions_accepted == 0
    finally:
        batch.close()


def test_look_echo_from_another_generation_is_rejected_as_stale():
    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=11, accepted=1,
                look_yaw=0.0, look_pitch=0.0, action_generation=9,
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
        assert batch.metrics.stale_echoes_rejected == 1
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


def test_same_tick_engine_action_change_returns_nontrainable_resync():
    events = []
    generation_flags = ((10 % 192) + 1) << ML_ACTION_GENERATION_SHIFT
    env = _FakeEnv(
        "client-a",
        0,
        [
            _telemetry(
                "client-a", 0, 2, 11, echo_tick=10, accepted=1,
                look_yaw=9.81, look_pitch=40.18, pitch=-89.0,
                forward=-0.03, right=0.71,
                vertical_intent=VerticalIntent.JUMP_OR_UP, hook=1, weapon=7,
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
                look_yaw=21.66, look_pitch=15.61,
                vertical_intent=VerticalIntent.JUMP_OR_UP, hook=1, weapon=7,
            )],
            policy_version=8,
        )
        assert result.infos[0]["action_state_resync"] is True
        assert result.infos[0]["trainable_transition"] is False
        assert result.infos[0]["action_dispatched"] is True
        assert batch.metrics.action_state_resyncs == 1
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
            "client-a", 0, 22, 31, echo_tick=31, accepted=1, forward=0.3,
            action_generation=30,
        )],
        events,
        preflight=backlog_a,
    )
    env_b = _FakeEnv(
        "client-b",
        1,
        [_telemetry(
            "client-b", 1, 19, 28, echo_tick=28, accepted=1, forward=0.3,
            action_generation=27,
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
                    action_generation=1,
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
                action_generation=1,
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


def test_synchronized_pre_boundary_gap_waits_without_dispatch_or_failure():
    events = []
    envs = [
        _FakeEnv(f"client-{slot}", slot, [], events)
        for slot in range(4)
    ]
    batch = Q2NetworkClientBatch(
        envs, round_timeout=0.01, max_rejected_echoes=0
    )
    try:
        batch.reset()

        # Some generated-map downloads silence every conduit before the first
        # intermission/new-map packet. The dispatched round is discarded, but
        # this synchronized gap is not a corrupt partial action round.
        gap = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=90
        )
        assert all(info["telemetry_gap_resync"] for info in gap.infos)
        assert all(not info["trainable_transition"] for info in gap.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4
        assert batch.metrics.telemetry_gap_resyncs == 1
        assert batch.metrics.echo_timeouts == 0
        assert batch.metrics.failed_rounds == 0

        paused = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=91
        )
        assert all(info["telemetry_gap_resync"] for info in paused.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        for env in envs:
            env.script.append(_telemetry(
                env.client_id, env.slot, 2, 1, map_name="mllive_uncached"
            ))
        changed = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=92
        )
        assert all(info["map_epoch_resync"] for info in changed.infos)
        assert all(not info["map_epoch_pending"] for info in changed.infos)
        assert sum(event[0] == "dispatch" for event in events) == 4

        for env in envs:
            env.script.append(_telemetry(
                env.client_id,
                env.slot,
                3,
                2,
                echo_tick=2,
                accepted=1,
                action_generation=1,
                forward=0.4,
                map_name="mllive_uncached",
            ))
        accepted = batch.collect_round(
            [Action(move_forward=0.4)] * 4, policy_version=93
        )
        assert all(info["trainable_transition"] for info in accepted.infos)
        assert sum(event[0] == "dispatch" for event in events) == 8
        assert batch.metrics.transitions_accepted == 4
        assert batch.metrics.map_epoch_resyncs == 1
    finally:
        batch.close()


def test_partial_multi_client_timeout_remains_a_failed_round():
    events = []
    env_a = _FakeEnv(
        "client-a",
        0,
        [_telemetry(
            "client-a", 0, 2, 11, echo_tick=11, accepted=1, forward=0.4
        )],
        events,
    )
    env_b = _FakeEnv("client-b", 1, [], events)
    batch = Q2NetworkClientBatch(
        [env_a, env_b], round_timeout=0.01, max_rejected_echoes=0
    )
    try:
        batch.reset()
        with pytest.raises(AuthoritativeEchoError):
            batch.collect_round(
                [Action(move_forward=0.4)] * 2, policy_version=94
            )
        assert batch.metrics.failed_rounds == 1
        assert batch.metrics.echo_timeouts == 1
        assert batch.metrics.telemetry_gap_resyncs == 0
        assert batch.metrics.transitions_accepted == 0
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
                forward=1.0, right=-1.0,
                vertical_intent=VerticalIntent.JUMP_OR_UP, fire=True,
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
            [np.array([2, -2, 0, 0, 2, 1, 0, 0], dtype=np.float32)],
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


def test_reset_slot_returns_cached_admitted_vector_without_env_or_provider_call():
    events = []
    env = _FakeEnv("client-a", 0, [], events)
    adapter = Q2NetworkClientMultiEnv([env], initial_policy_version=1)
    try:
        initial = adapter.reset_all()[0].copy()

        def forbidden_reset():
            raise AssertionError("reset_slot sampled the environment/provider")

        env.reset_episode_vector = forbidden_reset
        first = adapter.reset_slot(0)
        assert np.array_equal(first, initial)
        first[:] = -999.0
        assert np.array_equal(adapter.reset_slot(0), initial)
    finally:
        adapter.close()


@pytest.mark.parametrize("terminal,max_ep_steps", [(True, 100), (False, 1)])
def test_multires_terminal_and_truncation_never_finalize_legacy_spatial(
    terminal, max_ep_steps
):
    class ForbiddenSpatial:
        def finalize_episode(self, **_kwargs):
            raise AssertionError("multires transition reached legacy finalizer")

    events = []
    env = _FakeEnv(
        "client-a",
        0,
        [_telemetry(
            "client-a", 0, 2, 11, echo_tick=11, accepted=1,
            terminal=terminal,
        )],
        events,
    )
    env.uses_multires_spatial = True
    env._spatial = ForbiddenSpatial()
    adapter = Q2NetworkClientMultiEnv(
        [env], max_ep_steps=max_ep_steps, initial_policy_version=1
    )
    try:
        adapter.reset_all()
        neutral_action = np.zeros(8, dtype=np.float32)
        neutral_action[4] = int(VerticalIntent.NEUTRAL)
        _observation, reward, terminated, truncated, _info = adapter.step_all(
            [neutral_action], policy_version=1
        )[0]
        assert reward == pytest.approx(0.0 if terminal else 1.5)
        assert terminated is terminal
        assert truncated is (not terminal)
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
    telemetry_secret = "credential-that-must-never-appear-in-argv-123"

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
        telemetry_token=telemetry_secret,
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
        assert "ml_telemetry_token" not in args
        assert telemetry_secret not in args
        assert telemetry_secret not in " ".join(args)
        assert kwargs["env"]["Q2_ML_CLIENT_TELEMETRY_TOKEN"] == telemetry_secret
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


def test_direct_batch_builder_preserves_distinct_per_client_factory_binding(tmp_path):
    class Factory:
        def __init__(self):
            self.resets = 0

        def create(self, telemetry, *, map_epoch):
            raise AssertionError("builder must not create providers before reset")

        def reset_session(self):
            self.resets += 1

    factories = [Factory(), Factory()]
    kwargs = dict(
        n_clients=2,
        server="127.0.0.1:28000",
        telemetry_server="127.0.0.1:28049",
        telemetry_token="secret",
        client_binary=str(tmp_path / "quake2"),
        client_root=str(tmp_path / "runtime"),
        harness_port_base=41000,
        qport_base=51000,
        client_id_prefix="direct",
        multires_spatial_provider_factories=factories,
        expected_runtime_manifest_sha256="a" * 64,
    )
    batch = build_network_client_batch(**kwargs)
    try:
        assert isinstance(batch, Q2NetworkClientBatch)
        assert batch.vector is True
        assert [env.client_id for env in batch.envs] == ["direct-00", "direct-01"]
        assert [env.harness_port for env in batch.envs] == [41000, 41001]
        assert [env.qport for env in batch.envs] == [51000, 51001]
        assert [
            env._multires_spatial_provider_factory for env in batch.envs
        ] == factories
    finally:
        batch.close()

    with pytest.raises(ValueError, match="cannot start as observers"):
        build_network_client_batch(
            **kwargs,
            deterministic_frame_barrier=True,
            start_observer=True,
        )
        batch.close()
    assert [factory.resets for factory in factories] == [1, 1]

    with pytest.raises(ValueError, match="own provider factory"):
        build_network_client_batch(
            **{
                **kwargs,
                "multires_spatial_provider_factories": [factories[0]] * 2,
            }
        )
