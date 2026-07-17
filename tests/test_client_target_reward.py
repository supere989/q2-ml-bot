from types import SimpleNamespace

import numpy as np

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_env import Q2NetworkClientEnv, TARGET_ACQUIRE_REWARD
from harness.client_protocol import ClientTelemetry
from harness.protocol import ActionDebugIndex, ML_FIRE_GATE_PROTECTED, ML_FIRE_GATE_TARGET


def _sample(frame: int, flags: int = 0, *, terminal: bool = False):
    debug = np.zeros(len(ActionDebugIndex), dtype=np.float32)
    debug[ActionDebugIndex.FLAGS] = flags
    obs = SimpleNamespace(
        action_debug=debug,
        self_state=np.array([0, 0, 0, 0, 0, 0, 100, 0, 0, 0], dtype=np.float32),
        is_terminal=terminal,
        terminal_reason=1 if terminal else 0,
        reward_damage_dealt=0.0,
        reward_damage_taken=0.0,
        reward_kill=0.0,
        reward_death=0.0,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
        rune_flags=np.zeros(5, dtype=np.float32),
        self_debug=np.zeros(4, dtype=np.uint32),
        standing_blocked=0.0,
    )
    causal = CausalTelemetry(
        tick=frame,
        client_life_epoch=1,
        target_id=0,
        target_epoch=0,
        environmental_source_id=0,
        environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0,
        crouch_edge_id=0,
        crouch_edge_epoch=0,
        echo_tick=0,
        action_generation=0,
        hook_zone_id=0,
        hook_attempt_tick=0,
        hook_action_generation=0,
        flags=CausalFlags.FACTS_COMPLETE,
    )
    return ClientTelemetry(
        sequence=frame,
        client_slot=0,
        server_frame=frame,
        client_id="target-reward",
        map_name="q2dm1",
        observation=obs,
        causal=causal,
    )


def _env(tmp_path):
    return Q2NetworkClientEnv(
        server="127.0.0.1:28000",
        telemetry_server="127.0.0.1:28049",
        telemetry_token="x" * 48,
        client_binary=str(tmp_path / "quake2"),
        client_root=str(tmp_path),
        client_id="target-reward",
    )


def test_target_acquisition_is_rewarded_once_until_alignment_is_lost(tmp_path):
    env = _env(tmp_path)
    env.initial_result(_sample(1))

    _obs, reward, _term, _trunc, info = env.transition_result(
        _sample(2, ML_FIRE_GATE_TARGET)
    )
    assert reward == TARGET_ACQUIRE_REWARD
    assert info["target_acquired"] is True
    assert info["target_aligned"] is True

    _obs, reward, _term, _trunc, info = env.transition_result(
        _sample(3, ML_FIRE_GATE_TARGET)
    )
    assert reward == 0.0
    assert info["target_acquired"] is False


def test_target_bonus_is_cooldown_bounded_and_rejects_protected_targets(tmp_path):
    env = _env(tmp_path)
    env.initial_result(_sample(1))
    env.transition_result(_sample(2, ML_FIRE_GATE_TARGET))
    env.transition_result(_sample(3))

    _obs, reward, *_ = env.transition_result(_sample(4, ML_FIRE_GATE_TARGET))
    assert reward == 0.0

    env.transition_result(_sample(21))
    _obs, reward, _term, _trunc, info = env.transition_result(
        _sample(22, ML_FIRE_GATE_TARGET)
    )
    assert reward == TARGET_ACQUIRE_REWARD
    assert info["target_acquired"] is True

    env.transition_result(_sample(23))
    _obs, reward, _term, _trunc, info = env.transition_result(
        _sample(42, ML_FIRE_GATE_TARGET | ML_FIRE_GATE_PROTECTED)
    )
    assert reward == 0.0
    assert info["target_aligned"] is False


def test_focus_streak_offense_is_reserved_for_spatial_reward_path(tmp_path):
    env = _env(tmp_path)
    env.initial_result(_sample(1))
    sample = _sample(2)
    sample.observation.reward_offense = 25.0

    _obs, reward, _term, _trunc, info = env.transition_result(sample)
    # VoxelSpatialReward.update consumes this channel with R_OFFENSE_RUNE.
    # It must not also enter the authoritative base reward.
    assert reward == 0.0
    assert info["offense"] == 25.0
