from types import SimpleNamespace

import numpy as np
import pytest

from harness.protocol import R_HOOK
from harness.spatial import HOOK_REQUIRED, SessionMemoryCell, VoxelSpatialReward


def _obs(
    *,
    tick: int,
    pos=(128.0, 128.0, 128.0),
    hook: int = 0,
    damage_taken: float = 0.0,
    required: bool = True,
):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = 100.0
    self_state[7] = 50.0
    self_state[8] = 8.0
    self_state[9] = 10.0
    action_debug = np.zeros(12, dtype=np.float32)
    action_debug[4] = 1.0
    action_debug[10] = hook
    hook_zones = np.zeros((4, 8), dtype=np.float32)
    # The landing and hot cell are both lattice cell (2, 0, 0).  The anchor
    # is visible/reachable from the start and the landing removes 384 units of
    # the 512-unit distance to the hot cell center.
    hook_zones[0] = (
        384.0, 128.0, 384.0,
        512.0, 128.0, 128.0,
        362.0, float(HOOK_REQUIRED if required else 0),
    )
    return SimpleNamespace(
        tick=tick,
        yaw=0.0,
        pitch=0.0,
        self_state=self_state,
        entity_count=0,
        entities=np.zeros((8, 9), dtype=np.float32),
        hook_zone_count=1,
        hook_zones=hook_zones,
        audio=np.zeros(5, dtype=np.float32),
        action_debug=action_debug,
        rune_flags=np.zeros(5, dtype=np.float32),
        inbound_dmg_recency=0.0,
        reward_damage_dealt=0.0,
        reward_damage_taken=damage_taken,
        reward_kill=0.0,
        reward_death=0.0,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
    )


def _reward() -> VoxelSpatialReward:
    reward = VoxelSpatialReward(
        new_cell_reward=0.0,
        engagement_reward=0.0,
        aim_alignment_reward=0.0,
        nominal_speed_reward=0.0,
        slow_movement_penalty=0.0,
        overspeed_penalty=0.0,
        threat_engagement_reward=0.0,
        threat_aim_reward=0.0,
        threat_fire_reward=0.0,
        threat_damage_reward=0.0,
        threat_kill_reward=0.0,
        threat_ignore_penalty=0.0,
        survival_tick_reward=0.0,
        survival_threat_reward=0.0,
        survival_low_health_reward=0.0,
        exchange_jitter=0.0,
        exchange_aggression_mag=0.0,
    )
    reward.reset("hook-correction", _obs(tick=0))
    reward.session_memories["hook-correction"][(2, 0, 0)] = SessionMemoryCell(
        prior_opportunity=8.0
    )
    reward._invalidate_feature_cache()
    return reward


def test_hook_starts_one_concrete_correction_toward_heated_lattice_cell():
    reward = _reward()

    _bonus, info = reward.update(_obs(tick=1, hook=1))

    assert info["hook_correction_available"] == 1.0
    assert info["hook_correction_needed"] == 1.0
    assert info["hook_correction_started"] == 1.0
    assert info["hook_correction_active"] == 1.0
    assert info["hook_correction_heat"] > 0.0
    assert info["hook_correction_target_x"] == pytest.approx(512.0)
    assert info["hook_correction_target_y"] == pytest.approx(128.0)
    assert info["hook_correction_target_z"] == pytest.approx(128.0)
    assert info["hook_correction_hot_x"] == pytest.approx(640.0)
    assert info["hook_correction_hot_y"] == pytest.approx(128.0)
    assert info["hook_correction_hot_z"] == pytest.approx(128.0)
    # Starting the actuator has a safety cost; no positive hook-use reward.
    assert info["hook_correction_progress_reward"] == 0.0
    assert info["hook_discipline"] == pytest.approx(-reward.hook_cost)


def test_hook_reward_is_bounded_new_best_progress_then_one_completion():
    reward = _reward()
    reward.update(_obs(tick=1, hook=1))

    _bonus, progressed = reward.update(
        _obs(tick=2, pos=(384.0, 128.0, 128.0))
    )
    assert progressed["hook_correction_progress"] == pytest.approx(256.0)
    assert 0.0 < progressed["hook_correction_progress_reward"] < (
        reward.hook_correction_progress_reward
    )
    assert progressed["hook_correction_success"] == 0.0

    # Moving backward and then re-covering the same ground cannot farm the
    # correction: only a new best distance to the fixed target is rewarded.
    reward.update(_obs(tick=3, pos=(128.0, 128.0, 128.0)))
    _bonus, replayed = reward.update(
        _obs(tick=4, pos=(384.0, 128.0, 128.0))
    )
    assert replayed["hook_correction_progress"] == 0.0
    assert replayed["hook_correction_progress_reward"] == 0.0

    _bonus, arrived = reward.update(
        _obs(tick=5, pos=(500.0, 128.0, 128.0), hook=3)
    )
    assert arrived["hook_correction_success"] == 1.0
    assert arrived["hook_correction_active"] == 0.0
    assert arrived["hook_discipline"] > reward.hook_correction_complete_reward

    _bonus, after = reward.update(
        _obs(tick=6, pos=(500.0, 128.0, 128.0))
    )
    assert after["hook_correction_success"] == 0.0
    assert after["hook_correction_progress_reward"] == 0.0


def test_legacy_hook_rate_knobs_and_engine_channel_cannot_restore_rewards(
    monkeypatch,
):
    monkeypatch.setenv("R_HOOK", "99")
    monkeypatch.setenv("R_HOOK_ENEMY_REWARD", "99")
    monkeypatch.setenv("R_HOOK_NO_AMMO_REWARD", "99")
    monkeypatch.setenv("R_HOOK_RELEASE_OVERSPEED", "99")

    reward = VoxelSpatialReward.from_env(seed=3)

    assert R_HOOK == 0.0
    assert reward.hook_enemy_reward == 0.0
    assert reward.hook_no_ammo_reward == 0.0
    assert reward.hook_release_overspeed_reward == 0.0
    assert reward.hook_required_reward == 0.0


def test_escape_hook_still_requires_a_heated_reachable_destination():
    reward = VoxelSpatialReward(
        new_cell_reward=0.0,
        engagement_reward=0.0,
        aim_alignment_reward=0.0,
        nominal_speed_reward=0.0,
        slow_movement_penalty=0.0,
    )
    reward.reset("no-heat", _obs(tick=0, required=False))

    _bonus, info = reward.update(
        _obs(tick=1, hook=1, damage_taken=10.0, required=False)
    )

    assert info["hook_correction_needed"] == 1.0
    assert info["hook_correction_available"] == 0.0
    assert info["hook_correction_started"] == 0.0
    assert info["hook_blind"] == 1.0
    assert info["hook_discipline"] < -reward.hook_cost
