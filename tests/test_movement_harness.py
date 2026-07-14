import math
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from harness.spatial import VoxelSpatialReward
from maps.generator import (
    ARENA_STYLES,
    DM_SPAWN_COUNT,
    MIN_SPAWN_SEPARATION,
    MapGenerator,
)
if torch is not None:
    from models.policy import Q2BotPolicy


def _movement_obs(*, speed, forward=1.0, right=0.0, jump=0, pitch=0.0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[3] = speed
    action_debug = np.zeros(12, dtype=np.float32)
    action_debug[4] = forward
    action_debug[5] = right
    action_debug[8] = jump
    return SimpleNamespace(
        self_state=self_state,
        action_debug=action_debug,
        pitch=pitch,
        entity_count=0,
        entities=np.zeros((8, 9), dtype=np.float32),
    )


def test_generated_maps_have_spare_clear_spawns_for_six_players():
    for seed in range(12):
        generator = MapGenerator(seed=seed)
        generator.generate()
        assert len(generator.spawn_points) == DM_SPAWN_COUNT
        nearest = min(
            math.dist(first[:2], second[:2])
            for index, first in enumerate(generator.spawn_points)
            for second in generator.spawn_points[index + 1:]
        )
        assert nearest >= MIN_SPAWN_SEPARATION


@pytest.mark.parametrize("style", ARENA_STYLES)
def test_arena_styles_keep_eight_clear_spawns(style):
    for seed in range(3):
        generator = MapGenerator(seed=seed, style=style)
        generator.generate()
        assert generator.style == style
        assert len(generator.spawn_points) == DM_SPAWN_COUNT
        assert sum(room.kind == "arena" for room in generator.rooms) >= 3
        stats = generator.stats()
        assert stats["hallways"] >= 2
        assert stats["corner_pockets"] >= 2
        assert stats["corners"] >= 6
        assert stats["large_buildings"] >= 1
        assert all(stats["ceiling_bands"][band] >= 1
                   for band in ("low", "mid", "high"))
        nearest = min(
            math.dist(first[:2], second[:2])
            for index, first in enumerate(generator.spawn_points)
            for second in generator.spawn_points[index + 1:]
        )
        assert nearest >= MIN_SPAWN_SEPARATION


def test_nominal_forward_speed_is_rewarded_without_rewarding_strafe_only():
    reward = VoxelSpatialReward()
    forward = reward.movement_context(_movement_obs(speed=300, forward=1.0))
    strafe = reward.movement_context(_movement_obs(speed=300, forward=0.0, right=1.0))
    assert forward["movement_nominal"] == 1.0
    assert forward["movement_discipline"] > 0.0
    assert strafe["movement_nominal"] == 1.0
    assert strafe["movement_discipline"] == 0.0


def test_backpedaling_is_not_rewarded_as_forward_traversal():
    reward = VoxelSpatialReward()
    forward = reward.movement_context(_movement_obs(speed=300, forward=1.0))
    backward = reward.movement_context(_movement_obs(speed=300, forward=-1.0))

    assert forward["forward_intent"] == 1.0
    assert forward["backward_intent"] == 0.0
    assert backward["forward_intent"] == 0.0
    assert backward["backward_intent"] == 1.0
    assert backward["movement_discipline"] < 0.0
    assert forward["movement_discipline"] > backward["movement_discipline"]


def test_no_target_downlook_gets_bounded_horizon_penalty():
    reward = VoxelSpatialReward()
    level = reward.movement_context(_movement_obs(speed=300, pitch=0.0))
    down = reward.movement_context(_movement_obs(speed=300, pitch=25.0))

    assert level["horizon_pitch_penalty"] == 0.0
    assert reward.horizon_pitch_limit == 15.0
    assert down["view_pitch_abs"] == 25.0
    assert 0.0 < down["horizon_pitch_penalty"] <= reward.horizon_pitch_penalty
    assert down["movement_discipline"] < level["movement_discipline"]


def test_level_aim_reward_requires_forward_travel_and_level_posture():
    reward = VoxelSpatialReward()
    level = reward.movement_context(
        _movement_obs(speed=180, forward=1.0, pitch=0.0)
    )
    down = reward.movement_context(
        _movement_obs(speed=180, forward=1.0, pitch=25.0)
    )
    backward = reward.movement_context(
        _movement_obs(speed=180, forward=-1.0, pitch=0.0)
    )

    assert level["level_aim_movement_reward"] == pytest.approx(
        reward.level_aim_movement_reward
    )
    assert down["level_aim_movement_reward"] == 0.0
    assert backward["level_aim_movement_reward"] == 0.0
    assert level["movement_discipline"] > down["movement_discipline"]


def test_slow_jump_and_hook_overspeed_inputs_are_detected():
    reward = VoxelSpatialReward()
    slow_jump = reward.movement_context(_movement_obs(speed=40, jump=1))
    fast = reward.movement_context(_movement_obs(speed=520))
    assert slow_jump["movement_slow"] == 1.0
    assert slow_jump["jump_slow"] == 1.0
    assert slow_jump["movement_discipline"] < -reward.jump_cost
    assert fast["movement_overspeed"] == 1.0
    assert fast["level_aim_movement_reward"] == 0.0
    assert fast["movement_discipline"] < 0.0


def test_fresh_policy_starts_grounded_and_hook_idle():
    if torch is None:
        pytest.skip("torch is not installed")
    policy = Q2BotPolicy()
    jump_probs = policy.actor_jump.bias.softmax(dim=0)
    hook_probs = policy.actor_hook.bias.softmax(dim=0)
    assert float(jump_probs[0]) > 0.95
    assert float(hook_probs[0]) > 0.85
    assert float(hook_probs[2]) < 0.01  # protocol class 2 is a C-side no-op
