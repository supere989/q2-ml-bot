import math
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from harness.spatial import VoxelSpatialReward
from maps.generator import DM_SPAWN_COUNT, MIN_SPAWN_SEPARATION, MapGenerator
if torch is not None:
    from models.policy import Q2BotPolicy


def _movement_obs(*, speed, forward=1.0, right=0.0, jump=0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[3] = speed
    action_debug = np.zeros(12, dtype=np.float32)
    action_debug[4] = forward
    action_debug[5] = right
    action_debug[8] = jump
    return SimpleNamespace(self_state=self_state, action_debug=action_debug)


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


def test_nominal_forward_speed_is_rewarded_without_rewarding_strafe_only():
    reward = VoxelSpatialReward()
    forward = reward.movement_context(_movement_obs(speed=300, forward=1.0))
    strafe = reward.movement_context(_movement_obs(speed=300, forward=0.0, right=1.0))
    assert forward["movement_nominal"] == 1.0
    assert forward["movement_discipline"] > 0.0
    assert strafe["movement_nominal"] == 1.0
    assert strafe["movement_discipline"] == 0.0


def test_slow_jump_and_hook_overspeed_inputs_are_detected():
    reward = VoxelSpatialReward()
    slow_jump = reward.movement_context(_movement_obs(speed=40, jump=1))
    fast = reward.movement_context(_movement_obs(speed=520))
    assert slow_jump["movement_slow"] == 1.0
    assert slow_jump["jump_slow"] == 1.0
    assert slow_jump["movement_discipline"] < -reward.jump_cost
    assert fast["movement_overspeed"] == 1.0
    assert fast["movement_discipline"] < 0.0


def test_fresh_policy_starts_grounded_and_hook_idle():
    if torch is None:
        pytest.skip("torch is not installed")
    policy = Q2BotPolicy()
    jump_probs = policy.actor_jump.bias.softmax(dim=0)
    hook_probs = policy.actor_hook.bias.softmax(dim=0)
    assert float(jump_probs[0]) > 0.95
    assert float(hook_probs[0]) > 0.85
