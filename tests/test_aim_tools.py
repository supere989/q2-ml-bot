from types import SimpleNamespace
from pathlib import Path

import numpy as np

from tools.behavior_clone_aim import (
    OBS_DIM,
    _fire_after_command,
    _next_policy_name,
    _teacher_action,
    _wrap_degrees,
    synthetic_aim,
)
from tools.evaluate_aim import _desired_look_delta, _nearest_visible_enemy as _eval_nearest_enemy


def _obs(rel_xyz=None, pitch=0.0, yaw=0.0, control_source=3):
    entities = np.zeros((8, 9), dtype=np.float32)
    entity_debug = np.zeros((8, 4), dtype=np.uint32)
    count = 0
    if rel_xyz is not None:
        entities[0, :3] = rel_xyz
        entities[0, 7:9] = 1.0
        entity_debug[0, 2] = control_source
        count = 1
    return SimpleNamespace(
        entities=entities,
        entity_count=count,
        pitch=pitch,
        yaw=yaw,
        entity_debug=entity_debug,
        self_state=np.array([0, 0, 0, 0, 0, 0, 100, 0, 0, 0], dtype=np.float32),
        is_terminal=False,
    )


def test_teacher_uses_quake_local_yaw_sign_and_post_turn_fire():
    # Positive local Y is Quake's right-basis direction, so the engine must
    # receive a negative yaw delta. A 40-degree target is aligned after the
    # same tick's look command and may fire; a 90-degree target is not.
    rel_40 = np.array([np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0.0])
    action_40 = _teacher_action(_obs(rel_40))
    assert np.isclose(action_40[2], -40.0, atol=1e-4)
    assert action_40[5] == 1.0

    action_90 = _teacher_action(_obs([0.0, 100.0, 0.0]))
    assert action_90[2] == -45.0
    assert action_90[5] == 0.0


def test_teacher_does_not_fire_at_target_behind_bot():
    action = _teacher_action(_obs([-100.0, 0.0, 0.0]))
    assert abs(action[2]) == 45.0
    assert action[5] == 0.0


def test_fire_alignment_is_post_command_residual():
    assert _fire_after_command(-40.0, 0.0, -40.0, 0.0, 12.0, 14.0)
    assert not _fire_after_command(-90.0, 0.0, -45.0, 0.0, 12.0, 14.0)
    assert not _fire_after_command(
        0.0, 30.0, 0.0, 30.0, 12.0, 14.0, current_pitch=85.0
    )


def test_evaluator_does_not_double_subtract_global_yaw():
    # rel_pos is already local. A target straight ahead remains zero-error
    # even when the separately reported global yaw is 90 degrees.
    obs = _obs([100.0, 0.0, 0.0], yaw=90.0)
    yaw_delta, pitch_delta = _desired_look_delta(obs, obs.entities[0, :3])
    assert np.isclose(yaw_delta, 0.0)
    assert np.isclose(pitch_delta, 0.0)


def test_evaluator_selects_real_legacy_opponents_only():
    assert _eval_nearest_enemy(_obs([100.0, 0.0, 0.0], control_source=3)) is not None
    assert _eval_nearest_enemy(_obs([100.0, 0.0, 0.0], control_source=1)) is None


def test_nonzero_pitch_geometry_and_yaw_wrap():
    pitch = 10.0
    pitch_rad = np.deg2rad(pitch)
    # A world-horizontal target observed while looking 10 degrees down has
    # local coordinates (cos(p), 0, sin(p)) and requires -10 degrees pitch.
    obs = _obs([np.cos(pitch_rad), 0.0, np.sin(pitch_rad)], pitch=pitch)
    yaw_delta, pitch_delta = _desired_look_delta(obs, obs.entities[0, :3])
    assert np.isclose(yaw_delta, 0.0, atol=1e-5)
    assert np.isclose(pitch_delta, -10.0, atol=1e-5)
    assert _wrap_degrees(181.0) == -179.0


def test_output_name_preserves_numeric_ppo_step():
    assert _next_policy_name(Path("policy_39929600.pt")) == "policy_39929601"


def test_synthetic_vectors_match_policy_normalization_and_fire_subset():
    args = SimpleNamespace(
        seed=1337,
        synthetic_samples=2000,
        synthetic_visible_rate=0.65,
        synthetic_yaw_abs=180.0,
        aim_yaw_deg=12.0,
        aim_pitch_deg=14.0,
    )
    obs, actions, stats = synthetic_aim(args)

    assert obs.shape[0] == 2000
    assert np.max(np.abs(obs[:, 0:3])) < 1.0       # self position / 4096
    assert np.max(np.abs(obs[:, 3:6])) < 1.0       # velocity / 1000
    assert np.max(obs[:, 6]) <= 0.5                # health / 200
    assert np.max(np.abs(obs[:, 10:13])) < 1.0     # entity xyz / 4096
    ray_distances = obs[:, 85:146:4]
    assert np.max(ray_distances) <= 2048.0 / 4096.0 + 1e-6
    assert np.min(ray_distances) >= -1.0 / 4096.0 - 1e-6
    if OBS_DIM > 209:
        assert np.all(obs[:, 193] == -1.0)

    entities = obs[:, 10:82].reshape(-1, 8, 9)
    visible = ((entities[:, :, 7] > 0.5) & (entities[:, :, 8] > 0.0)).any(axis=1)
    fires = actions[:, 5] > 0.5
    assert int(visible.sum()) == stats["visible_samples"]
    assert int(fires.sum()) == stats["fire_samples"]
    assert np.all(~fires | visible)
    assert 0 < fires.sum() < visible.sum()
