from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.policy import ENT_DIM, ENT_OFF, OBS_DIM, Q2BotPolicy
from tools.behavior_clone_aim import _teacher_action
from train.ppo import (
    DEFAULT,
    _AIM_PITCH_OBS_INDEX,
    _aim_anchor_loss,
    _aim_anchor_targets,
    _balanced_binary_class_weights,
)


def _raw_obs(rel_xyz=None, pitch=0.0):
    entities = np.zeros((8, 9), dtype=np.float32)
    count = 0
    if rel_xyz is not None:
        entities[0, :3] = rel_xyz
        entities[0, 6] = 100.0
        entities[0, 7:9] = 1.0
        count = 1
    return SimpleNamespace(
        entities=entities,
        entity_count=count,
        pitch=pitch,
    )


def _policy_obs(rel_xyz=None, pitch=0.0, alive=True, speed=0.0):
    obs = torch.zeros(OBS_DIM, dtype=torch.float32)
    obs[6] = 0.5 if alive else 0.0
    obs[3] = speed / 1000.0
    obs[_AIM_PITCH_OBS_INDEX] = pitch / 90.0
    if rel_xyz is not None:
        ent = obs[ENT_OFF:ENT_OFF + ENT_DIM]
        ent[:3] = torch.as_tensor(rel_xyz) / 4096.0
        ent[6] = 0.5
        ent[7:9] = 1.0
    return obs


def _set_exposure(obs, exposure):
    obs[ENT_OFF + 8] = float(exposure)
    return obs


@pytest.mark.parametrize(
    "rel_xyz,pitch",
    (
        ((100.0, 0.0, 0.0), 0.0),
        ((100.0, 83.91, 0.0), 0.0),
        ((0.0, 100.0, 0.0), 0.0),
        ((-100.0, 0.0, 0.0), 0.0),
        ((98.48, 0.0, 17.36), 10.0),
    ),
)
def test_anchor_teacher_matches_behavior_clone_geometry(rel_xyz, pitch):
    expected = _teacher_action(_raw_obs(rel_xyz, pitch=pitch))
    targets = _aim_anchor_targets(
        _policy_obs(rel_xyz, pitch=pitch).view(1, 1, -1)
    )
    actual_look = targets["look_target"][0, 0].cpu().numpy()

    assert np.allclose(actual_look, expected[2:4], atol=1e-3)
    assert int(targets["fire_target"][0, 0]) == int(expected[5])


def test_anchor_masks_and_nearest_live_visible_enemy():
    obs = _policy_obs(None).view(1, 1, -1)
    # Far valid enemy in slot 0.
    first = obs[0, 0, ENT_OFF:ENT_OFF + ENT_DIM]
    first[:3] = torch.tensor([300.0, 0.0, 0.0]) / 4096.0
    first[6:] = torch.tensor([0.5, 1.0, 1.0])
    # Near valid enemy in slot 1 should win.
    second = obs[0, 0, ENT_OFF + ENT_DIM:ENT_OFF + 2 * ENT_DIM]
    second[:3] = torch.tensor([100.0, 83.91, 0.0]) / 4096.0
    second[6:] = torch.tensor([0.5, 1.0, 1.0])
    targets = _aim_anchor_targets(obs)
    assert torch.isclose(targets["look_target"][0, 0, 0], torch.tensor(-40.0), atol=0.1)
    assert bool(targets["look_mask"][0, 0])
    assert bool(targets["fire_mask"][0, 0])

    hidden = _aim_anchor_targets(_policy_obs(None).view(1, 1, -1))
    assert not bool(hidden["look_mask"][0, 0])
    assert bool(hidden["fire_mask"][0, 0])
    assert not bool(hidden["posture_mask"][0, 0])
    assert int(hidden["fire_target"][0, 0]) == 0

    pitched = _aim_anchor_targets(
        _policy_obs(None, pitch=24.0, speed=120.0).view(1, 1, -1)
    )
    assert bool(pitched["posture_mask"][0, 0])
    assert torch.isclose(
        pitched["posture_pitch_target"][0, 0], torch.tensor(-24.0)
    )

    visible = _aim_anchor_targets(
        _policy_obs((100.0, 0.0, 0.0), pitch=24.0).view(1, 1, -1)
    )
    assert not bool(visible["posture_mask"][0, 0])

    dead = _aim_anchor_targets(
        _policy_obs((100.0, 0.0, 0.0), alive=False).view(1, 1, -1)
    )
    assert not bool(dead["look_mask"][0, 0])
    assert not bool(dead["fire_mask"][0, 0])
    assert not bool(dead["posture_mask"][0, 0])


def test_anchor_tracks_protected_target_without_teaching_fire():
    obs = _set_exposure(
        _policy_obs((100.0, 0.0, 0.0)), -0.5
    ).view(1, 1, -1)
    targets = _aim_anchor_targets(obs)

    assert bool(targets["look_mask"][0, 0])
    assert bool(targets["has_target"][0, 0])
    assert int(targets["fire_target"][0, 0]) == 0


def test_anchor_emits_only_executable_pitch_near_engine_limit():
    obs = _policy_obs((100.0, 0.0, -100.0), pitch=86.0).view(1, 1, -1)
    targets = _aim_anchor_targets(obs)

    assert float(targets["look_target"][0, 0, 1]) <= 3.0 + 1e-5
    assert float(targets["look_target"][0, 0, 1]) >= -30.0


def test_balanced_binary_weights_are_finite_and_equalize_class_mass():
    targets = torch.tensor([0] * 9 + [1])
    mask = torch.ones(10, dtype=torch.bool)
    weights = _balanced_binary_class_weights(targets, mask)
    assert torch.isfinite(weights).all()
    assert torch.isclose(
        weights[0] * (targets == 0).sum(),
        weights[1] * (targets == 1).sum(),
    )

    all_zero = _balanced_binary_class_weights(
        torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.bool)
    )
    assert torch.equal(all_zero, torch.tensor([1.0, 0.0]))
    no_valid = _balanced_binary_class_weights(
        torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.bool)
    )
    assert torch.equal(no_valid, torch.zeros(2))


def test_anchor_gradients_reach_only_intended_actor_heads_directly():
    policy = Q2BotPolicy()
    obs = torch.stack(
        (
            _policy_obs((100.0, 83.91, 0.0)),
            _policy_obs(None, speed=120.0),
        )
    ).view(1, 2, -1)
    params, _value, _hx = policy(obs, policy.init_hidden(1))
    actions = torch.zeros(1, 2, 8)
    actions[0, 1, 0] = 1.0
    actions[..., 7] = torch.tensor([[3.0, 4.0]])
    targets = _aim_anchor_targets(obs)
    weights = _balanced_binary_class_weights(
        targets["fire_target"], targets["fire_mask"]
    )
    metrics = _aim_anchor_loss(
        policy,
        params,
        actions,
        targets,
        weights,
        look_weight=16.0,
        posture_weight=4.0,
        fire_weight=1.0,
    )
    metrics["inner"].backward()

    assert policy.encoder.rest[0].weight.grad.norm() > 0
    assert policy.lstm.weight_ih_l0.grad.norm() > 0
    assert policy.actor_cont.weight.grad[2:4].norm() > 0
    assert torch.equal(
        policy.actor_cont.weight.grad[:2],
        torch.zeros_like(policy.actor_cont.weight.grad[:2]),
    )
    assert policy.actor_fire.weight.grad.norm() > 0
    assert policy.weapon_embed.weight.grad[3:5].norm() > 0
    assert policy.actor_jump.weight.grad is None
    assert policy.actor_hook.weight.grad is None
    assert policy.actor_weapon.weight.grad is None
    assert policy.critic.weight.grad is None
    assert policy.predict_next.weight.grad is None


def test_anchor_is_default_off():
    assert DEFAULT["aim_anchor_coef"] == 0.0
    assert DEFAULT["aim_anchor_posture_weight"] == 4.0
