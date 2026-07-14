import numpy as np
import pytest
import torch

from models.policy import (
    ACTION_DIM,
    ENT_DIM,
    ENT_OFF,
    HIDDEN_DIM,
    OBS_DIM,
    Q2BotPolicy,
    target_fire_allowed,
)
from train.ppo import (
    DEFAULT,
    RolloutBuffer,
    _reconcile_server_fire_suppressions,
)


PITCH_INDEX = ENT_OFF + 8 * ENT_DIM + 16 * 4 + 4 * 8 + 5 + 1


def _observation(*, rel_xyz=None, alive=True, pitch=0.0):
    obs = torch.zeros(OBS_DIM, dtype=torch.float32)
    obs[6] = 0.5 if alive else 0.0
    obs[PITCH_INDEX] = float(pitch) / 90.0
    if rel_xyz is not None:
        obs[ENT_OFF:ENT_OFF + 3] = torch.as_tensor(
            rel_xyz, dtype=torch.float32
        )
        obs[ENT_OFF + 6] = 0.5
        obs[ENT_OFF + 7:ENT_OFF + 9] = 1.0
    return obs


def test_target_fire_gate_uses_post_command_quake_alignment():
    angle = np.deg2rad(40.0)
    target_right = [np.cos(angle), np.sin(angle), 0.0]
    observations = torch.stack([
        _observation(),
        _observation(rel_xyz=target_right),
        _observation(rel_xyz=target_right),
        _observation(rel_xyz=[-1.0, 0.0, 0.0]),
        _observation(rel_xyz=[1.0, 0.0, 0.0], alive=False),
    ])
    look = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [-40.0, 0.0],
        [-45.0, 0.0],
        [0.0, 0.0],
    ])

    allowed = target_fire_allowed(observations, look)

    assert allowed.tolist() == [False, False, True, False, False]


def test_target_fire_gate_accepts_any_aligned_visible_enemy():
    obs = _observation(rel_xyz=[1.0, 1.0, 0.0])
    second = ENT_OFF + ENT_DIM
    obs[second:second + 3] = torch.tensor([1.0, 0.0, 0.0])
    obs[second + 6] = 0.5
    obs[second + 7:second + 9] = 1.0

    assert target_fire_allowed(obs.unsqueeze(0), torch.zeros(1, 2)).item()


def test_gated_batch_sampling_and_ppo_logprob_replay_match():
    torch.manual_seed(7)
    policy = Q2BotPolicy()
    with torch.no_grad():
        policy.actor_fire.weight.zero_()
        policy.actor_fire.bias.copy_(torch.tensor([-10.0, 10.0]))
        policy.actor_cont.weight.zero_()
        policy.actor_cont.bias.zero_()

    observations = torch.stack([
        _observation(),
        _observation(rel_xyz=[1.0, 0.0, 0.0]),
    ])
    obs_np = observations.numpy()
    initial_hx = [policy.init_hidden(1) for _ in range(2)]

    actions, _values, old_log_probs, _new_hx, metadata = policy.act_batch(
        obs_np,
        initial_hx,
        deterministic=True,
        gate_fire=True,
        return_fire_metadata=True,
    )

    assert metadata["fire_allowed"].tolist() == [False, True]
    assert actions[:, 5].tolist() == [0.0, 1.0]
    assert metadata["raw_fire_probability"][0] > 0.999

    obs_t = observations.unsqueeze(1)
    h_stack = torch.cat([state[0] for state in initial_hx], dim=1)
    c_stack = torch.cat([state[1] for state in initial_hx], dim=1)
    params, _value, _ = policy(obs_t, (h_stack, c_stack))
    replay_log_probs, replay_entropy = policy.action_log_prob_entropy(
        params,
        torch.from_numpy(actions).unsqueeze(1),
        fire_allowed=torch.from_numpy(metadata["fire_allowed"]).unsqueeze(1),
    )

    assert torch.isfinite(replay_log_probs).all()
    assert torch.isfinite(replay_entropy).all()
    assert replay_log_probs.squeeze(1).detach().numpy() == pytest.approx(
        old_log_probs, abs=1e-6
    )
    ratio = torch.exp(
        replay_log_probs.squeeze(1)
        - torch.from_numpy(old_log_probs)
    )
    assert ratio.detach().numpy() == pytest.approx(np.ones(2), abs=1e-6)


def test_closed_gate_does_not_create_actor_fire_gradient():
    policy = Q2BotPolicy()
    obs = _observation().reshape(1, 1, -1)
    params, _value, _ = policy(obs, policy.init_hidden(1))
    action = torch.zeros(1, 1, ACTION_DIM)
    log_prob, entropy = policy.action_log_prob_entropy(
        params,
        action,
        fire_allowed=torch.zeros(1, 1, dtype=torch.bool),
    )
    (-(log_prob + 0.01 * entropy).sum()).backward()

    assert policy.actor_fire.weight.grad is not None
    assert torch.count_nonzero(policy.actor_fire.weight.grad) == 0
    assert torch.count_nonzero(policy.actor_fire.bias.grad) == 0


def test_server_suppression_reconciles_applied_action_and_exact_logprob():
    actions = np.zeros((1, ACTION_DIM), dtype=np.float32)
    actions[0, 5] = 1.0
    log_probs = np.array([-35.5], dtype=np.float32)
    fire_allowed = np.array([True], dtype=np.bool_)
    metadata = {
        # Deliberately below the former 1e-8 probability clamp.
        "raw_fire_log_probability": np.array([-35.0], dtype=np.float32),
        "raw_fire_probability": np.array([np.exp(-35.0)], dtype=np.float32),
    }
    step_results = [
        (0, [(None, 0.0, False, False, {"fire_gate_suppressed": True})])
    ]

    count = _reconcile_server_fire_suppressions(
        actions,
        log_probs,
        fire_allowed,
        metadata,
        step_results,
        n_ml=1,
    )

    assert count == 1
    assert actions[0, 5] == 0.0
    assert fire_allowed.tolist() == [False]
    assert log_probs[0] == pytest.approx(-0.5)


def test_server_reconciliation_metadata_exists_with_proactive_gate_disabled():
    policy = Q2BotPolicy()
    with torch.no_grad():
        policy.actor_fire.weight.zero_()
        policy.actor_fire.bias.copy_(torch.tensor([-4.0, 4.0]))
    actions, _values, log_probs, _new_hx, metadata = policy.act_batch(
        _observation().unsqueeze(0).numpy(),
        [policy.init_hidden(1)],
        deterministic=True,
        gate_fire=False,
        return_fire_metadata=True,
    )
    fire_allowed = metadata["fire_allowed"].copy()

    assert actions[0, 5] == 1.0
    assert fire_allowed.tolist() == [True]
    assert np.isfinite(metadata["raw_fire_log_probability"]).all()
    assert _reconcile_server_fire_suppressions(
        actions,
        log_probs,
        fire_allowed,
        metadata,
        [(0, [(None, 0.0, False, False, {"fire_gate_suppressed": True})])],
        n_ml=1,
    ) == 1
    assert actions[0, 5] == 0.0
    assert fire_allowed.tolist() == [False]


def test_reconciled_logprob_replays_with_closed_server_mask():
    policy = Q2BotPolicy()
    with torch.no_grad():
        policy.actor_fire.weight.zero_()
        policy.actor_fire.bias.copy_(torch.tensor([-3.0, 3.0]))
        policy.actor_cont.weight.zero_()
        policy.actor_cont.bias.zero_()
    observation = _observation(rel_xyz=[1.0, 0.0, 0.0])
    hx = [policy.init_hidden(1)]
    actions, _values, log_probs, _new_hx, metadata = policy.act_batch(
        observation.unsqueeze(0).numpy(),
        hx,
        deterministic=True,
        gate_fire=True,
        return_fire_metadata=True,
    )
    fire_allowed = metadata["fire_allowed"].copy()
    assert actions[0, 5] == 1.0

    _reconcile_server_fire_suppressions(
        actions,
        log_probs,
        fire_allowed,
        metadata,
        [(0, [(None, 0.0, False, False, {"fire_gate_suppressed": True})])],
        n_ml=1,
    )

    params, _value, _ = policy(
        observation.reshape(1, 1, -1),
        hx[0],
    )
    replay_log_prob, _entropy = policy.action_log_prob_entropy(
        params,
        torch.from_numpy(actions).unsqueeze(1),
        fire_allowed=torch.from_numpy(fire_allowed).unsqueeze(1),
    )
    assert replay_log_prob.item() == pytest.approx(float(log_probs[0]), abs=1e-6)


def test_server_suppression_rejects_unrecorded_fire_mismatch():
    with pytest.raises(RuntimeError, match="outside the recorded"):
        _reconcile_server_fire_suppressions(
            np.zeros((1, ACTION_DIM), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.bool_),
            {
                "raw_fire_log_probability": np.zeros(1, dtype=np.float32),
            },
            [(0, [(None, 0.0, False, False, {
                "fire_gate_suppressed": True,
            })])],
            n_ml=1,
        )


def test_rollout_buffer_defaults_open_and_records_network_gate():
    buffer = RolloutBuffer(2, 1, OBS_DIM, ACTION_DIM, HIDDEN_DIM, torch.device("cpu"))
    zeros = torch.zeros
    buffer.add(
        zeros(2, OBS_DIM),
        zeros(2, ACTION_DIM),
        zeros(2),
        zeros(2),
        zeros(2),
        zeros(2),
        zeros(2, HIDDEN_DIM),
        zeros(2, HIDDEN_DIM),
        torch.tensor([False, True]),
    )
    assert buffer.fire_allowed[0].tolist() == [False, True]
    assert DEFAULT["target_fire_gate"] == 1
