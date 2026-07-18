import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harness.multires_contract import OBS_DIM, POSTURE_NEUTRAL
from harness.multires_lineage import (
    LineageError,
    load_attested_checkpoint,
    save_attested_checkpoint,
)
from harness.multires_training_config import MultiresTrainingConfiguration
from models.multires_policy import MultiresQ2BotPolicy
from train.multires_ppo import (
    MultiresPPOConfig,
    MultiresPPOTrainer,
    MultiresRolloutBatch,
)


TRAINING_CONFIG = MultiresTrainingConfiguration.create(
    reward={"damage_reward": 0.003},
    guide_dropout={"global_probability": 0.1},
    ppo={"clip_coef": 0.2},
)


def test_fresh_policy_is_298_input_and_neutral_posture_biased():
    policy = MultiresQ2BotPolicy()
    obs = torch.zeros(2, 1, OBS_DIM)
    parameters, value, _state = policy(obs, policy.init_hidden(2))
    assert parameters["posture_logits"].shape == (2, 1, 3)
    assert value.shape == (2, 1, 1)
    probabilities = policy.actor_posture.bias.softmax(dim=0)
    assert probabilities.argmax().item() == POSTURE_NEUTRAL
    assert probabilities[POSTURE_NEUTRAL] > 0.85
    assert not hasattr(policy, "actor_jump")


def test_deterministic_action_uses_three_way_posture_slot():
    policy = MultiresQ2BotPolicy()
    actions, values, log_probabilities, states = policy.act_batch(
        np.zeros((2, OBS_DIM), dtype=np.float32),
        [policy.init_hidden(1), policy.init_hidden(1)],
        deterministic=True,
    )
    assert actions.shape == (2, 8)
    assert actions[:, 4].tolist() == [float(POSTURE_NEUTRAL)] * 2
    assert values.shape == (2,)
    assert np.isfinite(log_probabilities).all()
    assert len(states) == 2


def test_action_metadata_records_the_distribution_used_for_fire_replay():
    policy = MultiresQ2BotPolicy()
    actions, _values, _log_probabilities, _states, metadata = policy.act_batch(
        np.zeros((2, OBS_DIM), dtype=np.float32),
        [policy.init_hidden(1), policy.init_hidden(1)],
        deterministic=True,
        gate_fire=True,
        return_fire_metadata=True,
    )
    assert actions[:, 5].tolist() == [0.0, 0.0]
    assert metadata["fire_allowed"].tolist() == [False, False]
    assert metadata["raw_fire_probability"].shape == (2,)
    assert np.isfinite(metadata["raw_fire_log_probability"]).all()


def test_checkpoint_loader_rejects_raw_legacy_state_and_accepts_attested(tmp_path):
    catalog = "a" * 64
    runtime = "b" * 64
    source = MultiresQ2BotPolicy()
    source_optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    attested = tmp_path / "attested.pt"
    manifest = save_attested_checkpoint(
        attested,
        source,
        atlas_catalog_sha256=catalog,
        runtime_manifest_sha256=runtime,
        training_config=TRAINING_CONFIG,
        initialization="random",
        training_step=12,
        optimizer=source_optimizer,
    )
    target = MultiresQ2BotPolicy()
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-4)
    loaded = load_attested_checkpoint(
        attested,
        target,
        expected_atlas_catalog_sha256=catalog,
        expected_runtime_manifest_sha256=runtime,
        expected_training_config=TRAINING_CONFIG,
        optimizer=target_optimizer,
    )
    assert loaded.lineage_root_sha256 == manifest.lineage_root_sha256
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in source.state_dict()
    )

    legacy = tmp_path / "legacy.pt"
    torch.save(source.state_dict(), legacy)
    legacy_target = MultiresQ2BotPolicy()
    legacy_optimizer = torch.optim.Adam(legacy_target.parameters(), lr=1e-4)
    with pytest.raises(LineageError, match="legacy/raw"):
        load_attested_checkpoint(
            legacy,
            legacy_target,
            expected_atlas_catalog_sha256=catalog,
            expected_runtime_manifest_sha256=runtime,
            expected_training_config=TRAINING_CONFIG,
            optimizer=legacy_optimizer,
        )


def test_ppo_kernel_accepts_three_way_posture_and_rejects_binary_legacy_shape():
    torch.manual_seed(4)
    policy = MultiresQ2BotPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    observations = torch.zeros(2, 3, OBS_DIM)
    with torch.no_grad():
        parameters, values, _ = policy(observations)
        actions = torch.zeros(2, 3, 8)
        actions[..., 4] = POSTURE_NEUTRAL
        log_probabilities, _entropy = policy.action_log_prob_entropy(
            parameters,
            actions,
            fire_allowed=torch.ones(2, 3, dtype=torch.bool),
            obs=observations,
        )
    batch = MultiresRolloutBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=log_probabilities,
        old_values=values.squeeze(-1),
        advantages=torch.linspace(-1.0, 1.0, 6).reshape(2, 3),
        returns=values.squeeze(-1) + 0.1,
        fire_allowed=torch.ones(2, 3, dtype=torch.bool),
        valid=torch.ones(2, 3, dtype=torch.bool),
        recurrent_reset=torch.tensor(
            [[True, False, False], [True, False, True]], dtype=torch.bool
        ),
    )
    metrics = MultiresPPOTrainer(
        policy, optimizer, MultiresPPOConfig(epochs=1)
    ).update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["optimizer_steps"] == 1

    batch.actions = torch.zeros(2, 3, 7)
    with pytest.raises(ValueError, match="actions must have shape"):
        batch.validate()
