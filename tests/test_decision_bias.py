"""Fight decision bias tests.

The bias is advisory: reward shaping + typed lattice deposits only. It must
never mask or alter actions, must respect the lattice_reward gate, and its
decision-quality deposits must be typed (bad_fight_taken / good_disengage),
checkpoint-stable, and readable only through the existing threat channel.
"""

import copy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from harness.spatial import (
    SessionMemoryCell,
    VoxelSpatialReward,
    load_lattice_state,
    save_lattice_state,
)


def _obs(*, pos=(0.0, 0.0, 0.0), tick=0, health=100.0, entities=None,
         entity_count=0, damage_taken=0.0, deaths=0.0, self_exposure=0.0,
         hook_zone_count=0, fired=False):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = health
    self_state[7] = 50.0
    self_state[9] = 10.0
    if entities is None:
        entities = np.zeros((8, 9), dtype=np.float32)
    action_debug = np.zeros(12, dtype=np.float32)
    action_debug[9] = 1.0 if fired else 0.0  # engine-applied fire echo
    return SimpleNamespace(
        tick=tick,
        yaw=0.0,
        pitch=0.0,
        self_state=self_state,
        entities=entities,
        entity_count=entity_count,
        entity_debug=np.zeros((8, 4), dtype=np.uint32),
        rune_flags=np.zeros(5, dtype=np.float32),
        audio=np.zeros(5, dtype=np.float32),
        action_debug=action_debug,
        hook_zones=np.zeros((4, 8), dtype=np.float32),
        hook_zone_count=hook_zone_count,
        inbound_dmg_dist=-1.0,
        inbound_dmg_recency=0.0,
        is_terminal=False,
        terminal_reason=0,
        reward_damage_dealt=0.0,
        reward_damage_taken=damage_taken,
        reward_kill=0.0,
        reward_death=deaths,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
        last_damage_mod=0,
        last_death_mod=0,
        last_hit_target_edict=0,
        last_hit_target_epoch=0,
        self_exposure=self_exposure,
    )


def _enemy(index=0, rel=(300.0, 0.0, 0.0), exposure=0.5):
    entities = np.zeros((8, 9), dtype=np.float32)
    entities[index] = (*rel, 0.0, 0.0, 0.0, 100.0, 1.0, exposure)
    return entities


def _two_enemies():
    entities = _enemy(0, rel=(300.0, 0.0, 0.0), exposure=0.8)
    entities[1] = (0.0, 250.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 0.6)
    return entities


def _cell_of(reward, pos=(0.0, 0.0, 0.0)):
    return reward.cell_for(_obs(pos=pos))


# ── bias sign per component ──────────────────────────────────────────────

def test_winning_margin_makes_bias_positive():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    # Enemy DPS is zero and we are fully geared: margin projection is +1.
    _, info = reward.update(_obs(
        tick=1, entities=_enemy(), entity_count=1, self_exposure=0.0,
    ))
    assert info["bias_margin"] == pytest.approx(0.5)  # w_margin * (+1)
    assert info["fight_bias"] > 0.3


def test_exposed_outnumbered_hazardous_makes_bias_negative():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    # Two visible enemies who see me fully while I barely see them.
    reward._dps_enemy = 30.0  # measured incoming DPS sinks the projection
    _, info = reward.update(_obs(
        tick=1, entities=_two_enemies(), entity_count=2, self_exposure=1.0,
    ))
    assert info["bias_exposure"] < 0.0
    assert info["bias_outnumbered"] == pytest.approx(-0.15)
    assert info["fight_bias"] < -0.2


def test_hazard_history_lowers_bias_vs_combat_only_history():
    hazard_reward = VoxelSpatialReward()
    hazard_reward.reset("biasmap", _obs(tick=0))
    cell = _cell_of(hazard_reward)
    hazard_reward._memory_for_map("biasmap")[cell] = SessionMemoryCell(
        hazard_damage=150.0, hazard_deaths=2.0,
    )
    combat_reward = VoxelSpatialReward()
    combat_reward.reset("biasmap", _obs(tick=0))
    combat_reward._memory_for_map("biasmap")[cell] = SessionMemoryCell(
        damage_taken=150.0, deaths=2.0,
    )
    _, hazard_info = hazard_reward.update(_obs(tick=1))
    _, combat_info = combat_reward.update(_obs(tick=1))
    assert hazard_info["bias_hazard"] < 0.0
    assert combat_info["bias_hazard"] == 0.0
    assert hazard_info["fight_bias"] < combat_info["fight_bias"]


# ── reward terms ─────────────────────────────────────────────────────────

def _negative_bias_state(reward, tick):
    reward._dps_enemy = 30.0
    return _obs(
        tick=tick, entities=_two_enemies(), entity_count=2,
        self_exposure=1.0,
    )


def test_bad_fight_penalty_fires_when_engaging_at_negative_bias():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    obs = _negative_bias_state(reward, 1)
    obs.action_debug[9] = 1.0  # fired at the visible enemy
    _, info = reward.update(obs)
    assert info["fight_bias"] < -0.2
    assert info["bias_engaging"] == 1.0
    assert info["bad_fight_penalty"] == pytest.approx(
        reward.bad_fight_penalty * abs(info["fight_bias"])
    )
    assert reward.episode_bad_fights == 1.0


def test_disengage_reward_fires_when_enemy_sight_drops():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    reward.update(_negative_bias_state(reward, 1))  # enemies visible
    reward._dps_enemy = 30.0
    # Tick 2: sight dropped (last_visible_count=1 -> 0), hazard pushes the
    # bias negative even with no entities on screen.
    cell = _cell_of(reward)
    reward._memory_for_map("biasmap")[cell] = SessionMemoryCell(
        hazard_damage=200.0, hazard_deaths=3.0,
    )
    _, info = reward.update(_obs(tick=2))
    assert info["fight_bias"] < -0.2
    assert info["disengage_reward"] == pytest.approx(
        reward.disengage_reward * abs(info["fight_bias"])
    )
    assert reward.episode_good_disengages == 1.0
    entry = reward._memory_for_map("biasmap")[cell]
    assert entry.good_disengage == 1.0


def test_good_fight_reward_fires_when_engaging_at_positive_bias():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    obs = _obs(tick=1, entities=_enemy(exposure=0.9), entity_count=1,
               self_exposure=0.0, hook_zone_count=1)
    obs.action_debug[9] = 1.0
    _, info = reward.update(obs)
    assert info["fight_bias"] > 0.3
    assert info["good_fight_reward"] == pytest.approx(
        reward.good_fight_reward * info["fight_bias"]
    )


def test_reward_terms_respect_lattice_reward_gate():
    reward = VoxelSpatialReward(lattice_reward_enabled=False)
    reward.reset("biasmap", _obs(tick=0))
    obs = _negative_bias_state(reward, 1)
    obs.action_debug[9] = 1.0
    _, info = reward.update(obs)
    # Metric stays observable; reward contributions gate off.
    assert info["fight_bias"] < -0.2
    assert info["bad_fight_penalty"] == 0.0
    assert info["good_fight_reward"] == 0.0
    assert reward.episode_bad_fights == 0.0


# ── action authority ─────────────────────────────────────────────────────

def test_update_never_modifies_obs_and_returns_only_reward_and_info():
    reward = VoxelSpatialReward()
    obs = _obs(tick=0, entities=_two_enemies(), entity_count=2,
               self_exposure=0.7, hook_zone_count=1, fired=True)
    reward.reset("biasmap", obs)
    obs = _obs(tick=1, entities=_two_enemies(), entity_count=2,
               self_exposure=0.7, hook_zone_count=1, fired=True)
    snapshot = copy.deepcopy(obs)
    result = reward.update(obs)
    assert isinstance(result, tuple) and len(result) == 2
    bonus, info = result
    assert isinstance(bonus, float)
    assert isinstance(info, dict)
    # obs is untouched: no fire masking, no movement override, no writes.
    assert obs.tick == snapshot.tick
    np.testing.assert_array_equal(obs.self_state, snapshot.self_state)
    np.testing.assert_array_equal(obs.entities, snapshot.entities)
    np.testing.assert_array_equal(obs.action_debug, snapshot.action_debug)
    assert obs.entity_count == snapshot.entity_count
    assert obs.self_exposure == snapshot.self_exposure


# ── deposits + threat read + checkpoint ──────────────────────────────────

def test_death_at_negative_bias_deposits_bad_fight_taken_at_death_cell():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    reward._dps_enemy = 30.0
    _, pre = reward.update(_obs(
        tick=1, entities=_two_enemies(), entity_count=2, self_exposure=1.0,
        deaths=1.0, health=0.0,
    ))
    assert pre["fight_bias"] < -0.2
    entry = reward._memory_for_map("biasmap")[_cell_of(reward)]
    assert entry.bad_fight_taken == 1.0
    # The deposit feeds only the existing threat channel: same cell without
    # the decision deposit reads exactly R_BAD_FIGHT_THREAT_WEIGHT lower.
    twin = copy.copy(entry)
    twin.bad_fight_taken = 0.0
    assert reward._threat_score(entry) == pytest.approx(
        reward._threat_score(twin) + reward.bad_fight_threat_weight * 1.0
    )


def test_death_at_positive_bias_leaves_no_bad_fight_deposit():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    # Winning projection, no hazards: death is a combat loss, not a bad call.
    reward.update(_obs(tick=1, deaths=1.0, health=0.0))
    entry = reward._memory_for_map("biasmap")[_cell_of(reward)]
    assert entry.bad_fight_taken == 0.0


def test_corpse_frames_read_zero_bias():
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    _, info = reward.update(_obs(tick=1, health=0.0))
    assert info["fight_bias"] == 0.0
    assert info["bad_fight_penalty"] == 0.0


def test_checkpoint_round_trip_preserves_decision_fields(tmp_path):
    reward = VoxelSpatialReward()
    reward.reset("biasmap", _obs(tick=0))
    reward._memory_for_map("biasmap")[(1, 2, 3)] = SessionMemoryCell(
        bad_fight_taken=4.0, good_disengage=2.0,
    )
    path = save_lattice_state([reward], tmp_path / "lattice.json.gz")
    fresh = VoxelSpatialReward()
    load_lattice_state([fresh], path)
    entry = fresh._memory_for_map("biasmap")[(1, 2, 3)]
    assert entry.bad_fight_taken == 4.0
    assert entry.good_disengage == 2.0


def test_old_checkpoint_without_decision_fields_loads(tmp_path):
    path = tmp_path / "old.json.gz"
    payload = {
        "version": 1,
        "env_steps": 5,
        "instances": [{"maps": {"biasmap": [
            {"cell": [0, 0, 0], "deaths": 1.0},
        ]}}],
    }
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(payload))
    reward = VoxelSpatialReward()
    load_lattice_state([reward], path)
    entry = reward._memory_for_map("biasmap")[(0, 0, 0)]
    assert entry.bad_fight_taken == 0.0
    assert entry.good_disengage == 0.0
