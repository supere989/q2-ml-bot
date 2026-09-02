"""Role-gate integrity tests for the lattice refactor.

Design contract: one physical store, per-role channels, gates at the
consumption points. Storage (deposits) stays on during ablations; the
policy-input tail, the lattice-derived reward terms, the immediate
engagement slice, and external directives gate independently.
"""

import json
from types import SimpleNamespace

import numpy as np

from harness.spatial import (
    IMMEDIATE_ENGAGEMENT_SLICE,
    OBS_SESSION_MEMORY_DIM,
    VoxelSpatialReward,
    load_lattice_state,
    save_lattice_state,
)


def _obs(*, pos=(0.0, 0.0, 0.0), tick=0, health=100.0, entities=None,
         entity_count=0, damage_dealt=0.0, kills=0.0):
    self_state = np.zeros(10, dtype=np.float32)
    self_state[:3] = pos
    self_state[6] = health
    self_state[7] = 50.0
    self_state[9] = 10.0
    if entities is None:
        entities = np.zeros((8, 9), dtype=np.float32)
    return SimpleNamespace(
        tick=tick,
        yaw=0.0,
        pitch=0.0,
        self_state=self_state,
        entities=entities,
        entity_count=entity_count,
        rune_flags=np.zeros(5, dtype=np.float32),
        audio=np.zeros(5, dtype=np.float32),
        action_debug=np.zeros(12, dtype=np.float32),
        hook_zones=np.zeros((4, 8), dtype=np.float32),
        hook_zone_count=0,
        inbound_dmg_dist=-1.0,
        inbound_dmg_recency=0.0,
        is_terminal=False,
        reward_damage_dealt=damage_dealt,
        reward_damage_taken=0.0,
        reward_kill=kills,
        reward_death=0.0,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
    )


def _enemy_obs(tick=0, health=100.0):
    entities = np.zeros((8, 9), dtype=np.float32)
    entities[0] = (300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 1.0)
    return _obs(tick=tick, health=health, entities=entities, entity_count=1)


def _seeded_memory(reward, map_name="gatemap"):
    """One real deposit so later assertions have live state to read."""
    reward.reset(map_name, _obs(tick=0))
    reward._update_session_memory(
        obs=_enemy_obs(tick=1),
        cell=reward.cell_for(_obs()),
        visible_count=1,
        fired=True,
        hook_enemy=False,
        fire_audio_contact=False,
        audio_contact=False,
    )


def test_obs_gate_zeroes_public_tail_but_storage_accumulates():
    reward = VoxelSpatialReward(
        lattice_obs_enabled=False, lattice_preload_enabled=False,
        lattice_routes_enabled=False,
    )
    _seeded_memory(reward)
    memory = reward._memory_for_map("gatemap")
    assert len(memory) > 0, "storage must stay live under an obs ablation"

    public = reward.memory_features(_obs(tick=2))
    assert public.shape == (OBS_SESSION_MEMORY_DIM,)
    assert float(np.abs(public).sum()) == 0.0

    internal = reward._memory_features_internal(_obs(tick=2))
    assert float(np.abs(internal).sum()) > 0.0


def test_reward_gate_removes_lattice_terms_keeps_gameplay_terms():
    kwargs = dict(
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    on = VoxelSpatialReward(lattice_reward_enabled=True, **kwargs)
    off = VoxelSpatialReward(lattice_reward_enabled=False, **kwargs)
    for reward in (on, off):
        _seeded_memory(reward)

    # Same trajectory: step into a new cell with an engagement pull present.
    obs = _obs(pos=(400.0, 0.0, 0.0), tick=2)
    bonus_on, info_on = on.update(obs)
    bonus_off, info_off = off.update(obs)

    assert info_on["session_memory_bonus"] != 0.0
    assert info_off["session_memory_bonus"] == 0.0
    # Gameplay terms (exploration) survive the lattice ablation.
    assert info_off["voxel_new"] == 1.0
    assert bonus_off != bonus_on


def test_immediate_engagement_gate_zeroes_slice_regardless_of_producer():
    hot = _enemy_obs(tick=1)
    on = VoxelSpatialReward(
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    off = VoxelSpatialReward(
        immediate_engagement_enabled=False,
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    for reward in (on, off):
        reward.reset("gatemap", _obs(tick=0))

    tail_on = on.memory_features(hot)
    assert float(np.abs(tail_on[IMMEDIATE_ENGAGEMENT_SLICE]).sum()) > 0.0

    tail_off = off.memory_features(hot)
    assert float(np.abs(tail_off[IMMEDIATE_ENGAGEMENT_SLICE]).sum()) == 0.0
    # The rest of the tail is untouched by the slice gate.
    remaining = np.delete(tail_off, np.s_[IMMEDIATE_ENGAGEMENT_SLICE])
    assert remaining.shape == (OBS_SESSION_MEMORY_DIM - 4,)


def test_session_memory_zero_still_disables_everything_by_default(monkeypatch):
    monkeypatch.setenv("Q2_SESSION_MEMORY", "0")
    monkeypatch.delenv("Q2_LATTICE_OBS", raising=False)
    monkeypatch.delenv("Q2_LATTICE_REWARD", raising=False)
    reward = VoxelSpatialReward.from_env(seed=0)
    assert not reward.session_memory_enabled
    assert not reward.lattice_obs_enabled
    assert not reward.lattice_reward_enabled


def test_consumption_gates_override_session_memory_default(monkeypatch):
    monkeypatch.setenv("Q2_SESSION_MEMORY", "0")
    monkeypatch.setenv("Q2_LATTICE_OBS", "1")
    monkeypatch.setenv("Q2_LATTICE_REWARD", "1")
    reward = VoxelSpatialReward.from_env(seed=0)
    assert not reward.session_memory_enabled   # storage off
    assert reward.lattice_obs_enabled          # consumption independent
    assert reward.lattice_reward_enabled


def test_restore_remerges_priors_instead_of_shadowing(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_LATTICE_DIR", str(tmp_path))
    map_name = "restoremap"
    (tmp_path / f"{map_name}.lattice.json").write_text(json.dumps({
        "cell_size": 256,
        "objectives": [
            {"item": "item_quad", "x": 640, "y": 128, "z": 128, "value": 1.0}
        ],
        "danger": [],
        "spawns": [],
        "items": [],
    }))

    source = VoxelSpatialReward(
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    source.reset(map_name, _obs(tick=0))
    cell = source.cell_for(_obs())
    source._memory_cell(cell, 0).kills = 2.0
    path = save_lattice_state([source], tmp_path / "lattice.json.gz")

    restored = VoxelSpatialReward(
        lattice_preload_enabled=True, lattice_routes_enabled=False,
    )
    load_lattice_state([restored], path)
    assert map_name not in restored.preloaded_maps, (
        "restored maps must re-merge priors, not shadow them"
    )
    restored.reset(map_name, _obs(tick=0))
    merged = restored._memory_for_map(map_name)
    prior_cell = restored.cell_for_pos((640, 128, 128))
    assert merged[prior_cell].prior_opportunity > 0.0
    assert merged[cell].kills == 2.0


def test_directives_gated_and_logged():
    off = VoxelSpatialReward(
        directives_enabled=False,
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    off.reset("gatemap", _obs(tick=0))
    assert off.apply_directive("gatemap", "seek", 100, 100, 100) is False
    assert len(off._memory_for_map("gatemap")) == 0

    on = VoxelSpatialReward(
        lattice_preload_enabled=False, lattice_routes_enabled=False,
    )
    on.reset("gatemap", _obs(tick=0))
    assert on.apply_directive("gatemap", "seek", 100, 100, 100) is True
    cell = on.cell_for_pos((100, 100, 100))
    assert on._memory_for_map("gatemap")[cell].kills == 3.0
