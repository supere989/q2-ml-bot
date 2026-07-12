from types import SimpleNamespace

import numpy as np
import pytest

q2_lattice_rs = pytest.importorskip("q2_lattice_rs")

from harness.rust_lattice import (
    CHANNELS,
    StatefulLatticeIndex,
    nearest_signals,
    pack_cells,
)
from harness.spatial import (
    SessionMemoryCell,
    VoxelSpatialReward,
    load_lattice_state,
    save_lattice_state,
)


def _obs(position=(0.0, 0.0, 0.0)):
    state = np.zeros(10, dtype=np.float32)
    state[:3] = position
    state[6] = 100.0
    state[9] = 10.0
    return SimpleNamespace(self_state=state)


def test_rust_nearest_signals_match_python_randomized():
    rng = np.random.default_rng(8128)
    reward = VoxelSpatialReward.from_env(seed=8128)
    reward.map_name = "parity"
    memory = reward._memory_for_map("parity")
    for _ in range(400):
        cell = tuple(int(value) for value in rng.integers(-10, 11, size=3))
        memory[cell] = SessionMemoryCell(
            engagement_count=float(rng.uniform(0, 8)),
            enemy_seen=float(rng.uniform(0, 5)),
            damage_dealt=float(rng.uniform(0, 200)),
            damage_taken=float(rng.uniform(0, 200)),
            kills=float(rng.integers(0, 4)),
            deaths=float(rng.integers(0, 4)),
            self_fire=float(rng.uniform(0, 5)),
            hook_engagement=float(rng.uniform(0, 3)),
            item_contested=float(rng.uniform(0, 3)),
            successful_escape=float(rng.uniform(0, 3)),
        )
    obs = _obs((128.0, -256.0, 64.0))

    python_result = reward._nearest_memory_signals(obs, CHANNELS)
    rust_result = nearest_signals(reward, obs)

    assert pack_cells(reward).shape == (len(memory), 9)
    for kind in CHANNELS:
        assert rust_result[kind] == pytest.approx(
            python_result[kind], rel=2e-5, abs=2e-5
        )


def test_rust_binding_rejects_wrong_cell_width():
    with pytest.raises(ValueError, match="shape"):
        q2_lattice_rs.nearest_signals(
            np.zeros((2, 8), dtype=np.float32),
            (0.0, 0.0, 0.0),
            2048.0,
            256.0,
            8.0,
        )


def test_rust_binding_rejects_noncontiguous_cells():
    cells = np.zeros((2, 18), dtype=np.float32)[:, ::2]
    assert cells.shape == (2, 9)
    assert not cells.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        q2_lattice_rs.nearest_signals(
            cells,
            (0.0, 0.0, 0.0),
            2048.0,
            256.0,
            8.0,
        )


def test_stateful_index_matches_full_python_features_and_incremental_updates():
    reward = VoxelSpatialReward.from_env(seed=44)
    reward.map_name = "stateful"
    memory = reward._memory_for_map(reward.map_name)
    memory[(0, 0, 0)] = SessionMemoryCell(
        engagement_count=3.0,
        damage_taken=40.0,
        item_contested=2.0,
        self_fire=1.0,
    )
    memory[(2, -1, 0)] = SessionMemoryCell(
        kills=2.0,
        damage_dealt=75.0,
        successful_escape=1.0,
    )
    obs = _obs((64.0, 64.0, 64.0))
    obs.tick = 10
    obs.rune_flags = np.zeros(5, dtype=np.float32)
    obs.entities = np.zeros((0, 10), dtype=np.float32)
    obs.entity_count = 0

    stateful = StatefulLatticeIndex(reward)
    expected = reward.memory_features(obs)
    actual = stateful.features(reward, obs)
    assert len(stateful) == len(memory)
    assert actual == pytest.approx(expected, rel=2e-5, abs=2e-5)

    memory[(0, 0, 0)].damage_taken += 80.0
    assert stateful.apply_cells(reward, [(0, 0, 0)]) == 1
    reward._invalidate_feature_cache()
    assert stateful.features(reward, obs) == pytest.approx(
        reward.memory_features(obs), rel=2e-5, abs=2e-5
    )


def test_stateful_index_snapshot_and_removal_round_trip():
    reward = VoxelSpatialReward.from_env(seed=45)
    reward.map_name = "snapshot"
    reward._memory_for_map(reward.map_name)[(1, 2, 3)] = SessionMemoryCell(
        kills=1.0, deaths=2.0
    )
    stateful = StatefulLatticeIndex(reward)
    encoded = stateful.dumps()
    restored = StatefulLatticeIndex.loads(reward, encoded)
    assert restored.dumps() == encoded
    assert len(restored) == 1
    assert restored.remove_cells([(1, 2, 3)]) == 1
    assert len(restored) == 0
    with pytest.raises(ValueError, match="header"):
        StatefulLatticeIndex.loads(reward, b"bad")


def test_opt_in_spatial_path_flushes_only_dirty_cells():
    reward = VoxelSpatialReward.from_env(seed=46)
    reward.map_name = "opt_in"
    reward.rust_lattice_enabled = True
    reward._memory_for_map(reward.map_name)[(0, 0, 0)] = SessionMemoryCell(
        engagement_count=2.0
    )
    obs = _obs((64.0, 64.0, 64.0))
    obs.tick = 20
    obs.rune_flags = np.zeros(5, dtype=np.float32)
    obs.entities = np.zeros((0, 10), dtype=np.float32)
    obs.entity_count = 0

    rust_features = reward.memory_features(obs).copy()
    assert len(reward._rust_indices[reward.map_name]) == 1
    reward.rust_lattice_enabled = False
    reward._invalidate_feature_cache()
    python_features = reward.memory_features(obs).copy()
    assert rust_features == pytest.approx(python_features, rel=2e-5, abs=2e-5)

    reward.rust_lattice_enabled = True
    changed = reward._memory_cell((0, 0, 0), obs.tick)
    changed.damage_taken += 100.0
    reward._invalidate_feature_cache()
    rust_changed = reward.memory_features(obs).copy()
    reward.rust_lattice_enabled = False
    reward._invalidate_feature_cache()
    assert rust_changed == pytest.approx(
        reward.memory_features(obs), rel=2e-5, abs=2e-5
    )


def test_opt_in_spatial_path_falls_back_when_extension_unavailable(monkeypatch):
    import harness.rust_lattice as adapter

    monkeypatch.setattr(adapter, "q2_lattice_rs", None)
    reward = VoxelSpatialReward.from_env(seed=47)
    reward.map_name = "fallback"
    reward.rust_lattice_enabled = True
    obs = _obs()
    obs.tick = 30
    obs.rune_flags = np.zeros(5, dtype=np.float32)
    obs.entities = np.zeros((0, 10), dtype=np.float32)
    obs.entity_count = 0
    assert reward.memory_features(obs).shape == (24,)
    assert not reward.rust_lattice_enabled
    assert "not installed" in reward._rust_fallback_reason


def test_python_checkpoint_remains_authoritative_for_rust_index(tmp_path):
    original = VoxelSpatialReward.from_env(seed=48)
    original.map_name = "checkpoint"
    original.rust_lattice_enabled = True
    original._memory_for_map(original.map_name)[(3, 2, 1)] = SessionMemoryCell(
        kills=2.0, damage_dealt=50.0, prior_opportunity=1.5
    )
    path = save_lattice_state([original], tmp_path / "lattice.json.gz", 123)

    restored = VoxelSpatialReward.from_env(seed=48)
    restored.rust_lattice_enabled = True
    result = load_lattice_state([restored], path)
    restored.map_name = "checkpoint"
    obs = _obs((0.0, 0.0, 0.0))
    obs.tick = 40
    obs.rune_flags = np.zeros(5, dtype=np.float32)
    obs.entities = np.zeros((0, 10), dtype=np.float32)
    obs.entity_count = 0

    assert result["env_steps"] == 123
    rust_features = restored.memory_features(obs).copy()
    restored.rust_lattice_enabled = False
    restored._invalidate_feature_cache()
    assert rust_features == pytest.approx(
        restored.memory_features(obs), rel=2e-5, abs=2e-5
    )
