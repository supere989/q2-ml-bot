from types import SimpleNamespace

import numpy as np
import pytest

q2_lattice_rs = pytest.importorskip("q2_lattice_rs")

from harness.rust_lattice import CHANNELS, nearest_signals, pack_cells
from harness.spatial import SessionMemoryCell, VoxelSpatialReward


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
