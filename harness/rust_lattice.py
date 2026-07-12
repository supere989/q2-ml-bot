"""Optional adapter for the experimental q2-lattice Rust extension.

This is intentionally not enabled in the live spatial path yet. Packing a
Python dict into an ndarray on every tick could erase the kernel win; the Rust
engine must own or incrementally update packed state before it becomes the
default. This adapter exists for parity and boundary-cost benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .protocol import Observation
    from .spatial import VoxelSpatialReward

try:
    import q2_lattice_rs
except ImportError:
    q2_lattice_rs = None


CHANNELS = ("engagement", "threat", "opportunity", "self_fire", "deaths")
PACKED_CELL_WIDTH = 9


def available() -> bool:
    return q2_lattice_rs is not None


def pack_cells(reward: "VoxelSpatialReward") -> np.ndarray:
    """Pack Python session memory into the Rust prototype's stable row ABI."""
    memory = reward._memory_for_map(reward.map_name)
    packed = np.empty((len(memory), PACKED_CELL_WIDTH), dtype=np.float32)
    for index, (cell, entry) in enumerate(memory.items()):
        packed[index, :3] = cell
        packed[index, 3:8] = [
            reward._memory_score(entry, kind) for kind in CHANNELS
        ]
        packed[index, 8] = reward._memory_confidence(entry)
    return packed


def nearest_signals(
    reward: "VoxelSpatialReward", obs: "Observation"
) -> dict[str, tuple[float, float, float, float]]:
    if q2_lattice_rs is None:
        raise RuntimeError(
            "q2_lattice_rs is not installed; build crates/q2-lattice with Maturin"
        )
    result = q2_lattice_rs.nearest_signals(
        pack_cells(reward),
        tuple(float(value) for value in obs.self_state[:3]),
        float(reward.session_memory_search_radius),
        float(reward.voxel_size),
        float(reward.session_memory_score_scale),
    )
    array = np.asarray(result, dtype=np.float32)
    if array.shape != (len(CHANNELS), 4):
        raise RuntimeError(f"unexpected Rust lattice output shape: {array.shape}")
    return {
        kind: tuple(float(value) for value in array[index])
        for index, kind in enumerate(CHANNELS)
    }
