"""Optional adapter for the experimental q2-lattice Rust extension.

The stateful adapter owns a Rust-side cell index and transfers only changed
cells. It remains opt-in while rollout parity and combat regression gates run.
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
SCORE_EVENT_WIDTH = 11


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


class StatefulLatticeIndex:
    """Incremental Python facade over ``q2_lattice_rs.LatticeIndex``."""

    def __init__(self, reward: "VoxelSpatialReward", sync: bool = True):
        if q2_lattice_rs is None:
            raise RuntimeError(
                "q2_lattice_rs is not installed; build q2-lattice with Maturin"
            )
        self._index = q2_lattice_rs.LatticeIndex(
            float(reward.session_memory_search_radius),
            float(reward.voxel_size),
            float(reward.session_memory_score_scale),
        )
        if sync:
            self.sync_full(reward)

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _pack_selected(
        reward: "VoxelSpatialReward", cells
    ) -> np.ndarray:
        memory = reward._memory_for_map(reward.map_name)
        selected = [(cell, memory.get(cell)) for cell in cells]
        selected = [(cell, entry) for cell, entry in selected if entry is not None]
        packed = np.empty((len(selected), PACKED_CELL_WIDTH), dtype=np.float32)
        for index, (cell, entry) in enumerate(selected):
            packed[index, :3] = cell
            packed[index, 3:8] = [
                reward._memory_score(entry, kind) for kind in CHANNELS
            ]
            packed[index, 8] = reward._memory_confidence(entry)
        return packed

    def sync_full(self, reward: "VoxelSpatialReward") -> int:
        self._index.clear()
        packed = pack_cells(reward)
        return int(self._index.apply_packed(packed))

    def apply_cells(self, reward: "VoxelSpatialReward", cells) -> int:
        packed = self._pack_selected(reward, cells)
        if not len(packed):
            return 0
        return int(self._index.apply_packed(packed))

    def apply_score_events(self, events) -> int:
        """Apply coalesced score/sample/confidence events for changed cells."""
        rows = list(events.items())
        if not rows:
            return 0
        packed = np.empty((len(rows), SCORE_EVENT_WIDTH), dtype=np.float32)
        for index, (cell, delta) in enumerate(rows):
            packed[index, :3] = cell
            packed[index, 3:] = delta
        return int(self._index.apply_score_events(packed))

    def remove_cells(self, cells) -> int:
        return sum(
            bool(self._index.remove(tuple(int(value) for value in cell)))
            for cell in cells
        )

    def nearest_signals(
        self, obs: "Observation"
    ) -> dict[str, tuple[float, float, float, float]]:
        result = np.asarray(
            self._index.nearest_signals(
                tuple(float(value) for value in obs.self_state[:3])
            ),
            dtype=np.float32,
        )
        return {
            kind: tuple(float(value) for value in result[index])
            for index, kind in enumerate(CHANNELS)
        }

    def features(self, reward: "VoxelSpatialReward", obs: "Observation") -> np.ndarray:
        features, _ = self.features_with_deaths(reward, obs)
        return features

    def features_with_deaths(
        self, reward: "VoxelSpatialReward", obs: "Observation"
    ) -> tuple[np.ndarray, float]:
        position = tuple(float(value) for value in obs.self_state[:3])
        current = tuple(int(value) for value in reward.cell_for(obs))
        bundle = np.asarray(
            self._index.feature_bundle(position, current, (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        features = bundle[:24]
        features[21:24] = reward._win_margin(obs, float(features[16]))
        return features, float(bundle[24])

    def dumps(self) -> bytes:
        return bytes(self._index.dumps())

    @classmethod
    def loads(cls, reward: "VoxelSpatialReward", data: bytes):
        if q2_lattice_rs is None:
            raise RuntimeError("q2_lattice_rs is not installed")
        instance = cls.__new__(cls)
        instance._index = q2_lattice_rs.LatticeIndex.loads(
            data,
            float(reward.session_memory_search_radius),
            float(reward.voxel_size),
            float(reward.session_memory_score_scale),
        )
        return instance
