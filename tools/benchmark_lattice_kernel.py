#!/usr/bin/env python3
"""Reproduce Python/Rust nearest-signal boundary benchmarks."""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.rust_lattice import CHANNELS, available, nearest_signals, pack_cells
from harness.spatial import SessionMemoryCell, VoxelSpatialReward


def timed(label, iterations, function):
    for _ in range(min(100, iterations)):
        function()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    elapsed = time.perf_counter() - started
    print(f"{label:28s} {elapsed / iterations * 1e6:9.2f} us/call")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=287)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    reward = VoxelSpatialReward.from_env(seed=1)
    reward.map_name = "benchmark"
    memory = reward._memory_for_map(reward.map_name)
    width = max(1, int(np.ceil(args.cells ** (1.0 / 3.0))))
    for index in range(args.cells):
        cell = (
            index % width,
            (index // width) % width,
            index // (width * width),
        )
        memory[cell] = SessionMemoryCell(
            engagement_count=1.0 + index % 5,
            enemy_seen=float(index % 3),
            damage_dealt=float((index * 3) % 100),
            damage_taken=float(index % 100),
            kills=float(index % 11 == 0),
            deaths=float(index % 13 == 0),
            self_fire=float(index % 4),
        )
    state = np.zeros(10, dtype=np.float32)
    state[6] = 100.0
    state[9] = 10.0
    obs = SimpleNamespace(self_state=state)

    timed(
        "python single pass",
        args.iterations,
        lambda: reward._nearest_memory_signals(obs, CHANNELS),
    )
    if not available():
        print("rust extension unavailable (build q2-lattice with feature 'python')")
        return

    import q2_lattice_rs

    packed = pack_cells(reward)
    rust_args = (
        packed,
        (0.0, 0.0, 0.0),
        reward.session_memory_search_radius,
        reward.voxel_size,
        reward.session_memory_score_scale,
    )
    timed(
        "rust packed kernel",
        args.iterations,
        lambda: q2_lattice_rs.nearest_signals(*rust_args),
    )
    timed(
        "rust with Python repack",
        max(100, args.iterations // 5),
        lambda: nearest_signals(reward, obs),
    )


if __name__ == "__main__":
    main()
