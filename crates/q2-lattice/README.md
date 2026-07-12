# q2-lattice

Rust hot-path kernels for the q2-ml-bot Vector Lattice. The crate is an
experiment on `feature/rust-lattice`; Python remains the authoritative engine
until the following merge gates pass:

1. Exact channel-selection parity with `VoxelSpatialReward` on recorded and
   randomized cell sets.
2. No observation-shape or checkpoint change: output remains five nearest
   signals feeding the existing 24-dimensional memory tail.
3. End-to-end generated-map reward/feature parity within float tolerance.
4. Higher real-engine SPS than the optimized Python single-pass path.
5. Clean behavior/combat regression gates before enabling Rust by default.

Build the Python extension with Maturin:

```bash
python -m pip install maturin
maturin develop --manifest-path crates/q2-lattice/Cargo.toml --features python
```

The initial packed cell ABI is nine `float32` values per row:

```text
voxel_x, voxel_y, voxel_z,
engagement, threat, opportunity, self_fire, deaths,
confidence
```

## First benchmark

On the procreator host with 287 populated cells:

| Path | Time per five-channel query |
|---|---:|
| Original Python repeated scans | ~1,229 µs |
| Optimized Python single pass | ~173 µs |
| Rust over an already-packed array | ~2.65 µs |
| Rust plus Python dict→array repack | ~425 µs |

The kernel is about 65× faster than optimized Python, but a naive per-tick
binding is slower because packing dominates. Therefore this extension is not
connected to the live path yet. The next Rust milestone is a stateful index
that owns cells and accepts incremental deposits/updates, allowing queries to
cross the Python boundary without rebuilding the map.
