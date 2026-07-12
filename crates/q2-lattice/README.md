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
binding is slower because packing dominates.

## Stateful prototype

`q2_lattice_rs.LatticeIndex` now owns a deterministic sparse cell map. It
supports incremental upsert/remove operations, GIL-free nearest-signal and
24-float feature queries, and a versioned deterministic binary snapshot. The
Python harness enables it only with `Q2_RUST_LATTICE=1` and automatically
falls back when the extension is unavailable.

Combat and readiness updates are coalesced per voxel and applied as additive
score/sample events. This keeps Python's checkpoint cells as the compatibility
oracle while Rust owns live accumulation and confidence evolution.

With 400 populated cells, a complete stateful policy-tail query is ~7.4 µs,
a full one-cell replacement is ~2.4 µs, and an event-native update is ~0.9 µs.
An isolated four-slot q2ded A/B measured ~607 transitions/sec for optimized
Python and ~1,878 for event-native Rust. A live run applied 300 event rows with
maximum feature error `1.79e-07` over 600 comparisons.
