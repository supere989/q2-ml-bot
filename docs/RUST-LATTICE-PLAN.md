# Rust Lattice Optimization and Distributed Rollouts

Branch: `feature/rust-lattice`. Merge target: `master` only after the gates
below pass. The active trainer continues using the published Python code.

## Measured optimization sequence

The original lattice performed five full nearest-channel scans inside
`memory_features()`, called that method twice per transition, and performed a
separate death scan. At roughly 287 cells this cost about 1.52 ms per feature
call and 1.23 ms for five direct searches.

The branch now:

- computes engagement, threat, opportunity, self-fire, and deaths in one pass;
- caches the exact same-tick 24-dimensional feature vector;
- refreshes item readiness every five ticks unless a pickup or route-axis
  change forces an immediate update;
- services every deterministic cold-start ML socket until all slots reach one
  common engine tick.

Microbenchmark results at 287 cells:

- uncached Python features: 1.52 ms → 0.185 ms (8.2× faster);
- cached second request: approximately 0.14 µs;
- Rust packed nearest kernel: approximately 2.65 µs;
- Rust with a fresh Python dict→array pack: approximately 425 µs.

The Rust kernel is fast; Python packing is not. Rust must own incrementally
updated cell state before it is enabled in `harness/spatial.py`.

## Real-engine proof

All tests used the canonical fixed runtime on isolated ports while the active
trainer remained untouched:

| Topology | Result |
|---|---:|
| 1 server × 1 ML slot | 40–50 SPS |
| 1 server × 4 ML slots | 145–157 SPS |
| 4 servers × 4 ML slots | 97–99 SPS |

Every deterministic four-slot server aligned all slots at tick 134. The
previous stagger (`123/112/101`) and first-rollout deadlock did not recur.
A live four-slot rotation from `mltrain_00005208` to
`mltrain_00005209` also re-aligned every slot at one tick without restarting
q2ded. The rotation path drains queued old-map datagrams before issuing the
command; it does not assume that new-map frame numbers must be smaller than
old-map frame numbers.

## Rust ownership milestone

The branch now includes a per-environment sparse index:

```text
LatticeIndex.upsert/remove/apply_packed(changed_cells)
LatticeIndex.features(position, current_cell, survivability) -> [f32; 24]
LatticeIndex.dumps/loads()
```

`Q2_RUST_LATTICE=1` enables the stateful path. Python sends only dirty cells;
the Rust index owns all unchanged cells and releases the GIL for queries and
snapshot encoding. If the extension is missing or rejects a query, the reward
instance records the reason and falls back to Python without changing the
observation shape. Existing Python JSON/gzip checkpoints remain authoritative;
restoring one invalidates the derived Rust index and rebuilds it lazily.

At 400 populated cells, the measured boundary costs are:

- optimized Python five-channel traversal: approximately 572 µs;
- Rust query with a full Python repack: approximately 574 µs (rejected design);
- stateful Rust 24-float policy tail: approximately 7.4 µs;
- incremental one-cell update: approximately 2.3 µs.

An isolated one-server/four-ML q2ded A/B with 400 cells per bot sustained
approximately 608 transitions/sec on Python and 1,797 transitions/sec on the
stateful Rust path (about 3×). Across 600 live per-frame comparisons, maximum
absolute feature error was `5.96e-08`; no slot fell back.

A CUDA PPO smoke run then completed one 64-transition update with four ML
slots, saved its PyTorch checkpoint and ONNX export, and exited cleanly. A
subsequent live map rotation retained separate Rust indices for both maps and
re-aligned all four slots at tick 134 without falling back or restarting.

Remaining ownership work is event-native score accumulation and readiness
refresh inside Rust. The present bridge still derives changed-cell scores in
Python, but its cost scales with changed cells rather than total lattice size.

## Multi-host rollout design

Rust runs locally on each rollout host beside q2ded. Never make per-frame LAN
calls to a centralized lattice service. The RTX/WSL machine remains the PPO
learner; procreator and nobara can run versioned rollout workers and upload
complete trajectory batches.

The first distributed implementation should be synchronous PPO:

1. Learner publishes policy version N.
2. Workers load N and collect fixed-size recurrent rollouts.
3. Workers upload batches tagged N.
4. Learner accepts a quorum, rejects stale versions, and performs one update.
5. Version N+1 is broadcast.

Batch transport can use ZeroMQ or gRPC over the LAN. Policy inference and Rust
lattice state stay local to each worker. Fully asynchronous rollout requires
V-trace/APPO-style corrections and is explicitly out of scope for the first
prototype.

## Merge gates

1. Randomized and recorded feature/reward parity within documented tolerance.
   **Prototype passed** for randomized cells and 600 live frames.
2. Rust state save/load round-trip and Python fallback compatibility.
   **Prototype passed** with deterministic binary snapshots and authoritative
   Python checkpoint rebuilds.
3. No observation or checkpoint-shape change.
4. End-to-end SPS improvement over optimized Python, including boundary cost.
   **Prototype passed** on the populated four-slot isolated A/B (~3×).
5. Deterministic same-seed rollout hashes where supported.
6. Fixed combat, aim-preservation, and lattice-direction regression gates.
