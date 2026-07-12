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

The next crate API should own a per-environment sparse index:

```text
LatticeIndex.apply(events/deposits)
LatticeIndex.refresh_readiness(tick, route_state)
LatticeIndex.features(position, survivability) -> [f32; 24]
LatticeIndex.save/load()
```

Python should send compact events, not repack every cell. PyO3 methods must
release the GIL for batch updates/queries. The Python implementation remains a
parity oracle and fallback.

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
2. Rust state save/load round-trip and Python fallback compatibility.
3. No observation or checkpoint-shape change.
4. End-to-end SPS improvement over optimized Python, including boundary cost.
5. Deterministic same-seed rollout hashes where supported.
6. Fixed combat, aim-preservation, and lattice-direction regression gates.
