# Distributed Trainer Cutover Assessment — 2026-07-11

## Decision

**No-go for immediate primary-trainer cutover. Ready for a shadow/canary
deployment.** The data plane and PPO integration work, but operational and
training-equivalence gaps could silently degrade the policy or strand the only
active run.

## What is proven

- Exact policy/config generation checks, stale rejection, quorum sealing, and
  pickle-free PPO schema validation pass.
- WSL and Nobara completed a real two-host synthetic LAN quorum with identical
  deterministic hashes.
- WSL completed a real CUDA/q2ded recurrent upload accepted by the strict PPO
  schema.
- The embedded learner consumed remote arrays with the existing PPO optimizer,
  published version 16, consumed a second batch from the same persistent q2ded
  and lattice, then saved policy version 32.
- A restarted one-ML deterministic audit reproduces exactly. Four-ML fresh
  launch replay does not, and is rejected.

## Cutover blockers

| Priority | Blocker | Evidence / consequence |
|---|---|---|
| P0 | Remote rollout hosts are not provisioned | Nobara has CUDA PyTorch but no q2ded/runtime checkout. Procreator has neither PyTorch nor q2ded. Only WSL can produce real batches today, so cutover adds protocol complexity without adding compute. |
| P0 | Remote telemetry is incomplete | The two-generation learner logged episode/base/spatial/KD fields as `nan` because batches do not carry episode summaries or behavior/outcome metrics. Aim, reward, curriculum, and regression drift would be substantially less visible than in the primary trainer. |
| P0 | Worker recovery is not learner-owned | Persistent workers checkpoint their lattice locally, but the learner does not publish/retain a lattice artifact or lease its version. Replacing a failed host can mix fresh and persistent spatial memory or lose learning state. |
| P0 | No worker lease/heartbeat/retry policy | A missing quorum currently times out and aborts training. Workers do not reconnect with bounded backoff, and the learner cannot replace a dead assignment while preserving seed/rollout identity. |
| P1 | Four-ML replay is nondeterministic | Two fresh four-slot runs diverged in observations, rewards, actions, and recurrent state even with seeded game/Python/Torch and deterministic actions. Normal training can use unique rollout keys, but strict reproducibility and duplicate-work verification are unavailable for the high-throughput topology. |
| P1 | Multi-host performance is unmeasured | Cross-host transport was proven synthetically; real q2 collection ran only on WSL. The two-generation smoke included startup and reported ~1 SPS, so it is a correctness result, not evidence that distributed training beats the current ~18–20 SPS run. |
| P1 | Runtime attestation is weak | Workers echo the learner config hash but do not independently prove game.so, map, reward environment variables, Rust extension, or observation layout checksums. A mismatched host can submit shape-valid but semantically incompatible data. |
| P1 | No soak or policy-quality gate | Only two optimizer generations have run. There is no 12–24 hour shadow comparison for KL, clip rate, reward composition, aim preservation, KD, failure/retry behavior, or memory growth. |
| P2 | Transport security assumes a trusted fabric | Bearer authentication exists, but HTTP is unencrypted. This is acceptable only behind Tailscale/LAN ACLs; public exposure requires mTLS or a secure proxy. |

## Required path to primary

1. Provision Nobara with checksum-identical q2 runtime, maps, Python tree,
   policy dependencies, and the Rust extension. Run a real batch and compare
   its semantic manifest to WSL.
2. Add a signed runtime manifest covering game.so, protocol/observation shape,
   maps, reward variables, policy architecture, and lattice build.
3. Carry episode summaries and the current behavior/outcome telemetry in each
   batch; restore TensorBoard parity before evaluating policy quality.
4. Add worker leases, heartbeats, reconnect/backoff, deterministic assignment
   IDs, and learner-owned versioned lattice snapshots/recovery.
5. Diagnose or explicitly bound four-ML nondeterminism. Deterministic audit
   workers may remain one-ML, but high-throughput workers need repeated-run
   statistical equivalence gates.
6. Run distributed collection in shadow mode for at least 12–24 hours while
   the current primary trainer continues. Compare usable SPS, stale/rejected
   work, GPU utilization, PPO diagnostics, aim, KD, and reward composition.
7. Canary one learner update stream with automatic rollback to local
   collection, then promote only if throughput and policy-quality gates pass.

The active master trainer should remain primary until P0 items are closed and
the shadow run demonstrates a real capacity gain.
