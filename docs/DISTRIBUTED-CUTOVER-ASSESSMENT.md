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
- Nobara was provisioned from WSL with checksum-identical q2ded/game.so/Rust
  binaries and all maps, then completed a real four-ML RTX 3070 rollout that
  passed the strict PPO schema.
- The embedded learner consumed remote arrays with the existing PPO optimizer,
  published version 16, consumed a second batch from the same persistent q2ded
  and lattice, then saved policy version 32.
- The q2/Rust four-slot path reproduces bit-exactly with fixed actions. Full
  four-slot policy replay also reproduces on CPU; fresh CUDA policy inference
  is the isolated source of strict replay variance.
- WSL and Nobara concurrently sustained a conservative 671.01 rollout SPS
  against the current 18--20 SPS baseline, with zero timeouts and a matching
  signed semantic runtime manifest.

## Cutover blockers

| Priority | Blocker | Evidence / consequence |
|---|---|---|
| Closed | First spare CUDA host provisioning | Nobara now has the isolated runtime, feature branch, Rust extension, state/log scaffold, and a disabled user service. A real four-ML batch was accepted. See `docs/ROLLOUT-HOST-PROVISIONING.md`. |
| Closed | Remote telemetry parity | Strict batches now carry episode summaries plus 45 behavior/outcome aggregates under `ppo-telemetry-v1`; the learner restores the corresponding console and TensorBoard series. |
| Closed in code | Learner-owned worker recovery | The embedded learner now creates deterministic lane assignments, serves TTL leases/heartbeats and versioned lattice artifacts, fences stale submissions, and adopts each checksum-chained lattice before acknowledging the batch. `--continuous --leased` workers reconnect with bounded backoff and recover the exact learner reference. |
| P0 | Recovery has no live fault-injection proof | Lease/assignment state is process-local and the leased path has only focused/loopback tests. Kill/restart the learner during collection and replace a worker on real q2; prove stale work is fenced, the learner-owned lattice chain resumes exactly, and no optimizer generation is duplicated before relying on it. |
| P0 | Multi-map attestation/recovery mismatch | Assignments can select a map pool, but the worker currently rebuilds attestation for only its assigned map. An approved multi-map manifest therefore cannot match. Shadow recovery must stay on one fixed map until the worker attests the full approved map set and rejects out-of-set assignments. |
| Closed | Four-ML deterministic boundary | Eight fixed-action four-slot q2/Rust trajectories were bit-exact, and full CPU-policy replay matched exactly. CUDA inference remains unsuitable for a byte-exact audit, so production CUDA workers use unique rollout keys and a CPU audit lane verifies replay. |
| Closed | Concurrent multi-host capacity | WSL and Nobara supplied the same live generation concurrently: 335.50 and 536.60 SPS respectively, 671.01 conservative aggregate SPS, zero timeouts. See `docs/RUNTIME-ATTESTATION-THROUGHPUT-2026-07-12.md`. |
| Closed | Signed runtime attestation | The learner pins the manifest digest in every policy, workers independently rebuild and verify the signed semantic manifest before q2 startup, and the coordinator rejects mismatches as `wrong_runtime`. Signed shadow lanes must configure the HMAC-key variable on learner/coordinator/workers; updated ops templates fail closed when it is missing. |
| P1 | No soak or policy-quality gate | Only two optimizer generations have run. There is no 12–24 hour shadow comparison for KL, clip rate, reward composition, aim preservation, KD, failure/retry behavior, or memory growth. |
| P2 | Transport security assumes a trusted fabric | Bearer authentication exists, but HTTP is unencrypted. This is acceptable only behind Tailscale/LAN ACLs; public exposure requires mTLS or a secure proxy. |

## Required path to primary

1. **Completed:** provision Nobara with checksum-identical q2 runtime, maps,
   policy dependencies, and the Rust extension; accept a real four-ML batch.
2. **Completed:** signed runtime manifest and per-generation digest enforcement.
3. **Completed:** remote episode and behavior telemetry parity.
4. Run a real isolated learner-kill/worker-replacement rehearsal and close the
   durable restart/fencing gate; do not infer this from loopback tests.
5. Make map-pool attestation compatible with leased curriculum assignments and
   validate it with at least two maps. Until then use one fixed shadow map.
6. **Bounded:** use a CPU four-ML lane for exact replay audits; CUDA collection
   uses unique rollout keys and statistical policy-quality gates.
7. Run distributed collection in shadow mode for at least 12–24 hours while
   the current primary trainer continues. Compare usable SPS, stale/rejected
   work, GPU utilization, PPO diagnostics, aim, KD, and reward composition.
8. Canary one learner update stream with automatic rollback to local
   collection, then promote only if throughput and policy-quality gates pass.

The active master trainer should remain primary until P0 items are closed and
the shadow run demonstrates a real capacity gain.
