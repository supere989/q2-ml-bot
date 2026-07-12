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
| Closed | Learner-owned durable recovery | Real-q2 fault injection proved lease expiry/replacement with the identical assignment ID. A learner was then killed with one accepted lane: the restarted coordinator restored that batch, receipt, completed lease, determinism identity, and lattice artifact from its fsynced journal; only the missing Nobara lane ran, quorum updated exactly once, and policy advanced `40,410,882 → 40,412,930`. Distributed policy and Adam state are now atomically checkpointed before publishing a new generation. |
| Closed in code | Multi-map attestation/recovery | Workers rebuild the complete operator-approved manifest map pool, not only the assigned map, and reject assignments outside that pool. A multi-map live curriculum soak remains a promotion gate rather than a protocol mismatch. |
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
4. **Completed:** real worker replacement and mid-quorum learner-kill/restart;
   durable journal restored one accepted lane and produced one optimizer update.
5. **Completed in code:** full approved map-pool attestation and out-of-pool
   rejection. Include at least two maps in the shadow soak before promotion.
6. **Bounded:** use a CPU four-ML lane for exact replay audits; CUDA collection
   uses unique rollout keys and statistical policy-quality gates.
7. Run distributed collection in shadow mode for at least 12–24 hours while
   the current primary trainer continues. Compare usable SPS, stale/rejected
   work, GPU utilization, PPO diagnostics, aim, KD, and reward composition.
8. Canary one learner update stream with automatic rollback to local
   collection, then promote only if throughput and policy-quality gates pass.

The active master trainer should remain primary until the shadow soak and
policy-quality gate demonstrate that the proven capacity gain is sustainable.
