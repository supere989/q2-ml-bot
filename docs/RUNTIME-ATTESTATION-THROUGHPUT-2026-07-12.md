# Runtime attestation and concurrent rollout gate — 2026-07-12

## Outcome

The standalone attestation and capacity gate are implemented and a concurrent
real-q2 WSL + Nobara probe passed. This closes the narrow questions “can the
two hosts prove semantic runtime identity?” and “can they collect from the
same live policy generation concurrently faster than the current trainer's
18–20 SPS?” It does **not** by itself authorize primary-trainer cutover; seasonal quality,
recovery, and learner telemetry gates remain separate.

## Tools

- `harness/runtime_attestation.py` builds a canonical semantic payload and
  verifies its SHA-256 plus an optional HMAC-SHA256 signature.
- `tools/runtime_attestation.py` provides `build`, `verify`, and `validate`
  commands.
- `harness/throughput_gate.py` validates real-q2 producer identity, runtime and
  policy equality, timeouts, collection overlap, and conservative aggregate
  SPS.
- `tools/rollout_throughput_probe.py` instruments one cold-start real rollout,
  separates setup from collection time, and submits the strict PPO batch.
- `tools/rollout_throughput_gate.py` evaluates JSON probe records or spooled
  `.q2rb` batches.

The semantic manifest covers hashes/sizes for `q2ded`, `lithium/game.so`, the
loaded Rust extension, and all selected map assets; observation/packet
dimensions; the policy's complete name/shape/dtype schema and parameter count;
reward/runtime environment values; normalized runtime configuration; git
revision; and a selected runtime-source tree hash. Host paths, ports, worker
IDs, credentials, output directories, and GPU model are diagnostics rather
than semantic inputs, so identical installations at different paths match.

## Concurrent proof

Both machines used an isolated rollout tree and disjoint ports. The active WSL
trainer and its 12 q2ded children were not stopped or modified. Nobara's
provisioned worker service remained disabled/inactive.

Configuration:

- policy version `40000258`, SHA-256
  `d3f20b65324b2278c8353f0ae45cc0b7c3ec6a1c2f7932b720f71712cb86c859`;
- map `mltrain_00005208`, 4 ML slots/host, 512 steps, timescale 10;
- CUDA inference, stochastic policy actions, lockstep bridge;
- `Q2_EXT_OBS=1`, `Q2_RUST_LATTICE=1`, 219 observations;
- 2,048 transitions per host, strict `ppo-telemetry-v1` schema;
- WSL ports 39400/39500; Nobara ports 40600/40700.

The independently built semantic manifests matched exactly at
`b7cc88540ac69f88a6486d2c7ee944d350405b0e15cef0aed5ae1d90ecbefc89`.
Key identities within that probe snapshot:

| Input | SHA-256 |
|---|---|
| q2ded | `f3c645bada641937bd4f3ff4b97e0bf64338d277172d5f1bc534c038d01973d3` |
| game.so | `4733303035801ce9fa678fd90da25417c1c6edf2390e68731a9553dbbbcaf78e` |
| q2_lattice_rs.so | `e226a7f24b441f215316c8a806b7a53d4f98aa52b5d71d352c27bd0109f8ab9a` |
| map BSP | `4201f5b03a52d4234397fbaa059843be29a77d9132f319033500aa998ac871af` |
| map JSON | `a4b37962b215862a59cd38f1d1fd2ac7ae3d8bafaab4942fa56a4e6d88b53b43` |
| policy architecture schema | `c3a5bf2fe38f5e8ea00c591cdf63807251b54ece13a52725df0c8d2b4be27416` |
| runtime source tree | `a52bdfba00992cbba8c63e33eb9cd6fd574d5c059046a28403636064872a9334` |

Measured collection-only results (first policy inference through final q2
step; setup is reported separately):

| Host | Setup | Collection | Transitions | SPS | Timeouts |
|---|---:|---:|---:|---:|---:|
| WSL RTX 2080 | 2.191 s | 6.104 s | 2,048 | 335.50 | 0 |
| Nobara RTX 3070 | 1.123 s | 3.817 s | 2,048 | 536.60 | 0 |

The shorter worker's collection interval overlapped fully with WSL's. The
gate conservatively divides all 4,096 transitions by the 6.104-second union,
yielding **671.01 aggregate SPS**. Against a 20 SPS baseline and required
1.25× improvement (25 SPS), the gate passed by 33.55×. Both batches were
accepted into the same quorum and persisted by the coordinator.

The first Nobara attempt found UDP 39706 already occupied. Nothing was killed;
the probe moved to a preflighted free slab and the clean concurrent generation
above is the reported result.

## Interpretation and integration

This is a collection-capacity result, not end-to-end learner throughput. It
excludes cold setup, upload, quorum wait after the final collection, PPO update,
checkpointing, and long-run failure/retry costs. The fixed four-slot map also
has much less orchestration overhead than the active 12-server trainer. Use the
result to justify shadow seasons, not to predict 671 training SPS.

The standalone contract was subsequently integrated into the distributed
rollout protocol:

1. The learner loads an operator-approved manifest, includes its digest in the
   published policy generation/config identity, and exposes the expected
   digest in coordinator status.
2. Each worker rebuilds its manifest before q2 startup, verifies digest and
   required HMAC, and refuses collection on any mismatch.
3. Real-PPO metadata now requires `runtime_manifest_sha256`, and
   `rollout_hash()` includes it in its stable fields. The throughput probe adds
   its separate `collection` timing object; normal training batches need not.
4. `RolloutCoordinator` is configured with the expected digest, rejects missing
   or different values as `wrong_runtime`, and `merge_ppo_batches()` requires
   one identical digest across the quorum.
5. Workers attest at process/runtime creation. Treat the attested runtime tree
   as immutable for that process and restart/re-attest whenever binaries, maps,
   extension, source, observation mode, reward environment, runtime arguments,
   or policy architecture change.

Signed lanes must pass the HMAC-key environment-variable name on the learner,
coordinator, throughput probe, and every worker. Digest pinning still detects
semantic drift without HMAC, but it must not be described as signed
attestation. See `docs/DISTRIBUTED-ROLLOUTS.md` for the complete commands.

The manifest reports git revision plus the exact source-tree hash because WSL's
operational Python tree has no `.git`, and this test intentionally used an
uncommitted cross-host source snapshot. A future committed deployment should
still keep both fields; revision alone cannot prove a clean working tree.
