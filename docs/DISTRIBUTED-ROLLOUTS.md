# Synchronous LAN Rollout Prototype

Branch: `feature/rust-lattice`. The active trainer on `master` does not use
this protocol.

## Generation contract

1. The learner publishes policy version N, its SHA-256, and a training-config
   hash.
2. Workers fetch and verify that exact artifact before starting q2ded.
3. Each worker uploads a pickle-free rollout tagged with N and both hashes.
4. The coordinator accepts only N, waits for the configured quorum, then seals
   N when the learner consumes it.
5. The learner merges the quorum along the environment axis, updates once, and
   publishes N+1. Any later N batch is rejected as `generation_closed`; after
   N+1 is published it is rejected as `stale`.

The coordinator also rejects malformed PPO shapes, wrong config hashes,
duplicate payloads, reused worker sequences, and two different rollout hashes
claiming the same determinism key.

## Wire format

`harness/rollout_protocol.py` defines deterministic policy (`Q2PL0001`) and
rollout (`Q2RB0001`) envelopes. Each consists of an eight-byte magic value, a
canonical JSON manifest, and raw contiguous array bytes. No pickle or remote
code loading is used for rollout data. The configured maximums are 128 MiB per
policy and 512 MiB per batch.

A PPO batch contains:

- observations, actions, rewards, dones, values, and behavior log-probs;
- recurrent H/C state at every step;
- final observations and H/C state for learner-side value bootstrapping;
- worker/sequence, policy/config hashes, game and Python seeds, rollout index,
  map, device, and determinism key.

`merge_ppo_batches()` validates a complete quorum and concatenates time-major
arrays along their environment axis.

## Commands

Coordinator on the learner host:

```bash
python tools/rollout_coordinator.py \
  --policy checkpoints/lattice_aim_v1/policy_40000258.pt \
  --version 40000258 \
  --config run-config.json \
  --bind 0.0.0.0 --port 38888 --quorum 2 \
  --token "$Q2_ROLLOUT_TOKEN" --schema ppo --once
```

Real WSL worker (isolated ports; the Rust lattice extension must be installed):

```bash
Q2_ROOT=~/q2_lithium_merge Q2_EXT_OBS=1 Q2_RUST_LATTICE=1 \
python tools/rollout_worker.py \
  --mode q2 --coordinator http://LEARNER:38888 \
  --token "$Q2_ROLLOUT_TOKEN" --worker-id wsl-rtx2080 \
  --sequence 1 --seed 929 --game-seed 929 --rollout-index 0 \
  --steps 256 --map-name mltrain_00005208 --n-bots 4 --n-ml 4 \
  --sv-port-base 37000 --ml-port-base 37100
```

Use `--verify-determinism --deterministic-actions --n-ml 1` for the strict
same-host replay gate. Synthetic mode is for transport/LAN validation and does
not produce training data.

## Validation completed 2026-07-11

- Loopback: two independent workers produced one identical deterministic hash;
  quorum two was accepted and spooled.
- LAN: WSL/RTX2080 and Nobara fetched version 8 over separate paths and each
  produced a `256 × 397` synthetic validation batch with identical hash
  `f7b4e90b...`; procreator accepted and persisted the two-host quorum.
- Real rollout: WSL fetched the 3 MiB version-40000258 policy, loaded it on
  CUDA, launched isolated four-slot q2ded, and uploaded a valid `32 × 4 × 219`
  recurrent PPO batch. Actions were `32 × 4 × 8`; H/C states were
  `32 × 4 × 256`.
- Determinism: two fresh one-ML q2ded launches with identical seeds and
  deterministic actions produced the same full rollout hash. Four-ML fresh
  launches did not; the worker correctly refused submission and reported
  differences across observations, actions, rewards, and recurrent state.

The four-ML finding means deterministic audit jobs must currently use one ML
slot. Normal training workers may use four slots, but should use distinct
seed/rollout-index keys rather than falsely claiming cross-run equivalence.

The CLI worker currently starts a fresh per-session lattice for each batch and
records `lattice_mode=fresh_worker_session`. Before long-running distributed
training replaces local collection, either keep workers persistent across
generations or publish a versioned lattice snapshot beside the policy. Mixing
fresh and persistent lattice modes in one quorum is not allowed by the intended
learner configuration hash.

## Deployment and security

Keep the service on the trusted LAN/Tailscale fabric and set a long bearer
token. The prototype is HTTP rather than TLS; do not expose it publicly. A
production service should run behind WireGuard/Tailscale ACLs or an mTLS proxy.
Policy checkpoints are still loaded by PyTorch on workers, so workers must
trust the learner. Rollout batches themselves remain non-executable.

The coordinator CLI intentionally stops after `--once` in the demonstrated
flow. Continuous learning should embed `RolloutCoordinator`, call
`wait_for_quorum(N)`, merge/update, and only then call `publish(N + 1)`.

## Embedded learner and persistent worker

`train/ppo.py` now has an opt-in embedded coordinator. It does not construct
local q2ded servers; instead it validates a quorum, copies the merged arrays
into the existing `RolloutBuffer`, computes final values on the learner, runs
the unchanged PPO update, and publishes the new total-step version.

```bash
Q2_DISTRIBUTED_LEARNER=1 \
Q2_ROLLOUT_BIND=0.0.0.0 Q2_ROLLOUT_PORT=38888 \
Q2_ROLLOUT_TOKEN="$Q2_ROLLOUT_TOKEN" \
Q2_ROLLOUT_QUORUM=2 Q2_ROLLOUT_ENVS_PER_WORKER=4 \
python -m train.ppo --n_steps 256 --total_steps 20000000
```

Workers use `--continuous` to keep q2ded, recurrent state, and the Rust lattice
alive while policy versions change. `--lattice-dir` writes a versioned Python
checkpoint after every accepted generation and restores `lattice_latest` when
the worker restarts.

```bash
python tools/rollout_worker.py --mode q2 --continuous \
  --coordinator http://LEARNER:38888 --token "$Q2_ROLLOUT_TOKEN" \
  --worker-id nobara-0 --steps 256 --n-ml 4 \
  --lattice-dir worker-state/nobara-0
```

An isolated two-generation CUDA rehearsal completed versions 0 and 16 with a
single persistent q2ded launch, performed two learner PPO updates, published
both policy generations, saved worker lattice snapshots for both, and produced
learner checkpoint version 32.
