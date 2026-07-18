# Synchronous LAN Rollout Prototype

Branch: `feature/rust-lattice`. The active trainer on `master` does not use
this protocol.

## Generation contract

1. The learner publishes policy version N, its SHA-256, a training-config
   hash, and the operator-approved runtime-manifest digest.
2. Workers fetch that exact artifact, rebuild their semantic runtime manifest,
   and verify its digest plus required HMAC before starting q2ded.
3. Each worker uploads a pickle-free rollout tagged with N and both hashes.
4. The coordinator accepts only N, waits for the configured quorum, then seals
   N when the learner consumes it.
5. The learner merges the quorum along the environment axis, updates once, and
   publishes N+1. Any later N batch is rejected as `generation_closed`; after
   N+1 is published it is rejected as `stale`.

The coordinator also rejects malformed PPO shapes, wrong config/runtime
hashes, duplicate payloads, reused worker sequences, and two different rollout
hashes claiming the same determinism key.

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
- worker/sequence, policy/config/runtime hashes, game and Python seeds, rollout
  index, map, device, telemetry schema, and determinism key.

`merge_ppo_batches()` validates a complete quorum and concatenates time-major
arrays along their environment axis.

## Commands

Build and sign the manifest from the exact worker runtime. The HMAC key is a
separate long random secret shared through mode-600 environment files; do not
reuse the HTTP bearer token. WSL's operational tree has no `.git`, so pass the
approved revision explicitly there.

```bash
export Q2_ROLLOUT_ATTESTATION_KEY="$(openssl rand -hex 32)"
export Q2_ROLLOUT_ATTESTATION_KEY_ENV=Q2_ROLLOUT_ATTESTATION_KEY
export Q2_SOURCE_REVISION="$(git rev-parse HEAD)"  # set explicitly on WSL
export Q2_ROOT=/absolute/path/to/isolated/runtime
export Q2_EXT_OBS=1 Q2_RUST_LATTICE=1 Q2_ML_ASYNC=0 Q2_POLICY_STATEFUL=1
export Q2_RUST_EXTENSION_PATH=/absolute/path/to/q2_lattice_rs.so
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
python tools/runtime_attestation.py build \
  --q2-root "$Q2_ROOT" --source-root . \
  --source-revision "$Q2_SOURCE_REVISION" \
  --map mltrain_00005208 \
  --rust-extension "$Q2_RUST_EXTENSION_PATH" \
  --runtime n_bots=4 --runtime n_ml=4 --runtime timescale=10.0 \
  --runtime max_ep_steps=1000 --runtime steps=256 \
  --runtime deterministic=true --runtime deterministic_actions=false \
  --runtime 'device="cuda"' \
  --hmac-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --output runtime-manifest.json
python tools/runtime_attestation.py validate \
  --manifest runtime-manifest.json \
  --hmac-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --require-signature
```

Build independently on each host first: semantic digests must match. Then
distribute one operator-approved manifest without editing it. A worker rebuilds
the current runtime and compares it to that file before q2 startup. Runtime
arguments are type-strict and lane-specific: a 512-step throughput probe or a
CPU audit lane needs a manifest built with those exact `steps`/`device` values,
not the 256-step CUDA example above.

Coordinator on the learner host:

```bash
python tools/rollout_coordinator.py \
  --policy checkpoints/lattice_aim_v1/policy_40000258.pt \
  --version 40000258 \
  --config run-config.json \
  --runtime-manifest runtime-manifest.json \
  --attestation-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --require-attestation-signature \
  --bind 0.0.0.0 --port 38888 --quorum 2 \
  --token "$Q2_ROLLOUT_TOKEN" --schema ppo --once
```

Real WSL worker (isolated runtime/ports; the Rust extension must be installed):

```bash
Q2_ROOT="$HOME/q2-rollout/runtime" \
Q2_EXT_OBS=1 Q2_RUST_LATTICE=1 Q2_ML_ASYNC=0 Q2_POLICY_STATEFUL=1 \
Q2_RUST_EXTENSION_PATH="$HOME/q2-rollout/python/q2_lattice_rs.so" \
PYTHONPATH="$HOME/q2-rollout/python" \
python tools/rollout_worker.py \
  --mode q2 --coordinator http://LEARNER:38888 \
  --token "$Q2_ROLLOUT_TOKEN" --worker-id wsl-rtx2080 \
  --runtime-manifest runtime-manifest.json \
  --attestation-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --sequence 1 --seed 929 --game-seed 929 --rollout-index 0 \
  --steps 256 --map-name mltrain_00005208 --n-bots 4 --n-ml 4 \
  --sv-port-base 37000 --ml-port-base 37100
```

Use `--verify-determinism --deterministic-actions --n-ml 1` for the strict
same-host replay gate. Synthetic mode is for transport/LAN validation and does
not produce training data.

For an instrumented concurrent capacity run, pass the same manifest and key
name to each `tools/rollout_throughput_probe.py` process. Verify the resulting
records and signed expected manifest together:

```bash
python tools/rollout_throughput_gate.py \
  wsl-probe.json nobara-probe.json \
  --expected-manifest runtime-manifest.json \
  --attestation-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --require-attestation-signature \
  --baseline-sps 20 --min-speedup 1.25 --min-overlap-ratio 0.75
```

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

Follow-up isolation showed the q2/Rust four-slot trajectory is bit-exact with
fixed actions, and full policy replay is exact on CPU. The variance is in fresh
CUDA policy inference. Use a CPU four-slot lane for byte-exact audit jobs;
normal CUDA workers use distinct seed/rollout-index keys.

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
trust the learner. Rollout batches themselves remain non-executable. The
runtime-manifest HMAC is mandatory for a signed lane; configure
`Q2_ROLLOUT_ATTESTATION_KEY_ENV` on the embedded learner and pass
`--attestation-key-env` to every worker. Merely placing a `signature` object in
JSON is not verification—the key must be present.

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
Q2_EXT_OBS=1 Q2_RUST_LATTICE=1 Q2_POLICY_STATEFUL=1 \
Q2_ROLLOUT_BIND=0.0.0.0 Q2_ROLLOUT_PORT=38888 \
Q2_ROLLOUT_TOKEN="$Q2_ROLLOUT_TOKEN" \
Q2_ROLLOUT_RUNTIME_MANIFEST="$PWD/runtime-manifest.json" \
Q2_ROLLOUT_ATTESTATION_KEY_ENV=Q2_ROLLOUT_ATTESTATION_KEY \
Q2_ROLLOUT_QUORUM=2 Q2_ROLLOUT_ENVS_PER_WORKER=4 \
Q2_ROLLOUT_RECOVERY=1 Q2_ROLLOUT_LEARNER_ID=q2-ppo-shadow \
Q2_ROLLOUT_LATTICE_DIR="$PWD/learner-state/distributed-lattice" \
Q2_ROLLOUT_LEASE_TTL=45 Q2_ROLLOUT_MAX_ATTEMPTS=3 \
python -m train.ppo --n_steps 256 --map_name mltrain_00005208 \
  --deterministic 1 --total_steps 20000000
```

Workers use `--continuous --leased` to keep q2ded, recurrent state, and the
Rust lattice alive while policy versions change. The coordinator assigns a
stable lane/seed/map, fences it with a TTL heartbeat lease, and adopts the
submitted checksum-chained lattice before acknowledging the batch. A
replacement worker fetches that exact learner-owned lattice; `--lattice-dir`
is a local incoming/outgoing mirror, not the authority.

```bash
Q2_ROOT="$HOME/q2-rollout/runtime" \
Q2_EXT_OBS=1 Q2_RUST_LATTICE=1 Q2_ML_ASYNC=0 Q2_POLICY_STATEFUL=1 \
Q2_RUST_EXTENSION_PATH="$HOME/q2-rollout/python/q2_lattice_rs.so" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
PYTHONPATH="$HOME/q2-rollout/python" \
python tools/rollout_worker.py --mode q2 --continuous --leased \
  --coordinator http://LEARNER:38888 --token "$Q2_ROLLOUT_TOKEN" \
  --worker-id nobara-0 --steps 256 --n-ml 4 \
  --runtime-manifest runtime-manifest.json \
  --attestation-key-env "$Q2_ROLLOUT_ATTESTATION_KEY_ENV" \
  --lattice-dir worker-state/nobara-0
```

Worker attestation hashes the complete map pool in the operator-approved
manifest and rejects assignments outside it. This keeps one semantic runtime
identity while leased curriculum assignments rotate among approved maps.

An earlier isolated two-generation CUDA rehearsal completed versions 0 and 16
with a single persistent q2ded launch, performed two learner PPO updates,
published both policy generations, saved worker lattice snapshots for both,
and produced learner checkpoint version 32. That predates leased recovery and
was followed by a real fault-injection proof on 2026-07-12. A dead worker's
lane was reissued with the identical assignment ID and a newer lease epoch. In
a separate run the learner was killed after accepting one of two lanes; after
restart its fsynced generation journal restored that batch, receipt, completed
lease, determinism identity, and lattice artifact. Only the missing lane ran,
quorum updated once, and policy advanced `40,410,882 → 40,412,930` at 63 SPS.
Distributed policy and optimizer checkpoints are atomically written before a
new policy generation is published.

Nobara's operational scaffold lives under `~/q2-rollout`; repo templates are
in `ops/q2-rollout-worker.{service,env.example}`. The installed user service is
deliberately disabled until a shadow learner token is assigned and an isolated
learner-kill/worker-replacement rehearsal proves fencing and lattice recovery.
Provisioning paths, immutable hashes, and the accepted real RTX 3070 rollout
are recorded in `docs/ROLLOUT-HOST-PROVISIONING.md`.
