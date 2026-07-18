# Rollout Host Provisioning — Nobara RTX 3070

Provisioned 2026-07-11 from the working WSL runtime template.

## Layout

```text
/home/raymond/q2-rollout/
  runtime/              # exact WSL q2_lithium_merge copy (548 MiB)
  python/               # ABI3 q2_lattice_rs.so
  q2-ml-bot/            # feature/rust-lattice checkout
  state/                # local lattice transfer mirror; learner store is authoritative
  logs/                 # worker service output
  runtime-manifest.json # operator-approved signed semantic runtime
  worker.env            # mode 0600; token intentionally UNSET
```

The user service is installed at
`~/.config/systemd/user/q2-rollout-worker.service`, daemon-reloaded, disabled,
and inactive. It must not be enabled until `worker.env` contains the real
learner address/token, manifest path, and attestation-key variable, the signed
manifest has been verified locally, and a shadow run is authorized. Keep the
bearer token and HMAC attestation key distinct.
Set `Q2_SOURCE_REVISION` to the approved full commit hash on operational trees
without `.git`; workers use it as measured provenance and still hash the exact
runtime source tree to catch overlays or drift.
The installed service uses `--continuous --leased`; its learner must enable
`Q2_ROLLOUT_RECOVERY=1` and a durable learner-owned lattice directory. Do not
point it at a legacy coordinator that cannot issue assignments.

## Verified template identity

| Artifact | SHA-256 |
|---|---|
| `q2ded` | `f3c645bada641937bd4f3ff4b97e0bf64338d277172d5f1bc534c038d01973d3` |
| `lithium/game.so` | `4733303035801ce9fa678fd90da25417c1c6edf2390e68731a9553dbbbcaf78e` |
| `q2_lattice_rs.so` | `e226a7f24b441f215316c8a806b7a53d4f98aa52b5d71d352c27bd0109f8ab9a` |

All copied runtime files matched WSL before the proof run. The proof run then
created/updated the expected machine-local `lithium/ml_server_38400.cfg`; this
generated config is mutable and must be excluded from immutable-runtime
attestation.

Nobara environment:

- NVIDIA RTX 3070, 8 GiB;
- PyTorch 2.12.0+cu130 with CUDA available;
- Python 3.13.5 and NumPy 2.4.6;
- Cargo installed;
- ABI3 Rust extension imported without rebuilding;
- observation dimension 219 and policy parameter count 781,916;
- all 32 `mltrain_*.bsp` maps present.

## Real rollout proof

Using isolated bases 38400/38500, Nobara launched the copied runtime, aligned
four ML slots at tick 134, ran CUDA inference with the Rust lattice, and
uploaded a strict 32-step/four-env PPO batch. The coordinator accepted policy
version 40000258 with rollout hash
`720efecd0470f5a8913bc45b51a7b6817eec156b6f9c1db05cb9ed330f1bf1c0`.

## Refresh procedure

1. Stop/disable the worker service.
2. Pull `feature/rust-lattice` in `q2-ml-bot`.
3. Rsync WSL `~/q2_lithium_merge/` to `runtime/` while no worker q2ded exists.
4. Copy/build `q2_lattice_rs.so` and verify the three hashes above.
5. Preserve `state/`, `logs/`, and the mode-600 `worker.env`.
6. Rebuild/sign `runtime-manifest.json` with the exact runtime, source, map
   pool, environment, and worker configuration; copy the same approved
   manifest and HMAC key to the learner and every worker.
7. Run `tools/runtime_attestation.py verify` with `--hmac-key-env`, then one
   isolated strict-schema batch before re-enabling the service.
