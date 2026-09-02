# Phase B multi-lane distributed training (2026-07-25)

Two concurrent network-native rollout workers feeding one embedded PPO
learner on wsl-box. Infrastructure validation run (from zero, not a science
run): throughput and correctness of the distributed path, wire v5 lane.

Status: RUNNING (acceptance window results below).

## Topology

```
                ┌────────────────────────────────────────────┐
                │ learner  train.ppo (Q2_DISTRIBUTED_LEARNER)│
                │ coordinator http://127.0.0.1:38888         │
                │ quorum=2, envs/worker=8, n_steps=128       │
                │ batch_size 2048 (see note), from zero      │
                └───────▲───────────────────────▲────────────┘
              policy vN │         batch vN      │ policy vN
         ┌──────────────┴───┐             ┌─────┴────────────┐
         │ worker w1        │             │ worker w2        │
         │ rollout_worker.py│             │ rollout_worker.py│
         │ --mode network   │             │ --mode network   │
         │ --continuous     │             │ --continuous     │
         │ 8 ML clients     │             │ 8 ML clients     │
         └──────┬───────────┘             └────────┬─────────┘
         q2ded 28310/28311                  q2ded 28320/28321
         ml2sk1 x2 bots                     ml2sk1 x2 bots
         mllive_44987431                    mllive_44987431
```

All on one WSL host, loopback only. GPU shared (RTX 2080, ~11% busy).
Ports are deliberately disjoint from the local single-lane (28300/28301,
harness 39000+, qports 49000+): w1 uses harness 39100+/qports 49100+,
w2 uses 39200+/49200+; client data roots
`.../client-data/phaseb_w1` and `.../client-data/phaseb_w2`.

## Secrets (mode-600 env files under /home/raymond/q2-rollout)

- `local-pilot-telemetry.env` — `Q2_ML_CLIENT_TELEMETRY_TOKEN` (game conduit).
- `phaseb-rollout-token.env` — `Q2_ROLLOUT_TOKEN` (coordinator HTTP bearer).
- `phaseb-attestation-key.env` — `Q2_ROLLOUT_ATTESTATION_KEY` (manifest HMAC).

The rollout token and the HMAC key are separate long random secrets
(`openssl rand -base64 33` / `openssl rand -hex 32`). The game-telemetry
token is a different secret again. None are ever logged; all env dumps in
run logs filter `TOKEN|KEY`.

## Manifest

`runtime-manifest-phaseb.json` in the repo root, built per
docs/DISTRIBUTED-ROLLOUTS.md with the worker env (Q2_EXT_OBS=1,
Q2_RUST_LATTICE=1, Q2_ML_ASYNC=0, Q2_POLICY_STATEFUL=1, rust extension path,
CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED, Q2_SOURCE_REVISION), runtime args
`n_bots=8 n_ml=8 timescale=1.0 max_ep_steps=1000 steps=128 deterministic=false
deterministic_actions=false device="cuda"`, map `mllive_44987431`, q2-root =
the staging server runtime. Semantic digest:
`a558ea2fe60f5426c6d057e8eb187aac4786ad89eb8449d67c1d0f7469edc27f`
(HMAC-signed; revalidate with `tools/runtime_attestation.py validate
--require-signature`). Rebuild after ANY source-tree or game.so change —
the semantic hash covers the Python source files.

Two attestation gaps were fixed for this topology (in
`harness/runtime_attestation.py`, mirrored to nobara):
`Q2_NETWORK_` prefix is now non-semantic (lane ports/IDs are deployment
detail; two workers must share one manifest), and
`Q2_ML_CLIENT_TELEMETRY_TOKEN` is a non-semantic key (credential; must never
be attested or recorded).

## Launch / stop

```bash
# 1. learner (publishes policy version 0, listens on 127.0.0.1:38888)
ssh wsl-box 'tmux new-session -d -s q2_phaseb_learner \
  /home/raymond/q2-rollout/q2-ml-bot/ops/phaseb_learner.sh'
# 2. workers (each starts its own q2ded, waits for coordinator TCP first)
ssh wsl-box 'tmux new-session -d -s q2_phaseb_w1 \
  /home/raymond/q2-rollout/q2-ml-bot/ops/phaseb_worker.sh w1'
ssh wsl-box 'tmux new-session -d -s q2_phaseb_w2 \
  /home/raymond/q2-rollout/q2-ml-bot/ops/phaseb_worker.sh w2'
```

Scripts: `ops/phaseb_learner.sh` (md5 1fc719fe9933b191b73c78f53538e17c),
`ops/phaseb_worker.sh` (md5 af75451653ec7dd6de581d21a2a8edc6), mirrored to
nobara `/home/raymondj/q2-ml-bot/ops/`.

Stop: kill the worker python PIDs and the learner python PID by exact PID
(`ps -C python3 -o pid,cmd`); each worker script's trap stops its own q2ded.
Never pkill by pattern. Learner checkpoints:
`training-data/checkpoints/phaseb_zero_v1/`; worker lattice snapshots under
`.../phaseb_zero_v1/worker_state/<worker-id>/`.

## batch_size note

The learner crashed on first launch:
`recurrent aim anchor currently requires one full-rollout minibatch ...
need batch_size >= 2048`. With quorum 2 x 8 envs x 128 steps = 2048
transitions per generation and `aim_anchor_coef 0.02`, batch_size must be
2048 (the approved config said 1024; `train/ppo.py` rejects it — the
"check ppo.py for exact requirements" clause applies).

## Monitoring

```bash
ssh wsl-box 'tmux attach -t q2_phaseb_learner'   # updates: [ N] steps= ... sps= ...
ssh wsl-box 'tail -f /home/raymond/q2-rollout/q2-ml-bot/logs/phaseb/phaseb-w1.log'
ssh wsl-box 'tail -f /home/raymond/q2-rollout/q2-ml-bot/logs/phaseb/phaseb-w2.log'
curl -s localhost:6006/data/plugin/scalars/tags   # via wsl-tunnel on nobara
/tmp/q2_phaseb_sample.sh                          # one-shot health sampler (WSL)
```

Generation contract per docs/DISTRIBUTED-ROLLOUTS.md: learner publishes
policy vN (= total env steps), workers fetch, collect 128 steps x 8 envs,
submit; quorum 2 seals the generation, learner merges along the env axis,
one PPO update, publishes v(N+2048). Late batches rejected stale/closed.

## Acceptance window (2026-07-25, 30 min)

Generation-0 quorum sealed ~35 s after worker start; update [1] at
elapsed 1.1 min. All counters below are from the live window; the only
tracebacks in the logs are from the two pre-fix launch attempts (manifest
rebuilds), proven by after-last-session-header greps.

| t (learner elapsed) | update | steps | sps (aggregate accepted) |
|---|---|---|---|
| 12.3 min | 28 | 57,344 | 78 |
| 21.1 min | 52 | 106,496 | 84 |
| 30.5 min | 77 | 157,696 | 86 |

- Submissions: 78 per worker, every one `accepted` (quorum_count 1/2 as
  expected; latest seen policy_version 157,696+).
- Failures since successful launch (10:38:18): ZERO attestation failures,
  ZERO failed_rounds, ZERO echo timeouts, ZERO telemetry-gap resyncs,
  zero tracebacks in learner/worker logs.
- Client CPU: 16 quake2 clients, mean 1.5% each (idle-sleep patch).
- Server CPU: 2 q2ded, ~5% each. GPU: 10-15% / ~1.55 GiB (RTX 2080).
- Full pytest on WSL after all changes: 184 passed, 1 skipped.

Completion bar met: >=80 accepted t/s sustained >=30 min, zero failures,
suite green. Left RUNNING (tmux q2_phaseb_learner / q2_phaseb_w1 /
q2_phaseb_w2); stop by exact python PIDs when done.
