# Network-native public trainer handoff — 2026-07-13

This supersedes the runtime/process sections of
`HANDOFF-2026-07-11.md`. The older document remains the historical record for
the aim, reward, movement, and first lattice investigations.

## Active topology

- The public VPS (`valheim-server`, Tailscale `100.101.57.114`) runs a
  server-only `q2mlbot.service`. Normal game traffic is UDP 28000. The private
  per-client observation conduit is UDP 28049.
- WSL (`rtx2080-wsl`, Tailscale `100.86.206.50`) runs four headless Yamagi
  clients plus PPO in tmux session `q2_public_train`. These are ordinary
  protocol-34 player connections, not in-process bot entities.
- Two of the public server's six client slots remain available for humans.
- The existing TensorBoard process still watches
  `/home/raymond/q2-ml-bot/runs`; that directory contains only the current
  `ppo_public_network_live_v1_*` run.
- The separate teacher server remains active on VPS loopback UDP 28001. The
  WSL map farm remains the source of generated `mllive_*` maps.

## Exact source/runtime state

- `q2-ml-bot`, branch `feature/rust-lattice`: `429a8e3` (pushed).
- `q2-lithium-3zb2`, branch `ml-wip-20260611`: `0712a224` (pushed).
  Deployed `game.so` SHA-256 is
  `a0396183ff3fa5738de40e58e73e628ce33cd300d93e2ad8177b80f15045a8b1`.
- `supere989/yquake2`, branch `feature/ml-client-harness`: `0382c0c0`
  (pushed). The staged WSL binary SHA-256 is
  `fdd996a5880b589fff6f0046e5739705213a7dc980d9dd5a5423d30128fa0e3b`.
- WSL staged source/runtime root:
  `/home/raymond/q2-network-client-staging-20260713`.
- The shared conduit secret is intentionally absent from git. The VPS copy is
  root-owned mode 0600 at `/etc/q2mlbot/network-client-harness.env`; the WSL
  copy is mode 0600 at `~/.config/q2-network-live-trainer.env`.

## Training start and rollback

The public trainer initially warm-started from
`checkpoints/movement_reset_v2/policy_04008192.pt` at 4,008,192 environment
steps with the matching 42,639-cell lattice. It uses four clients, 128-step
rollouts, two PPO epochs, batch 256, LR `1e-5`, `vf_coef=0.1`, entropy
`0.005`, auxiliary coefficient `0.01`, lattice-direction coefficient `0.02`,
stateful policy, `Q2_EXT_OBS=1`, and live `timescale=1` pacing. Its first
complete network checkpoint is `policy_04014336.pt` with the matching
optimizer and lattice under
`training-data/checkpoints/public_network_live_v1`; the active supervised
restart resumed all three at 4,014,336.

The previous movement run was stopped only after that checkpoint was written.
Its log, TensorBoard event, policy, lattice, ONNX, and checksums are archived
under:

`/home/raymond/q2-network-client-staging-20260713/archive/movement_reset_v2-20260713T2115-pdt`

The canonical movement checkpoint directory remains in place because it is
the new run's warm-start source and the trainer rollback point. The VPS
rollback module, unit, config, firewall, and old launcher are under:

`/home/q2mlbot/staging/network-client-57d4190-20260714/rollback`

## Admission and validation evidence

- Four WSL clients against the isolated VPS candidate produced 219-feature
  vectors and 64/64 accepted transitions with zero echo timeouts.
- The map-lifecycle canary crossed `q2dm1 -> q2dm3 -> q2dm1` with two clients,
  preserved monotonic route sequences, surfaced a zero-reward nontrainable
  boundary, and resumed collection by server frame 5 on each map.
- The first live PPO update admitted 512 transitions in 128 complete rounds,
  with zero failed rounds/timeouts. TensorBoard exposed 112 scalar tags,
  including `network_client/*` provenance/admission counters.
- Initial live throughput is about 18 environment steps/sec at normal public
  pacing. This is expected to be below the retired 56 steps/sec `timescale=8`
  local simulation.
- The first PT/optimizer/lattice/ONNX export exposed a real-time backlog: the
  fail-closed collector rejected the next action after more than 16 queued
  frames. No stale transition entered PPO. Commit `2d70b8c` added a
  nonblocking pre-dispatch drain; any catch-up now returns a zero-reward,
  nontrainable boundary so PPO recomputes the action from fresh state. A live
  four-client reproduction paused four seconds, drained 171 packets without
  dispatch, and resumed admitted transitions with zero timeouts. The active
  run logs this under `realtime_catchup_resyncs` and
  `preflight_packets_drained`.

## Security boundary

The conduit socket binds all IPv4 interfaces in the legacy game module, so
the firewall is part of the correctness boundary. VPS INPUT rules must remain
ordered as:

1. accept UDP 28049 on `tailscale0` from `100.86.206.50/32`;
2. drop UDP 28049 from every other source.

Both rules are persisted in `/etc/iptables/rules.v4`. Game and telemetry must
both use the VPS Tailscale address from WSL because registration validates the
normal Quake source address/port against the conduit datagram.

## Current caveats and next checks

- WSL's user systemd bus is absent. The versioned user unit is installed but
  the active trainer uses the established tmux fallback. Do not assume it will
  restart after a WSL reboot until the user bus or a system-level unit is
  repaired.
- The first few live updates were finite and advancing, but early public
  scores were negative from deaths/suicides and no combat kill had yet been
  observed. This is a quality signal to monitor, not a transport failure:
  transitions, action echoes, movement, and telemetry were all active.
- Confirm the next periodic checkpoint crosses its catch-up boundary without
  a process restart; the direct four-client checkpoint-pause reproduction has
  already passed.
- At the first automatic stock/generated map transition, confirm
  `network_client/map_epoch_resyncs` increments once, echo timeouts remain
  zero, and training continues.
- Watch movement speed, visible-target engagement, damage, kills/deaths, and
  action rates across the seasonal soak before promoting this checkpoint to
  public inference. Loss convergence alone is not a quality gate.

## Process checks

```bash
ssh rtx2080-wsl 'tmux list-sessions; tail -n 20 \
  /home/raymond/q2-network-client-staging-20260713/logs/live-trainer.log'

ssh valheim-server 'systemctl is-active q2mlbot.service \
  q2-teacher-server.service; ss -lun | grep -E ":(28000|28001|28049)\\b"'
```

Do not restart or replace the live `game.so` merely to inspect it. Use the
staged canary runtime and disjoint ports for subsequent C changes.
