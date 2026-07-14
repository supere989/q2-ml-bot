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
  `ppo_public_network_engagement_anchor_v3_*` run.
- The separate teacher server remains active on VPS loopback UDP 28001. The
  WSL map farm remains the source of generated `mllive_*` maps. The live queue
  holds two ready bundles and the teacher queue four, all using generator v6
  and the complete bundle-v2 lattice/lighting contract.

## Exact source/runtime state

- `q2-ml-bot`, branch `feature/rust-lattice`: implementation through
  `fb85aa7` (pushed; this documentation commit follows it).
- `q2-lithium-3zb2`, branch `ml-wip-20260611`: `f49caee` (pushed).
  Deployed `game.so` SHA-256 is
  `f81ca42f000c0263cab44cabe2cb61881b485af31d584c04cfe7d850339ee2c8`
  in both public and teacher runtimes. Rollback artifacts are under
  `/home/q2mlbot/staging/ml-view-angle-f49caee-20260714/rollback`.
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
stateful policy, `Q2_EXT_OBS=1`, and live `timescale=1` pacing. The clean
engagement-anchor segment retains those settings except for batch 512 and
`aim_anchor_coef=0.02`, which keep each recurrent rollout intact while using
the now-verified true-view target labels. The clean
targeting/hook restart point is `policy_04030720.pt` with the matching optimizer
and 43,845-cell lattice under
`training-data/checkpoints/public_network_live_v1`. The active supervised
restart resumes all three at 4,030,720; short post-checkpoint segments made
before the target gate and lattice-hook correction were archived and are not
resume candidates.

The immutable resume set is `policy_04055296.pt` with matching optimizer and
lattice files. It is copied alone under
`training-data/resume/public_network_live_v1_04055296`; do not resume directly
from the rolling checkpoint directory because `--resume` always selects its
latest lexicographic policy file. ONNX export is non-fatal but currently skipped
because the WSL training Python environment does not have `onnx` installed.

The current clean TensorBoard segment is
`ppo_public_network_engagement_anchor_v3_1784015091`, resumed from step
4,055,296 after the true-view, dense-posture, and telemetry-audit fixes. It runs
at 17--18 steps/sec. The new
audit immediately proved that target data is present: mean entity/enemy count
4.793, a visible enemy on 32.42% of frames, 0.383 visible enemies/frame, nonzero
fire permission/execution, and 0.03125 damage dealt per accepted step. The same
update measured the inherited policy's remaining quality problem: 63.67%
backward commands, only 0.040 tracking quality, and no kills. TensorBoard
watches only this run; every superseded launch is archived beneath the staging
tree. Because the audit proved that visible target labels are populated, this
segment enables the recurrent aim anchor at a conservative coefficient 0.02
with one complete 512-sample minibatch. Its first update logged 65.63% visible
alive frames, finite look/fire losses, and 3.52% fire-gate permission; those are
pipeline proofs, not yet a combat-quality promotion.

The previous `public_network_live_v1` segment crossed the real
`mllive_33336964 -> q2dm5` transition after `ff3ad84`, incremented the map epoch
once, admitted no stale transition, continued training, and saved a complete
4,055,296 policy/optimizer/lattice set. That is the operational proof of the
persistent download barrier and the source of the immutable resume pin.

The previous movement run was stopped only after that checkpoint was written.
Its log, TensorBoard event, policy, lattice, ONNX, and checksums are archived
under:

`/home/raymond/q2-network-client-staging-20260713/archive/movement_reset_v2-20260713T2115-pdt`

The canonical movement checkpoint directory remains in place because it is
the new run's warm-start source and the trainer rollback point. The original
VPS network-cutover rollback is under:

`/home/q2mlbot/staging/network-client-57d4190-20260714/rollback`

Later targeted rollback artifacts are under
`/home/q2mlbot/staging/network-client-controls-20260714`,
`network-targeting-20260714`, and `lattice-hook-cd4395f`. The ML sound,
respawn, and map-bundle-v2 deployments have rollback copies under
`ml-pain-sound-8c2a5d9`, `ml-respawn-fa1c749`, and
`map-bundle-v2-200d28d`, respectively.

## Admission and validation evidence

- Four WSL clients against the isolated VPS candidate produced 219-feature
  vectors and 64/64 accepted transitions with zero echo timeouts.
- The map-lifecycle canary crossed `q2dm1 -> q2dm3 -> q2dm1` with two clients,
  preserved monotonic route sequences, surfaced a zero-reward nontrainable
  boundary, and resumed collection by server frame 5 on each map. Commit
  `f930daf` also recognizes live intermission telemetry before stale-echo
  exhaustion, preserving the same fail-closed admission rule.
- The first real stock-to-generated transition showed that one nontrainable
  intermission boundary was insufficient: the next PPO call could dispatch
  while clients were still downloading, then exhaust all four echo budgets.
  No stale sample entered PPO, but the process exited. Commit `ff3ad84` makes
  the map epoch persistent, sends no actions while it is pending, tolerates a
  download pause, and requires one common playable target map before clearing.
  The four-client staggered-arrival regression completes with zero failed
  rounds and zero echo timeouts; the next automatic live transition remains
  the operational proof point.
- The targeting/hook restart admitted 512 transitions in 128 complete rounds
  at about 18 steps/sec with no failed round or process restart. TensorBoard
  exposes 132 scalar tags, including target-gate/acquisition and lattice-hook
  correction metrics.
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
- Python masks fire unless sampled look ends within 12 degrees yaw / 14 degrees
  pitch of a visible enemy. C independently requires a live hostile,
  damageable, unprotected LOS target within 14/16 degrees and echoes a
  suppression bit; PPO replaces the stored action and removes the exact sampled
  fire log-probability when that last-moment shield triggers.
- Target alignment has a 0.02 rising-edge reward with a 20-frame cooldown.
  Same-target damaging hits within 30 frames add offense credit from 0.25x to
  a 1.5x cap and reset on switch, timeout, or death.
- Hook has no positive usage-rate reward. It selects reachable hook-zone
  landings that advance toward opportunity/readiness heat and pays only bounded
  new-best correction progress plus one arrival bonus. The no-op, blind, idle
  release, and overspeed costs remain.
- The zero-speed/action-state failure was a respawn deadlock, not missing
  registration or dead usercmd transport. The old run accepted full rounds and
  echoed about 88% forward intent, but the target gate also removed Quake's
  death-screen attack button. `fa1c749` injects only the lifecycle respawn
  input after recording the policy echo. A live proof showed repeated
  `Lattice-* melted` deaths followed by continued episodes and 123--148
  units/sec through updates 3--6; the old run was at exactly zero by update 3.
- Generator v6 rejects overlapping slab/roof/platform pairs with 56--95 units
  of free player-admitting headroom, and every spawn needs a clear 96-unit
  column plus one supported 96-unit escape route. A 21-map matrix (seven styles
  times seeds 1/7/42) had zero unsafe sandwiches, blocked columns, or trapped
  starts.
- Generator v6 also computes the union of playable floor rectangles and emits
  a solid 96-unit guard wall on every union edge that faces lethal void. The
  21-map style/seed matrix had 21/21 `lethal_drop_ok`, zero missing guards, and
  at least 20 guard walls per map. The current public server has armed
  `mllive_44987431`, a v6 bundle with 26 lethal edges, 26 exact guard walls,
  100% floor-light coverage, and 45/45 interior lights.
- Lighting v2 requires 98% direct floor coverage, minimum counted value 650,
  world ambient 180, 900-value overhead sources, and a dedicated 850-value
  internal source in each enterable room-like zone. The first real WSL
  BSP/VIS/qrad build had 24 floor regions, 30 floor sources, 33/33 interior
  sources, and a valid 1,381,476-byte lightdata lump below the 2 MiB limit.
- Farm bundle v2 carries and verifies four artifacts: BSP, hook zones, lattice
  priors, and routes/item timing. The WSL mirror publishes all files before the
  queue item becomes visible; the public consumer installs the same attested
  set atomically. A direct preload of `mllive_14732349` loaded seven prior
  cells, 49 route nodes, and the item-timing table from its verified manifest.
- Connected ML pain sounds now use explicit stock male samples; human clients
  keep model-sexed sounds and legacy disconnected 3ZB2 bots remain silent.
  This bypasses Yamagi's entity-state sexed lookup, so no client update or
  warning suppression is required.
- `ML_PackObs` previously used `ent->s.angles[PITCH]`, which Quake sets to one
  third of `client->v_angle[PITCH]` solely for the rendered player model. The
  observation pitch and entity-local basis therefore disagreed with the
  full-resolution server fire gate. C commit `f49caee` makes pitch, target
  coordinates, and the gate share `client->v_angle`. Python commit `f197fba`
  adds a dense forward-only 96--360 units/s level-posture reward that increases
  continuously toward level, with a separate penalty beyond 15 degrees,
  plus entity count, enemy visibility, aim error/tracking, fire quality, and
  damage-event TensorBoard tags. The observation already contains relative
  enemy position/velocity/health/hostility/visibility, current view, weapon and
  ammo, damage dealt/taken, kills/deaths, audio, rays, and lattice memory; after
  the view-frame correction this is sufficient for the engagement prototype.
  Follow-up `fb85aa7` makes the level reward dense across the full pitch range
  and adds visibility-conditional aim-error tags so sparse-contact maps no
  longer dilute the targeting audit.

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
- The first target/hook updates on stock `q2dm7` were finite and advancing, but
  contact remains sparse and the map exposes no eligible
  heated hook-zone landing. Treat zero target acquisition/hook progress as an
  active quality blocker until generated-map/contact evidence changes it; do
  not infer success from loss or transport health.
- Restore the optional `onnx` Python package before an inference promotion if
  a fresh ONNX artifact is required; PPO/checkpoint training is unaffected.
- At the next automatic `q2dm7 -> mllive_44987431` transition, verify both the
  already-proven persistent barrier and the first live v6 lethal-guard bundle.
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
staged canary runtime and disjoint ports for subsequent C changes. Never print
the telemetry cvar or full ML-client command lines: the prototype client still
receives the shared secret as a cvar argument. Rotate both mode-0600 env copies
and restart the server/clients if that value is exposed.
