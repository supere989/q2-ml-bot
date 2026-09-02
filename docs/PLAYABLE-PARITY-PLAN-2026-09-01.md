# Playable Parity Plan — good maps, real-AI bots (2026-09-01)

> **Read this before starting any code session on the trainer, maps, or live
> deployment.** It is the current campaign plan. It exists because the
> project's recurring failure mode is drift: sessions re-litigate settled
> findings and re-try already-falsified approaches. The evidence for every
> claim below is in the referenced docs and handoffs.

## Ground truth (do not re-litigate)

- **Infrastructure is done and validated.** Network-native client harness
  (wire v5), Phase B distributed rollouts (quorum=2, attestation, 86
  accepted t/s sustained, zero failures, completed at 2,000,896 steps on
  2026-07-25), map farm with bundle-v2 contract, teacher demo collection,
  telemetry hygiene, season quality gate. See `PHASE-B-MULTILANE-2026-07-25.md`,
  `DISTRIBUTED-ROLLOUTS.md`, `HANDOFF-2026-07-13-NETWORK.md`,
  `SEASON-QUALITY-GATE.md`.
- **No policy has ever passed the stock-map combat quality gate.** The
  39.9M-step June lineage was measured at 100% down-look and ~66-70%
  backward commands and is quality-invalid; the 4,063,488 and 4,055,296
  warm-starts reproduced the same pathology and are archived. Never resume
  or promote them (`HANDOFF-2026-07-13-NETWORK.md`, repo `AGENTS.md`).
- **Imitation works; from-zero PPO does not (yet).** Every from-scratch run
  reproduced degenerate behavior. The BC-distilled `bc_live_v2` clone passed
  its holdout (2.23° yaw / 1.69° pitch MAE, 92.7% aligned-fire precision)
  and showed real combat signal (49 hits / 2 kills in 16,384 live
  transitions). The clone is parked under `training-data/retired-evidence/`
  on the WSL staging tree — retired as a *run*, still valid as a *warm-start*.
- **The bottleneck is the learning signal, not compute or plumbing.** RL
  policies here learn what is measurable, not what is good. Stop tuning
  rewards from zero; give the policy "good" to start from via imitation.

## Campaign phases (in order)

### Phase 1 — Restore the fast lane (bulk pretraining engine)
- **DONE 2026-09-01.** The wire-v5 C changes (`ML_CLIENT_WIRE_VERSION` 5,
  `ml_obs_t` 1060 bytes) were uncommitted in BOTH engine trees; committed
  as `q2-lithium-3zb2` `3b67d7d` on `ml-wip-20260611` and pushed.
  Rebuilt `game.so` (`make clean && make -j8` in `~/merge_mod/lithium` on
  wsl-box) and deployed to `~/q2_lithium_merge/lithium/game.so`
  (Jul-11 wire-v4 build backed up as `game.so.bak-wire4-20260901`).
- Lane verified live: trainer relaunched (tmux `q2_ppo`), ZERO invalid-obs
  errors, ~87 accepted env steps/sec (12 servers × 4 ML bots + 4 AI each,
  timescale 8), real episode returns from update 1. Checkpoints save every
  100k steps to `~/q2-ml-bot/checkpoints/`.
- This run is from-zero with the fixed (post-July) reward/view stack —
  treat it as the clean-room test of whether the July fixes cured the
  down-look/backward-command pathology, and as the bulk pretraining
  engine for Phase 3. It does NOT replace the Phase 2 BC baseline.

### Phase 2 — Imitation corpus and BC baseline
- Maximize collection from BOTH existing pipes: 3ZB2 teacher demos
  (receiver on wsl-box:32511, batches under
  `~/q2-rollout/live-3zb2/teacher_batches`) and human telemetry from real
  public-server sessions (007Bond, LordNiKON).
- Behavior-clone to the EXISTING holdout gates (yaw/pitch MAE, aligned-fire
  precision, movement drift). Do not invent new gates mid-campaign.

### Phase 3 — PPO polish under the season gate
- Short PPO seasons warm-started from the BC baseline, judged by
  `tools/season_quality_gate.py` (100 generations / 1M steps / per-map
  episode coverage / KL and clip bounds / no-regression ladder).
- Bulk pretraining on the fast lane; verification and fine-tuning on the
  network-native lane (real engine conditions). The RTX 2080 runs ~15% GPU
  with two Phase B workers — 3-4 workers fit.
- Causal ladder order is fixed: posture → movement → aim → combat.

### Phase 4 — Maps and public play
- Maps are a curation problem now, not a generation problem. Generator v6 +
  lighting v2 + lethal-edge guards are in production. Promote a fixed pool
  of 4-6 maps passing BOTH the safety contract and the map-judgment
  pipeline, anchored by the stock competitive set (q2dm1-8).
- **Per-round live map generation is training-only from now on.** It is
  good for generalization, bad for humans learning a map and for policy
  consistency. Public server runs the curated pool; live-maps on
  designated chaos/training windows only.
- Public roster: 2 ML bots + 2 3ZB2 fillers, humans anytime
  (`maxclients=6` leaves slots by design).

## Anti-goals (things that will not help)

- More from-zero reward shaping — the falsified treadmill.
- Promoting a checkpoint because loss converged — gates or it didn't happen.
- Reverting to a static map pool for training — generalization regresses.
- Any resume from pre-2026-07-14 checkpoints — quality-invalid, archived.

## Current repo/topology facts (2026-09-01)

- Canonical working tree: `wsl-box:~/q2-ml-bot` (real clone,
  branch `feature/rust-lattice`). Procreator checkout and the nobara
  15-min backup sync are downstream. The old `yquake-ml-workspace` repo on
  nobara is frozen at 2026-07-11 and marked superseded.
- Engine: `q2-lithium-3zb2` `ml-wip-20260611` @ `1e659aa` (bootstrap race
  fix committed and pushed).
- Live services on the VPS (`valheim-server`): `q2mlbot.service`
  (public, UDP 28000 + telemetry 28049), `q2-teacher-server.service`
  (loopback 28001), `q2mlbot-gamedata.service` (HTTP 32494). WSL user
  services: `q2-map-farm`, `q2-teacher-map-farm`, `q2-teacher-receiver`.
- Restore runbook: `q2-training-box` skill. `launch-trainer.sh` targets
  the classic lane and is safe again since the Phase 1 rebuild
  (2026-09-01).
