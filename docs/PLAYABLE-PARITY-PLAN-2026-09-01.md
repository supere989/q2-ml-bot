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
- **IN PROGRESS 2026-09-01, two pipes fixed/built:**
  - **Teacher pipe was silently dead since 2026-07-14.** Root causes found
    and fixed: (1) `teacher_server.py` (and `network_public_server.py`) only
    armed `sv_maplist` when a generated map was staged — during the
    2026-08-27→30 farm outage the teacher lane hit timelimit with no armed
    map and **wedged in intermission for 5 days** (zero bots, zero
    telemetry); both controllers now fall back to stock→stock rotation
    (commit `3334596`). (2) The teacher runtime's `game.so` still emitted
    the 1032-byte wire-v4 obs while the receiver expects 1148-byte packets
    — all packets rejected; rebuilt from `q2-lithium-3zb2` `3b67d7d` and
    deployed. Collection verified live (~55 samples/s, rejected=0 lost=0).
  - **Human pipe BUILT (it never existed):** teacher capture only ran in
    the 3ZB2 `Bot_Think` path and the receiver rejected
    `ML_CONTROL_HUMAN`. Added opt-in `ml_teacher_humans` cvar
    (`ML_TeacherSendHuman`, engine commit `9b91b91`) taking the action
    straight from the player's usercmd in `ClientThink`; receiver
    `--allow_humans` + per-row `control` column in the npz
    (commit `ca13ef1`). Enabled on the public lane. **End-to-end human
    verification is pending a real player session** (007Bond/LordNiKON) —
    confirm `control == 1` rows appear in the next batches after they play.
- **BC tooling:** `tools/behavior_clone_demos.py` (new, tested) — the
  teacher_batches corpus previously had NO consumer; this is the demo-BC
  path. Gates match the bc_live_v2 acceptance bar (no new gates invented).
- **First BC run (`bc_demos_v1`, 32 epochs, 248k rows):** yaw MAE 10.61°,
  pitch MAE 4.73°, move drift 0.238, fire P/R 73.4%/59.6%, hidden-fire
  7.7%, weapon top-1 57.9% — **MISSES the bc_live_v2 gate** (2.23°/1.69°/
  0.0041/92.7%/0). Not promoted. Confounds: the corpus was only 6 episodes
  (6 bots × 1 map, q2dm2 — collection had been dead since Jul 14) and 3ZB2
  snap-aim is a harder imitation target than a distilled policy. Train loss
  was still descending at epoch 32.
- **Second BC run (`bc_demos_v2`, 96 epochs, 308k rows, 2026-09-02):** yaw
  MAE 13.51°, pitch 5.13°, drift 0.247, fire P/R 75.2%/60.5%, hidden-fire
  7.4%, weapon top-1 61.8% — **MISSES, worse than v1.** Train loss fell
  98→47 while holdout-adjacent MAE *rose* monotonically from epoch 1:
  textbook overfitting to a corpus with zero diversity. Conclusion: the
  binding constraint is corpus diversity, not model capacity or epochs.
- **Rotation root cause found 2026-09-02 (why the corpus is 100% q2dm2):**
  the in-engine `sv_maplist` rotation on this lithium/3ZB2 build has NEVER
  worked — journal shows zero `advanced to` lines in 5 days. With no human
  to press attack, intermission never exits (the wedge); the rare in-engine
  `gamemap` it does attempt **segfaults** (Signal 11, twice on 2026-09-02),
  and every systemd restart re-seeded the deterministic rotation back to
  q2dm2. **Fix:** wrapper-driven process-restart rotation in
  `tools/teacher_server.py` (commits `72de5a5`, `ca55961`): watch q2ded
  stdout for `Timelimit hit.`/`Fraglimit hit.`, relaunch on the next map
  (stock↔generated interleave kept), persist the stock draw count so
  service restarts resume mid-rotation, skip unmountable maps, crash
  backoff. Smoke-tested on the VPS: 4 clean rotations in 4 minutes,
  generated maps load and serve. **Deployed to `q2-teacher-server.service`.**
  Latent follow-up: the in-engine gamemap segfault also lurks on the public
  lane's human-driven rotations — needs a C-level investigation (backtrace
  capture was inconclusive; no core). Not blocking while the public lane
  rotates rarely and has crashed 0 times in 3 days.
- **Third BC run (`bc_demos_v3`, 96 epochs, 1.73M rows, 150 episodes,
  2026-09-02):** first run on the diversified corpus. yaw MAE 9.63° (best
  epoch ~8.5° around epoch 15), pitch 3.26°, drift 0.215, fire P/R
  78.9%/68.0%, hidden-fire 4.8%, weapon top-1 75.1% — **still MISSES every
  gate**, but improves on v2 across the board (13.51°→9.63°, 5.13→3.26,
  0.247→0.215, 61.8%→75.1%) with 5.6× the rows and 25× the episodes, and
  the holdout now includes generated maps. Diversity was real leverage.
- **Ceiling analysis (2026-09-02):** yaw MAE plateaus at ~8.5–9.6° while
  train loss keeps falling — the residual is irreducible per-tick jitter in
  the 3ZB2 snap-aim target, not a data or capacity shortfall. The
  bc_live_v2 bar (2.23°) was set cloning a *distilled policy* (smooth
  trajectories); 3ZB2 is a different, spikier distribution. Per the rule
  above, gates are NOT relaxed. **Open decision for the project owner:**
  (a) keep 3ZB2 as teacher and accept a re-based Phase 2 bar measured
  against 3ZB2's own conditional predictability, (b) switch the teacher to
  human capture (pipe is deployed; needs real sessions on the public
  lane), (c) treat `bc_demos_v3` as a good-enough prior and let Phase 3
  PPO do the polishing, or (d) distill from the fast-lane PPO policy once
  it passes the season gate. Default recommendation: (c)+(d) — v3 is a
  sane movement/posture prior, and aim quality should come from RL, not
  from imitating snap-aim jitter.
- **Corpus state:** ~860 batches / 1.73M rows, rotating every ~10 min
  across q2dm2/4/6/8 + mlteacher_*; human rows still 0 (no public-lane
  session since capture went live).

### Phase 3 — PPO polish under the season gate
- **Direction approved by project owner 2026-09-02:** option (c)+(d) from
  the Phase 2 decision point — `bc_demos_v3` is the movement/posture prior,
  aim quality comes from RL, and the fast-lane PPO policy becomes the
  distillation teacher once it passes the season gate. The 3ZB2 gates stay
  as-is (historical bar, not a blocker).
- **Season 1 (`bc_v3`) LAUNCHED 2026-09-02:** PPO warm-started from
  `checkpoints/bc_demos_v3/policy_bc_final.pt` (copied to
  `checkpoints/warm_bc_v3/policy_00000000.pt`), `--resume
  --reset_optimizer 1` (fresh Adam moments), `Q2_EXT_OBS=1` (219-dim input
  to match the prior — the ext block is always on the wire; the from-zero
  run consumes the same packets at 209-dim), `Q2_RUN_TAG=bc_v3` (isolated
  ckpt dir + TB run), disjoint port slabs (`Q2_SV_PORT_BASE=28410`,
  `Q2_ML_PORT_BASE=28600`). Half-size fleet: `--n_servers 6
  --n_bots_per_server 8 --n_ml_bots 2`, same reward stack and
  `mltrain_*` curriculum as the canonical run, tmux `q2_ppo_bc` on wsl-box,
  log `/tmp/q2_train.log.ppo_bc`. Runs alongside the from-zero control
  (`q2_ppo`, 209-dim), which continues as the Phase 1 clean-room test and
  the option-(d) distillation source. Warm start verified in log ("Resumed
  from ... policy_00000000.pt (env_steps=0)"); both lanes hold ~77 sps.
- Short PPO seasons warm-started from the BC baseline, judged by
  `tools/season_quality_gate.py` (100 generations / 1M steps / per-map
  episode coverage / KL and clip bounds / no-regression ladder).
- Bulk pretraining on the fast lane; verification and fine-tuning on the
  network-native lane (real engine conditions). The RTX 2080 runs ~15% GPU
  with two Phase B workers — 3-4 workers fit.
- Causal ladder order is fixed: posture → movement → aim → combat.
- **Season `bc_v3` progress (2026-09-02 13:30 PDT):** 618k steps at 137 min
  (~75 sps alongside the control). Latest windows show ep_r spikes to
  +13.08 with kd 0.67 (2/3) — early but clearly ahead of where the
  from-zero control was at comparable wall time. Control healthy at 4.77M
  steps (78 sps). Season gate runs at ~1M steps. Teacher corpus: 2.85M
  rows, 42 distinct maps, still 0 human rows (no public-lane human session
  yet; the AI-bot lane on 28002 now also captures humans).

### Phase 4 — Maps and public play
- Maps are a curation problem now, not a generation problem. Generator v6 +
  lighting v2 + lethal-edge guards are in production. Promote a fixed pool
  of 4-6 maps passing BOTH the safety contract and the map-judgment
  pipeline, anchored by the stock competitive set (q2dm1-8).
- **Per-round live map generation is training-only from now on.** It is
  good for generalization, bad for humans learning a map and for policy
  consistency. Public server runs the curated pool; live-maps on
  designated chaos/training windows only.
- **Public roster directive (2026-09-02, supersedes the old 2ML+2-3ZB2
  mix): the game initializes with AI (policy-driven) bots, NOT 3ZB2.**
  - Capability verified 2026-09-02: `tools/live_match_onnx.py` (ONNX
    Runtime, CPU-only, no torch on the VPS) boots a dedicated server with
    policy bots in the top slots; probe showed navigation (~170 ups mean),
    target acquisition, aligned fire, damage, kills and deaths inside 90s.
  - **Deployed: `q2-ai-bots.service` on the VPS, public UDP 28002** —
    4 AI bots (bc_demos_v3 clone, `serve_bc_v3.onnx`) + 2 human slots,
    mllive-farm maps interlaced with q2dm1/2/4/6/8, fast downloads via the
    gamedata service, human-capture cvars armed (bots excluded engine-side
    by the `zc.ml_enabled` guard). First live combat confirmed in the
    service journal. Note: bot entities are spawned through the mod's bot
    scaffolding, but slots >= ml_bot_slot are 100% policy-driven — 3ZB2 AI
    never thinks for them.
  - **Verified 2026-09-02 ~19:25 UTC:** bots fight (kills/deaths accrue),
    round ends at timelimit, and the engine rotates stock↔generated
    in-engine (`sv_maplist` + ML intermission auto-exit) — this works on
    ml_enabled lanes, unlike the pure-3ZB2 teacher lane. Two deployment
    fixes landed: the VPS harness/ tree was stale wire-v4 (expected 1032;
    synced to HEAD, 1060) and bot-only intermission can wedge — the lane
    runs `--intermission_maxtime 30` as the force-exit lever, with q2ded
    stdout captured via `Q2_SERVER_STDOUT_LOG=/home/q2mlbot/q2-ai-bots-server.log`.
    The unit currently runs timelimit 5 during the soak window.
  - Transitional: this lane currently interlaces live maps (a chaos window
    per the rule above). When the curated pool lands, it becomes the map
    source. Policy served is the current best checkpoint's ONNX; promotion
    = export a gated checkpoint, restart the unit. When a season-gated
    policy exists, this roster becomes the competitive target.
  - 28000 (`q2mlbot.service`, human/network-native lane) is untouched; the
    AI-bot lane proves itself on 28002 before any merge onto 28000.

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
