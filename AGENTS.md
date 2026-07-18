# Repository Guidelines

ML-driven bot system for Quake 2 (Yamagi engine + Lithium II mod + 3ZB2 bots).
A PPO-trained LSTM policy controls in-game bots over a UDP bridge; a voxel
spatial-memory system supplies tactical context. See `docs/ML_FRAMEWORK.md`.

## Project Structure

- `q2-ml-bot/` — Python training stack (canonical).
  `train/ppo.py` (PPO trainer), `models/policy.py` (LSTM actor-critic),
  `harness/` (env, UDP protocol, voxel/spatial rewards, tactical-LLM sidecar),
  `maps/generator.py` (procedural training maps), `tools/` (eval, monitoring).
- `engine/lithium/` — game-mod C source. **Standalone git repo**
  (`github.com/supere989/q2-lithium-3zb2`, branch `ml-wip-20260611`); the
  ML modules are `ml_bridge.c`, `ml_obs.c`, `ml_sensors.c`.
- `q2_lithium_merge/` — game runtime install (binaries/assets, untracked;
  only the `lithium/*.cfg` ML server configs are versioned).
- `docs/` — design notes, plus `architecture-map.svg` (full system diagram).

A production live-deployment instance (`tools/live_match_onnx.py`, ONNX
Runtime, human-joinable) has run on a separate Hetzner VPS since 2026-07 —
see `q2-ml-bot/README.md` § Live Deployment / § Known Issues for topology
and current findings.

**Live map farm (added 2026-07-12):** radiosity must not run on the production
VPS because it disturbs 10 Hz frame pacing. WSL runs the enabled user service
`q2-map-farm.service`, bound only to its Tailscale address at
`http://100.86.206.50:32510`, and maintains two checksum-attested compiled
map bundles. Generator v6 rejects player-admitting 56--95-unit gaps between
overlapping horizontal surfaces, requires a clear 96-unit spawn column and
escape path, and gives each enterable room-like zone a dedicated internal
light, and places a solid 96-unit guard wall on every playable floor-union edge
that faces lethal void. Spawn-clear floor regions require at least 98% direct coverage from
tagged lights of value 650 or higher. The validator recomputes the contract
from emitted brushes/entities and checks compiled BSP lightdata. Bundle v2
atomically carries BSP, hook zones, lattice priors, routes/item timing, and an
installed checksum manifest; the trainer reads sidecars from
`~/q2-rollout/runtime/baseq2/maps`. Production uses `--map_farm_url` to pull
and install the same bundle. `/health` reports queue depth/build state. If WSL
is unavailable, the
live server keeps its current/armed map and retries; never restore local VPS
compilation as an automatic fallback. The worker's ignored `q2tool` asset is
copied into `~/q2-rollout/q2-ml-bot/maps/q2tools/` from the canonical WSL tree.

**Interlaced live/teacher rotations (added 2026-07-12):** both lanes alternate
one stock map with one remotely compiled map so the WSL farm has a complete
stock-map round to replenish its queue. The public lane uses the shuffled stock
pool `q2dm1,q2dm3,q2dm5,q2dm7` plus `mllive_*` from port 32510. The isolated
3ZB2 teacher lane uses `q2dm2,q2dm4,q2dm6,q2dm8` plus `mlteacher_*` from a
separate queue on port 32513. The farms use disjoint seed ranges, independent
shuffle seeds, randomized queue claims, and strict consumer-side prefix checks.
Do not merge these queues or add `q2dm1` to the teacher pool.

The teacher q2ded is a separate VPS system service `q2-teacher-server.service`
on loopback port 28001, using `/home/q2mlbot/q2_teacher_runtime`; it must never
replace the public runtime on port 28000. Six legacy 3ZB2 bots use the dedicated
`ml6sk1` section (installed idempotently by the teacher controller) and send
pre-action observations plus their genuine actions over Tailscale UDP to WSL
port 32511. Keep `maxclients=8`: 3ZB2 needs two spare engine slots even though
the active player count is six. WSL's `q2-teacher-receiver.service` writes atomic batches beneath
`~/q2-rollout/live-3zb2/teacher_batches`. The dedicated receiver and map-farm
services are enabled user units on WSL.

The public `q2mlbot.service` uses `maxclients=6`; four network ML clients leave
two slots available for human players. Do not assume fixed slot numbers because
normal engine registration assigns whichever client slots are free.

**Network-native client harness successor architecture:** a separate
Yamagi checkout at `/home/raymondj/q2-ml-client`, pushed to
`supere989/yquake2` branch `feature/ml-client-harness`, now runs policies as
real protocol-34 player connections. `game.so` supplies privileged state
through a default-off, token-authenticated UDP conduit routed by the client's
`ml_client_id`; each client receives only its own `ml_obs_t`. Python support
is in `harness/client_{protocol,env}.py`; design and cutover gates are in
`docs/NETWORK-CLIENT-HARNESS.md`. The conduit is enabled only on the public
VPS/Tailscale lane. The architecture has demonstrated synchronous collection,
action/policy-version admission, stale rejection, map-epoch recovery, and
deterministic checks, but there is currently no admitted or running trainer.
Only a fresh B2-through-B6 evidence chain may authorize the next WSL trainer;
passing component tests or an isolated staging checkout cannot. The old
in-process public ONNX runtime is retired and has no operational selector or
rollback role.

**Current B2 authority (updated 2026-07-18):** final cohorts through 71453 are
permanently retired. Cohort 71453's first and only authorized final gate
invocation used the current alias while compiled-CM evidence bound the
byte-identical immutable path; `compiled-CM declaration binding differs` was
terminal and no gate was published. A second invocation was an unauthorized
diagnostic replay only. It exposed a separate blocker: the generic compact
JSON loader rejected the exact committed stock provenance writer bytes
(pretty/sorted JSON plus LF, SHA-256
`3ed2e930dcccf3abdabc7b5e1d9a1a95d74db4915a481bd523c51688c2bad030`).
Its canonical terminal authority is
`docs/multires/B2-GENERATED-COHORT-71453-FAILURE.json`; its named declaration,
current alias, maps, seeds, stages, reports, and artifacts are forensic only.
`ACTIVE_FINAL_AUTHORITY = None`. The 71454 lane is pre-declaration: its final
plan must preauthorize the exact immutable declaration path, and the
dedicated hash-pinned provenance writer-format loader and non-vacuous real-byte
regressions must first pass a completely fresh disposable qualification with
all eight retained infrastructure checks, including
`stock-provenance-writer-format`. Only a later, separate commit may add and
activate a fresh disjoint 71454 declaration. Gate publication, deployment,
trainer, and TensorBoard remain forbidden until that successor and every
B2-B6 cutover gate pass.

## Training Topology

Training runs on the Windows RTX 2080 box (DESKTOP-RTX2080), inside WSL.

**SSH — use the direct WSL path** (`~/.ssh/config` has both entries):
- `ssh wsl-box` → WSL's own tailscaled at `100.86.206.50` (preferred).
- `ssh win-box` → Windows host `100.104.16.95:2222`. **Trap:** this port
  reaches WSL's sshd only via localhost-forwarding; when the WSL VM is
  down (e.g. after a Windows update reboot — WSL never auto-starts), it
  silently falls through to the **Windows** sshd. Symptoms: host-key-change
  warning, cmd.exe errors like `'grep' is not recognized`, and `wsl -l -v`
  reporting "no installed distributions" (the ssh `raymond` profile does
  not own the distro — it belongs to another Windows profile; this does
  NOT mean the distro is gone).

**Recovery after a Windows reboot:** start WSL from the desktop session
(run `wsl` / launch Ubuntu). systemd inside the distro brings sshd and
tailscaled back automatically. Then relaunch tmux sessions:
- `q2_tb`: `python3 -m tensorboard.main --logdir runs --bind_all --port 6006`
  (plain `tensorboard` is not on tmux's PATH).
- `q2_ppo`: trainer command — recover it from `/tmp/q2_train.log.ppo` or
  the previous tmux session's `bash -c` line in `ps`.

**TensorBoard renders blank (board up, no data):** TB 2.13 crashes on
every request under protobuf ≥5.26 — `MessageToJson() got an unexpected
keyword argument 'including_default_value_fields'` (the arg was renamed
to `always_print_fields_with_no_presence`). The event files under
`runs/` are fine; only the server is broken. Confirm with
`grep -i traceback /tmp/tb.log`. Patched in place at three call sites
(`.bak-protobuf7` copies beside each) — re-apply if a `pip install
--upgrade tensorboard` reverts them:
```
cd ~/.local/lib/python3.10/site-packages/tensorboard
sed -i 's/including_default_value_fields=True/always_print_fields_with_no_presence=True/g' \
  plugins/hparams/hparams_plugin.py plugins/custom_scalar/custom_scalars_plugin.py
find plugins/hparams plugins/custom_scalar -name '*.pyc' -delete   # clear stale bytecode
```
Then restart the `q2_tb` tmux session. Verify data loads:
`curl -s localhost:6006/data/plugin/scalars/tags` should list the runs
and their tags. Permanent fix (do it during a training gap, not live):
bump TB to a protobuf-7-aware 2.16+ release. Note the trainer's protobuf
is untouched by this patch — only TB plugin files change.

**Local viewing from the Nobara workstation:** a `systemd --user`
service `wsl-tunnel.service` holds an SSH local-forward
(`-L 6006:localhost:6006 wsl-box`) with `Restart=always` + keepalives,
so `http://127.0.0.1:6006` is always live (auto-reconnects after drops
or while the WSL VM is down). `systemctl --user {status,restart}
wsl-tunnel`; logs via `journalctl --user -u wsl-tunnel -f`.

**Gotcha, fixed 2026-07-11**: this service had been silently broken (stuck
in an `activating (auto-restart)` loop, TensorBoard never actually
reachable) on nobara this whole time, independent of anything above — its
`ControlPath=%d/.ssh/cm-wsl-box.sock` referenced systemd's per-unit
credentials directory (`%d`), which is only created when the unit has a
`LoadCredential=`/`SetCredential=` directive; without one, `%d` resolves to
a path that's never created and ssh's `ControlMaster` socket bind fails
with `unix_listener: cannot bind to path ... No such file or directory`.
Fixed by switching to `%t` (the runtime directory, always exists) instead.
The same service now also runs on procreator (same fix applied there from
the start). If you ever add a similar persistent-tunnel unit, use `%t`, not
`%d`, unless you're also adding `LoadCredential=`.

**Training-data sync (added 2026-07-11)**: nobara pulls `checkpoints/` and
`runs/` from the WSL box automatically via `systemd --user`
`q2ml-training-sync.timer` (every 15 min, `~/sync_q2ml_training_data.sh`,
one-way `rsync -a` with `--delete` deliberately OFF — nobara has its own
unique recorded human-play session logs under `runs/` that must never be
clobbered). WSL box remains the sole active training location; this is
backup/visibility only, not a compute migration.

- `~/merge_mod/lithium` is the C git checkout on branch
  `ml-wip-20260611`. `~/q2-ml-bot` is currently an operational mirror with
  **no `.git` directory**; GitHub/local canonical changes must be copied there
  deliberately and checksum-verified. Do not assume `git pull` works in that
  Python tree, and do not rsync it blindly over its gitignored checkpoints,
  runs, maps, or machine-local diagnostics.
- Trainer runs in tmux session `q2_ppo`; log at `/tmp/q2_train.log.ppo`.
- TensorBoard on port 6006 (`http://100.86.206.50:6006` direct, the
  Windows portproxy at `http://100.104.16.95:6006`, or `http://127.0.0.1:6006`
  via the wsl-tunnel service above); events in `~/q2-ml-bot/runs/`.
- Headless `q2ded` servers on ports 27910+, configs `ml_server_*.cfg`.

**Current public-training state (updated 2026-07-14):** every model lineage
described below is retired and must not be resumed, exported, or deployed. The
old `q2_ppo` and `q2_tb` sessions and four network clients are stopped, and the
former current checkpoint/event directories have been moved out of operational
selectors without making a fallback copy. The public and teacher server
services remain online only until the multires protocol generation is replaced
atomically. Follow
`docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md` and its execution plan;
the paragraphs below are baseline evidence.

The
`public_network_engagement_anchor_v3` trainer and four clients are stopped. Its
step-4,063,488 checkpoint is quality-invalid: true pitch averaged about +86
degrees, down-look rate was 100%, and backward commands were 70.5%. Never
resume or promote that checkpoint. A second target/thermal canary loaded the
formerly eligible immutable step-4,055,296 policy with a fresh optimizer and
lattice, but reproduced 100% down-look and about 66% backward commands. It is
also rejected and archived under the staging tree.

The fresh `public_network_thermal_fresh_v1` experiment is stopped and archived.
It repaired posture and forward motion, but its step-49,152 zero-state holdout
still measured 39.6-degree yaw and 16.9-degree pitch MAE. The retired
`public_network_thermal_bc_live_v2` lineage starts from a 100k-sample,
32-epoch distilled-backbone clone of that exact step. It passed the fixed
holdout at 2.23-degree yaw, 1.69-degree pitch, 30.0% post-command alignment,
92.7% aligned-fire precision, zero hidden fire, and 0.0041 movement drift.
Head-only and 20k/8-epoch clones did not learn the geometry; do not promote
them merely because their loss converged.

The retired run used `Q2_EXT_OBS=1`, the Rust persistent lattice, deterministic
seed `7142026`, two PPO epochs, one full 512-sample recurrent minibatch, LR
`1e-5`, `aim_anchor_coef=0.02`, `aim_anchor_fire_weight=0`, and
`--reset_optimizer 1`. It resumes the matching 398-cell/four-client
`lattice_00049152.json.gz`; do not reset that lattice or point the resume path
at the archived fresh directory. Exact target vectors mean
eye-to-chest-first damage points clear from eye and common muzzle; entity
velocity is local relative velocity. Exposure magnitude is exact; positive is
fire-actionable and negative is shooter-protected tracking. A separate
per-client 64-unit thermal overlay cools for at most five ticks and is never
checkpointed. Old dynamic lattice state is invalid because it contains
pre-transform-fix deposits; rebuild it from attested map sidecars. See
`docs/TARGET-THERMAL-LATTICE-PROTOTYPE-2026-07-14.md`.

There is no active trainer or TensorBoard process. The former
`training-data/runs/current_public_network_thermal_bc_live_v2` and matching
checkpoint root are absent from operational paths. The public
client-telemetry credential is durably mirrored on WSL, mode 0600, at
`/home/raymond/q2-rollout/public-client-telemetry.env`. It is intentionally
different from `Q2_ROLLOUT_TOKEN`; sourcing the rollout-worker secret for the
public client conduit causes registration timeouts. Never print either value
or full client command lines.

The first 16,384 live generated-map transitions after the BC cutover had 1,539
aligned frames, 1,117 aligned shots, 49 hit events, 15 repeat-hit events, two
kills, 4.2% down-look, and zero failed rounds or echo timeouts. Forward
commands were 53.0% versus 33.3% backward. The generated-map prototype gate is
passed, but this is not a promotion. Require the same causal ladder on a
stock-map season before declaring combat quality.

The first fresh process correctly failed closed when all four conduits went
silent before emitting a map-boundary packet during an uncached generated-map
download. Python now classifies only a whole-batch timeout as a nontrainable
`telemetry_gap_resync`, holds dispatch while the conduit is silent, and then
enters the ordinary map-epoch barrier when packets resume. A partial-client
timeout remains fatal. TensorBoard exposes
`network_client/telemetry_gap_resyncs`; do not turn arbitrary one-client echo
timeouts into map boundaries.

The public client conduit is wire version 4. Its protocol-34 impulse and five
unused button bits carry a modulo-192 action generation plus hook/weapon
request; the server strips the private button bits before gameplay. Same-frame
action echoes are admissible only when that generation matches; C records the
latest intended angle relative to that decision generation's initial view in
the frame across multiple `ClientThink` calls. Never reuse the prior
generation's base or sum per-call deltas: same-frame prior actions and duplicate
held usercmds can otherwise contaminate or multiply the current action.
When same-generation movement and reliable commands prove delivery but an
intervening engine view/lifecycle change alters look or buttons, the result is
a nontrainable whole-batch resync (`network_client/action_state_resyncs`), not
a PPO transition and not a fatal transport error. Movement or hook/weapon
corruption remains fatal.
Do not downgrade the client, game module, or Python parser independently.

Do not point `Q2_RESUME_DIR` at the rolling checkpoint directory: `--resume`
always chooses its latest lexicographic triple. The immutable three-file pin
under `training-data/resume/public_network_live_v1_04055296` is retained only
for historical comparison; it is not a source for the active fresh lineage. Network
collection enters a persistent map-epoch barrier at intermission: it dispatches
no actions through BSP download/load, tolerates staggered clients without an
echo timeout, and clears only when every client reports the same non-intermission
target map. The first real generated-to-stock transition passed this barrier
and continued saving checkpoints with zero failed rounds or echo timeouts.

Quake stores the visible player-model pitch in `ent->s.angles[PITCH]` at one
third of the true view pitch. ML observations and target-local coordinates must
use `ent->client->v_angle`; C commit `f49caee` fixes that frame mismatch while
the server fire gate continues to use the same full-resolution view. Python
commit `f197fba` adds a forward-only level-posture reward plus explicit
entity/visibility/aim/damage audit tags. Never regress observation packing to
`s.angles[PITCH]` for normal network clients. Follow-up `fb85aa7` makes the
level-posture reward dense across the full pitch range while keeping the
15-degree penalty threshold and visible-target exemption. The true-view audit
proved 54--66% visible-contact frames on `mllive_44987431`, so the former
default-off anchor was promoted conservatively at coefficient 0.02. Keep batch
512 with a positive recurrent anchor; smaller minibatches break hidden-state
continuity.

The server engagement gate must not own the death-screen lifecycle. Registered
ML clients receive an internal attack button after their one-second death delay
so Quake can respawn them, while the recorded policy echo remains the gated
combat action. The 2026-07-13 proof held 123--148 units/s through updates 3--6,
where the broken run fell to zero after every client died.

Network fire is hard-masked unless sampled look ends within 12/14 degrees of
an authoritative visible enemy; the C server independently enforces a looser
14/16-degree shield and reports suppression for exact PPO reconciliation.
Target acquisition receives a bounded edge reward, and repeated hits on the
same target receive escalating offense credit until death/switch/timeout.

Hook is not a positive usage-rate objective. A live hook-zone landing must
advance toward positive lattice heat, and correction starts only for required
traversal, stuck/slow movement, or escape pressure. It pays bounded new-best
progress plus one arrival reward; blind/no-op/idle/overspeed costs remain.
Lithium's attached hook overwrites the complete velocity at
`hook_pullspeed=1700`, which can look like low gravity even though normal maps
reset to `sv_gravity=800`. Do not attempt to tune the old Python-emitted
`hook_gravity_comp`, `hook_min_lift`, `hook_pullscale`, or
`hook_pullspeed_max` cvars: the C hook never implemented them.

**Reproducible ablations (added 2026-07-11):** use all three controls together:
`--seed N --game_seed N --deterministic 1`, and keep `Q2_ML_ASYNC=0`.
`--seed` covers Python/NumPy/Torch/CUDA plus an independent RNG per spatial
reward instance; `--game_seed` enables the C `ml_game_seed` stream (unique per
server); `--deterministic 1` selects deterministic Torch/CUDA kernels. The
harness also pre-binds UDP and removes the boot sleep in seeded mode. Named
gameplay repeatability is lockstep-only; async mode still races wall time. The
default game seed is `-1`, preserving normal behavior. A 500-transition proof
matched SHA-256 across fresh same-seed launches and differed when only the game
seed changed. `train/ppo.py` now registers exception-time q2ded cleanup too;
if an older/abruptly-killed run leaves ports occupied, identify exact isolated
PIDs/ports before killing anything and never use a pattern that can touch the
active `~/q2_lithium_merge` servers.

## Live Deployment Gotchas

**Solo/few-bot deployments need a real timeout on `ML_RecvAction`'s bootstrap
path** (2026-07-10, fixed in `ml_bridge.c`). The "self-arming lockstep"
first-frame check originally used a non-blocking (`MSG_DONTWAIT`) recv to
avoid delaying other bots' spawn during multi-bot cold start. That path can
only ever succeed if the harness's reply is *already* sitting in the socket
buffer at the exact instant it's checked — during 40+-bot batched training
there's enough incidental inter-bot processing time within one frame for
that race to resolve favorably; a solo live bot has none of that slack, so
the check fails forever and the bot silently applies the zero-initialized
fallback action every tick ("stuck at spawn" — looks like a lattice/map
problem, isn't). Fixed with a bounded 50ms real timeout on the bootstrap
check specifically (not the normal-path timeout, to avoid reintroducing the
original multi-bot startup-delay problem). **If you ever see a live/solo
deployment produce a perfectly healthy, non-timing-out obs/action loop
where the bot simply never moves, check `action_debug`'s `accepted`/
`echo_move` fields (in `harness/protocol.py`'s `Observation.action_debug`,
exposed via a small `harness/env.py` info-dict patch) before assuming it's
spatial-reward or map-generation related** — dump a few frames and look
for the applied action actually being zero.

## Build, Test, and Development Commands

- Engine: from the standalone C clone, run `make` in
  `/home/raymondj/q2-lithium-3zb2` (procreator) or `~/merge_mod/lithium`
  (WSL); `engine/lithium` is only the path inside nobara's top-level workspace.
  This builds `lithium/gamex86_64.so`. The
  Makefile does not generate header dependencies: after changing `botstr.h`
  or any shared layout/header, run **`make clean && make -j4`**. A plain
  incremental build after the 2026-07-11 `zgcl_t` change linked mixed struct
  offsets and crashed in the first `G_RunFrame`.
  **Deploy**: copy the build to `q2_lithium_merge/lithium/game.so` — that is
  the filename `q2ded` actually dlopens; `gamex86_64.so` in the runtime dir is
  dead weight. Servers respawn per round, so a copied `game.so` goes live on
  the next round; create a separate scratch runtime while training is active.
- Train (on the WSL box): `python3 -m train.ppo --n_servers N --n_bots_per_server M --map_glob 'mltrain_*.bsp' --resume`.
- Maps: `python3 maps/generator.py` then `maps/compile.sh` (q2tools/ericw).
- Map sanity: `python3 tools/validate_maps.py` (4-player playability).
- Protocol check: `python3 tools/verify_protocol.py`.

## Conventions

- Observation/action structs are versioned PODs (`ml_bridge.h` ↔
  `harness/protocol.py`); any layout change requires a version bump on both
  sides and a note in `docs/ML_FRAMEWORK.md`.
- Reward weights are env-var overrides (`R_*`) today; record the full set in
  the run log when launching. Do not tune rewards mid-run.
- Checkpoints (`*.pt`, `*.onnx`) and `runs/` stay out of git.
- Commit style: short imperative subject, body explains the why
  (see `engine/lithium` history).
