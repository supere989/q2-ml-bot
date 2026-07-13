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
map bundles. Production uses `--map_farm_url` to pull and atomically install
them. `/health` reports queue depth/build state. If WSL is unavailable, the
live server keeps its current/armed map and retries; never restore local VPS
compilation as an automatic fallback. The worker's ignored `q2tool` asset is
copied into `~/q2-rollout/q2-ml-bot/maps/q2tools/` from the canonical WSL tree.

## Training Topology

Training runs on the Windows RTX 2080 box (DESKTOP-KDLBAE7), inside WSL.

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

**Active runtime (updated 2026-07-11):** the reward/terminal-fixed runtime was
promoted to the canonical `~/q2_lithium_merge` path. The active `q2_ppo`
migration uses that runtime, `Q2_EXT_OBS=1`, and disjoint port bases
33400/33500. Do not replace its `lithium/game.so` while this run is active:
servers reload it on round/map restart, so a copy silently changes reward
semantics mid-run. The pre-fix runtime is retained only for forensic rollback
at `~/q2_lithium_merge_DEPRECATED_pre_fixed_20260711`; never launch training
or evaluation from it. Production remains separate and was not touched.

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
