# q2-ml-bot

ML-based bot engine for Quake 2 (Lithium II + 3ZB2 base).

**Status:** The network-native lattice trainer is live on the public server.
Four headless Yamagi clients connect as ordinary protocol-34 players while a
private per-client conduit supplies authoritative 219-feature observations to
PPO on WSL. It warm-started at 4,008,192 steps from
`movement_reset_v2/policy_04008192.pt`; two of six slots remain available for
humans. The old in-process ONNX runtime is archived/rollback-only. Transport
and action admission are proven. The fixed run started from the clean
4,030,720-step policy/optimizer/lattice checkpoint and produced its first
validated post-fix checkpoint at 4,038,912 steps, with a target-aligned fire
gate, bounded target-acquisition reward, escalating same-target hit credit,
automatic network-client respawn, attested generated-map lattice sidecars, and
lattice-directed hook corrections. Combat quality remains below the
promotion target and must be judged by the seasonal soak, not loss convergence.
**See `docs/HANDOFF-2026-07-13-NETWORK.md` for the exact active topology,
commits, rollback locations, validation evidence, and next checks.**

## Architecture

```
q2_plugin/ml_bridge.{c,h}   C bridge in game.so: packs observations, applies actions
harness/protocol.py          Python-side struct serialization (must match bridge.h)
harness/env.py               Gymnasium environment wrapping a q2ded subprocess
harness/client_env.py        Real protocol-34 client backend + private authoritative telemetry
harness/spatial.py           Voxel/spatial reward shaping from existing observations ("Voxel Lattice")
models/policy.py             LSTM actor-critic policy + ONNX export
train/ppo.py                 PPO training loop (parallel envs, overnight on Vega 10)
tools/live_match_onnx.py     ONNX-Runtime live match server — human-joinable, --live_maps
tools/network_public_server.py Server-only public lane for ordinary training clients
harness/client_batch.py      Synchronous multi-client action/echo admission and map resync
maps/                        Procedurally generated maps + hook-zone annotations
data/                        Recorded human demonstrations for imitation learning
```

A full visual map (training pipeline, C bridge internals, production topology,
and current known issues) lives at `docs/architecture-map.svg`.

## Observation space (185 base, up to 219 with extensions)

| Component | Dims | Description |
|---|---|---|
| self_state | 10 | pos, vel, health, armor, weapon, ammo |
| entities | 72 | 8 visible entities × 9 features |
| rays | 64 | 16 directional depth traces × 4 |
| hook_zones | 32 | 4 nearest hook annotations × 8 |
| audio | 5 | sound direction, age, alert level |
| facing | 2 | normalised yaw/pitch |
| **base subtotal** | **185** | always present |
| ext_obs *(Q2_EXT_OBS=1)* | 10 | rune_flags[5] + inbound_dmg_dir[3] + dist/recency[2] |
| session_memory | 24 | per-voxel-cell lattice: engagement/threat/opportunity/self-fire scores + nearest-signal pull vectors + survivability projection (win_margin, effective-HP, DPS-share) — see `harness/spatial.py` |

`OBS_DIM = 185 + 24 + (10 if Q2_EXT_OBS else 0)`. The active public trainer
and its `policy_04008192.pt` warm start use `Q2_EXT_OBS=1` → 219 dims. Loading
that checkpoint with the variable unset fails at the encoder with a ten-input
shape mismatch; see `harness/protocol.py` and the handoff gotcha.

## Action space (8 dims)

| Index | Range | Meaning |
|---|---|---|
| 0 | [-1, 1] | move forward/back |
| 1 | [-1, 1] | strafe right/left |
| 2 | [-45, 45] | look yaw delta (°/tick) |
| 3 | [-30, 30] | look pitch delta (°/tick) |
| 4 | [0, 1] | jump |
| 5 | [0, 1] | fire |
| 6 | [0, 3] | hook (0=idle 1=fire 2=hold 3=release) |
| 7 | [0, 9] | weapon select (0=no change) |

## Setup

```bash
# Dependencies
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install onnxruntime gymnasium stable-baselines3

# Verify protocol/policy
python tools/verify_protocol.py

# Generate and install procedural maps
bash maps/compile.sh --batch 8

# Train on the generated-map curriculum
python -m train.ppo --n_servers 4 --n_bots_per_server 4 \
  --map_glob 'mltrain_*.bsp' --map_change_episodes 1 \
  --total_steps 20000000 --resume

# Managed RTX/WSL service with TensorBoard
bash tools/train_service.sh start
```

## Generated-Map Curriculum

`maps/generator.py` creates Lithium/3ZB2 deathmatch maps with spawns, weapons,
items, platforms, hook affordances, and measured floor lighting. Spawn-clear
base floors are divided into deterministic 512-unit regions and sampled every
128 units. At least 90% of every region must have a direct tagged point light;
the generator adds lights below platforms and building roofs when those block
the ceiling source. `maps/compile.sh` installs compiled BSPs, hook zones,
lattice priors, and route/item-timing sidecars into
`q2_lithium_merge/baseq2/maps/`.

Use `--map_glob 'mltrain_*.bsp'` to train from the installed generated maps. Use
`--map_change_episodes 1` to restart each server onto a new sampled map after
each ML episode. Keep `--map_name q2dm1` for fixed benchmark runs.

Run `python tools/validate_maps.py --glob 'mltrain_*.map' --runtime` before
overnight training to check spawn spread, map scale, pickups, hook zones, and
q2ded loadability with a four-bot setup. Static validation recomputes lighting
from the point lights in the `.map` and the floor samples/occluders in
`.meta.json`; missing, weak, moved, or platform-occluded lights fail the map.
Use `--min-light-coverage` only to raise the default 0.90 promotion floor.

## KD Evaluation Gate

The project target is one ML bot versus three Lithium/3ZB2 opponents with a
15:1 kill/death ratio. Evaluate the latest checkpoint with:

```bash
Q2_ROOT=/home/raymond/q2_lithium_merge Q2_EXT_OBS=1 \
python tools/evaluate_kd.py \
  --checkpoint checkpoints/aim_ppo_canary_v2/policy_39940354.pt \
  --map_glob 'mltrain_*.bsp' --max_maps 1 --steps 5000 --n_bots 4 \
  --game_seed 8801
```

The command exits non-zero until `kd_ratio >= 15.0`, so it can be used as the
promotion gate before treating a training run as successful. The canonical
WSL runtime now contains the reward/terminal fixes; use disjoint port bases for
evaluation so it cannot collide with active training.

## Live Deployment

`tools/live_match_onnx.py` runs a real-time, human-joinable server: ONNX
Runtime inference (CPU-only, ~66MB RSS vs. ~2.1GB for the torch+CUDA path —
this is what makes it viable on a small no-GPU VPS), a custom-built
`bot-compat-patches` yquake2 client/server (stock builds mismatch this
server's raised `MAX_MSGLEN`), and optionally `--live_maps`: a fresh
procedurally-generated, fully-lit map compiled in the background every round
instead of cycling a fixed pool.

```bash
python tools/live_match_onnx.py --onnx checkpoints/policy_43507714.onnx \
  --live_maps --map_farm_url http://100.86.206.50:32510 \
  --dlserver http://<host>:<port>
```

Key flags: `--live_maps` (fresh-map rotation armed through
`sv_maplist`/`EndDMLevel` since `use_mapqueue=0`; standalone mode compiles
locally);
`--map_farm_url` (consume checksum-attested maps from the private WSL worker
instead of compiling on the live VPS; production keeps a two-map reserve);
`--dlserver` (HTTP `sv_downloadserver` instead of the legacy in-game UDP map
transfer — required once maps are never pre-bundled); `Q2_BIND_IP=""` (empty
string — some builds hit a `getaddrinfo` error on a literal `+set ip`, empty
string takes the working bind-all path instead). Production also sets
`Q2_ML_STEP_TIMEOUT_MS=75`: normal ONNX replies take 1–14 ms, so an inference
failure falls back within the current 100 ms frame instead of stalling the
server for the training-oriented one-second default.

A production instance (Hetzner VPS, `q2mlbot.service` + `q2mlbot-gamedata.service`
systemd units) has been running since 2026-07 with real external players. The
versioned live unit is [`ops/q2mlbot-live.service`](ops/q2mlbot-live.service).

## Known Issues / Roadmap

Findings below are from live-deployment investigation on 2026-07-10
(instrumented C-level tracing + quantitative behavioral analysis against a
fixed checkpoint). Ordered by what's blocking what.

1. **[FIXED]** `ml_bridge.c :: ML_RecvAction()` bootstrap race — the
   first-frame "self-arming lockstep" check used a non-blocking
   (`MSG_DONTWAIT`) recv to avoid delaying other bots' spawn. Escaping that
   bootstrap path required Python's reply to already be buffered at the exact
   instant it ran; training's 48-concurrent-bot batched timing gave enough
   incidental slack for that race to resolve, but a solo live bot never
   could, so every action applied as the zero-initialized fallback forever
   ("stuck at spawn," reported first as a live-maps/lattice suspicion, root
   cause was unrelated to either). Fixed with a bounded 50ms real timeout on
   the bootstrap check. Deployed to production.

2. **[IN PROGRESS — corrected BC, fixed-runtime PPO, and seeded ablations
   validated 2026-07-11]** Aim/fire remains the main policy-quality blocker,
   but the first investigation's 97.5°→81.2° telemetry must not be reused:
   that scratch measurement subtracted global yaw even though the C bridge's
   `rel_pos` is already bot-local, and the first BC run trained on
   unnormalised synthetic vectors with an almost-always-fire teacher. The
   formal tools now use the exact Quake `AngleVectors` inverse, post-command
   alignment (look is applied before attack), pitch clamping, real legacy-bot
   identity, and the policy's actual normalised input layout.

   `tools/behavior_clone_aim.py` was rebuilt around those invariants. It now
   uses `fire_logits_for()`, preserves non-aim behavior by distillation,
   supports exact real-rollout observations plus synthetic replay, writes
   isolated numeric PT/ONNX pairs atomically, and never collides with the
   active trainer's `policy_latest.onnx`. The best balanced warm start is
   `checkpoints/aim_bc_rollout_blend_v1/policy_39929602.pt` (WSL box,
   gitignored): on a fixed 10k synthetic holdout it has 3.06° yaw MAE, 1.88°
   pitch MAE, 28.96% predicted-post alignment, 92.51% aligned-fire precision,
   0% hidden fire, and preserves the reference non-aim heads. Real q2dm1
   rollouts are much harder/noisier (roughly 8–14% post alignment), so this
   is a warm start, not a deployable bot.

   PPO canary v2 started from that balanced BC checkpoint on the isolated,
   reward/terminal-fixed runtime. It completed 10,752 environment steps at
   `checkpoints/aim_ppo_canary_v2/policy_39940354.pt` with bounded optimizer
   traces (loss 0.38–2.07; returns no longer contain 100,000-point sentinels).
   Synthetic aim was retained: yaw MAE 3.05°, pitch MAE 1.93°, aligned-fire
   precision 92.43%, hidden fire 0%, and 99.3–99.8% non-aim agreement. On the
   same fixed generated-map 5k-step K/D gate it improved the BC baseline from
   3/64 (0.0469, 373 damage) to 4/60 (0.0667, 552 damage); episodes exactly
   matched deaths and both runs had zero timeouts. This is a real, modest
   improvement, still nowhere near the 15:1 target. Do not extend canary v1:
   it trained its critic/shared state on corrupted sentinel returns.

   The follow-up `vf_coef` A/B is complete and **does not support lowering the
   value weight**. Three deterministic training-seed pairs (3101/4201/5301),
   each gated on gameplay seeds 8801 and 9901, produced fixed-map totals of
   11/405 for `vf_coef=0.1` and 13/469 for `0.01`; real-q2dm1 totals were
   7/118 and 5/96. Pair directions were inconsistent. Lowering the coefficient
   cut weighted value gradients 90.4%, shared movement 35.3%, and clipping
   pressure, but actor parameter movement stayed flat. Keep `0.1` by inertia,
   not because three pairs prove it optimal; none of these artifacts is a
   deployment candidate.

   Reproducibility is now explicit: `--seed` covers Python/NumPy/Torch and
   per-venv spatial shaping, `--game_seed` names the C gameplay RNG stream,
   and `--deterministic 1` enables deterministic Torch/CUDA kernels. A
   500-transition lockstep replay matched byte-for-byte across two launches
   and changed when only the game seed changed. Repeatability is lockstep-only;
   `Q2_ML_ASYNC=1` remains wall-scheduling-dependent.

   A default-off recurrent aim/fire anchor (`--aim_anchor_coef`) was also
   tested because the 3° synthetic holdout uses zero LSTM state while live
   recurrent chunks initially showed roughly 34° yaw error. At coefficient
   `0.1`, a seed-5301 canary retained the synthetic gate and improved pooled
   real-q2dm1 post-alignment from 13.45% to about 21.35%, but failed the fixed
   generated-map gate: 5/129 fell to 2/162. It remains an experimental switch,
   default `0`; do not extend that checkpoint. If revisiting it, isolate a
   weaker or look-only anchor and require the same multi-seed fixed-combat gate
   before any longer PPO run. Leave expensive gradient diagnostics disabled
   for ordinary training.

3. **[PROTOTYPE 2026-07-11]** Lattice pull signals in the old checkpoint are
   weak, and one is inverted from intent.
   During pure exploration (no visible enemy — the lattice is the only
   spatial signal): "opportunity" pull (`tools/steer.py`'s "value/target is
   here") correlates weakly *positively* with movement (cos-sim ≈ +0.11,
   n=75). "Engagement" pull ("expect combat here") correlates *negatively*
   (cos-sim ≈ −0.49, n=75) — the bot tends to move away from cells with
   past-engagement history, opposite of the concept's intent. During active
   combat, movement vs. opportunity pull is also mildly negative (≈ −0.22).
   PPO now includes a yaw-aware lattice-direction objective (default coefficient
   `0.02`) that attracts toward engagement/opportunity, repels from threat, and
   masks itself during visible combat. This is implemented and unit-gated but
   still needs a newly-trained checkpoint to prove the measured inversion fixed.

4. **[FIXED/PROTOTYPE 2026-07-11]** `maps/generator.py`'s `.lattice.json`
   objective/lava priors are now installed and preloaded by
   `VoxelSpatialReward.reset()`. `.routes.json` now seeds a per-bot item timing
   clock whose readiness and selected offense/survival/balanced route appear
   through the existing opportunity/threat vectors. Learned lattice cells are
   saved beside policy checkpoints as `lattice_*.json.gz` and restored on
   `--resume`. Engine item-visibility IDs are still absent, so the prototype
   re-phases timers from the bot's own nearest pickup plus predicted respawns;
   enemy pickup inference remains future wire work.

5. **[FIXED 2026-07-11]** `tools/behavior_clone_aim.py`'s `train_bc()` crashed
   immediately on `KeyError: 'fire_logits'` — it predates `models/policy.py`'s
   fire decision becoming an autoregressive head conditioned on the chosen
   weapon (`Q2BotPolicy.fire_logits_for(feat, weapon_idx)`, not a plain key
   in `forward()`'s output dict). This means the tool could never have run
   successfully since that architecture change — it existed as dead code.
   Fixed with teacher-forcing on the demonstration's ground-truth weapon
   choice. If you hit the same `KeyError` elsewhere, the fix pattern is:
   compute `weapon_idx = <ground-truth-or-sampled-weapon>.long().unsqueeze(1)`
   then `policy.fire_logits_for(act_params["feat"], weapon_idx)`.

6. **[environment gotcha, not a bug]** The `ml2sk1` botlist (2-bot server
   configs, `n_bots=2`) doesn't spawn anything on the WSL/RTX2080 box —
   server initializes cleanly, no error, but zero bots ever connect and the
   harness times out waiting for obs. Root cause not investigated (likely a
   missing/misconfigured `3zb2/` botlist file specific to that size). Every
   working config this session used `n_bots=4`/`ml4sk1`-or-larger sizing
   instead — e.g. a clean 1v1 ML-vs-ML match currently requires borrowing
   the 4-bot config with 2 slots redirected to ML (`num_ml_bots=2`), which
   leaves 2 incidental 3ZB2 opponents also in the match (see
   `tools/ml_vs_ml.py`, added this session — pits N ONNX policy instances
   against each other, zero 3ZB2 by default if `n_bots=num_ml_bots`, but see
   this gotcha before using a small `n_bots`).

7. **[FIXED AND DEPLOYED TO TRAINING AND PRODUCTION]** ML reward accounting
   used raw Quake damage. Kill planes,
   telefrags, crushers, and the out-of-bounds fallback deliberately pass
   `100000`, which entered PPO returns unchanged; corpse hits could also add
   damage and repeat ML kill credit. The impossible totals were exact
   sentinel multiples (`3,704,740 = 37 × 100,000 + 4,740`). `T_Damage()` now
   caps reward damage to actual live health removed, ignores corpse/gib
   reward, requires a client attacker for proximity direction, and awards a
   kill only on an alive→dead transition. Gameplay damage is unchanged. On
   the same 5k gate, the old runtime reported 18/70 and 3,704,740 damage
   taken; the fixed isolated runtime reported 1/70 and 7,121, proving that
   17 of 18 apparent kills were corpse credits and the critic spikes were
   sentinel returns, not useful learning.

8. **[FIXED, VERIFIED, AND DEPLOYED TO TRAINING AND PRODUCTION]** Terminal
   delivery had two independent bugs: lockstep sent the same death terminal
   from both the frame pre-pass and dead `Bot_Think`, while every intermission
   frame was terminal. The latter produced runs with 2,769 episode endings in
   3,072 transitions and mean episode length 1.027. C now success-gates a
   one-shot death/intermission flag on every sync/async send path and exits a
   bot-only intermission only after every ML slot has sent its boundary.
   Python discards a true same-tick replay but preserves a valid same-tick
   nonterminal→terminal promotion, lower ticks after map reload, and
   same-new-tick split deltas. Integration checks produced
   three intermission terminals followed by three clean rounds, and 2 death
   terminals for exactly 2 death rewards, with zero timeouts. Formal gates now
   report episodes equal to deaths (64/64 and 60/60).

9. **[build gotcha]** This C Makefile does not track header dependencies.
   After changing `botstr.h`/another shared struct, plain `make` can link a
   mixed-layout `game.so` that crashes immediately. Use `make clean && make
   -j4` for header/layout changes. Continue copying only the resulting
   `lithium/gamex86_64.so` to an isolated runtime's `lithium/game.so` while
   the main trainer is live.

See `docs/architecture-map.svg` for the full system diagram with these
findings annotated in place.

## Tactical LLM Sidecar

`harness/tactical.py` can query Ollama at a low frequency and produce a compact
tactical intent packet for logging or optional action biasing. The policy input
shape and checkpoints are unchanged.

Recommended model roles:

- `llama3.2:3b` for live local tactical reasoning.
- `qwen3-8b-gemini-3-pro-preview-high-reasoning-distill-q4_k_m:custom` for
  higher-quality live reasoning when GPU budget allows.
- `qwen2.5-coder:7b` for offline replay, map, and reward analysis.
- On the `proxy` Ollama endpoint, `deepseek-r1:8b` and `qwen2.5-coder:7b`
  produced the strongest tactical packets in the initial compact benchmark;
  `lfm2.5:1.2b-instruct` was fast but more aggressive.

Low health is only one input. `engage` can still be correct when the estimated
damage race is favorable due to opponent weakness, exposure, reload timing,
weapon mismatch, or skill estimate.

Benchmark local Ollama:

```bash
python tools/benchmark_tactical_models.py \
  --models llama3.2:3b qwen2.5-coder:7b
```

Use the `proxy` Ollama endpoint through SSH:

```bash
ssh -N -L 11435:127.0.0.1:11434 proxy
Q2_OLLAMA_HOST=http://127.0.0.1:11435 \
  python tools/benchmark_tactical_models.py --host http://127.0.0.1:11435
```

Run 1v1 with sidecar logging only:

```bash
python tools/evaluate_1v1.py --max_maps 1 --steps 1000 \
  --tactical_model llama3.2:3b --tactical_interval 20
```

Add `--tactical_apply` only for experiments where intent packets should
conservatively modify actions.

## Spatial Reward Shaping

`harness/spatial.py` adds a training-side voxel reward without changing the C
wire protocol or checkpoint shape. It rewards entering new position voxels and
visible tactical engagement ranges while lightly penalizing stagnation.
Generated-map objective/danger priors and live route/item readiness feed the
existing 24-dimensional memory tail. The trainer
checkpoints learned cells alongside the policy and explicitly supervises
movement along the pull vectors outside visible combat.
The implementation contract and isolated real-engine smoke results are in
`docs/LATTICE-PROTOTYPE-2026-07-11.md`.

Fresh policies also use a grounded-locomotion curriculum. Their categorical
heads start jump-off/hook-idle instead of the old uniform 50% jump / 75% hook
distribution. The spatial reward targets 220–360 units/s when meaningful
movement is requested, rewards forward traversal inside that window, and
penalizes slow jumping plus unnecessary hook overspeed. TensorBoard exposes
the result under `movement/*`, `behavior/jump_*`, and
`behavior/hook_overspeed_rate`. Generated maps now carry eight geometry-checked,
384-unit-separated deathmatch starts, leaving spare starts for six-player live
matches instead of forcing spawn reuse.

Three arena-focused presets provide deliberate spatial composition:
`arena_open` is flat and cover-rich, `arena_vertical` combines arena bowls
with terraces/stairs/platforms, and `arena_lanes` emphasizes ground-level
lanes and chokepoints. Arena presets enforce at least two 256-unit-wide
through-hallways, repeated L-shaped corner pockets, low/mid/high ceiling
bands, and one or more enterable 384–448-unit buildings with opposed doors,
interior ceilings, and playable roofs. Their counts are recorded in each
map's metadata as `hallways`, `corner_pockets`, `corners`, `large_buildings`, and
`ceiling_bands`. `mixed` includes these presets alongside the original four.
The live and teacher map farms additionally maintain a 50% arena-style quota
in each ready queue (one of two public bundles and two of four teacher
bundles), while the remaining capacity preserves the original style family.

Generator v5 treats lighting and movement clearance as promotion contracts.
It rejects overlapping horizontal surfaces with a player-admitting 56--95-unit
gap, requires each spawn to have a clear 96-unit column and supported escape
path, raises direct floor-light coverage to 98%, and assigns every enterable
room, hallway, corner pocket, building, and safe under-platform space its own
internal light. Map-farm bundle v2 delivers the compiled BSP together with
hook zones, lattice priors, routes/item timing, and a checksum manifest; the
trainer fails closed on missing or corrupt farm sidecars. See
`docs/MAP-LIGHTING-GEOMETRY-CONTRACT.md` for the exact thresholds.

Hook telemetry distinguishes fire, release, and the reserved class-2 no-op,
but hook use is no longer a positive rate/value objective. A correction can
start only for required traversal, stuck/slow movement, or escape pressure,
and only when a live hook-zone landing advances toward a positive heated
lattice cell. One fixed correction pays bounded new-best displacement plus one
arrival bonus; replaying the same ground cannot farm reward. Blind fire, the
class-2 no-op, idle release, and overspeed retain their costs. TensorBoard's
`behavior/hook_action_rate` remains diagnostic only; authoritative progress is
under `hook/target_*`, `hook/progress_*`, and `hook/correction_*`.

The live server still uses Lithium's real `hook_pullspeed 1700` actuator. In
the current C implementation an attached hook replaces the player's complete
velocity every frame, so it can look like low gravity even though maps without
a gravity key reset to normal `sv_gravity 800`. The former
`hook_gravity_comp`, `hook_min_lift`, `hook_pullscale`, and
`hook_pullspeed_max` config lines were inert and are no longer emitted.

Prototype smoke test and checkpoint regression gate:

```bash
python tools/evaluate_lattice.py --map mltrain_00005200
Q2_EXT_OBS=1 python tools/evaluate_lattice.py \
  --checkpoint checkpoints/<run>/policy_<steps>.pt \
  --require_mean_cosine 0.25
```

The map check exits non-zero if lattice/routes sidecars or their live deposits
are missing. The checkpoint gate probes opportunity, engagement, and threat
across cardinal directions and four yaw angles. Use it with the fixed combat
gate; vector alignment alone is not a promotion criterion.

Useful knobs:

- `Q2_SPATIAL_REWARD=0` disables the spatial bonus.
- `Q2_VOXEL_SIZE=256` sets world units per voxel.
- `R_VOXEL_NEW_CELL=0.02` controls exploration reward.
- `R_TACTICAL_ENGAGEMENT=0.01` controls visible-enemy engagement reward.
- `Q2_LATTICE_PRELOAD=0` disables generated prior loading.
- `Q2_LATTICE_ROUTES=0` disables live route/item readiness deposits.
- `--lattice_direction_coef 0.02` controls direct pull-vector supervision
  (`0` disables it for an A/B control).
- `Q2_NOMINAL_SPEED_MIN=220` / `Q2_NOMINAL_SPEED_MAX=360` define the ordinary
  traversal window; `R_MOVE_*`, `R_JUMP_*`, and `R_HOOK_OVERSPEED` tune its
  reward terms.
- `R_KILL=5.0` and `R_DEATH=3.0` emphasize the KD objective during training.

## Related repos

- [q2-lithium-3zb2](https://github.com/supere989/q2-lithium-3zb2) — game mod (game.so)
- [yquake2 fork](https://github.com/supere989/yquake2) — engine patches

## License

GPL v2 (engine-side C code) / GPL v3 (Lithium-derived mod code).
Python harness and model code: MIT.
