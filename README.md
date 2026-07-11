# q2-ml-bot

ML-based bot engine for Quake 2 (Lithium II + 3ZB2 base).

**Status:** Live-deployed. `checkpoints/policy_39929600.onnx` (39.9M training
steps) runs a public 1v1 human-vs-bot server via `tools/live_match_onnx.py`
(see **Live Deployment**). Movement/control is solid as of 2026-07-10. Aim
never converged through pure RL (root-caused and quantified 2026-07-11); a
behavior-cloning warm-start fix is validated and improving it, in progress —
see **Known Issues / Roadmap** item 2. **See `docs/HANDOFF-2026-07-11.md`
for the current state and next steps if picking this up fresh.**

## Architecture

```
q2_plugin/ml_bridge.{c,h}   C bridge in game.so: packs observations, applies actions
harness/protocol.py          Python-side struct serialization (must match bridge.h)
harness/env.py               Gymnasium environment wrapping a q2ded subprocess
harness/spatial.py           Voxel/spatial reward shaping from existing observations ("Voxel Lattice")
models/policy.py             LSTM actor-critic policy + ONNX export
train/ppo.py                 PPO training loop (parallel envs, overnight on Vega 10)
tools/live_match_onnx.py     ONNX-Runtime live match server — human-joinable, --live_maps
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

`OBS_DIM = 185 + 24 + (10 if Q2_EXT_OBS else 0)`. The deployed live checkpoint
(`policy_39929600.onnx`) was trained with `Q2_EXT_OBS=1` → 219 dims; running it
with the env var unset is a silent shape mismatch, not a training bug — see
`harness/protocol.py` and the `Q2_EXT_OBS` gotcha noted in project memory.

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
items, platforms, and hook affordances. `maps/compile.sh` installs compiled BSPs
and hook-zone sidecars into `q2_lithium_merge/baseq2/maps/`.

Use `--map_glob 'mltrain_*.bsp'` to train from the installed generated maps. Use
`--map_change_episodes 1` to restart each server onto a new sampled map after
each ML episode. Keep `--map_name q2dm1` for fixed benchmark runs.

Run `python tools/validate_maps.py --glob 'mltrain_*.map' --runtime` before
overnight training to check spawn spread, map scale, pickups, hook zones, and
q2ded loadability with a four-bot setup.

## KD Evaluation Gate

The project target is one ML bot versus three Lithium/3ZB2 opponents with a
15:1 kill/death ratio. Evaluate the latest checkpoint with:

```bash
python tools/evaluate_kd.py --map_glob 'mltrain_*.bsp' --steps 5000 --n_bots 4
```

The command exits non-zero until `kd_ratio >= 15.0`, so it can be used as the
promotion gate before treating a training run as successful.

## Live Deployment

`tools/live_match_onnx.py` runs a real-time, human-joinable server: ONNX
Runtime inference (CPU-only, ~66MB RSS vs. ~2.1GB for the torch+CUDA path —
this is what makes it viable on a small no-GPU VPS), a custom-built
`bot-compat-patches` yquake2 client/server (stock builds mismatch this
server's raised `MAX_MSGLEN`), and optionally `--live_maps`: a fresh
procedurally-generated, fully-lit map compiled in the background every round
instead of cycling a fixed pool.

```bash
python tools/live_match_onnx.py --onnx checkpoints/policy_39929600.onnx \
  --live_maps --dlserver http://<host>:<port>
```

Key flags: `--live_maps` (background `bsp -vis -fast -rad` compile per round,
armed via `sv_maplist`/`EndDMLevel` fallback since `use_mapqueue=0`);
`--dlserver` (HTTP `sv_downloadserver` instead of the legacy in-game UDP map
transfer — required once maps are never pre-bundled); `Q2_BIND_IP=""` (empty
string — some builds hit a `getaddrinfo` error on a literal `+set ip`, empty
string takes the working bind-all path instead).

A production instance (Hetzner VPS, `q2mlbot.service` + `q2mlbot-gamedata.service`
systemd units) has been running since 2026-07 with real external players.

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

2. **[IN PROGRESS, root cause confirmed, fix validated 2026-07-11]**
   Aim/yaw-tracking of visible targets is essentially non-functional.
   Measured across 318 frames with a visible enemy: mean facing error 97.5°
   (median 97.8°); only 6% of frames within the training reward's own 12°
   alignment threshold (`Q2_AIM_YAW_DEG`); 0 of 40 fire events happened
   while aligned. Confirmed via four independent measurements: this live
   telemetry, the historical TensorBoard training curve (`combat/kd_ratio`
   plateaued at ~1.5:1 for the entire second half of the `CUR` run, `40M`
   steps, no late-training improvement), a controlled `tools/evaluate_kd.py`
   re-run (0.0 kd, confirming this is a policy-quality issue, not the
   control-plane bug above), and an ML-vs-ML self-play match (1 kill/24
   deaths vs. 5 kills/26 deaths — fails identically against its own
   checkpoint, ruling out opponent-specific explanations).
   **Root cause**: pure PPO exploration never discovered the aim→fire
   association across 40M steps — the reward function has dozens of
   competing terms, and movement/survival/exploration reward is easier to
   accumulate than the sparse, precise skill of landing hits, so gradient
   never favored aim. **Fix**: `tools/behavior_clone_aim.py` — a supervised
   warm-start from a scripted geometric teacher (`atan2` toward nearest
   visible enemy) — already existed in the repo but had never been
   successfully run (see gotcha below). Fixed and run 2026-07-11
   (`--synthetic`, default hyperparameters, ~20s of training): mean facing
   error 97.5°→81.2°, within-12° rate 6.0%→8.0%, **fires-while-aligned
   0.0%→7.7%**. Directionally correct on the first, cheapest attempt — not
   yet good enough to deploy. New checkpoint: `checkpoints/policy_39929601.pt`
   (+ `checkpoints/policy_latest.onnx`), does not overwrite `CUR`.
   **Next steps**: more epochs/samples on `--synthetic`, then `collect()`
   mode (real rollouts, in-distribution data) instead of synthetic, then
   resume PPO fine-tuning *from this checkpoint* (not from scratch) so RL's
   job becomes integrating already-working aim rather than discovering it —
   watch `combat/kd_ratio` for a real break from the ~1.5:1 plateau, and
   watch for regression (RL overwriting the imitation-learned behavior) if
   fine-tuning LR is too high.

3. **[OPEN]** Lattice pull signals are weak, one is inverted from intent.
   During pure exploration (no visible enemy — the lattice is the only
   spatial signal): "opportunity" pull (`tools/steer.py`'s "value/target is
   here") correlates weakly *positively* with movement (cos-sim ≈ +0.11,
   n=75). "Engagement" pull ("expect combat here") correlates *negatively*
   (cos-sim ≈ −0.49, n=75) — the bot tends to move away from cells with
   past-engagement history, opposite of the concept's intent. During active
   combat, movement vs. opportunity pull is also mildly negative (≈ −0.22).

4. **[OPEN]** `maps/generator.py`'s `.lattice.json` sidecar (objective sites
   as opportunity priors, lava pools as danger priors) has been written for
   every generated map since the map-gen commit that added it, explicitly
   "so the bot starts with the map sense the generator already has" — but
   nothing in `harness/spatial.py` or `harness/env.py` has ever read it
   back. Wiring this preload in would give freshly-generated maps (and
   `--live_maps` rounds especially) a populated lattice from spawn instead
   of building purely from in-episode contact.

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
wire protocol or checkpoint shape. It rewards entering new position voxels,
visible tactical engagement ranges, and proximity to required hook zones, while
lightly penalizing stagnation.

Useful knobs:

- `Q2_SPATIAL_REWARD=0` disables the spatial bonus.
- `Q2_VOXEL_SIZE=256` sets world units per voxel.
- `R_VOXEL_NEW_CELL=0.02` controls exploration reward.
- `R_TACTICAL_ENGAGEMENT=0.01` controls visible-enemy engagement reward.
- `R_KILL=5.0` and `R_DEATH=3.0` emphasize the KD objective during training.

## Related repos

- [q2-lithium-3zb2](https://github.com/supere989/q2-lithium-3zb2) — game mod (game.so)
- [yquake2 fork](https://github.com/supere989/yquake2) — engine patches

## License

GPL v2 (engine-side C code) / GPL v3 (Lithium-derived mod code).
Python harness and model code: MIT.
