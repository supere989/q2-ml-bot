# q2-ml-bot

ML-based bot engine for Quake 2 (Lithium II + 3ZB2 base).

**Status:** Live-deployed. `checkpoints/policy_39929600.onnx` (39.9M training
steps) runs a public 1v1 human-vs-bot server via `tools/live_match_onnx.py`
(see **Live Deployment**). Movement/control is solid as of 2026-07-10; aim and
lattice-driven tactics are not — see **Known Issues / Roadmap**.

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

2. **[OPEN, likely highest-impact]** Aim/yaw-tracking of visible targets is
   essentially non-functional post-fix. Measured across 318 frames with a
   visible enemy (two live runs): mean facing error 97.5° (median 97.8°);
   only 6% of frames within the training reward's own 12° alignment
   threshold (`Q2_AIM_YAW_DEG`); **0 of 40 recorded fire events happened
   while aligned within that threshold.** Movement and target-detection
   clearly work (distance/contact correctly modulate behavior) — the
   yaw-to-target coupling specifically does not. This is very likely why the
   **KD Evaluation Gate** (15:1) has never been confirmed passed: an
   unaimed bot cannot land hits at any real rate regardless of positioning
   quality. Worth root-causing before further training runs — possibly
   related to `ML_FRAMEWORK.md`'s planned target-sensing extraction
   (`ml_sensors.c`) and richer exposure/confidence fields.

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
