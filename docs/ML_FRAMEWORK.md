# ML Framework Plan

This project should treat ML bot support as an engine subsystem, not as scattered
bridge code inside the 3ZB2/Lithium bot path. The first target is still Quake II
Lithium training, but the boundaries should support future Yamagi engine work,
offline replay analysis, ONNX inference, and richer sensors.

## Runtime Architecture

The game module owns the authoritative simulation state. Each ML-enabled bot
flows through:

1. `ML_Config`: cvars, feature flags, protocol version, and backend selection.
2. `ML_Sensors`: observations derived from game state, traces, sound, map zones,
   voxel/spatial context, target exposure, and future lighting probes.
3. `ML_Transport`: backend-agnostic action exchange. Current backend is UDP to
   Python; future backends can be in-process ONNX or replay capture.
4. `ML_Actions`: converts model actions into Quake movement, view, weapon,
   grapple, and tactical commands.
5. `ML_Rewards`: damage, kills, deaths, pickups, exploration, visibility,
   positioning, and objective-specific shaping.
6. `ML_TacticalReasoner`: optional low-frequency tactical intent sidecar. This
   stays outside `game.so` and emits structured intent modifiers for the fast
   policy, replay critique, or reward labeling.

`Bot_Think` should only orchestrate this pipeline. It should not own sensor,
transport, reward, or policy details.

## Observation Standards

Keep observations versioned and fixed-layout for reproducibility. When possible,
add richer meaning without changing shape. Breaking changes require:

- a protocol version bump,
- Python parser update,
- checkpoint compatibility notes,
- evaluator baseline reruns.

Current compatibility-preserving fields:

- `entities[*].visible`: now represents target exposure from multiple body rays,
  while retaining `> 0.5` visible semantics.
- `rays[*]`: navigation depth rays.
- `audio`: recent sound direction, age, and alert strength.
- `hook_zones`: nearest annotated traversal opportunities.

Planned fields for a future protocol version:

- per-target exposure score and occlusion class,
- light level at bot and target,
- last-known target confidence,
- projectile danger vectors,
- local voxel occupancy/cover summary,
- semantic sound class and intensity.

**2026-07-10 evidence this is the right next investment**: live-deployment
measurement (post-bootstrap-fix, see `q2-ml-bot/README.md` § Known Issues)
found mean facing error to a visible, `entities[*].visible`-flagged target of
97.5° across 318 frames, with 0 of 40 fire events landing within the
training reward's own 12° alignment window. The raw `entities[*]` signal the
bot receives today (rel_pos/vel/health/is_enemy/visible) is apparently not
enough for the policy to reliably close the aim loop — richer exposure/
confidence/occlusion fields (as already planned above) are a plausible fix,
but it's also worth checking whether this is a sensing gap or a training
gap (i.e. does `R_FIRE_ALIGNED_REWARD`/`R_FIRE_UNALIGNED_PENALTY` actually
train the coupling, or is aim reward too weak relative to move/survive
reward to shape it) before assuming richer sensors alone will fix it.

## Modernization Goals

- Isolate ML code into `ml_*` modules with small headers.
- Keep protocol structs POD and explicitly versioned.
- Prefer deterministic evaluators and reusable scripts over ad hoc probes.
- Maintain local and WSL runtime parity.
- Build tests around observation packing, map validation, and 1v1/4-player
  evaluator baselines.
- Avoid hard-coding training-only behavior in general bot code; gate it behind
  ML cvars or ML-enabled bot state.

## Tactical LLM Sidecar

The fast LSTM/RL policy remains responsible for frame-level movement, aim, fire,
weapon, and grapple actions. A small language model can run at a lower rate to
summarize tactical intent:

- live default: `llama3.2:3b`,
- higher-quality local/proxy option:
  `qwen3-8b-gemini-3-pro-preview-high-reasoning-distill-q4_k_m:custom`,
- offline replay/map/reward analysis: `qwen2.5-coder:7b`.

The sidecar returns a compact packet:

```json
{"t":"take_cover","a":0.2,"c":0.8,"cb":0.9,"sw":0.5,"pw":0.6}
```

`engage` is valid even at low health when the estimated damage race is favorable:
for example, the opponent is weak, exposed, reloading, badly armed, or has shown
low accuracy. The prompt and scoring should treat low health as a risk factor,
not a hard retreat rule.

Python harness knobs:

- `Q2_TACTICAL_MODEL`: Ollama model name.
- `Q2_OLLAMA_HOST`: Ollama API endpoint, default `http://127.0.0.1:11434`.
- `Q2_TACTICAL_INTERVAL`: steps between LLM updates.
- `Q2_TACTICAL_TIMEOUT`: skip slow responses without blocking the match.

For the `proxy` host, open a tunnel before evaluation:

```bash
ssh -N -L 11435:127.0.0.1:11434 proxy
Q2_OLLAMA_HOST=http://127.0.0.1:11435 python tools/benchmark_tactical_models.py
```

Initial `proxy` results:

- `deepseek-r1:8b`: strongest tactical packet, high cold-load cost.
- `qwen2.5-coder:7b`: strong tactical packet, high cold-load cost.
- `lfm2.5:1.2b-instruct`: fast and valid, but more aggressive.
- `gemma4:e2b-agent`: timed out in the compact sidecar benchmark.

## First Milestone

Extract target sensing into `ml_sensors.c/.h` and make `ml_obs.c` consume it.
This makes ray-emission based targeting a real subsystem boundary and prepares
the code for lighting, sound, and voxel-aware target confidence without growing
`ml_obs.c` further. **Now has direct empirical motivation** (see Observation
Standards above) — this was a design-quality goal before 2026-07-10; it's now
also the leading suspect for why the bot doesn't aim.

## Roadmap Snapshot (2026-07-10)

Detailed findings/measurements live in `q2-ml-bot/README.md` § Known Issues;
summarized here for framework-level prioritization:

1. ~~Live single-bot deployment permanently inert~~ — **fixed**
   (`ml_bridge.c :: ML_RecvAction` bootstrap race).
2. **Aim/target-tracking** — not functioning; see evidence above. Likely
   next real milestone, ahead of further target-sensing extraction work
   *or* motivating it directly, depending on root cause (sensing vs. reward
   shaping vs. undertrained checkpoint).
3. **Lattice pull fidelity — prototype implemented, training proof pending.**
   The old checkpoint's "opportunity" pull is weak and "engagement" is
   inverted. PPO now has a yaw-aware direction objective over the existing
   24-d memory tail, masked during visible combat. A new checkpoint must pass
   `tools/evaluate_lattice.py` plus the fixed combat gate before this is closed.
4. ~~**`.lattice.json` preload**~~ — **implemented 2026-07-11.** Generator
   priors and route graphs are installed, preloaded, converted into live
   opportunity/threat readiness heat, and persisted beside policy checkpoints.
   Own pickups re-phase item clocks; enemy item visibility still needs an
   engine observation field for full contention inference.
