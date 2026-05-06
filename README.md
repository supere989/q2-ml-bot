# q2-ml-bot

ML-based bot engine for Quake 2 (Lithium II + 3ZB2 base).

**Status:** Early development — protocol and training harness scaffolded.

## Architecture

```
q2_plugin/ml_bridge.{c,h}   C bridge in game.so: packs observations, applies actions
harness/protocol.py          Python-side struct serialization (must match bridge.h)
harness/env.py               Gymnasium environment wrapping a q2ded subprocess
models/policy.py             LSTM actor-critic policy + ONNX export
train/ppo.py                 PPO training loop (parallel envs, overnight on Vega 10)
maps/                        Procedurally generated maps + hook-zone annotations
data/                        Recorded human demonstrations for imitation learning
```

## Observation space (185 floats)

| Component | Dims | Description |
|---|---|---|
| self_state | 10 | pos, vel, health, armor, weapon, ammo |
| entities | 72 | 8 visible entities × 9 features |
| rays | 64 | 16 directional depth traces × 4 |
| hook_zones | 32 | 4 nearest hook annotations × 8 |
| audio | 5 | sound direction, age, alert level |
| facing | 2 | normalised yaw/pitch |

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

# Train (ROCm GPU)
HSA_OVERRIDE_GFX_VERSION=9.0.0 python -m train.ppo --n_envs 4 --map_name q2dm1

# Train (CPU fallback)
python -m train.ppo --n_envs 4
```

## Related repos

- [q2-lithium-3zb2](https://github.com/supere989/q2-lithium-3zb2) — game mod (game.so)
- [yquake2 fork](https://github.com/supere989/yquake2) — engine patches

## License

GPL v2 (engine-side C code) / GPL v3 (Lithium-derived mod code).
Python harness and model code: MIT.
