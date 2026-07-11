# Vector Lattice Prototype — 2026-07-11

This pass closes the first end-to-end Vector/Voxel Lattice prototype without
changing the C wire protocol or the policy's 24-dimensional session-memory
tail.

## Implemented path

1. `maps/compile.sh` installs `.lattice.json` and `.routes.json` beside each
   generated BSP.
2. `VoxelSpatialReward.reset()` searches installed/generated map roots and
   preloads objective cells as opportunity plus lava cells as threat.
3. A per-bot `ItemTimingTable` consumes route nodes. Its readiness plumes and
   health-selected offense/survival/balanced route bias are rebuilt each tick
   into the existing opportunity/threat channels. The bot's nearest own pickup
   re-phases a timer.
4. PPO's yaw-aware direction objective converts world pulls into Quake's real
   local basis: `forward=(cos(yaw), sin(yaw))`,
   `right=(sin(yaw), -cos(yaw))`. Engagement/opportunity attract, threat
   repels, and visible-combat/dead transitions are masked.
5. Every policy checkpoint gets a matching `lattice_<steps>.json.gz` plus
   `lattice_latest.json.gz`; `--resume` restores the matching per-bot cells.
6. `tools/evaluate_lattice.py` smoke-tests a map and direction-gates a policy.

## Verified prototype

An isolated WSL/RTX run used the evaluation `game.so`, disjoint ports, seed
7711, `Q2_EXT_OBS=1`, one generated map, and 32 real q2ded transitions. It
reported:

- lattice prior loaded rate: `1.0`;
- route graph loaded rate: `1.0`;
- live dynamic readiness cells: `112`;
- active route rate: `1.0`;
- direction samples per minibatch: `16`;
- policy and `lattice_00000032.json.gz` both saved.

A second isolated run resumed at step 32, restored 15 persisted cells, trained
to step 48, and emitted the next paired checkpoints. The production/active
trainer remained running and its runtime binary was not touched.

The existing aim PPO canary is a useful failing baseline for the new gate:

```text
opportunity_cosine=+0.002
engagement_cosine=+0.015
threat_cosine=+0.005
mean_cosine=+0.007  min_cosine=-0.984
```

It correctly fails `--require_mean_cosine 0.25`. In an isolated synthetic
learnability check, 240 optimizer steps over all three channels, two cardinal
targets, and four yaw angles moved mean cosine from approximately `0.000` to
`1.000`. That proves the objective and coordinate transform can teach the
contract; it does not replace real rollout and combat regression gates.

## Recommended first controlled run

Warm-start from the selected balanced-BC/PPO checkpoint and keep the lattice
objective small:

```bash
Q2_EXT_OBS=1 Q2_RUN_TAG=lattice_dir_s7711 \
python3 -m train.ppo --resume \
  --seed 7711 --game_seed 7711 --deterministic 1 \
  --vf_coef 0.1 --n_epochs 1 --lattice_direction_coef 0.02 \
  --map_glob 'mltrain_*.bsp'
```

Gate the resulting checkpoint with both:

```bash
Q2_EXT_OBS=1 python tools/evaluate_lattice.py \
  --checkpoint checkpoints/lattice_dir_s7711/policy_<steps>.pt \
  --require_mean_cosine 0.25
```

and the fixed generated-map combat gate from `README.md`. Run a coefficient-0
control with the same seeds before promoting anything.

## Known prototype limit

The engine observation does not expose visible item spawn IDs. Therefore the
timing table can learn this bot's nearest pickup and advance predicted respawns,
but cannot yet distinguish an enemy pickup from an unseen item. Adding that
field is the next item-timing fidelity step, not a blocker for this prototype.
